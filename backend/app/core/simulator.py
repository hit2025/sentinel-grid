"""
simulator.py
------------
Why this file exists: a judge should be able to clone the repo, run one
command, and watch every alert type fire *deterministically* -- without
needing us to source a real border-CCTV dataset (which is restricted/
sensitive by nature, exactly per the problem statement) or without the
demo's outcome depending on a shaky webcam feed during a live pitch.

This module scripts a small cast of synthetic "actors" (as Detection-style
centroids, frame by frame) that:
  - actor 1: loiters inside the restricted buffer zone past the threshold
  - actors 2-4: walk independently, converge into a group near the fence,
    then disperse -- exercising DBSCAN-based group detection
  - actor 5: walks up to the border line and crosses it in the
    unauthorized direction
  - actor 6: sprints erratically across the frame

The exact same PersonTracker + GroupBehaviorAnalyzer + alerting code path
used for real detector output consumes this synthetic stream, so the
"intelligence" being demonstrated is identical to production -- only the
detector is swapped for a scripted source. In a real deployment this file
is deleted; nothing else changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from app.core.detector import Detection


@dataclass
class Actor:
    actor_id: int
    path_fn: Callable[[int], tuple]  # frame_index -> (x, y)
    label: str = "person"
    active_range: tuple = (0, 10_000)  # (start_frame, end_frame)

    def position_at(self, frame_index: int):
        if frame_index < self.active_range[0] or frame_index > self.active_range[1]:
            return None
        return self.path_fn(frame_index - self.active_range[0])


def _line_crosser_path(line, start_offset: float = 220.0, speed: float = 4.5):
    """A path_fn that walks straight through the GIVEN line, starting
    comfortably on its "unauthorized_side" start side and ending on the
    end side, crossing near its midpoint -- computed from the line's own
    geometry (p1/p2/side_of), not assumed to be vertical or in any fixed
    position. This is what makes a freshly-calibrated line get walked
    across automatically, the same way the original hardcoded default
    line always did, instead of needing a manual Test click because the
    scripted actors had no way to know a new line existed."""
    (x1, y1), (x2, y2) = line.p1, line.p2
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1.0
    px, py = -dy / length, dx / length
    start_side, _, end_side = line.unauthorized_side.partition("_to_")
    # Query the line's OWN side_of() to find which perpendicular
    # direction is actually the "start" side -- same technique already
    # proven correct for the calibration Test button, works regardless
    # of the real BorderLine implementation's sign convention.
    side_at_plus = line.side_of((mx + px, my + py))
    sign = 1.0 if side_at_plus == start_side else -1.0
    start_x, start_y = mx + sign * px * start_offset, my + sign * py * start_offset
    end_x, end_y = mx - sign * px * start_offset, my - sign * py * start_offset
    total = math.hypot(end_x - start_x, end_y - start_y)
    steps = max(1.0, total / speed)

    def fn(f):
        t = min(1.0, f / steps)
        return (start_x + (end_x - start_x) * t, start_y + (end_y - start_y) * t)
    return fn


def build_demo_actors(frame_w: int, frame_h: int, lines: Optional[list] = None,
                       zones: Optional[list] = None) -> List[Actor]:
    fence_x = frame_w * 0.55  # fallback used only when no custom line/zone is configured

    # ---- loiterer: targets the CURRENTLY ACTIVE zone's own centroid
    # when one exists (default or operator-calibrated) -- previously
    # always loitered relative to a hardcoded fence_x offset, completely
    # unaware that a zone might have been calibrated somewhere else.
    if zones:
        zone = zones[-1]
        zx = sum(p[0] for p in zone.polygon) / len(zone.polygon)
        zy = sum(p[1] for p in zone.polygon) / len(zone.polygon)
    else:
        zx, zy = fence_x + 60, frame_h * 0.4

    def loiterer(f):  # stands near the active zone's centroid, small jitter
        return (zx + 6 * math.sin(f * 0.15), zy + 4 * math.cos(f * 0.2))

    def wanderer(offset_y, speed):
        def fn(f):
            x = 40 + (speed * f) % (fence_x - 80)
            return (x, offset_y)
        return fn

    def group_member(start_x, start_y, target_x, target_y, start_frame_local, speed=3.0):
        def fn(f):
            t = max(0, f - start_frame_local)
            dx, dy = target_x - start_x, target_y - start_y
            dist = math.hypot(dx, dy)
            steps = max(1, dist / speed)
            progress = min(1.0, t / steps)
            return (start_x + dx * progress, start_y + dy * progress)
        return fn

    # ---- crosser: walks straight through the CURRENTLY ACTIVE border
    # line (default or operator-calibrated) -- previously a hardcoded
    # horizontal walk toward a fixed fence_x, which only ever crossed a
    # vertical line sitting at exactly that x position (i.e. only the
    # original default line). A line drawn anywhere else, at any angle,
    # was never actually walked across by anything.
    if lines:
        crosser_fn = _line_crosser_path(lines[-1])
    else:
        def crosser_fn(f):  # approaches then crosses the fence line left -> right
            x = 30 + 4.5 * f
            return (min(x, frame_w - 20), frame_h * 0.7)

    def sprinter(f):  # fast erratic diagonal run
        x = (frame_w * 0.1 + 14 * f) % frame_w
        y = (frame_h * 0.2 + 9 * math.sin(f * 0.3) * 10 + f * 1.2) % frame_h
        return (x, y)

    def vehicle_road(f):  # drives along a road strip near the bottom of frame
        x = (20 + 3.2 * f) % (frame_w + 160) - 80
        return (x, frame_h * 0.88)

    actors = [
        Actor(1, loiterer, active_range=(0, 400)),
        Actor(2, wanderer(frame_h * 0.2, 2.0), active_range=(0, 400)),
        Actor(3, group_member(frame_w * 0.35, frame_h * 0.55, fence_x - 40, frame_h * 0.55, 60),
              active_range=(0, 400)),
        Actor(4, group_member(frame_w * 0.30, frame_h * 0.65, fence_x - 60, frame_h * 0.6, 60),
              active_range=(0, 400)),
        Actor(5, group_member(frame_w * 0.25, frame_h * 0.60, fence_x - 30, frame_h * 0.58, 60),
              active_range=(0, 400)),
        Actor(6, crosser_fn, active_range=(150, 400)),
        Actor(7, sprinter, active_range=(200, 400)),
        Actor(8, vehicle_road, label="vehicle", active_range=(0, 400)),
    ]
    return actors


def frame_detections(actors: List[Actor], frame_index: int, box_size: int = 40) -> List[Detection]:
    dets: List[Detection] = []
    for actor in actors:
        pos = actor.position_at(frame_index)
        if pos is None:
            continue
        x, y = pos
        if actor.label == "vehicle":
            half_w, half_h = 55, 26
            dets.append(Detection(
                x1=int(x - half_w), y1=int(y - half_h),
                x2=int(x + half_w), y2=int(y + half_h),
                confidence=0.9, label="vehicle",
            ))
        else:
            half = box_size // 2
            dets.append(Detection(
                x1=int(x - half), y1=int(y - box_size),
                x2=int(x + half), y2=int(y),
                confidence=0.95, label=actor.label,
            ))
    return dets


NIGHT_SEGMENT = (280, 360)  # frame range where the demo scenario simulates low light


def is_scripted_night(frame_index: int) -> bool:
    return NIGHT_SEGMENT[0] <= frame_index <= NIGHT_SEGMENT[1]

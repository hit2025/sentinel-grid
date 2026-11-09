"""
cross_camera.py
----------------
A deliberately lightweight cross-camera appearance matching heuristic --
NOT real re-identification. Real person/vehicle re-ID uses a learned deep
embedding (trained on paired-camera datasets) that's robust to lighting,
pose, and viewpoint changes. This uses an HSV color histogram of each
track's bounding-box crop instead: cheap, needs no model download or
GPU, and good enough to demo the *concept* of "is this the same vehicle
seen on a different camera" -- but it WILL confuse two objects that
simply happen to be similarly colored (two white sedans, two people in
dark clothing), and it has no notion of shape, texture, or identity.
Documented here and in the API response itself as a heuristic, not a
capability claim.

Why this belongs in its own module rather than inside pipeline.py:
matching happens ACROSS two independent CameraSession instances, so it
naturally sits a level above any single session's own pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class TrackSignature:
    camera_id: str
    track_id: int
    label: str
    histogram: np.ndarray


@dataclass
class CrossCameraMatch:
    camera_a: str
    track_id_a: int
    camera_b: str
    track_id_b: int
    label: str
    similarity: float


def _extract_histogram(frame: np.ndarray, bbox: tuple) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    # A small blur before histogramming matters more than it looks: a
    # near-uniform crop (a plain-coloured car door, a shirt) produces a
    # histogram that's almost a single-bin spike, and two genuinely
    # similar colours can still land in ADJACENT quantization bins
    # (e.g. saturation 222 vs 229 with 8-wide bins) -- comparison metrics
    # then see zero overlap between two "spikes" that are visually
    # nearly identical. Blurring spreads each colour's mass across
    # neighbouring bins before comparison, which is what makes the match
    # robust to this instead of just being lucky about bin edges.
    crop = cv2.GaussianBlur(crop, (5, 5), 0)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Hue + Saturation only (skip Value/brightness) -- makes the match
    # somewhat more tolerant of the two cameras having different
    # exposure/lighting, which is otherwise the single biggest reason a
    # naive histogram match fails across real cameras. Coarser bins
    # (16x16, not 30x32) for the same reason as the blur above.
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten()


def _session_signatures(camera_id: str, frame: np.ndarray, tracks: list) -> List[TrackSignature]:
    sigs = []
    for t in tracks:
        hist = _extract_histogram(frame, t.bbox)
        if hist is not None:
            sigs.append(TrackSignature(camera_id, t.track_id, t.label, hist))
    return sigs


def find_cross_camera_matches(
    camera_a_id: str, frame_a: Optional[np.ndarray], tracks_a: list,
    camera_b_id: str, frame_b: Optional[np.ndarray], tracks_b: list,
    threshold: float = 0.65,
) -> List[CrossCameraMatch]:
    """Compares every current track in session A against every current
    track in session B (same label only -- a person is never "matched"
    to a vehicle) via HSV histogram correlation, returning pairs above
    `threshold`. Returns [] immediately, rather than raising, if either
    session has no cached frame yet (nothing has been streamed from it
    this run)."""
    if frame_a is None or frame_b is None:
        return []

    sigs_a = _session_signatures(camera_a_id, frame_a, tracks_a)
    sigs_b = _session_signatures(camera_b_id, frame_b, tracks_b)

    matches = []
    for sa in sigs_a:
        for sb in sigs_b:
            if sa.label != sb.label:
                continue
            score = float(cv2.compareHist(sa.histogram, sb.histogram, cv2.HISTCMP_CORREL))
            if score >= threshold:
                matches.append(CrossCameraMatch(
                    camera_a=sa.camera_id, track_id_a=sa.track_id,
                    camera_b=sb.camera_id, track_id_b=sb.track_id,
                    label=sa.label, similarity=round(score, 3),
                ))
    matches.sort(key=lambda m: -m.similarity)
    return matches

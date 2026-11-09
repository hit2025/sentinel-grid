"""
zones.py
--------
Geometry primitives for the two spatial concepts a border-surveillance
deployment actually needs:

  * Zone       - a polygon (e.g. "restricted buffer strip") -- used for
                 dwell-time / loitering analysis and intrusion detection.
  * BorderLine - a directed line segment (e.g. the fence line / LoC marker)
                 -- used for crossing-direction detection (which side ->
                 which side, and which direction is "unauthorized").

Both are configured at runtime via the API (`/api/zones`, `/api/lines`) and
persisted to SQLite, so operators calibrate a new camera without touching
code -- a real requirement the MHA problem statement calls out (existing,
heterogeneous CCTV infrastructure).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


Point = Tuple[float, float]


@dataclass
class Zone:
    zone_id: int
    name: str
    polygon: List[Point]
    zone_type: str = "restricted"  # restricted | loiter_watch | general

    def contains(self, point: Point) -> bool:
        return _point_in_polygon(point, self.polygon)


@dataclass
class BorderLine:
    line_id: int
    name: str
    p1: Point
    p2: Point
    unauthorized_side: str = "b_to_a"  # which crossing direction raises an alert

    def side_of(self, point: Point) -> str:
        """Returns 'a' or 'b' depending which side of the line the point is on."""
        (x1, y1), (x2, y2) = self.p1, self.p2
        cross = (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1)
        return "a" if cross >= 0 else "b"


def _point_in_polygon(point: Point, polygon: List[Point]) -> bool:
    """Standard ray-casting point-in-polygon test."""
    x, y = point
    n = len(polygon)
    inside = False
    px1, py1 = polygon[0]
    for i in range(1, n + 1):
        px2, py2 = polygon[i % n]
        if y > min(py1, py2):
            if y <= max(py1, py2):
                if x <= max(px1, px2):
                    if py1 != py2:
                        xinters = (y - py1) * (px2 - px1) / (py2 - py1) + px1
                    if px1 == px2 or x <= xinters:
                        inside = not inside
        px1, py1 = px2, py2
    return inside


def default_demo_geometry(frame_w: int, frame_h: int) -> Tuple[List[Zone], List[BorderLine]]:
    """A reasonable default layout used when no operator calibration exists
    yet, so the live demo has *something* meaningful to alert on."""
    restricted = Zone(
        zone_id=1,
        name="Restricted Buffer Strip",
        polygon=[
            (frame_w * 0.55, 0),
            (frame_w, 0),
            (frame_w, frame_h),
            (frame_w * 0.55, frame_h),
        ],
        zone_type="restricted",
    )
    border = BorderLine(
        line_id=1,
        name="Border Fence Line",
        p1=(frame_w * 0.55, 0),
        p2=(frame_w * 0.55, frame_h),
        unauthorized_side="a_to_b",
    )
    return [restricted], [border]

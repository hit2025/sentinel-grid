"""
tracker.py
----------
Lightweight multi-object tracker that assigns a persistent track ID to each
detected person across frames using centroid distance + Hungarian assignment.

Why not DeepSORT/ByteTrack here? Those need an appearance embedding model
(extra weights, extra latency) which is overkill for a hackathon-tier CCTV
feed where cameras are mostly static and frame rate is modest. This tracker
is deliberately dependency-light (numpy + scipy only) so it runs anywhere,
and it's what actually powers the "group behavior" and "loitering" analytics
below -- both need stable IDs and per-track history, not perfect re-ID.

Swap-in note for production: this class implements the same
`update(detections) -> List[Track]` interface a DeepSORT wrapper would, so
upgrading later is a one-file change.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.core.detector import Detection


@dataclass
class TrackPoint:
    x: float
    y: float
    t: float


@dataclass
class Track:
    track_id: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    label: str
    first_seen: float
    last_seen: float
    misses: int = 0
    history: Deque[TrackPoint] = field(default_factory=lambda: deque(maxlen=90))

    def age_seconds(self) -> float:
        return self.last_seen - self.first_seen

    def speed_px_per_s(self) -> float:
        if len(self.history) < 2:
            return 0.0
        p0, p1 = self.history[-2], self.history[-1]
        dt = max(1e-3, p1.t - p0.t)
        dist = ((p1.x - p0.x) ** 2 + (p1.y - p0.y) ** 2) ** 0.5
        return dist / dt

    def heading_deg(self) -> float:
        if len(self.history) < 2:
            return 0.0
        p0, p1 = self.history[-2], self.history[-1]
        return float(np.degrees(np.arctan2(p1.y - p0.y, p1.x - p0.x)))


class CentroidTracker:
    def __init__(self, max_distance: float = 90.0, max_misses: int = 15):
        self.max_distance = max_distance
        self.max_misses = max_misses
        self._next_id = 1
        self.tracks: Dict[int, Track] = {}

    def update(self, detections: List[Detection], timestamp: float | None = None) -> List[Track]:
        ts = timestamp if timestamp is not None else time.time()

        if not self.tracks:
            for det in detections:
                self._register(det, ts)
            return list(self.tracks.values())

        track_ids = list(self.tracks.keys())
        track_centroids = np.array([self.tracks[tid].centroid for tid in track_ids])

        if not detections:
            self._age_all(track_ids, ts)
            return list(self.tracks.values())

        det_centroids = np.array([d.centroid for d in detections])

        # Cost matrix: Euclidean distance between existing tracks and new detections.
        cost = np.linalg.norm(
            track_centroids[:, None, :] - det_centroids[None, :, :], axis=2
        )
        row_ind, col_ind = linear_sum_assignment(cost)

        assigned_tracks, assigned_dets = set(), set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= self.max_distance:
                tid = track_ids[r]
                self._update_track(tid, detections[c], ts)
                assigned_tracks.add(tid)
                assigned_dets.add(c)

        # Unmatched existing tracks age out.
        for tid in track_ids:
            if tid not in assigned_tracks:
                self.tracks[tid].misses += 1

        # Unmatched detections become new tracks.
        for i, det in enumerate(detections):
            if i not in assigned_dets:
                self._register(det, ts)

        self.tracks = {
            tid: t for tid, t in self.tracks.items() if t.misses <= self.max_misses
        }
        return list(self.tracks.values())

    def _register(self, det: Detection, ts: float) -> None:
        tid = self._next_id
        self._next_id += 1
        t = Track(
            track_id=tid,
            centroid=det.centroid,
            bbox=det.bbox,
            label=det.label,
            first_seen=ts,
            last_seen=ts,
        )
        t.history.append(TrackPoint(det.centroid[0], det.centroid[1], ts))
        self.tracks[tid] = t

    def _update_track(self, tid: int, det: Detection, ts: float) -> None:
        t = self.tracks[tid]
        t.centroid = det.centroid
        t.bbox = det.bbox
        t.last_seen = ts
        t.misses = 0
        t.history.append(TrackPoint(det.centroid[0], det.centroid[1], ts))

    def _age_all(self, track_ids: List[int], ts: float) -> None:
        for tid in track_ids:
            self.tracks[tid].misses += 1
        self.tracks = {
            tid: t for tid, t in self.tracks.items() if t.misses <= self.max_misses
        }

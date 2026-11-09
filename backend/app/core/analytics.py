"""
analytics.py
------------
The actual "intelligence" layer of SentinelGrid, and the direct translation
of group-behavior-analysis-in-video-surveillance research into a deployable
rule+ML hybrid. Given a frame's set of tracked people, this module answers
four questions every border/perimeter watch commander actually asks:

  1. Is anyone loitering (dwelling) near the fence beyond a safe threshold?
  2. Did anyone cross the line in the unauthorized direction?
  3. Are people converging into an unusual GROUP near a sensitive zone --
     smuggling/infiltration is rarely a lone-person event, it clusters?
  4. Is anyone moving erratically (sudden sprint, sudden stop, reversal) --
     often the tell for "spotted, now running" or coordinated diversion?

Group detection uses density-based spatial clustering (DBSCAN) on live
centroids, which is exactly the technique behind crowd/group-behavior
literature (grouping by proximity + persistence rather than a fixed
headcount threshold), so the demo doubles as a legitimate research artifact
for the sponsoring authors' Group Behavior Analysis background.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.cluster import DBSCAN

from app.core.tracker import Track
from app.core.zones import BorderLine, Zone


@dataclass
class Alert:
    alert_type: str          # loitering | unauthorized_crossing | group_formation | erratic_motion | night_time_movement
    severity: str            # low | medium | high | critical
    message: str
    track_ids: List[int]
    timestamp: float = field(default_factory=time.time)
    camera_id: str = "cam-01"
    location: Optional[tuple] = None
    # Database row id, set by storage.save_alert() AFTER insert (via
    # pipeline.py's _persist()) -- None until then. Added as the LAST
    # field specifically so every existing call site that constructs an
    # Alert with positional args (alert_type, severity, message,
    # track_ids, timestamp, camera_id) keeps working unchanged; this is
    # what lets the live dashboard's "Acknowledge" button target a real
    # alert id instead of being permanently hidden.
    id: Optional[int] = None
    # Content-address hash of this alert's evidence snapshot (see
    # storage.save_snapshot_file / pipeline._persist), set the same way
    # and for the same reason as `id` above -- lets the live dashboard
    # show a "View Evidence" link the moment the alert arrives, without
    # a second round-trip to look it up.
    snapshot_hash: Optional[str] = None


class GroupBehaviorAnalyzer:
    def __init__(
        self,
        loiter_seconds: float = 8.0,
        group_eps_px: float = 80.0,
        group_min_size: int = 3,
        group_persist_seconds: float = 3.0,
        erratic_speed_px_s: float = 260.0,
    ):
        self.loiter_seconds = loiter_seconds
        self.group_eps_px = group_eps_px
        self.group_min_size = group_min_size
        self.group_persist_seconds = group_persist_seconds
        self.erratic_speed_px_s = erratic_speed_px_s

        # Track-side state that persists across frames.
        self._zone_entry_time: Dict[int, float] = {}
        self._last_side: Dict[int, str] = {}
        self._group_since: Optional[float] = None
        self._alerted_crossings: set = set()
        self._alert_cooldown: Dict[str, float] = {}
        self._cooldown_seconds = 10.0

    # -- public API ---------------------------------------------------
    def analyze(
        self,
        tracks: List[Track],
        zones: List[Zone],
        lines: List[BorderLine],
        camera_id: str = "cam-01",
        timestamp: Optional[float] = None,
    ) -> List[Alert]:
        # A virtual clock (frame_index / fps, passed in by the pipeline) is
        # used instead of wall-clock time.time() so that dwell/persistence
        # thresholds are correct whether frames arrive live at 12 fps or are
        # batch-processed from a video file in a fraction of a second.
        now = timestamp if timestamp is not None else time.time()
        alerts: List[Alert] = []

        alerts += self._check_loitering(tracks, zones, now, camera_id)
        alerts += self._check_crossings(tracks, lines, now, camera_id)
        alerts += self._check_group_formation(tracks, zones, now, camera_id)
        alerts += self._check_erratic_motion(tracks, now, camera_id)

        alerts = [a for a in alerts if self._pass_cooldown(a, now)]
        for a in alerts:
            a.timestamp = now
        return alerts

    # -- individual detectors ------------------------------------------
    def _check_loitering(self, tracks, zones, now, camera_id) -> List[Alert]:
        alerts = []
        active_ids = set()
        for t in tracks:
            in_watch_zone = any(z.contains(t.centroid) for z in zones)
            if in_watch_zone:
                active_ids.add(t.track_id)
                entered = self._zone_entry_time.setdefault(t.track_id, t.first_seen)
                dwell = now - entered
                if dwell >= self.loiter_seconds:
                    alerts.append(Alert(
                        alert_type="loitering",
                        severity="medium" if dwell < self.loiter_seconds * 2 else "high",
                        message=(f"{t.label.capitalize()} #{t.track_id} loitering in restricted zone "
                                 f"for {dwell:.0f}s"),
                        track_ids=[t.track_id],
                        camera_id=camera_id,
                        location=t.centroid,
                    ))
        # forget people who left the zone
        for tid in list(self._zone_entry_time):
            if tid not in active_ids:
                self._zone_entry_time.pop(tid, None)
        return alerts

    def _check_crossings(self, tracks, lines, now, camera_id) -> List[Alert]:
        alerts = []
        for t in tracks:
            for line in lines:
                side = line.side_of(t.centroid)
                key = (t.track_id, line.line_id)
                prev = self._last_side.get(key)
                self._last_side[key] = side
                if prev is None or prev == side:
                    continue
                direction = f"{prev}_to_{side}"
                if direction == line.unauthorized_side and key not in self._alerted_crossings:
                    self._alerted_crossings.add(key)
                    alerts.append(Alert(
                        alert_type="unauthorized_crossing",
                        severity="critical",
                        message=(f"{t.label.capitalize()} #{t.track_id} crossed '{line.name}' "
                                 f"in unauthorized direction"),
                        track_ids=[t.track_id],
                        camera_id=camera_id,
                        location=t.centroid,
                    ))
        return alerts

    def _check_group_formation(self, tracks, zones, now, camera_id) -> List[Alert]:
        alerts = []
        if len(tracks) < self.group_min_size:
            self._group_since = None
            return alerts

        pts = np.array([t.centroid for t in tracks])
        labels = DBSCAN(eps=self.group_eps_px, min_samples=self.group_min_size).fit_predict(pts)

        cluster_ids = set(labels[labels >= 0])
        if not cluster_ids:
            self._group_since = None
            return alerts

        for cid in cluster_ids:
            members = [tracks[i] for i in range(len(tracks)) if labels[i] == cid]
            centroid = tuple(np.mean([m.centroid for m in members], axis=0))
            near_sensitive_zone = any(z.contains(centroid) for z in zones)
            label = members[0].label

            self._group_since = self._group_since or now
            persisted = now - self._group_since
            if persisted >= self.group_persist_seconds:
                noun = "convoy" if label == "vehicle" else f"cluster of {label}s"
                alerts.append(Alert(
                    alert_type="group_formation",
                    severity="high" if near_sensitive_zone else "medium",
                    message=(f"Unusual {noun} ({len(members)}) formed"
                             + (" near restricted zone" if near_sensitive_zone else "")),
                    track_ids=[m.track_id for m in members],
                    camera_id=camera_id,
                    location=centroid,
                ))
        return alerts

    def _check_erratic_motion(self, tracks, now, camera_id) -> List[Alert]:
        alerts = []
        for t in tracks:
            speed = t.speed_px_per_s()
            if speed >= self.erratic_speed_px_s:
                alerts.append(Alert(
                    alert_type="erratic_motion",
                    severity="medium",
                    message=f"{t.label.capitalize()} #{t.track_id} moving erratically ({speed:.0f} px/s)",
                    track_ids=[t.track_id],
                    camera_id=camera_id,
                    location=t.centroid,
                ))
        return alerts

    def _pass_cooldown(self, alert: "Alert", now: float) -> bool:
        key = f"{alert.alert_type}:{','.join(map(str, alert.track_ids))}"
        last = self._alert_cooldown.get(key)
        if last is not None and now - last < self._cooldown_seconds:
            return False
        self._alert_cooldown[key] = now
        return True

    # -- night-time movement (called separately by the pipeline, which is
    # the one place that knows whether the current frame is dark) --------
    def check_night_movement(self, tracks: List[Track], camera_id: str,
                              timestamp: float) -> List[Alert]:
        alerts = []
        for t in tracks:
            alert = Alert(
                alert_type="night_time_movement",
                severity="high",  # night activity at a border post is inherently higher-risk
                message=f"{t.label.capitalize()} #{t.track_id} detected during low-light conditions",
                track_ids=[t.track_id],
                camera_id=camera_id,
                location=t.centroid,
                timestamp=timestamp,
            )
            if self._pass_cooldown(alert, timestamp):
                alerts.append(alert)
        return alerts

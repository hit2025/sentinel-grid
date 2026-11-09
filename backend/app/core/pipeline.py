"""
pipeline.py
-----------
Orchestrates one camera session end to end, covering all eight capabilities
listed in the official SIH26187 spec:

    human detection+tracking, vehicle detection+classification, face
    detection, ANPR, virtual fence intrusion, suspicious activity
    detection, night-time movement detection, real-time alerts + logging

    frame -> [night check + enhancement]
           -> person detector -> person tracker  -\
           -> vehicle detector -> vehicle tracker  >-- GroupBehaviorAnalyzer -> alerts
                                                   /
           -> face detector (real frames only)    -- events
           -> ANPR on vehicle crops                -- events
                                  -> annotated frame (for the live view)
                                  -> storage (SQLite: alerts + events)

Two entry points:
  * CameraSession.process_upload(path) - batch-processes an uploaded video
    file end to end, returns a summary + all alerts. The "run it on real
    footage" path -- this is where face detection actually runs, since
    synthetic simulation frames have no real face texture to find.
  * CameraSession.step_simulation()    - advances one tick of the
    deterministic demo scenario; used by the websocket live view. Person
    behavior (loitering/crossing/group/erratic) is scripted directly as
    track positions; the vehicle + its plate are rendered as real pixels
    onto the canvas, so vehicle detection and ANPR are genuinely running
    detection/OCR algorithms even in "simulation" mode -- not mocked.

Both paths share the same analyzer classes, so whatever the demo shows is
exactly what real footage would produce.
"""

from __future__ import annotations

import base64
import re
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.core.analytics import Alert, GroupBehaviorAnalyzer
from app.core.anpr import PlateReader
from app.core.detector import DetectorBackend, build_detector
from app.core.face import FaceDetector
from app.core.night import enhance_for_detection, is_night
from app.core.simulator import build_demo_actors, frame_detections, is_scripted_night
from app.core.storage import save_alert, save_event, save_snapshot_file, save_stat
from app.core.tracker import CentroidTracker, Track
from app.core.vehicle import BlobVehicleDetector, VehicleDetectorBackend, build_vehicle_detector
from app.core.zones import default_demo_geometry

COLORS = {
    "person": (60, 200, 60),
    "vehicle": (255, 190, 60),
    "face": (255, 255, 255),
    "loitering": (0, 165, 255),
    "unauthorized_crossing": (0, 0, 255),
    "group_formation": (0, 215, 255),
    "erratic_motion": (255, 0, 255),
    "night_time_movement": (200, 130, 255),
}

# Indian number-plate formats (HSRP), used to flag an OCR read as
# format-verified vs. not -- this is the concrete next step named on the
# project's own Feasibility slide ("ANPR reads are not yet validated
# against Indian plate formats"), now actually closed. Two patterns:
# the standard state-code format (e.g. DL1CN4521, MH12AB1234) and the
# newer BH ("Bharat") series introduced in 2021 (e.g. 21BH1234AB).
# Deliberately just a format check, not a checksum/RTO-database lookup --
# it catches OCR noise (wrong character count, garbled characters) but
# cannot confirm the plate is real or currently registered.
_INDIAN_PLATE_STANDARD = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")
_INDIAN_PLATE_BH_SERIES = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{2}$")


def is_valid_indian_plate_format(text: str) -> bool:
    cleaned = re.sub(r"[\s\-]", "", text.upper())
    return bool(_INDIAN_PLATE_STANDARD.match(cleaned) or _INDIAN_PLATE_BH_SERIES.match(cleaned))


class CameraSession:
    """Holds all mutable state for one camera feed: trackers, analyzers,
    zone config, per-track caches, frame counters. One instance per active
    camera/session."""

    def __init__(self, camera_id: str = "cam-01", frame_w: int = 960, frame_h: int = 540,
                 detector_backend: str = "auto", fps: float = 12.0):
        self.camera_id = camera_id
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.zones, self.lines = default_demo_geometry(frame_w, frame_h)
        self.frame_index = 0
        self.fps = fps
        self.total_alerts_fired = 0
        self.total_events_logged = 0

        # human pipeline
        self.tracker = CentroidTracker()
        self.analyzer = GroupBehaviorAnalyzer()
        self._detector: Optional[DetectorBackend] = None
        self._detector_backend_name = detector_backend

        # vehicle pipeline (looser group threshold -> "convoy" of 2+, higher
        # erratic-speed threshold since vehicles are naturally faster than
        # a walking/running person)
        self.vehicle_tracker = CentroidTracker(max_distance=140.0)
        self.vehicle_analyzer = GroupBehaviorAnalyzer(
            group_min_size=2, group_persist_seconds=2.0, erratic_speed_px_s=550.0,
            loiter_seconds=10.0,
        )
        self._vehicle_detector: Optional[VehicleDetectorBackend] = None
        # Simulation mode always uses the blob detector, regardless of what
        # production backend is configured for real footage. A real object
        # detector (YOLO) correctly refuses to call a plain rendered
        # rectangle a "car" -- it has no real car features to recognize --
        # which would silently break the synthetic demo's vehicle/ANPR
        # story the moment YOLO is installed. The two are genuinely
        # different jobs: the simulation's own simplified rendering needs
        # a detector built for that rendering; real footage needs the best
        # available real-world detector.
        self._sim_vehicle_detector = BlobVehicleDetector()

        # face + ANPR (stateless detectors, cheap to build eagerly)
        self.face_detector = FaceDetector()
        self.plate_reader = PlateReader()
        self._plate_cache: dict[int, str] = {}
        self._face_logged_cooldown: dict[int, float] = {}
        # Tracks which (label, track_id) pairs have already produced a
        # "first seen" detection-log entry -- without this, every person
        # and vehicle would log a new event on EVERY frame they're
        # tracked in, flooding the log instead of showing one clean entry
        # per object that actually appeared. This is what makes
        # "Detection log" genuinely show everything detected (person,
        # vehicle, face, plate), not just faces/ANPR.
        self._detection_logged: set[tuple[str, int]] = set()

        self._sim_actors = build_demo_actors(frame_w, frame_h, lines=self.lines, zones=self.zones)
        self.night_mode = False
        # Last rendered frame, cached for cross_camera.py's appearance
        # matching -- it needs an actual frame to crop track bounding
        # boxes out of, and sessions don't otherwise keep one around
        # after returning it to the caller.
        self._last_frame: Optional[np.ndarray] = None

    def _virtual_time(self) -> float:
        return self.frame_index / self.fps

    def rebuild_sim_actors(self) -> None:
        """Called after self.zones/self.lines change (see main.py's
        set_zones()) so the scripted actors immediately start walking
        toward the NEWLY calibrated line/zone instead of whatever was
        active when the session was first created. Without this, a
        freshly-drawn line would only ever be crossed by the Test button,
        never by the simulation actually running -- which is the entire
        point of making the actor paths geometry-aware in the first
        place. frame_index is deliberately left untouched: only the path
        targets change, not where the demo currently is in its loop."""
        self._sim_actors = build_demo_actors(self.frame_w, self.frame_h, lines=self.lines, zones=self.zones)

    @property
    def detector(self) -> DetectorBackend:
        if self._detector is None:
            self._detector = build_detector(self._detector_backend_name)
        return self._detector

    @property
    def vehicle_detector(self) -> VehicleDetectorBackend:
        if self._vehicle_detector is None:
            self._vehicle_detector = build_vehicle_detector(self._detector_backend_name)
        return self._vehicle_detector

    # ---- simulation mode --------------------------------------------
    def step_simulation(self) -> Tuple[np.ndarray, List[Track], List[Alert]]:
        vt = self._virtual_time()
        scripted_night = is_scripted_night(self.frame_index)

        canvas = self._blank_canvas(dim=scripted_night)
        self._draw_zones(canvas)
        self._draw_sim_vehicle(canvas)  # real pixels: body + plate text

        # vehicle detection genuinely runs MOG2 + contour analysis on the
        # canvas we just rendered -- not a shortcut. Deliberately uses the
        # simulation-only blob detector (see __init__) even if a YOLO
        # backend is configured for real footage -- see the comment there.
        vehicle_dets = self._sim_vehicle_detector.detect(canvas)
        vehicle_tracks = self.vehicle_tracker.update(vehicle_dets, timestamp=vt)

        # person behavior is scripted directly as track positions (see
        # module docstring for why: HOG can't detect abstract synthetic
        # shapes as people, and behavior-over-time is what's being
        # demonstrated here, not detection accuracy).
        person_dets = [d for d in frame_detections(self._sim_actors, self.frame_index)
                       if d.label != "vehicle"]
        person_tracks = self.tracker.update(person_dets, timestamp=vt)

        alerts = self._run_analytics(person_tracks, vehicle_tracks, vt, night=scripted_night)
        self._read_plates(vehicle_tracks, canvas, vt)
        self._log_new_detections(person_tracks, "person", vt)
        self._log_new_detections(vehicle_tracks, "vehicle", vt)

        self._draw_tracks(canvas, person_tracks, "person")
        self._draw_tracks(canvas, vehicle_tracks, "vehicle")
        self._draw_night_badge(canvas, scripted_night)
        self._draw_alert_banners(canvas, alerts)

        self._persist(alerts, len(person_tracks) + len(vehicle_tracks),
                      vehicle_count=len(vehicle_tracks), evidence_frame=canvas)
        self.frame_index += 1

        if self.frame_index > 420:  # loop the scripted scenario
            self._reset_session_state()

        self._last_frame = canvas
        return canvas, person_tracks + vehicle_tracks, alerts

    # ---- real video mode ----------------------------------------------
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Track], List[Alert]]:
        vt = self._virtual_time()

        # Guard against degenerate frames -- a solid black/uniform frame
        # (end-of-clip padding, a dropped frame, or real camera signal
        # loss in production) has near-zero pixel variance. Running the
        # full detector/analytics pipeline on one produces meaningless
        # results -- this is exactly what caused a false
        # 'night_time_movement' alert on a real daytime clip: its own
        # final frame was a solid black artifact, which is legitimately
        # "dark" by brightness alone but isn't a real scene at all.
        if float(frame.std()) < 3.0:
            annotated = frame.copy()
            cv2.putText(annotated, "LOW-SIGNAL FRAME - analysis skipped", (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
            self.frame_index += 1
            return annotated, [], []

        night = is_night(frame)
        detect_input = enhance_for_detection(frame) if night else frame

        person_dets = self.detector.detect(detect_input)
        person_tracks = self.tracker.update(person_dets, timestamp=vt)

        vehicle_dets = self.vehicle_detector.detect(detect_input)
        vehicle_tracks = self.vehicle_tracker.update(vehicle_dets, timestamp=vt)

        alerts = self._run_analytics(person_tracks, vehicle_tracks, vt, night=night)
        self._read_plates(vehicle_tracks, frame, vt)
        self._detect_faces(person_tracks, frame, vt)
        self._log_new_detections(person_tracks, "person", vt)
        self._log_new_detections(vehicle_tracks, "vehicle", vt)

        annotated = frame.copy()
        self._draw_zones(annotated)
        self._draw_tracks(annotated, person_tracks, "person")
        self._draw_tracks(annotated, vehicle_tracks, "vehicle")
        self._draw_night_badge(annotated, night)
        self._draw_alert_banners(annotated, alerts)

        self._persist(alerts, len(person_tracks) + len(vehicle_tracks),
                      vehicle_count=len(vehicle_tracks), evidence_frame=annotated)
        self.frame_index += 1
        self._last_frame = annotated
        return annotated, person_tracks + vehicle_tracks, alerts

    def process_upload(self, path: str, sample_every: int = 2, max_frames: int = 600) -> dict:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError("Could not open uploaded video")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.frame_w
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.frame_h
        self.frame_w, self.frame_h = w, h
        self.zones, self.lines = default_demo_geometry(w, h)

        all_alerts: List[Alert] = []
        processed = 0
        idx = 0
        while processed < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % sample_every == 0:
                _, _, alerts = self.process_frame(frame)
                all_alerts.extend(alerts)
                processed += 1
            idx += 1
        cap.release()

        return {
            "frames_processed": processed,
            "alerts_raised": len(all_alerts),
            "events_logged": self.total_events_logged,
            "plates_read": list(set(self._plate_cache.values())),
            "alerts": [
                {
                    "type": a.alert_type, "severity": a.severity,
                    "message": a.message, "track_ids": a.track_ids,
                    "timestamp": a.timestamp,
                }
                for a in all_alerts
            ],
        }

    # ---- shared analytics + logging -----------------------------------
    def _run_analytics(self, person_tracks, vehicle_tracks, vt, night: bool) -> List[Alert]:
        alerts = self.analyzer.analyze(person_tracks, self.zones, self.lines, self.camera_id, timestamp=vt)
        alerts += self.vehicle_analyzer.analyze(vehicle_tracks, self.zones, self.lines, self.camera_id, timestamp=vt)
        self.night_mode = night
        if night:
            alerts += self.analyzer.check_night_movement(person_tracks, self.camera_id, vt)
            alerts += self.vehicle_analyzer.check_night_movement(vehicle_tracks, self.camera_id, vt)
        return alerts

    def _read_plates(self, vehicle_tracks: List[Track], frame: np.ndarray, vt: float) -> None:
        for t in vehicle_tracks:
            if t.track_id in self._plate_cache:
                continue  # already read this vehicle this session
            x1, y1, x2, y2 = t.bbox
            x1, y1 = max(0, x1), max(0, y1)
            crop = frame[y1:y2, x1:x2]
            result = self.plate_reader.read(crop)
            if result:
                self._plate_cache[t.track_id] = result.text
                verified = is_valid_indian_plate_format(result.text)
                suffix = "" if verified else " (format not verified -- possible OCR noise)"
                save_event("anpr_read", self.camera_id,
                           f"Vehicle #{t.track_id} plate read: {result.text}{suffix}", vt)
                self.total_events_logged += 1

    def _log_new_detections(self, tracks: List[Track], label: str, vt: float) -> None:
        """Logs one 'first seen' detection-log entry per NEW track (not
        per frame) -- so Detection log genuinely reflects every person
        and vehicle the system has spotted, the same way it already did
        for faces and plates, without flooding the log with a repeat
        entry every single frame an object stays in view."""
        for t in tracks:
            key = (label, t.track_id)
            if key in self._detection_logged:
                continue
            self._detection_logged.add(key)
            save_event(f"{label}_detected", self.camera_id,
                       f"{label.capitalize()} #{t.track_id} detected", vt)
            self.total_events_logged += 1

    def _detect_faces(self, person_tracks: List[Track], frame: np.ndarray, vt: float) -> None:
        for t in person_tracks:
            last_logged = self._face_logged_cooldown.get(t.track_id)
            if last_logged is not None and vt - last_logged < 15.0:
                continue
            x1, y1, x2, y2 = t.bbox
            x1, y1 = max(0, x1), max(0, y1)
            crop = frame[y1:y2, x1:x2]
            faces = self.face_detector.detect(crop, offset=(x1, y1))
            if faces:
                self._face_logged_cooldown[t.track_id] = vt
                save_event("face_detected", self.camera_id,
                           f"Face detected on person #{t.track_id}", vt)
                self.total_events_logged += 1

    def _persist(self, alerts: List[Alert], occupancy: int, vehicle_count: int = 0,
                 evidence_frame: Optional[np.ndarray] = None) -> None:
        # One snapshot per frame (not per alert) -- every alert fired on
        # the same frame shares it, since they're evidence of the same
        # moment. save_snapshot_file() content-addresses by hash, so this
        # is also naturally deduplicated on disk.
        snapshot_hash = None
        if alerts and evidence_frame is not None:
            ok, buf = cv2.imencode(".jpg", evidence_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok:
                snapshot_hash = save_snapshot_file(buf.tobytes())

        for a in alerts:
            # Capture the database-assigned id back onto the in-memory
            # Alert object -- this is what the websocket payload (main.py)
            # needs to include so the dashboard's live "Acknowledge"
            # button has a real row to target instead of nothing.
            a.id = save_alert(a, snapshot_hash=snapshot_hash)
            a.snapshot_hash = snapshot_hash
            self.total_alerts_fired += 1
        # vehicle_count/night_mode were already being computed every frame
        # (used for the on-screen HUD and the alert banners) but were
        # never actually passed through to storage -- this is the fix for
        # that; occupancy (people+vehicles combined) keeps its existing
        # meaning unchanged, exactly as before.
        save_stat(self.camera_id, occupancy, self.frame_index,
                  vehicle_count=vehicle_count, night_mode=self.night_mode)

    def _reset_session_state(self) -> None:
        self.frame_index = 0
        self.tracker = CentroidTracker()
        self.analyzer = GroupBehaviorAnalyzer()
        self.vehicle_tracker = CentroidTracker(max_distance=140.0)
        self.vehicle_analyzer = GroupBehaviorAnalyzer(
            group_min_size=2, group_persist_seconds=2.0, erratic_speed_px_s=550.0,
            loiter_seconds=10.0,
        )
        self._plate_cache.clear()
        self._face_logged_cooldown.clear()
        self._detection_logged.clear()
        self._sim_vehicle_detector = BlobVehicleDetector()

    # ---- rendering helpers --------------------------------------------
    def _blank_canvas(self, dim: bool = False) -> np.ndarray:
        base = (14, 14, 16) if dim else (30, 30, 30)
        canvas = np.full((self.frame_h, self.frame_w, 3), base, dtype=np.uint8)
        cv2.putText(canvas, "SIMULATION MODE - synthetic scenario feed", (14, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        return canvas

    def _draw_sim_vehicle(self, canvas: np.ndarray) -> None:
        """Render the scripted vehicle actor as real pixels: a car body and
        a genuinely OCR-able plate. This is what lets vehicle detection and
        ANPR run as real algorithms even in simulation mode."""
        for actor in self._sim_actors:
            if actor.label != "vehicle":
                continue
            pos = actor.position_at(self.frame_index)
            if pos is None:
                continue
            x, y = int(pos[0]), int(pos[1])
            cv2.rectangle(canvas, (x - 55, y - 26), (x + 55, y + 26), (70, 70, 200), -1)
            cv2.rectangle(canvas, (x - 55, y - 26), (x + 55, y + 26), (30, 30, 120), 2)
            # plate: white background, black text -- real OCR target
            px1, py1, px2, py2 = x - 32, y + 6, x + 32, y + 22
            cv2.rectangle(canvas, (px1, py1), (px2, py2), (255, 255, 255), -1)
            cv2.putText(canvas, "DL1CN4521", (px1 + 2, py2 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1, cv2.LINE_AA)

    def _draw_zones(self, frame: np.ndarray) -> None:
        overlay = frame.copy()
        for z in self.zones:
            pts = np.array(z.polygon, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 0, 120))
            cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
            cv2.putText(frame, z.name, (pts[0][0] + 5, pts[0][1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        for line in self.lines:
            p1 = tuple(map(int, line.p1))
            p2 = tuple(map(int, line.p2))
            cv2.line(frame, p1, p2, (0, 0, 255), 3)
            # Label at the line's own midpoint, not a fixed y=40 -- that
            # only ever looked correct for the default line because it
            # happens to start at the very top of the frame (p1 y=0). A
            # custom line drawn anywhere else had its name stuck up near
            # the top of the screen, completely disconnected from where
            # the line actually is -- confusing regardless of how correct
            # the underlying alert logic is.
            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2
            label_y = max(18, mid_y - 10)
            (text_w, _), _ = cv2.getTextSize(line.name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_x = max(4, min(frame.shape[1] - text_w - 4, mid_x - text_w // 2))
            cv2.putText(frame, line.name, (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    def _draw_tracks(self, frame: np.ndarray, tracks: List[Track], kind: str) -> None:
        color = COLORS.get(kind, (60, 200, 60))
        for t in tracks:
            x1, y1, x2, y2 = t.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (int(t.centroid[0]), int(t.centroid[1])), 3, (255, 255, 255), -1)
            label = f"{kind[:3].upper()} {t.track_id}"
            plate = self._plate_cache.get(t.track_id)
            if plate:
                label += f" [{plate}]"
            cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            pts = [(int(p.x), int(p.y)) for p in t.history]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], (120, 120, 255), 1)

    def _draw_night_badge(self, frame: np.ndarray, night: bool) -> None:
        if not night:
            return
        cv2.putText(frame, "NIGHT MODE - low-light enhancement active",
                    (self.frame_w - 340, self.frame_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["night_time_movement"], 1, cv2.LINE_AA)

    def _draw_alert_banners(self, frame: np.ndarray, alerts: List[Alert]) -> None:
        y = self.frame_h - 14
        for a in alerts[:4]:
            color = COLORS.get(a.alert_type, (0, 0, 255))
            cv2.putText(frame, f"[{a.severity.upper()}] {a.message}", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
            y -= 20
        occ_text = f"People {len(self.tracker.tracks)} | Vehicles {len(self.vehicle_tracker.tracks)}"
        cv2.putText(frame, occ_text, (self.frame_w - 260, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def frame_to_jpeg_b64(frame: np.ndarray, quality: int = 75, max_dim: int = 1600) -> str:
    """Encodes a frame for the WebSocket payload. max_dim caps the
    LARGER side before encoding -- this is a real, previously-undiscovered
    bug fix, not a cosmetic tweak: a 4K (3840x2160) source frame's
    base64-encoded JPEG routinely exceeds the WebSocket protocol's default
    1MB single-message limit, which silently kills the connection
    mid-replay (websockets raises "message too big", frame exceeds limit
    of 1048576 bytes). If that happens before the ~25-frame detector
    warmup completes, the client never sees a single detection for the
    rest of that session -- which looks exactly like "detection doesn't
    work," when the actual detector/tracker never even got the chance.
    Detection itself already ran on the full-resolution frame before this
    function is called (see process_frame/step_simulation) -- this only
    resizes the copy being sent to the browser, so detection accuracy is
    completely unaffected."""
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")

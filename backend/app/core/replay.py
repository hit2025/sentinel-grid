"""
replay.py
---------
Manages "upload & replay" sessions: a video file saved server-side and
streamed to the dashboard frame by frame over a WebSocket, running through
the exact same CameraSession.process_frame() pipeline a live camera would
use. This is what lets the dashboard show real detection/analytics
overlays on a real uploaded video, not just a JSON report at the end.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import cv2

from app.core.pipeline import CameraSession
from app.core.zones import default_demo_geometry

REPLAY_DIR = Path(__file__).resolve().parents[3] / "data" / "replay_uploads"


@dataclass
class ReplaySession:
    session_id: str
    video_path: Path
    session: CameraSession
    cap: "cv2.VideoCapture" = field(repr=False)
    source_fps: float = 25.0
    total_frames: int = 0
    frame_pos: int = 0
    created_at: float = field(default_factory=time.time)


_replay_sessions: Dict[str, ReplaySession] = {}


def create_replay_session(camera_id: str, video_path: Path,
                           session: Optional[CameraSession] = None) -> ReplaySession:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Could not open uploaded video -- unsupported codec or corrupt file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 960
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 540

    session_id = uuid.uuid4().hex[:12]

    if session is not None:
        # Reuse the SAME CameraSession main.py's get_session(camera_id)
        # manages, instead of building a throwaway one -- this is the fix
        # for a real bug: a replay used to run on its own disconnected
        # session, so a newly-calibrated fence line (saved against the
        # get_session() copy via /api/zones/{camera_id}) was silently
        # invisible to the replay actually processing frames; alerts kept
        # citing the original default line no matter when you recalibrated.
        # If this video's resolution differs from what the session was
        # last configured for, its existing zones/lines are in the wrong
        # coordinate space for the new frame size, so those get reset to
        # defaults for the new dimensions -- a same-resolution re-upload
        # correctly keeps whatever calibration is already set.
        if session.frame_w != w or session.frame_h != h:
            session.frame_w, session.frame_h = w, h
            session.zones, session.lines = default_demo_geometry(w, h)
        cam_session = session
    else:
        cam_session = CameraSession(camera_id=camera_id, frame_w=w, frame_h=h, fps=fps or 25.0)

    replay = ReplaySession(
        session_id=session_id, video_path=video_path, session=cam_session,
        cap=cap, source_fps=fps, total_frames=total,
    )
    _replay_sessions[session_id] = replay
    return replay


def get_replay_session(session_id: str) -> Optional[ReplaySession]:
    return _replay_sessions.get(session_id)


def close_replay_session(session_id: str) -> None:
    replay = _replay_sessions.pop(session_id, None)
    if replay:
        replay.cap.release()
        replay.video_path.unlink(missing_ok=True)

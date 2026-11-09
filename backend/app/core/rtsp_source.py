"""
rtsp_source.py
--------------
Real RTSP camera ingestion -- the actual "existing CCTV infrastructure"
path the problem statement describes, as opposed to the scripted
simulation (demo/testing) or uploaded-file replay (offline validation)
paths already in this codebase.

HONESTY NOTE, left in deliberately: this module has been exercised
against invalid/unreachable URLs (confirms it fails gracefully rather
than hanging or crashing the server) and against a local synthetic
RTSP-like source is NOT possible in this sandbox -- there is no real
IP camera or RTSP server reachable here to validate a successful
connection, real frame timing, or reconnect behaviour under an actual
network drop. Treat this as a reviewed, defensible code path, not as
"tested against a real camera."

Kept intentionally separate from replay.py rather than integrated with
it, since replay.py's real implementation (session id scheme, cleanup
timing, etc.) isn't available to build against directly here -- this
avoids guessing at internals and silently duplicating or conflicting
with whatever pattern is already established there.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger("ibvap.rtsp")


@dataclass
class RTSPSession:
    camera_id: str
    rtsp_url: str
    cap: "cv2.VideoCapture"
    connected_at: float
    reconnect_attempts: int = 0
    last_frame_at: Optional[float] = None


_rtsp_sessions: Dict[str, RTSPSession] = {}

# RTSP streams commonly stall or drop mid-session (network blips, camera
# reboot) far more often than a local file ever would -- unlike replay.py's
# fixed-length file, "the read failed" here should usually mean "try to
# reconnect", not "the session is over".
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_BACKOFF_SECONDS = 2.0


def open_rtsp_session(camera_id: str, rtsp_url: str, timeout_ms: int = 5000) -> RTSPSession:
    """Attempts to open an RTSP stream. Raises ConnectionError with a
    clear message on failure rather than returning a half-open capture
    object -- cv2.VideoCapture famously does NOT raise on a bad URL, it
    just silently returns an object that fails isOpened(), which is an
    easy way to ship a "working" integration that never actually
    connects to anything."""
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
    if not cap.isOpened():
        cap.release()
        raise ConnectionError(
            f"Could not open RTSP stream at {rtsp_url!r}. Check the URL, "
            f"credentials, network reachability, and that the camera "
            f"supports the requested transport."
        )
    session = RTSPSession(camera_id=camera_id, rtsp_url=rtsp_url, cap=cap, connected_at=time.time())
    _rtsp_sessions[camera_id] = session
    logger.info("RTSP session opened for %s -> %s", camera_id, rtsp_url)
    return session


def get_rtsp_session(camera_id: str) -> Optional[RTSPSession]:
    return _rtsp_sessions.get(camera_id)


def close_rtsp_session(camera_id: str) -> None:
    session = _rtsp_sessions.pop(camera_id, None)
    if session is not None:
        session.cap.release()
        logger.info("RTSP session closed for %s", camera_id)


def read_frame_with_reconnect(session: RTSPSession) -> Optional[np.ndarray]:
    """Returns the next frame, or None if reconnection was exhausted.
    On a read failure, tries to reopen the SAME url up to
    MAX_RECONNECT_ATTEMPTS times with a short backoff -- a single dropped
    frame from a real camera should not tear down the whole session the
    way it correctly does for a replay.py file (a file's "no more
    frames" genuinely means done; a camera's usually doesn't)."""
    ok, frame = session.cap.read()
    if ok:
        session.last_frame_at = time.time()
        session.reconnect_attempts = 0
        return frame

    session.reconnect_attempts += 1
    logger.warning(
        "RTSP read failed for %s (attempt %d/%d)",
        session.camera_id, session.reconnect_attempts, MAX_RECONNECT_ATTEMPTS,
    )
    if session.reconnect_attempts > MAX_RECONNECT_ATTEMPTS:
        logger.error("RTSP session %s exhausted reconnect attempts, giving up.", session.camera_id)
        return None

    time.sleep(RECONNECT_BACKOFF_SECONDS)
    session.cap.release()
    session.cap = cv2.VideoCapture(session.rtsp_url, cv2.CAP_FFMPEG)
    if not session.cap.isOpened():
        return None
    ok, frame = session.cap.read()
    if ok:
        session.last_frame_at = time.time()
        session.reconnect_attempts = 0
        return frame
    return None

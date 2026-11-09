"""
main.py
-------
IBVAP API -- Intelligent Border Video Analytics Platform.
(SIH26187, Ministry of Home Affairs / Sashastra Seema Bal (SSB), Police II
Division. Category: Software. Theme: Blockchain & Cybersecurity.)

Run:
    uvicorn app.main:app --reload --port 8000

Endpoints
---------
GET  /health                      liveness probe
GET  /api/ws/live/{camera_id}     websocket: annotated frames + alerts (simulation mode
                                    by default; feeds real camera when configured)
POST /api/upload                  multipart video upload -> batch analysis report (no live view)
POST /api/replay/start            multipart video upload -> starts a replay session
GET  /api/ws/replay/{session_id}  websocket: annotated frames + alerts, streamed from
                                    the uploaded video -- lets the dashboard show real
                                    detection overlays on real footage, not just a report
GET  /api/alerts                  paginated alert history (filter by severity)
GET  /api/events                  face-detected / ANPR-read event log
GET  /api/analytics/summary       aggregate KPIs for the dashboard
GET  /api/zones/{camera_id}       current zone/line configuration
POST /api/zones/{camera_id}       operator-calibrated zone/line configuration
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import parse_qs

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from app.core.cross_camera import find_cross_camera_matches
from app.core.pipeline import CameraSession, frame_to_jpeg_b64
from app.core.rtsp_source import (
    close_rtsp_session, get_rtsp_session, open_rtsp_session, read_frame_with_reconnect,
)
from app.core.replay import (
    REPLAY_DIR, close_replay_session, create_replay_session, get_replay_session,
)
from app.core.storage import (
    fetch_alerts, fetch_events, fetch_summary, init_db, verify_all,
    seed_tamper_demo, simulate_visible_edit_attack, simulate_chain_rewrite_attack,
    acknowledge_alert, reset_all_data, fetch_alert_by_id, SNAPSHOT_DIR, prune_snapshots,
    save_alert, fetch_audit_trail, fetch_session_stats,
)
from app.core.zones import BorderLine, Zone

# --------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------
# A single shared key is the right level of auth for a prototype that
# still needs a real, demonstrable answer to "what stops an unauthorized
# person from hitting /api/data/reset right now?" -- previously nothing
# did. A full user/password/JWT system would be over-engineering for a
# single-operator demo deployment; this closes the actual gap without
# rewriting the app's shape. Override the default via the IBVAP_API_KEY
# environment variable before any real/public deployment -- the fallback
# below exists only so the app still runs out of the box for local
# testing, and is deliberately obvious in the source that it must change.
API_KEY = os.environ.get("IBVAP_API_KEY", "ibvap-demo-key-2026-CHANGE-ME")

# Evidence snapshots (data/snapshots/) grow without bound on a
# long-running session -- a live simulation generates a fresh, genuinely
# unique photo for every alert, and a single ongoing loitering situation
# alone re-fires every ~10s. Override via IBVAP_MAX_SNAPSHOT_MB before a
# real deployment if 500MB isn't the right ceiling for your disk.
MAX_SNAPSHOT_MB = int(os.environ.get("IBVAP_MAX_SNAPSHOT_MB", "500"))
SNAPSHOT_PRUNE_INTERVAL_SECONDS = 600  # 10 minutes


class APIKeyMiddleware:
    """Raw ASGI middleware (not the usual @app.middleware("http") /
    BaseHTTPMiddleware) specifically because BaseHTTPMiddleware never
    sees WebSocket connections -- and half of this API's real endpoints
    (the live camera feeds) are WebSocket. This single middleware covers
    every /api/* route, HTTP and WebSocket alike, in one place, rather
    than adding a Depends(...) to each of the ~20 existing endpoint
    functions individually. Accepts the key as an X-API-Key header
    (REST) or an api_key query parameter (WebSocket -- browsers cannot
    set custom headers during a WebSocket handshake). "/" (the dashboard
    shell) and "/health" (liveness probe) stay open on purpose."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if not path.startswith("/api/"):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        supplied = headers.get(b"x-api-key", b"").decode()
        if not supplied:
            qs = parse_qs(scope.get("query_string", b"").decode())
            supplied = (qs.get("api_key") or [""])[0]

        if supplied != API_KEY:
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 4401})
            else:
                response = JSONResponse(
                    {"detail": "Invalid or missing API key. Include an X-API-Key header "
                               "(or ?api_key=... for WebSocket connections)."},
                    status_code=401,
                )
                await response(scope, receive, send)
            return

        await self.app(scope, receive, send)

app = FastAPI(
    title="IBVAP API",
    description=(
        "Intelligent Border Video Analytics Platform -- SIH26187, "
        "Ministry of Home Affairs / Sashastra Seema Bal (SSB). "
        "AI-based video analytics for border/perimeter CCTV: human + "
        "vehicle detection, face detection, ANPR, virtual fence intrusion, "
        "suspicious activity, night-time movement, alerts + event logging."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(APIKeyMiddleware)

_sessions: Dict[str, CameraSession] = {}

# Serves frontend/index.html when it's a sibling of backend/ -- true both
# locally (sentinel-grid/backend/ and sentinel-grid/frontend/) and in the
# root Dockerfile's image layout. This is what lets a single hosted
# service serve the whole app (API + dashboard) from one origin, with no
# separate frontend host and no cross-origin WebSocket URL to configure.
_FRONTEND_INDEX = Path(__file__).resolve().parents[2] / "frontend" / "index.html"
_DB_VIEWER_PAGE = Path(__file__).resolve().parents[2] / "frontend" / "db_viewer.html"


@app.get("/", include_in_schema=False)
def serve_frontend():
    if _FRONTEND_INDEX.exists():
        return FileResponse(_FRONTEND_INDEX)
    return JSONResponse(
        {"detail": "Frontend not bundled with this deployment. API is still available at /docs."},
        status_code=404,
    )


@app.get("/db-viewer", include_in_schema=False)
def serve_db_viewer():
    """A standalone, read-only database browser -- for a hosted
    deployment, a jury can't open DB Browser for SQLite against a remote
    server's local file the way you can on your own machine. This is the
    web-based equivalent: same API key gate as the main dashboard, same
    /api/* endpoints, just a different, DB-focused view of the exact
    same live data."""
    if _DB_VIEWER_PAGE.exists():
        return FileResponse(_DB_VIEWER_PAGE)
    return JSONResponse({"detail": "db_viewer.html not found."}, status_code=404)


@app.on_event("startup")
async def on_startup() -> None:
    logging.basicConfig(level=logging.INFO)
    init_db()
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    # Prune once immediately (covers a session that's been sitting well
    # over the limit since before this server started), then keep
    # pruning periodically for the rest of this run -- a startup-only
    # prune wouldn't help a single long-running uvicorn process that
    # never restarts, which is exactly the "left running for a demo/day"
    # scenario this exists for.
    result = prune_snapshots(MAX_SNAPSHOT_MB * 1024 * 1024)
    if result["deleted"]:
        logging.getLogger("ibvap.snapshots").info(
            "Startup snapshot prune: deleted %d file(s), freed %.1f MB",
            result["deleted"], result["freed_bytes"] / 1024 / 1024,
        )
    asyncio.create_task(_snapshot_prune_loop())


async def _snapshot_prune_loop() -> None:
    logger = logging.getLogger("ibvap.snapshots")
    while True:
        await asyncio.sleep(SNAPSHOT_PRUNE_INTERVAL_SECONDS)
        try:
            result = prune_snapshots(MAX_SNAPSHOT_MB * 1024 * 1024)
            if result["deleted"]:
                logger.info(
                    "Periodic snapshot prune: deleted %d file(s), freed %.1f MB",
                    result["deleted"], result["freed_bytes"] / 1024 / 1024,
                )
        except Exception:  # noqa: BLE001 -- a failed prune should never crash the server
            logger.exception("Periodic snapshot prune failed")


def get_session(camera_id: str) -> CameraSession:
    if camera_id not in _sessions:
        _sessions[camera_id] = CameraSession(camera_id=camera_id)
    return _sessions[camera_id]


@app.get("/health")
def health():
    return {"status": "ok", "service": "ibvap-api"}


# --------------------------------------------------------------------
# Live view (simulation-mode by default -- swap to a real RTSP/webcam
# reader in `process_frame` for an actual camera without touching the
# websocket contract at all).
# --------------------------------------------------------------------
@app.websocket("/api/ws/live/{camera_id}")
async def ws_live(websocket: WebSocket, camera_id: str):
    await websocket.accept()
    session = get_session(camera_id)
    try:
        while True:
            try:
                frame, tracks, alerts = session.step_simulation()
            except Exception as e:  # noqa: BLE001
                logging.getLogger("ibvap.live").exception("Live simulation step failed")
                await websocket.send_text(json.dumps({
                    "type": "error", "message": f"{type(e).__name__}: {e}",
                }))
                break
            payload = {
                "type": "frame",
                "camera_id": camera_id,
                "frame_index": session.frame_index,
                "people_count": len(session.tracker.tracks),
                "vehicle_count": len(session.vehicle_tracker.tracks),
                "night_mode": session.night_mode,
                "jpeg_b64": frame_to_jpeg_b64(frame),
                "alerts": [
                    {
                        "id": a.id,
                        "snapshot_hash": a.snapshot_hash,
                        "alert_type": a.alert_type,
                        "severity": a.severity,
                        "message": a.message,
                        "track_ids": a.track_ids,
                        "timestamp": a.timestamp,
                    }
                    for a in alerts
                ],
            }
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(1 / 12)  # ~12 fps stream to the browser
    except WebSocketDisconnect:
        pass


# --------------------------------------------------------------------
# Batch analysis of an uploaded real video file
# --------------------------------------------------------------------
@app.post("/api/upload")
async def upload_video(camera_id: str = Query("cam-upload"), file: UploadFile = File(...)):
    if file.content_type not in (
        "video/mp4", "video/avi", "video/quicktime", "video/x-matroska", "application/octet-stream",
    ):
        raise HTTPException(400, f"Unsupported content type: {file.content_type}")

    tmp_path = None
    try:
        suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        session = CameraSession(camera_id=camera_id)
        report = session.process_upload(tmp_path)
    except Exception as e:  # noqa: BLE001
        logging.getLogger("ibvap.upload").exception("Upload processing failed")
        raise HTTPException(500, f"Processing failed: {type(e).__name__}: {e}")
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    return JSONResponse(report)


# --------------------------------------------------------------------
# Upload & replay: same real footage, but streamed frame-by-frame to the
# dashboard with live overlays -- so a real video "feels" like a live feed
# instead of only producing a JSON report.
# --------------------------------------------------------------------
_UPLOAD_CONTENT_TYPES = (
    "video/mp4", "video/avi", "video/quicktime", "video/x-matroska", "application/octet-stream",
)


@app.post("/api/replay/start")
async def start_replay(camera_id: str = Query("cam-replay"), file: UploadFile = File(...)):
    if file.content_type not in _UPLOAD_CONTENT_TYPES:
        raise HTTPException(400, f"Unsupported content type: {file.content_type}")

    try:
        REPLAY_DIR.mkdir(parents=True, exist_ok=True)
        suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
        dest = REPLAY_DIR / f"{uuid.uuid4().hex[:12]}{suffix}"
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        # Pass the SAME session get_session(camera_id) manages, so a
        # zone/line calibrated via /api/zones/{camera_id} -- before or
        # during this replay -- is actually visible to the frames this
        # replay processes, instead of silently landing on a disconnected
        # throwaway session (see replay.py's create_replay_session).
        existing_session = get_session(camera_id)
        replay = create_replay_session(camera_id, dest, session=existing_session)
    except Exception as e:  # noqa: BLE001
        logging.getLogger("ibvap.replay").exception("Failed to start replay session")
        raise HTTPException(500, f"Failed to start replay: {type(e).__name__}: {e}")

    return {
        "session_id": replay.session_id,
        "total_frames": replay.total_frames,
        "fps": replay.source_fps,
    }


@app.websocket("/api/ws/replay/{session_id}")
async def ws_replay(websocket: WebSocket, session_id: str):
    await websocket.accept()
    replay = get_replay_session(session_id)
    if replay is None:
        await websocket.send_text(json.dumps({
            "type": "error", "message": "Unknown or expired replay session",
        }))
        await websocket.close()
        return

    # Cap total processed frames around ~300 regardless of source length,
    # so a long clip still replays in a reasonable time on stage.
    sample_every = max(1, replay.total_frames // 300) if replay.total_frames else 3

    try:
        while True:
            frame = None
            for _ in range(sample_every):
                ok, f = replay.cap.read()
                if not ok:
                    frame = None
                    break
                frame = f
                replay.frame_pos += 1

            if frame is None:
                await websocket.send_text(json.dumps({
                    "type": "done", "camera_id": replay.session.camera_id,
                }))
                break

            try:
                annotated, tracks, alerts = replay.session.process_frame(frame)
            except Exception as e:  # noqa: BLE001
                logging.getLogger("ibvap.replay").exception("Replay frame processing failed")
                await websocket.send_text(json.dumps({
                    "type": "error", "message": f"{type(e).__name__}: {e}",
                }))
                break
            payload = {
                "type": "frame",
                "camera_id": replay.session.camera_id,
                "frame_index": replay.frame_pos,
                "total_frames": replay.total_frames,
                "people_count": len(replay.session.tracker.tracks),
                "vehicle_count": len(replay.session.vehicle_tracker.tracks),
                "night_mode": replay.session.night_mode,
                "jpeg_b64": frame_to_jpeg_b64(annotated),
                "alerts": [
                    {
                        "id": a.id,
                        "snapshot_hash": a.snapshot_hash,
                        "alert_type": a.alert_type,
                        "severity": a.severity,
                        "message": a.message,
                        "track_ids": a.track_ids,
                        "timestamp": a.timestamp,
                    }
                    for a in alerts
                ],
            }
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(1 / 12)
    except WebSocketDisconnect:
        pass
    finally:
        close_replay_session(session_id)


# --------------------------------------------------------------------
# Real RTSP camera ingestion -- the actual "existing CCTV infrastructure"
# path, parallel to (not integrated with) the scripted-simulation and
# uploaded-file-replay paths above. See rtsp_source.py's module
# docstring for exactly what has and hasn't been validated: connection
# failure handling has been tested against unreachable/invalid URLs;
# a successful real-camera connection has NOT been tested, since no
# real IP camera or RTSP server is reachable from this environment.
# --------------------------------------------------------------------
@app.post("/api/live/rtsp/connect")
def rtsp_connect(camera_id: str = Query(...), rtsp_url: str = Query(...)):
    try:
        session = open_rtsp_session(camera_id, rtsp_url)
    except ConnectionError as e:
        raise HTTPException(502, str(e))
    return {"status": "connected", "camera_id": camera_id, "connected_at": session.connected_at}


@app.post("/api/live/rtsp/disconnect")
def rtsp_disconnect(camera_id: str = Query(...)):
    close_rtsp_session(camera_id)
    return {"status": "disconnected", "camera_id": camera_id}


@app.websocket("/api/ws/live-rtsp/{camera_id}")
async def ws_live_rtsp(websocket: WebSocket, camera_id: str):
    """Same payload shape as /api/ws/live/{camera_id} and
    /api/ws/replay/{session_id} -- the dashboard doesn't need to know or
    care which of the three frame sources it's looking at. Requires
    /api/live/rtsp/connect to have been called first for this camera_id."""
    await websocket.accept()
    rtsp_session = get_rtsp_session(camera_id)
    if rtsp_session is None:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"No RTSP session for {camera_id!r}. Call POST /api/live/rtsp/connect first.",
        }))
        await websocket.close()
        return

    session = get_session(camera_id)  # the same CameraSession/analytics used everywhere else
    try:
        while True:
            frame = read_frame_with_reconnect(rtsp_session)
            if frame is None:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "RTSP stream lost and reconnect attempts exhausted.",
                }))
                break
            try:
                annotated, tracks, alerts = session.process_frame(frame)
            except Exception as e:  # noqa: BLE001
                logging.getLogger("ibvap.rtsp").exception("RTSP frame processing failed")
                await websocket.send_text(json.dumps({
                    "type": "error", "message": f"{type(e).__name__}: {e}",
                }))
                break
            payload = {
                "type": "frame",
                "camera_id": camera_id,
                "frame_index": session.frame_index,
                "people_count": len(session.tracker.tracks),
                "vehicle_count": len(session.vehicle_tracker.tracks),
                "night_mode": session.night_mode,
                "jpeg_b64": frame_to_jpeg_b64(annotated),
                "alerts": [
                    {
                        "id": a.id, "snapshot_hash": a.snapshot_hash,
                        "alert_type": a.alert_type, "severity": a.severity,
                        "message": a.message, "track_ids": a.track_ids, "timestamp": a.timestamp,
                    }
                    for a in alerts
                ],
            }
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(1 / 12)
    except WebSocketDisconnect:
        pass
    finally:
        close_rtsp_session(camera_id)


# --------------------------------------------------------------------
# Alerts + analytics
# --------------------------------------------------------------------
@app.get("/api/alerts")
def get_alerts(limit: int = 100, severity: Optional[str] = None, camera_id: Optional[str] = None):
    return fetch_alerts(limit=limit, severity=severity, camera_id=camera_id)


@app.post("/api/alerts/{alert_id}/acknowledge")
def ack_alert(alert_id: int):
    """Marks an alert as reviewed by an operator. This is an ordinary
    operational update, not a security-relevant one -- it does NOT
    touch audit_log or the hash chain, and verify_integrity deliberately
    still checks the OTHER fields (message/severity/etc.) against their
    original snapshot regardless of acknowledged state. See storage.py's
    acknowledge_alert() docstring for why that split is intentional."""
    ok = acknowledge_alert(alert_id)
    if not ok:
        raise HTTPException(404, f"No alert with id {alert_id}")
    return {"status": "acknowledged", "alert_id": alert_id}


@app.get("/api/alerts/{alert_id}/snapshot")
def get_alert_snapshot(alert_id: int):
    """Serves the evidence frame captured at the moment this alert fired.
    Recomputes the file's SHA-256 on every read and compares it against
    the hash embedded in the alert's own audit_log payload when the
    record was created -- so this isn't just "here's an image," it's
    "here's the image, and it's still cryptographically the same file
    referenced by the tamper-evident log." Swapping the file on disk
    (as opposed to editing the database) is caught here, not by
    /api/audit/verify, which never looks at image bytes."""
    alert = fetch_alert_by_id(alert_id)
    if alert is None:
        raise HTTPException(404, f"No alert with id {alert_id}")
    snapshot_hash = alert.get("snapshot_hash")
    if not snapshot_hash:
        raise HTTPException(404, "This alert has no evidence snapshot.")

    path = SNAPSHOT_DIR / f"{snapshot_hash}.jpg"
    if not path.exists():
        raise HTTPException(410, "Snapshot file is missing from disk (data/snapshots/).")

    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != snapshot_hash:
        raise HTTPException(
            409,
            f"TAMPER DETECTED: snapshot file no longer matches its audit-chain hash "
            f"(expected {snapshot_hash[:12]}..., found {actual_hash[:12]}...).",
        )
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/alerts/{alert_id}/report", response_class=HTMLResponse)
def get_alert_report(alert_id: int, api_key: str = Query(...)):
    """A styled, self-contained, printable incident report -- the
    "one-click PDF" is deliberately the browser's own Print -> Save as
    PDF rather than a server-side PDF library: no new dependency that
    might not install cleanly in the field, and the browser's PDF
    renderer is more reliable than most lightweight Python PDF libs for
    this kind of layout. api_key is required as a query param (not just
    the header) specifically so this URL is directly usable as the
    target of a link/new-tab open from the dashboard, and so the
    embedded evidence <img> tag -- which cannot send custom headers --
    can still authenticate against the snapshot endpoint."""
    alert = fetch_alert_by_id(alert_id)
    if alert is None:
        raise HTTPException(404, f"No alert with id {alert_id}")

    verify = verify_all()
    chain_ok = verify["chain"]["chain_intact"]
    row_ok = alert_id not in verify["integrity"]["tampered_alert_ids"]
    integrity_line = (
        "VERIFIED -- this record matches its tamper-evident audit-chain snapshot"
        if (chain_ok and row_ok) else
        "WARNING -- this record or the audit chain itself failed verification; "
        "treat this report as evidence of tampering, not as a clean incident record"
    )
    integrity_color = "#1e8a4c" if (chain_ok and row_ok) else "#c0392b"

    has_snapshot = bool(alert.get("snapshot_hash"))
    snapshot_html = (
        f'<img src="/api/alerts/{alert_id}/snapshot?api_key={api_key}" '
        f'alt="Evidence snapshot" style="max-width:100%;border:1px solid #ccc;border-radius:6px;" />'
        if has_snapshot else
        '<p style="color:#888;font-style:italic;">No evidence snapshot was captured for this alert.</p>'
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Incident Report #{alert_id} -- IBVAP</title>
<style>
  body{{font-family:Arial,Helvetica,sans-serif;color:#1a1a1a;max-width:800px;margin:32px auto;padding:0 20px;}}
  h1{{font-size:20px;border-bottom:2px solid #1f4e79;padding-bottom:8px;}}
  .badge{{display:inline-block;padding:3px 10px;border-radius:4px;color:#fff;font-size:12px;font-weight:bold;}}
  table{{width:100%;border-collapse:collapse;margin:16px 0;}}
  td{{padding:7px 10px;border-bottom:1px solid #e0e0e0;font-size:13px;vertical-align:top;}}
  td.label{{color:#666;width:160px;font-weight:bold;}}
  .integrity{{margin-top:20px;padding:12px 16px;border-radius:6px;border:1px solid {integrity_color};
              background:{integrity_color}15;color:{integrity_color};font-size:13px;font-weight:bold;}}
  .footer{{margin-top:24px;font-size:11px;color:#999;}}
  @media print {{ body{{margin:0;}} }}
</style></head>
<body>
  <h1>IBVAP Incident Report &mdash; Alert #{alert_id}</h1>
  <table>
    <tr><td class="label">Type</td><td>{alert['alert_type']}</td></tr>
    <tr><td class="label">Severity</td><td><span class="badge" style="background:#1f4e79;">{alert['severity'].upper()}</span></td></tr>
    <tr><td class="label">Message</td><td>{alert['message']}</td></tr>
    <tr><td class="label">Camera / Session</td><td>{alert['camera_id']}</td></tr>
    <tr><td class="label">Track ID(s)</td><td>{alert['track_ids']}</td></tr>
    <tr><td class="label">Timestamp</td><td>{alert['timestamp']}</td></tr>
    <tr><td class="label">Acknowledged</td><td>{'Yes' if alert['acknowledged'] else 'No'}</td></tr>
    <tr><td class="label">Snapshot hash</td><td style="font-family:monospace;font-size:11px;">{alert.get('snapshot_hash') or '(none)'}</td></tr>
  </table>
  <h3>Evidence</h3>
  {snapshot_html}
  <div class="integrity">Audit-chain status: {integrity_line}</div>
  <div class="footer">Generated by IBVAP (SIH26187) &middot; Use your browser's Print &rarr; Save as PDF to export this report.</div>
</body></html>"""
    return HTMLResponse(html)


@app.get("/api/events")
def get_events(limit: int = 100, event_type: Optional[str] = None, camera_id: Optional[str] = None):
    """Face-detected / ANPR-read event log -- capabilities that are
    informational (audit trail) rather than security escalations.
    Pass camera_id to scope to the current live/replay session -- without
    it, this returns events across every session ever run against this
    database file, which reads as confusing stale data on a fresh demo."""
    return fetch_events(limit=limit, event_type=event_type, camera_id=camera_id)


@app.get("/api/analytics/summary")
def get_summary(camera_id: Optional[str] = None):
    return fetch_summary(camera_id=camera_id)


@app.get("/api/audit/verify")
def audit_verify():
    """Verifies the tamper-evident hash chain covering every alert and
    event ever logged, plus cross-checks the visible alerts/events tables
    against their own chained snapshot. `ok: true` means nothing in the
    security log has been altered since it was written."""
    return verify_all()


@app.get("/api/db/audit_log")
def get_audit_trail(limit: int = 100):
    """Raw rows from the audit_trail_readable view -- what powers the
    standalone DB viewer page's Audit Log table."""
    return fetch_audit_trail(limit=limit)


@app.get("/api/db/session_stats")
def get_session_stats(limit: int = 100, camera_id: Optional[str] = None):
    """Raw session_stats rows -- what powers the standalone DB viewer
    page's Session Stats table."""
    return fetch_session_stats(limit=limit, camera_id=camera_id)


@app.post("/api/audit/demo/seed")
def audit_demo_seed():
    """Live jury-demo helper: wipes the database and inserts a small,
    fixed, realistic set of alerts/events (own camera_id, doesn't touch
    real session data) so the two attacks below always have a known
    clean starting point. A hash chain is deliberately append-only, so
    'reset' here means start a fresh chain, not patch history back to
    clean -- that's the honest behaviour to demonstrate."""
    return seed_tamper_demo()


@app.post("/api/audit/demo/attack1")
def audit_demo_attack1():
    """Runs the real 'visible edit' attack from the technical briefing
    against the live database: rewrites the newest demo alert's message
    and severity directly, without touching audit_log at all. Call
    GET /api/audit/verify afterward to see it caught."""
    return simulate_visible_edit_attack()


@app.post("/api/audit/demo/attack2")
def audit_demo_attack2():
    """Runs the real 'chain rewrite' attack: edits the earliest demo
    audit_log record's payload AND recomputes that row's own hash so it
    looks self-consistent. Still caught -- the next record's prev_hash
    already committed to the old hash. Call GET /api/audit/verify
    afterward to see it caught."""
    return simulate_chain_rewrite_attack()


@app.post("/api/data/reset")
def data_reset():
    """Wipes ALL tracking data (alerts/events/audit_log/session_stats) to
    a genuinely empty database. Distinct from /api/audit/demo/seed above:
    that endpoint fills the DB with a fixed fake dataset to demonstrate
    tamper-evidence; this one just clears everything so a live simulation
    or replay session starts clean -- e.g. right before a jury walks up,
    so DB Browser only shows real, current activity rather than leftover
    rows from earlier testing. Irreversible -- the frontend confirms with
    the user before calling this."""
    return reset_all_data()


@app.get("/api/snapshots/usage")
def snapshot_usage():
    """Current disk usage of data/snapshots/ and the configured ceiling,
    so the dashboard (or you, via curl) can see this before it becomes a
    problem instead of after."""
    files = list(SNAPSHOT_DIR.glob("*.jpg")) if SNAPSHOT_DIR.exists() else []
    total_bytes = sum(f.stat().st_size for f in files)
    return {
        "file_count": len(files),
        "total_mb": round(total_bytes / 1024 / 1024, 1),
        "limit_mb": MAX_SNAPSHOT_MB,
        "auto_prune_interval_seconds": SNAPSHOT_PRUNE_INTERVAL_SECONDS,
    }


@app.post("/api/snapshots/prune")
def snapshot_prune_now():
    """Manual trigger -- doesn't wait for the periodic background prune.
    Safe to call any time; see prune_snapshots()'s docstring for exactly
    what this does and does not affect (photos only, never the alert
    records or the hash chain)."""
    return prune_snapshots(MAX_SNAPSHOT_MB * 1024 * 1024)


# --------------------------------------------------------------------
# Zone / line calibration
# --------------------------------------------------------------------
class ZoneIn(BaseModel):
    name: str
    polygon: list[list[float]]
    zone_type: str = "restricted"


class LineIn(BaseModel):
    name: str
    p1: list[float]
    p2: list[float]
    unauthorized_side: str = "b_to_a"


class GeometryIn(BaseModel):
    zones: list[ZoneIn] = []
    lines: list[LineIn] = []


@app.get("/api/zones/{camera_id}")
def get_zones(camera_id: str):
    session = get_session(camera_id)
    return {
        "zones": [
            {"zone_id": z.zone_id, "name": z.name, "polygon": z.polygon, "zone_type": z.zone_type}
            for z in session.zones
        ],
        "lines": [
            {"line_id": l.line_id, "name": l.name, "p1": l.p1, "p2": l.p2,
             "unauthorized_side": l.unauthorized_side}
            for l in session.lines
        ],
    }


@app.post("/api/zones/{camera_id}")
def set_zones(camera_id: str, geometry: GeometryIn):
    session = get_session(camera_id)
    session.zones = [
        Zone(zone_id=i + 1, name=z.name, polygon=[tuple(p) for p in z.polygon], zone_type=z.zone_type)
        for i, z in enumerate(geometry.zones)
    ] or session.zones
    session.lines = [
        BorderLine(line_id=i + 1, name=l.name, p1=tuple(l.p1), p2=tuple(l.p2),
                   unauthorized_side=l.unauthorized_side)
        for i, l in enumerate(geometry.lines)
    ] or session.lines
    # Makes the scripted live-simulation actors immediately start walking
    # toward THIS geometry -- previously they were fixed at session-
    # creation time and never knew calibration had changed, so a custom
    # line only ever got crossed via the manual Test button, never
    # automatically like the built-in default line always did.
    session.rebuild_sim_actors()
    return {"status": "updated", "zones": len(session.zones), "lines": len(session.lines)}


@app.post("/api/zones/{camera_id}/test")
def test_zone_calibration(camera_id: str, kind: str = Query(..., pattern="^(line|zone)$")):
    """Generates a REAL alert through the actual analytics pipeline --
    same GroupBehaviorAnalyzer instance, same cooldown logic, same
    save_alert()/audit-chain path as a genuine detection -- against
    whichever line/zone was most recently saved for this camera. This
    exists because the scripted live-simulation actors follow fixed
    paths, so a freshly-calibrated line drawn somewhere they don't
    happen to walk would never fire naturally, which is a bad experience
    demoing this to a jury. The resulting alert's message is clearly
    prefixed [CALIBRATION TEST] and saved through the exact same path as
    everything else, so it's honestly visible in Alert Feed, Detection
    log and the audit chain -- never hidden, never faked separately."""
    session = get_session(camera_id)
    # Use the session's own virtual clock (frame_index / fps), not real
    # wall-clock time -- every other alert in this session (live
    # simulation or replay) is timestamped this way, and the dashboard's
    # fmtTime() renders it as elapsed mm:ss assuming a small number. A
    # real epoch timestamp here (e.g. 1790492561.5) broke that display
    # outright, showing nonsense like "29841545:05" instead of "t+00:25".
    now = session._virtual_time()

    @dataclass
    class _TestTrack:
        track_id: int
        centroid: tuple
        bbox: tuple = (0, 0, 20, 20)
        label: str = "person"
        first_seen: float = 0.0
        _speed: float = 0.0
        def speed_px_per_s(self) -> float:
            return self._speed

    try:
        if kind == "line":
            if not session.lines:
                raise HTTPException(400, "No border line is currently saved for this camera. Draw and Save one first.")
            line = session.lines[-1]
            x1, y1 = line.p1
            x2, y2 = line.p2
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            dx, dy = x2 - x1, y2 - y1
            length = (dx ** 2 + dy ** 2) ** 0.5 or 1.0
            # Perpendicular offset from the line's midpoint, in both
            # directions -- then ask the line's OWN side_of() which side
            # each landed on, rather than assuming a sign convention.
            # This is what makes the test correct regardless of the real
            # BorderLine implementation's internal geometry.
            px, py = -dy / length, dx / length
            offset = 40
            point_plus = (mx + px * offset, my + py * offset)
            point_minus = (mx - px * offset, my - py * offset)
            side_plus = line.side_of(point_plus)
            start_side, _, end_side = line.unauthorized_side.partition("_to_")
            start_point = point_plus if side_plus == start_side else point_minus
            end_point = point_minus if side_plus == start_side else point_plus

            # Use the REAL tracker's own next id, not a random 6-digit
            # number -- test alerts otherwise show as "Person #903156"
            # right next to genuinely-tracked people showing "PER 1",
            # "PER 3" on screen, which looks obviously synthetic to
            # anyone watching. Actually consuming the id (incrementing
            # the tracker's counter) keeps this collision-free: the next
            # REAL detection correctly gets the id after this one, not a
            # duplicate of it.
            test_id = session.tracker._next_id
            session.tracker._next_id += 1

            # A believable 3-beat mini-scenario instead of a single
            # repetitive "crossed" message every time -- appear, then a
            # sudden burst of speed heading toward the line (genuinely
            # triggers erratic_motion, not a fabricated label), then the
            # actual crossing. Every real border/perimeter incident tends
            # to look like this anyway (something startles them, they
            # bolt, they cross) -- this is a more honest demonstration of
            # what the analytics actually watch for, not just one rule
            # in isolation.
            # The burst point is deliberately only 30% of the way to
            # end_point, not the geometric midpoint -- the midpoint of
            # two points chosen to straddle the line symmetrically sits
            # right ON the line, which made side_of() occasionally
            # already register the crossing on this call instead of the
            # final one, collapsing both alerts onto the same timestamp.
            # 30% stays clearly on the starting side.
            burst_point = (
                start_point[0] + 0.3 * (end_point[0] - start_point[0]),
                start_point[1] + 0.3 * (end_point[1] - start_point[1]),
            )
            session.analyzer.analyze(
                [_TestTrack(track_id=test_id, centroid=start_point)],
                session.zones, session.lines, camera_id, timestamp=now - 1.0,
            )
            burst_alerts = session.analyzer.analyze(
                [_TestTrack(track_id=test_id, centroid=burst_point, _speed=420.0)],
                session.zones, session.lines, camera_id, timestamp=now - 0.5,
            )
            crossing_alerts = session.analyzer.analyze(
                [_TestTrack(track_id=test_id, centroid=end_point)],
                session.zones, session.lines, camera_id, timestamp=now,
            )
            alerts = burst_alerts + crossing_alerts
            target_name = line.name

        else:  # kind == "zone"
            if not session.zones:
                raise HTTPException(400, "No restricted zone is currently saved for this camera. Draw and Save one first.")
            zone = session.zones[-1]
            xs = [p[0] for p in zone.polygon]
            ys = [p[1] for p in zone.polygon]
            centroid = (sum(xs) / len(xs), sum(ys) / len(ys))
            if not zone.contains(centroid):
                raise HTTPException(
                    500,
                    "Could not find a point inside this zone's polygon to test with -- "
                    "the shape may be self-intersecting or too irregular. Try redrawing it.",
                )
            test_id = session.tracker._next_id
            session.tracker._next_id += 1
            alerts = session.analyzer.analyze(
                [_TestTrack(track_id=test_id, centroid=centroid, first_seen=now - 15.0)],
                session.zones, session.lines, camera_id, timestamp=now,
            )
            target_name = zone.name

        if not alerts:
            return {
                "status": "no_alert",
                "message": "The test ran but produced no alert -- this is likely a brief cooldown "
                           "from a very recent real or test alert on the same geometry. Wait ~10s and try again.",
            }

        saved = []
        for a in alerts:
            a.message = f"[CALIBRATION TEST] {a.message}"
            a.id = save_alert(a)
            saved.append({
                "id": a.id, "message": a.message, "severity": a.severity,
                "alert_type": a.alert_type, "track_ids": a.track_ids,
                "timestamp": a.timestamp, "snapshot_hash": a.snapshot_hash,
            })
        return {"status": "alert_generated", "target_name": target_name, "alerts": saved}

    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        # Without this, an unhandled exception here falls through to
        # FastAPI's default plain-text 500 response, which is not JSON --
        # the frontend's res.json() then chokes on it with a confusing
        # "Unexpected token" error that hides what actually went wrong.
        # This logs the full traceback server-side (check your terminal)
        # AND returns a real, readable JSON error either way. Moved to
        # wrap the alert-saving step too, not just the analysis step --
        # the original version of this fix still let a save_alert()
        # failure escape uncaught, which is exactly what happened here
        # (save_alert was never imported into main.py at all).
        logging.getLogger("ibvap.zones").exception("Calibration test failed")
        raise HTTPException(500, f"Calibration test failed: {type(e).__name__}: {e}")


# --------------------------------------------------------------------
# Cross-camera appearance matching (heuristic -- see cross_camera.py's
# module docstring for exactly what this is and, more importantly,
# is NOT: color-histogram similarity, not real re-identification)
# --------------------------------------------------------------------
@app.get("/api/cross-camera/match")
def cross_camera_match(
    camera_a: str = Query(...), camera_b: str = Query(...), threshold: float = Query(0.65),
):
    if camera_a == camera_b:
        raise HTTPException(400, "camera_a and camera_b must be different sessions.")
    session_a = get_session(camera_a)
    session_b = get_session(camera_b)
    tracks_a = list(session_a.tracker.tracks.values()) + list(session_a.vehicle_tracker.tracks.values())
    tracks_b = list(session_b.tracker.tracks.values()) + list(session_b.vehicle_tracker.tracks.values())

    matches = find_cross_camera_matches(
        camera_a, session_a._last_frame, tracks_a,
        camera_b, session_b._last_frame, tracks_b,
        threshold=threshold,
    )
    return {
        "camera_a": camera_a, "camera_b": camera_b, "threshold": threshold,
        "tracks_compared": {"camera_a": len(tracks_a), "camera_b": len(tracks_b)},
        "matches": [m.__dict__ for m in matches],
        "note": ("Appearance heuristic (HSV colour histogram correlation), not real "
                 "re-identification -- can confuse similarly-coloured objects. See "
                 "cross_camera.py."),
    }

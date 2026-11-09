"""
storage.py
----------
Minimal, dependency-light persistence (stdlib sqlite3) for alerts, events,
and per-session analytics, PLUS a tamper-evident audit log.

Why an audit log, on a hackathon prototype: the official theme for this
problem statement is "Blockchain & Cybersecurity", but the spec itself is
pure computer vision -- there's no real blockchain use case to force in
here. Rather than bolt on decorative blockchain buzzwords, this applies
blockchain's actual core mechanism honestly: every alert and event is
additionally written into `audit_log` as a hash-chained record, where each
row's hash depends on the previous row's hash. Change a single byte of any
past record and every hash after it stops matching -- exactly the
tamper-evidence property a real investigation would need from a border
security incident log. This is NOT a distributed ledger / consensus
network (that would be theatre for a single-server system) -- it's the
one piece of blockchain's design that is genuinely useful here:
append-only, cryptographically chained, tamper-evident storage.

Each alert/event row also stores which audit_log row is its own snapshot
(`audit_id`), so verification can check two different kinds of tampering:
  1. Was the hash chain itself rewritten? (`verify_chain`)
  2. Does the live alerts/events table still match its own snapshot?
     (`verify_integrity` -- catches someone editing the visible table
     directly without touching audit_log at all)

Deliberately not an ORM: this keeps the prototype trivial to run anywhere
while remaining a straightforward swap-in for Postgres/Timescale in
production (same SQL surface, different driver).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import List, Optional

from app.core.analytics import Alert

DB_PATH = Path(__file__).resolve().parents[3] / "data" / "sentinelgrid.db"
SNAPSHOT_DIR = Path(__file__).resolve().parents[3] / "data" / "snapshots"
GENESIS_HASH = "0" * 64


def save_snapshot_file(image_bytes: bytes) -> str:
    """Saves an alert's evidence frame to disk, content-addressed by its
    own SHA-256 hash (data/snapshots/<hash>.jpg) -- so the hash is known
    BEFORE any database row exists, and can be embedded straight into
    the same audit_log payload that alert's hash chain entry covers.
    That's what makes the image itself tamper-evident, not just the text
    description of the alert: this hash is exactly what
    GET /api/alerts/{id}/snapshot recomputes on every read to prove the
    file on disk hasn't been swapped since. Content-addressing also
    means two alerts firing on the same frame share one file instead of
    duplicating it."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    path = SNAPSHOT_DIR / f"{image_hash}.jpg"
    if not path.exists():
        path.write_bytes(image_bytes)
    return image_hash


def get_conn() -> sqlite3.Connection:
    """WAL mode lets reads proceed without blocking the writer -- this app
    has several read paths (REST polling, dashboard queries) hitting the
    DB while writes are happening continuously from a live/replay loop."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.row_factory = sqlite3.Row
    return conn


# ---- single shared write connection ------------------------------------
# WAL mode fixes reader-vs-writer contention, but SQLite still only allows
# ONE writer at a time -- and this app can have a live simulation loop and
# a replay session both trying to write several times per frame. Opening a
# fresh connection per insert meant every one of those writes had to
# separately acquire SQLite's file-level write lock, which under real
# concurrent load caused either "database is locked" errors or (once a
# busy_timeout was added) long stalls waiting to retry.
#
# The actual fix: only ONE connection in this process ever writes, guarded
# by a plain Python lock. Since it's the same process serializing its own
# writes in Python before they ever reach SQLite, there's no file-lock
# contention to retry on at all -- writes just queue briefly in memory,
# which is fast. Reads still use their own short-lived connections (via
# get_conn()) and, under WAL, don't block on this writer.
_write_lock = threading.Lock()
_write_conn: Optional[sqlite3.Connection] = None


def _get_write_conn() -> sqlite3.Connection:
    global _write_conn
    if _write_conn is None:
        _write_conn = get_conn()
    return _write_conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, coltype: str) -> None:
    """Adds `column` to `table` if it's missing -- handles a database file
    that was created by an older version of this schema. CREATE TABLE IF
    NOT EXISTS does nothing to a table that already exists, even if its
    columns are now out of date, which is exactly what caused a real
    crash here: an existing `alerts` table from before the audit-log
    feature was added had no `audit_id` column, and the first INSERT that
    tried to use it failed with 'no such column'. This makes startup
    self-healing for that class of schema change instead of requiring
    anyone to manually delete the database file."""
    cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


def init_db() -> None:
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            message TEXT NOT NULL,
            track_ids TEXT NOT NULL,
            camera_id TEXT NOT NULL,
            location_x REAL,
            location_y REAL,
            timestamp REAL NOT NULL,
            acknowledged INTEGER DEFAULT 0,
            audit_id INTEGER,
            snapshot_hash TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            people_count INTEGER NOT NULL,
            frame_index INTEGER NOT NULL,
            vehicle_count INTEGER DEFAULT 0,
            night_mode INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            camera_id TEXT NOT NULL,
            details TEXT NOT NULL,
            timestamp REAL NOT NULL,
            audit_id INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_type TEXT NOT NULL,
            payload TEXT NOT NULL,
            timestamp REAL NOT NULL,
            prev_hash TEXT NOT NULL,
            hash TEXT NOT NULL
        )
    """)
    # Self-healing migration for databases created before audit_id existed
    # (this is exactly the bug that crashed replay on a pre-existing DB file).
    _ensure_column(conn, "alerts", "audit_id", "INTEGER")
    _ensure_column(conn, "events", "audit_id", "INTEGER")
    # Self-healing migration for a DB file that predates evidence snapshots.
    _ensure_column(conn, "alerts", "snapshot_hash", "TEXT")
    # Same self-healing pattern for an existing DB file that predates
    # vehicle_count/night_mode on session_stats.
    _ensure_column(conn, "session_stats", "vehicle_count", "INTEGER DEFAULT 0")
    _ensure_column(conn, "session_stats", "night_mode", "INTEGER DEFAULT 0")

    # A DB-Browser-friendly view of the hash chain: raw sha256 hex strings
    # are unreadable to a non-technical jury and the point of this table
    # isn't to be read by eye anyway -- tamper detection is a recompute,
    # not a glance. This view just makes browsing pleasant: a short
    # payload preview instead of raw JSON, truncated hashes so the chain
    # linkage is visually obvious, and a human timestamp. Note the
    # CASE: simulation-mode alerts use a virtual clock (small numbers
    # like 3.25 = 3.25s into the demo), not real wall-clock time, so
    # treating every timestamp as a unix epoch would show 1970 dates for
    # those rows -- this only converts genuinely large (real epoch)
    # values and labels the rest as virtual-clock offsets instead.
    conn.execute("""
        CREATE VIEW IF NOT EXISTS audit_trail_readable AS
        SELECT
            id,
            record_type,
            substr(payload, 1, 90) AS payload_preview,
            CASE
                WHEN timestamp > 1000000000 THEN datetime(timestamp, 'unixepoch', 'localtime')
                ELSE printf('t+%.1fs (virtual clock)', timestamp)
            END AS logged_at,
            substr(prev_hash, 1, 12) || '...' AS prev_hash_short,
            substr(hash, 1, 12) || '...' AS hash_short,
            prev_hash,
            hash
        FROM audit_log
        ORDER BY id
    """)
    conn.commit()
    conn.close()


# ---- hash chain internals ---------------------------------------------
def _compute_hash(prev_hash: str, record_type: str, payload_json: str, timestamp: float) -> str:
    canonical = f"{prev_hash}|{record_type}|{payload_json}|{timestamp}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _append_audit(conn: sqlite3.Connection, record_type: str, payload: dict, timestamp: float) -> int:
    """Appends one row to the hash chain using the given (open, uncommitted)
    connection, so the alert/event row and its audit snapshot land in the
    same transaction. Returns the new audit_log row id."""
    row = conn.execute("SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
    prev_hash = row["hash"] if row else GENESIS_HASH
    payload_json = json.dumps(payload, sort_keys=True)
    h = _compute_hash(prev_hash, record_type, payload_json, timestamp)
    cur = conn.execute(
        "INSERT INTO audit_log (record_type, payload, timestamp, prev_hash, hash) VALUES (?, ?, ?, ?, ?)",
        (record_type, payload_json, timestamp, prev_hash, h),
    )
    return cur.lastrowid


def save_alert(alert: Alert, snapshot_hash: Optional[str] = None) -> int:
    """Returns the database-assigned alert id (see the docstring from the
    previous round for why). snapshot_hash, when provided, is baked into
    BOTH the alerts row (queryable directly) and the audit_log payload
    for this record -- meaning the evidence photo's integrity is covered
    by the exact same hash chain as the alert's text fields, not a
    separate, weaker mechanism bolted on afterward."""
    loc = alert.location or (None, None)
    track_ids_str = ",".join(map(str, alert.track_ids))
    with _write_lock:
        conn = _get_write_conn()
        cur = conn.execute(
            """INSERT INTO alerts
               (alert_type, severity, message, track_ids, camera_id,
                location_x, location_y, timestamp, snapshot_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (alert.alert_type, alert.severity, alert.message, track_ids_str,
             alert.camera_id, loc[0], loc[1], alert.timestamp, snapshot_hash),
        )
        alert_id = cur.lastrowid
        payload = {
            "id": alert_id, "alert_type": alert.alert_type, "severity": alert.severity,
            "message": alert.message, "track_ids": track_ids_str, "camera_id": alert.camera_id,
            "location_x": loc[0], "location_y": loc[1], "timestamp": alert.timestamp,
            "snapshot_hash": snapshot_hash,
        }
        audit_id = _append_audit(conn, "alert", payload, alert.timestamp)
        conn.execute("UPDATE alerts SET audit_id = ? WHERE id = ?", (audit_id, alert_id))
        conn.commit()
        return alert_id


def save_stat(camera_id: str, people_count: int, frame_index: int, ts: Optional[float] = None,
              vehicle_count: int = 0, night_mode: bool = False) -> None:
    """vehicle_count and night_mode are new, optional, keyword-friendly
    params appended after the original signature -- every existing call
    site (save_stat(camera_id, people_count, frame_index, ts)) keeps
    working unchanged and simply records vehicle_count=0/night_mode=0
    until its call site is updated to pass the real values it already
    computes each frame (see pipeline.py)."""
    with _write_lock:
        conn = _get_write_conn()
        conn.execute(
            """INSERT INTO session_stats
               (camera_id, timestamp, people_count, frame_index, vehicle_count, night_mode)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (camera_id, ts or time.time(), people_count, frame_index,
             vehicle_count, 1 if night_mode else 0),
        )
        conn.commit()


def save_event(event_type: str, camera_id: str, details: str, timestamp: Optional[float] = None) -> None:
    """Event log for detection capabilities that are informational rather
    than security escalations -- face-detected, vehicle-classified,
    plate-read. Satisfies the spec's 'real-time... event logging'
    requirement independent of the alert/severity pipeline. Also chained
    into audit_log for the same tamper-evidence guarantee as alerts."""
    ts = timestamp if timestamp is not None else time.time()
    with _write_lock:
        conn = _get_write_conn()
        cur = conn.execute(
            "INSERT INTO events (event_type, camera_id, details, timestamp) VALUES (?, ?, ?, ?)",
            (event_type, camera_id, details, ts),
        )
        event_id = cur.lastrowid
        payload = {"id": event_id, "event_type": event_type, "camera_id": camera_id,
                   "details": details, "timestamp": ts}
        audit_id = _append_audit(conn, "event", payload, ts)
        conn.execute("UPDATE events SET audit_id = ? WHERE id = ?", (audit_id, event_id))
        conn.commit()


def fetch_events(limit: int = 100, event_type: Optional[str] = None,
                  camera_id: Optional[str] = None) -> List[dict]:
    conn = get_conn()
    q = "SELECT * FROM events"
    clauses, params = [], []
    if event_type:
        clauses.append("event_type = ?")
        params.append(event_type)
    if camera_id:
        clauses.append("camera_id = ?")
        params.append(camera_id)
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    q += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def fetch_alerts(limit: int = 100, severity: Optional[str] = None,
                  camera_id: Optional[str] = None) -> List[dict]:
    conn = get_conn()
    q = "SELECT * FROM alerts"
    clauses, params = [], []
    if severity:
        clauses.append("severity = ?")
        params.append(severity)
    if camera_id:
        clauses.append("camera_id = ?")
        params.append(camera_id)
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    q += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def fetch_alert_by_id(alert_id: int) -> Optional[dict]:
    conn = get_conn()
    row = conn.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def fetch_audit_trail(limit: int = 100) -> List[dict]:
    """Raw rows from the audit_trail_readable VIEW (see init_db()) --
    the human-friendly window onto audit_log: truncated hashes, a
    payload preview, and a real/virtual-clock-aware timestamp, instead
    of raw 64-char hex strings. Powers the standalone DB viewer page."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM audit_trail_readable ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def fetch_session_stats(limit: int = 100, camera_id: Optional[str] = None) -> List[dict]:
    """Raw session_stats rows -- no endpoint exposed these before; the
    dashboard only ever showed derived summaries (peak/avg occupancy).
    Powers the standalone DB viewer page's Session Stats table."""
    conn = get_conn()
    if camera_id:
        rows = conn.execute(
            "SELECT * FROM session_stats WHERE camera_id = ? ORDER BY id DESC LIMIT ?",
            (camera_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM session_stats ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def acknowledge_alert(alert_id: int) -> bool:
    """Marks an alert as reviewed by an operator (`acknowledged = 1`).
    This is a deliberate, ordinary operational update -- not a security
    escalation -- so it does NOT go through the hash-chained audit log
    the way the alert's original creation did. The audit trail's job is
    to prove the underlying INCIDENT record (type, severity, message,
    track_ids, timestamp) was never retroactively altered; whether an
    operator has since looked at it is normal triage workflow, tracked
    here the same way any alerting/SOC tool tracks it. Returns True if
    a row was actually found and updated."""
    with _write_lock:
        conn = _get_write_conn()
        cur = conn.execute("UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,))
        conn.commit()
        return cur.rowcount > 0


def fetch_summary(camera_id: Optional[str] = None) -> dict:
    conn = get_conn()
    where = " WHERE camera_id = ?" if camera_id else ""
    params = [camera_id] if camera_id else []

    total = conn.execute(f"SELECT COUNT(*) c FROM alerts{where}", params).fetchone()["c"]
    by_type = conn.execute(
        f"SELECT alert_type, COUNT(*) c FROM alerts{where} GROUP BY alert_type", params
    ).fetchall()
    by_severity = conn.execute(
        f"SELECT severity, COUNT(*) c FROM alerts{where} GROUP BY severity", params
    ).fetchall()
    peak = conn.execute(
        f"SELECT MAX(people_count) m FROM session_stats{where}", params
    ).fetchone()["m"]
    avg = conn.execute(
        f"SELECT AVG(people_count) a FROM session_stats{where}", params
    ).fetchone()["a"]
    peak_vehicles = conn.execute(
        f"SELECT MAX(vehicle_count) m FROM session_stats{where}", params
    ).fetchone()["m"]
    avg_vehicles = conn.execute(
        f"SELECT AVG(vehicle_count) a FROM session_stats{where}", params
    ).fetchone()["a"]
    conn.close()
    return {
        "total_alerts": total,
        "by_type": {r["alert_type"]: r["c"] for r in by_type},
        "by_severity": {r["severity"]: r["c"] for r in by_severity},
        "peak_occupancy": peak or 0,
        "avg_occupancy": round(avg, 2) if avg else 0,
        "peak_vehicle_count": peak_vehicles or 0,
        "avg_vehicle_count": round(avg_vehicles, 2) if avg_vehicles else 0,
    }


# ---- tamper-evidence verification --------------------------------------
def verify_chain() -> dict:
    """Walks the entire audit_log and recomputes every hash from scratch.
    Detects an attacker who edited audit_log rows directly -- any single
    changed byte breaks every hash from that point forward."""
    conn = get_conn()
    rows = conn.execute("SELECT * FROM audit_log ORDER BY id ASC").fetchall()
    conn.close()

    prev_hash = GENESIS_HASH
    for row in rows:
        expected = _compute_hash(prev_hash, row["record_type"], row["payload"], row["timestamp"])
        if row["prev_hash"] != prev_hash or row["hash"] != expected:
            return {"chain_intact": False, "broken_at_id": row["id"], "records_checked": len(rows)}
        prev_hash = row["hash"]

    return {"chain_intact": True, "broken_at_id": None, "records_checked": len(rows)}


def verify_integrity(check_last: int = 200) -> dict:
    """Cross-checks the most recent alerts/events against their own
    audit_log snapshot. Detects an attacker who edited the visible
    alerts/events table directly WITHOUT touching audit_log -- the chain
    math alone can't see that; this comparison can."""
    conn = get_conn()

    tampered_alerts = []
    for row in conn.execute(
        "SELECT * FROM alerts WHERE audit_id IS NOT NULL ORDER BY id DESC LIMIT ?", (check_last,)
    ):
        audit_row = conn.execute("SELECT payload FROM audit_log WHERE id = ?", (row["audit_id"],)).fetchone()
        if not audit_row:
            tampered_alerts.append(row["id"])
            continue
        snap = json.loads(audit_row["payload"])
        live = {
            "id": row["id"], "alert_type": row["alert_type"], "severity": row["severity"],
            "message": row["message"], "track_ids": row["track_ids"], "camera_id": row["camera_id"],
            "location_x": row["location_x"], "location_y": row["location_y"], "timestamp": row["timestamp"],
            "snapshot_hash": row["snapshot_hash"],
        }
        if live != snap:
            tampered_alerts.append(row["id"])

    tampered_events = []
    for row in conn.execute(
        "SELECT * FROM events WHERE audit_id IS NOT NULL ORDER BY id DESC LIMIT ?", (check_last,)
    ):
        audit_row = conn.execute("SELECT payload FROM audit_log WHERE id = ?", (row["audit_id"],)).fetchone()
        if not audit_row:
            tampered_events.append(row["id"])
            continue
        snap = json.loads(audit_row["payload"])
        live = {
            "id": row["id"], "event_type": row["event_type"], "camera_id": row["camera_id"],
            "details": row["details"], "timestamp": row["timestamp"],
        }
        if live != snap:
            tampered_events.append(row["id"])

    conn.close()
    return {
        "tampered_alert_ids": tampered_alerts,
        "tampered_event_ids": tampered_events,
        "records_checked": check_last,
    }


def verify_all() -> dict:
    chain = verify_chain()
    integrity = verify_integrity()
    ok = chain["chain_intact"] and not integrity["tampered_alert_ids"] and not integrity["tampered_event_ids"]
    return {"ok": ok, "chain": chain, "integrity": integrity}


# ---- clean-slate reset for a live jury demo ----------------------------
def reset_all_data() -> dict:
    """Wipes alerts/events/audit_log/session_stats to a genuinely EMPTY
    state -- no re-seeded rows of any kind. This is deliberately
    different from seed_tamper_demo(): that function fills the database
    with a small FAKE dataset built specifically to demonstrate the
    tamper-evidence feature; this one just clears out old test/demo
    data so the next live simulation or replay session starts from a
    clean database, and DB Browser only shows real, current tracking
    activity when opened in front of the jury -- not leftover rows from
    earlier testing."""
    with _write_lock:
        conn = _get_write_conn()
        conn.execute("DELETE FROM alerts")
        conn.execute("DELETE FROM events")
        conn.execute("DELETE FROM audit_log")
        conn.execute("DELETE FROM session_stats")
        # Resets SQLite's AUTOINCREMENT counters back to 1 too, so the
        # first alert/event/audit row after a reset is genuinely id=1
        # again -- tidier to look at in DB Browser than picking up from
        # wherever a previous session's ids left off.
        conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('alerts', 'events', 'audit_log', 'session_stats')")
        conn.commit()
    return {"status": "reset", "message": "All tracking data cleared. Database is empty and ready for a fresh session."}


# ---- bounded disk usage for evidence snapshots --------------------------
def prune_snapshots(max_total_bytes: int) -> dict:
    """Deletes the OLDEST snapshot files (by filesystem mtime) until
    data/snapshots/ is back under max_total_bytes. Content-addressed
    storage already dedupes identical frames, but a long-running live
    simulation still generates a fresh, genuinely unique photo for every
    alert -- and loitering alerts alone re-fire every ~10s while the
    situation continues, so this grows without bound if nothing prunes
    it. Safe to run at any time: pruning a snapshot only means
    GET /api/alerts/{id}/snapshot returns 410 for that one alert
    afterward (already handled gracefully) -- the alert's own text
    record and its place in the tamper-evident hash chain are completely
    untouched; only the accompanying photo is removed."""
    if not SNAPSHOT_DIR.exists():
        return {"deleted": 0, "freed_bytes": 0, "remaining_bytes": 0}
    files = sorted(SNAPSHOT_DIR.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
    total = sum(f.stat().st_size for f in files)
    deleted = 0
    freed = 0
    for f in files:
        if total <= max_total_bytes:
            break
        size = f.stat().st_size
        f.unlink(missing_ok=True)
        total -= size
        freed += size
        deleted += 1
    return {"deleted": deleted, "freed_bytes": freed, "remaining_bytes": total}


# ---- live tamper-evidence demo -----------------------------------------
# Built for the jury demo itself, not just documentation: these let a
# presenter seed a small known-clean dataset, launch either of the two
# real attacks described in the technical briefing against the live
# database, and watch /api/audit/verify flip from verified to tampered
# on screen. "Reset" wipes and reseeds rather than trying to undo a
# tamper in place, since a hash chain is deliberately append-only -- you
# can't patch history back to clean, only start a fresh chain, which is
# the honest behaviour to demonstrate, not a bug to work around.
DEMO_CAMERA_ID = "cam-tamper-demo"


def seed_tamper_demo() -> dict:
    """Wipes alerts/events/audit_log/session_stats and inserts a small,
    fixed, realistic demo set under DEMO_CAMERA_ID, each with a proper
    hash-chained audit entry. Returns the seeded record ids so the
    frontend can show exactly what now exists."""
    with _write_lock:
        conn = _get_write_conn()
        conn.execute("DELETE FROM alerts")
        conn.execute("DELETE FROM events")
        conn.execute("DELETE FROM audit_log")
        conn.execute("DELETE FROM session_stats")
        # Also reset AUTOINCREMENT, matching reset_all_data()'s behaviour
        # -- without this, a SECOND seed() call within the same server
        # run (e.g. a presenter re-running the demo, or attack1/attack2
        # having already advanced the counter) produces alert ids that
        # keep climbing (5,6,7... instead of 1,2,3...) instead of the
        # same predictable 1-4 range every time, which is confusing
        # during a live demo and was catching my own regression test.
        conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('alerts', 'events', 'audit_log', 'session_stats')")
        conn.commit()

    now = time.time()
    demo_alerts = [
        Alert("unauthorized_crossing", "critical",
              "Vehicle #1 crossed 'Border Fence Line' in unauthorized direction",
              [1], now - 40, DEMO_CAMERA_ID),
        Alert("group_formation", "medium", "Unusual cluster of persons (3) formed",
              [2, 3, 4], now - 30, DEMO_CAMERA_ID),
        Alert("loitering", "medium", "Person #5 loitering in restricted zone for 12s",
              [5], now - 20, DEMO_CAMERA_ID),
        Alert("erratic_motion", "high", "Person #6 moving erratically (410 px/s)",
              [6], now - 10, DEMO_CAMERA_ID),
    ]
    for a in demo_alerts:
        save_alert(a)
    save_event("anpr_read", DEMO_CAMERA_ID, "Vehicle #1 plate read: DL1CN4521", now - 35)
    save_event("face_detected", DEMO_CAMERA_ID, "Face detected on person #2", now - 25)

    conn = get_conn()
    alert_rows = conn.execute(
        "SELECT id, alert_type, severity, message FROM alerts WHERE camera_id = ? ORDER BY id ASC",
        (DEMO_CAMERA_ID,),
    ).fetchall()
    event_rows = conn.execute(
        "SELECT id, event_type, details FROM events WHERE camera_id = ? ORDER BY id ASC",
        (DEMO_CAMERA_ID,),
    ).fetchall()
    conn.close()
    return {
        "seeded_alerts": [dict(r) for r in alert_rows],
        "seeded_events": [dict(r) for r in event_rows],
    }


def simulate_visible_edit_attack() -> dict:
    """Attack 1: directly rewrites the most recent demo alert's message
    and severity in the `alerts` table, leaving audit_log completely
    untouched -- an attacker (or insider) with raw DB access who edits
    only the table an operator would actually look at."""
    conn = get_conn()
    row = conn.execute(
        "SELECT id, message, severity FROM alerts WHERE camera_id = ? ORDER BY id DESC LIMIT 1",
        (DEMO_CAMERA_ID,),
    ).fetchone()
    conn.close()
    if not row:
        return {"error": "No demo data seeded yet -- call seed first."}

    original = {"id": row["id"], "message": row["message"], "severity": row["severity"]}
    tampered_message = "Person cleared -- no threat detected"
    tampered_severity = "medium"

    with _write_lock:
        conn = _get_write_conn()
        conn.execute(
            "UPDATE alerts SET message = ?, severity = ? WHERE id = ?",
            (tampered_message, tampered_severity, row["id"]),
        )
        conn.commit()

    return {
        "attack": "visible_edit",
        "target_alert_id": row["id"],
        "before": original,
        "after": {"id": row["id"], "message": tampered_message, "severity": tampered_severity},
        "note": "audit_log was left untouched -- verify_integrity's cross-check should now flag this row.",
    }


def simulate_chain_rewrite_attack() -> dict:
    """Attack 2: edits the EARLIEST demo audit_log record's payload AND
    recomputes that row's own hash with the exact same formula the app
    uses, so the row looks internally self-consistent to a naive
    per-row check. Still breaks the chain, because the next record's
    stored prev_hash already committed to the old hash value."""
    conn = get_conn()
    # earliest audit_log row belonging to the demo dataset
    row = conn.execute(
        """SELECT audit_log.* FROM audit_log
           JOIN alerts ON alerts.audit_id = audit_log.id
           WHERE alerts.camera_id = ?
           ORDER BY audit_log.id ASC LIMIT 1""",
        (DEMO_CAMERA_ID,),
    ).fetchone()
    conn.close()
    if not row:
        return {"error": "No demo data seeded yet -- call seed first."}

    original_payload = json.loads(row["payload"])
    tampered_payload = dict(original_payload)
    if "severity" in tampered_payload:
        tampered_payload["severity"] = "low"
    if "message" in tampered_payload:
        tampered_payload["message"] = "Routine activity, no action needed"
    new_payload_json = json.dumps(tampered_payload, sort_keys=True)
    new_hash = _compute_hash(row["prev_hash"], row["record_type"], new_payload_json, row["timestamp"])

    with _write_lock:
        conn = _get_write_conn()
        conn.execute(
            "UPDATE audit_log SET payload = ?, hash = ? WHERE id = ?",
            (new_payload_json, new_hash, row["id"]),
        )
        conn.commit()

    return {
        "attack": "chain_rewrite",
        "target_audit_log_id": row["id"],
        "before_payload": original_payload,
        "after_payload": tampered_payload,
        "note": "This row's own hash was recomputed to match -- it looks self-consistent. "
                "The break shows up one record downstream, where prev_hash still points to the old hash.",
    }

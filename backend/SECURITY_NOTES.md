# Security Hardening Notes (Tier 3)

These two items are documented rather than silently implemented, for a
specific reason explained under each one.

---

## 1. Encryption at rest for sentinelgrid.db

**Status: documented migration path, not implemented.**

### Why this isn't just done automatically

Real encryption at rest for SQLite means swapping the database driver to
**SQLCipher** (via the `sqlcipher3` or `pysqlcipher3` Python package).
That's a real dependency change, not a config flag:

- It requires the SQLCipher native library to be present on the machine
  (on Windows this usually means either a prebuilt wheel, which isn't
  guaranteed to exist for every Python version, or building from source,
  which needs a C compiler + OpenSSL dev headers).
- It is NOT guaranteed to `pip install` cleanly on a machine we haven't
  tested it on -- silently adding this to `requirements.txt` risks
  breaking `setup.bat` for teammates on a machine where it doesn't build,
  right before a demo.
- The existing `sentinelgrid.db` file would need a one-time migration
  (export/re-import under the new encrypted driver) -- not a drop-in
  replacement.

Given this project's own repeated principle so far ("do not change
anything from the current system" without it being verified), silently
swapping the DB driver is exactly the kind of change that shouldn't be
made without the team explicitly choosing to do it and testing it on
their actual machine first.

### The migration path, if/when you want it

1. `pip install sqlcipher3-binary` (the `-binary` variant bundles a
   prebuilt SQLCipher, which is the one most likely to actually install
   cleanly on Windows without a compiler toolchain -- verify this on
   your machine before relying on it for a demo).
2. In `storage.py`, `get_conn()` and `_get_write_conn()` currently do
   `sqlite3.connect(DB_PATH, ...)`. Swap the import from `sqlite3` to
   `sqlcipher3` (the API is designed to be a drop-in replacement) and add
   one line immediately after connecting:
   ```python
   conn.execute(f"PRAGMA key = '{ENCRYPTION_KEY}'")
   ```
   where `ENCRYPTION_KEY` should come from an environment variable (the
   same pattern already used for `IBVAP_API_KEY` in `main.py`), never
   hardcoded.
3. To migrate the existing plaintext `sentinelgrid.db`: open it with
   plain `sqlite3`, `ATTACH DATABASE 'encrypted.db' AS enc KEY '...'`,
   then `SELECT sqlcipher_export('enc')`, per SQLCipher's own documented
   migration procedure.
4. Test the full seed -> attack1 -> attack2 -> verify cycle again after
   migrating -- the hash-chain logic itself doesn't change, but this is
   exactly the kind of change worth re-confirming end to end before
   trusting it in front of a jury.

---

## 2. Load / performance benchmarking

**Status: deliberately not attempted, with a specific reason.**

I could run many concurrent requests against my own local instance and
report a number. I'm not doing that, because the result would be
actively misleading: this sandbox's detector/tracker/analyzer stack is
made of minimal stand-in modules built purely to let me test the *other*
changes in this session in isolation -- they return empty detections
instantly, doing essentially none of the real work (HOG/YOLO inference,
OpenCV frame processing, OCR) that dominates real request latency.

A benchmark run against that stack would measure the FastAPI/WebSocket
plumbing's overhead, not the system's real throughput -- and a number
like "500 req/s" reported without that caveat could easily get quoted
later as if it were a real performance claim, which would be a genuinely
bad thing to have said in front of a jury who then reasonably assumes it
reflects the real detector pipeline.

### If you want a real number

Run this against the actual deployment, with the real `pipeline.py` and
real detector backends loaded (lightweight or YOLOv8, whichever you're
demoing):

```bash
pip install --break-system-packages websockets
python3 -c "
import asyncio, websockets, time

async def one_connection(i):
    uri = 'ws://localhost:8000/api/ws/live/cam-loadtest-%d?api_key=YOUR_KEY' % i
    async with websockets.connect(uri) as ws:
        start = time.time()
        for _ in range(50):
            await ws.recv()
        return 50 / (time.time() - start)

async def main():
    results = await asyncio.gather(*[one_connection(i) for i in range(10)])
    print('per-connection frames/sec:', results)
    print('aggregate frames/sec:', sum(results))

asyncio.run(main())
"
```

Report whatever this actually gives you, on the actual hardware you'll
demo on, with the real detector backend active -- that number is real
and defensible; anything generated against the stub stack in this
session would not be.

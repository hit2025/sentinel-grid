# Deploying IBVAP to Render (Free Tier)

## Before anything else — the one thing you need to know

**Render's free tier does NOT give you persistent disk storage.** Every time
your service goes idle and spins back up (free services sleep after 15
minutes of no traffic), it restarts from a clean copy of your code — which
means `data/sentinelgrid.db` and `data/snapshots/` get **wiped back to
empty** every time that happens. This isn't a bug in anything we've built;
it's how Render's free tier works for every project, not just this one.

**What this means practically:**
- Alerts, the audit chain, evidence photos — none of it survives a
  sleep/wake cycle on the free tier.
- If a jury visits your link after it's been idle, they'll see an empty,
  freshly-initialized database, not whatever you demoed last.

**How to handle this for a jury demo — pick one:**
1. **Visit the link yourself 2-3 minutes before the jury does**, to wake
   the service up, then click **"Seed Clean Demo Data"** (Audit Demo panel)
   or run the live simulation for a minute so there's real data to look at.
2. **Keep the tab open and the simulation running** in the time leading up
   to the demo — an *active* connection prevents the free tier from
   sleeping in the first place.
3. If persistence genuinely matters for your submission (not just the live
   demo), Render's paid tier ($7/mo Starter plan) adds a real persistent
   disk — a one-line change to `render.yaml`, not a code change. Not
   required for what you asked for (a free live link), just worth knowing
   the option exists.

None of this is a limitation of your actual system — the tamper-evident
audit chain, the detection pipeline, all of it works exactly the same. It's
specifically about what "free" costs you on the hosting side.

---

## Option A — One-click deploy with the included `render.yaml`

1. Push your project to a GitHub repository (Render deploys from GitHub,
   GitLab, or Bitbucket — not a local zip upload).
2. Go to <https://dashboard.render.com> → **New** → **Blueprint**.
3. Connect the repository. Render will detect `render.yaml` at the repo
   root automatically and show you the service it's about to create.
4. Click **Apply**. Render builds and deploys automatically.
5. Once live, Render generated a **random, secure API key** for you
   (via `generateValue: true` in the blueprint) — find it under your
   service's **Environment** tab, variable `IBVAP_API_KEY`. That's the
   key you'll type into the dashboard's login prompt.

## Option B — Manual setup via the dashboard (if you'd rather see every step)

1. Push your project to GitHub, same as above.
2. Render dashboard → **New** → **Web Service** → connect your repo.
3. Fill in these fields exactly:

| Field | Value |
|---|---|
| **Root Directory** | `backend` |
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| **Instance Type** | Free |

4. Under **Environment Variables**, add:

| Key | Value |
|---|---|
| `IBVAP_API_KEY` | *(click "Generate" for a random secure value, or type your own)* |
| `IBVAP_MAX_SNAPSHOT_MB` | `200` *(lower than your local 500MB default — free tier's ephemeral disk is small, and it resets anyway; no reason to reserve more)* |
| `PYTHON_VERSION` | `3.12.0` *(match whatever you're running locally)* |

5. Click **Create Web Service**. First deploy takes a few minutes.

**Important — do NOT set `IBVAP_API_KEY` to the default
`ibvap-demo-key-2026-CHANGE-ME`** once this is on a public URL. Anyone with
that link and the default key has full access, including the "Reset DB"
button. Always generate or set a real key for anything public-facing.

---

## What your jury actually gets

- **Live dashboard**: `https://<your-service-name>.onrender.com/`
- **Live database viewer**: `https://<your-service-name>.onrender.com/db-viewer`
  — a read-only, web-based view of all four tables (alerts, events,
  audit_trail_readable, session_stats) plus the same tamper-verification
  status the main dashboard shows. No DB Browser install needed on their
  end — just the link and your API key.

Both pages ask for the same API key once per browser tab (stored only in
that tab's `sessionStorage`, never sent anywhere else).

---

## A few things worth setting expectations on

- **First request after sleeping is slow** (10-30 seconds) — Render's free
  tier "cold starts" the container. Worth mentioning to the jury so a slow
  first load doesn't read as the app being broken.
- **No YOLO/ultralytics on free tier** — 512MB RAM is too tight for it
  reliably. The offline HOG/MOG2 detectors (already your fallback) are
  what will actually run here, which is fine — that's exactly the
  "works anywhere, no GPU" story the problem statement asks for anyway.
  Don't add `ultralytics` to `requirements.txt` for this deployment.
- **RTSP won't reach anything** — Render's free tier has outbound network
  access, but a real camera's RTSP stream needs to be reachable from
  Render's servers, which it won't be for a camera on your home/college
  network. Not relevant for a jury demo either way.

---

## Honesty note on what I could and couldn't verify

I tested that your app starts correctly with a dynamic `$PORT` (exactly how
Render invokes it) and binds to `0.0.0.0` correctly — both confirmed
locally. I could not actually create a Render account and deploy from this
environment, so the dashboard field names/labels above are accurate as of
my knowledge but I'd recommend having your actual deploy screen open
side-by-side with this guide the first time, in case Render's UI has
shifted slightly since.

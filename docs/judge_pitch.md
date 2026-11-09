# IBVAP — judge pitch & demo script

Written the way a 3x SIH evaluator reads a pitch: does the team understand
the problem, is the demo real, and does the team know what they *haven't*
built yet. Use this as your run sheet, not a script to read verbatim.

**Sponsor:** Ministry of Home Affairs / Sashastra Seema Bal (SSB), Police II
Division. **Theme:** Blockchain & Cybersecurity. Keep that theme label in
mind if asked to justify the category — the honest answer is that this
statement's actual content is video analytics/AI, filed under that theme
by the sponsor; don't invent a blockchain angle that isn't there.

## 1. The 30-second framing (say this first, before any slide)

> "SSB posts already have CCTV at border out-posts and check posts. What
> they don't have is FRS, ANPR, or intrusion detection — because that
> normally means expensive dedicated hardware. IBVAP turns the cameras
> they already own into all of that, in software."

Do not open with your tech stack. Open with the operational gap the spec
itself names: "conventional CCTV... requiring continuous human observation."

## 2. Problem -> insight -> solution (60-90 seconds)

- **Problem**: the spec is explicit — no FRS/ANPR/smart-camera hardware,
  and eight named capabilities: human tracking, vehicle
  detection+classification, face detection, ANPR, virtual fence intrusion,
  suspicious activity, night-time movement, real-time alerts + logging.
- **Insight**: most hackathon video-analytics demos build one or two of
  these and call it done. The eight capabilities aren't independent
  features — they share one pipeline (detect -> track -> analyze -> log),
  and building them as one system instead of eight bolted-on modules is
  what actually makes this deployable, not just demoable.
- **Solution**: a single FastAPI pipeline where person and vehicle
  detection both feed the same behavioral analyzer (loitering, virtual
  fence crossing, group/convoy formation, erratic motion), face detection
  and ANPR run on top as targeted sub-analyses, and night-time conditions
  automatically engage low-light enhancement and raise the alert
  threshold — all through a backend that swaps its offline demo detectors
  for YOLOv8 in production without touching anything else.

## 3. Live demo (3-4 minutes — this is what actually wins the round)

Run `./run_local.sh`, open `frontend/index.html`. Narrate live, don't just
watch it:

1. **0:00** — "Live feed. Every box is a tracked person or vehicle with a
   persistent ID — that's what lets us reason about behavior over time,
   not just detect something in one frame."
2. **~0:08** — Loitering alert fires. "Person in the restricted buffer
   zone past a safe dwell time."
3. **~0:20** — Vehicle appears at the bottom, plate box drawn. "That
   plate read isn't hand-typed — it's Tesseract OCR actually reading the
   rendered plate live." Point at the track label showing the plate text.
4. **~0:35** — Group formation alert. "Three people converged near the
   fence — density-clustered, not a fixed headcount rule."
5. **~1:00** — Unauthorized crossing (critical, red). "This is the one
   that pages someone. Directional, calibrated per camera."
6. **~4:40** — Night segment: canvas dims, "NIGHT MODE" badge appears,
   night-time movement alerts fire. "Brightness-triggered — the system
   automatically enhanced the feed and raised alert severity, it isn't a
   separate mode someone has to switch on."
7. **Point at the alert rail vs. the detection log below it**: "Two
   different logs on purpose — the alert rail is what a watch commander
   acts on; the detection log is the audit trail an investigating officer
   pulls later. Faces and plates go there, not into the alert stream,
   because not every detection is an incident."
8. **Switch to real footage**: upload a real clip via `/docs` hitting
   `POST /api/upload`. "Same exact pipeline — face detection runs for
   real here, on real imagery, which the synthetic demo deliberately
   doesn't attempt." Show the JSON report with `plates_read` and
   `events_logged`.

## 4. Anticipate the four questions a judge asks

**"Does this actually work on real footage, or just your simulation?"**
Show `/api/upload` on a real video, returning real alerts and (if a
vehicle with a legible plate is in frame) a real OCR read. Say plainly:
person/vehicle detection is HOG/MOG2 offline-safe by default; the
production swap to YOLOv8 is one file per detector and doesn't touch
tracking, analytics, or storage — show `build_detector()` /
`build_vehicle_detector()` if pushed.

**"Is the ANPR read in the demo faked?"**
No — say so directly and explain why: the demo renders a real plate image
(pixels, not a string) onto the canvas each frame, and the same Haar
cascade + Tesseract OCR code that would run on a real camera crop reads
it. That's *why* the demo's OCR output has the occasional 0/O or 1/I
misread you'll notice — genuine OCR noise, not scripted text.

**"What about face recognition, not just detection?"**
We built detection (localizing a face), not recognition (matching against
a watchlist) — say this is a deliberate scope line, not a missing
feature: enrollment, a watchlist database, and a legal/privacy framework
for a border security deployment are a different, larger piece of work
than a hackathon prototype should claim to have solved.

**"What happens with 10 cameras instead of 1?"**
Each camera already runs its own independent session
(`get_session(camera_id)` in `main.py`) — that works today. What's *not*
built is cross-camera re-identification (recognizing the same
person/vehicle moving between camera views) — name that as phase 2.

## 5. If asked about accuracy numbers

Don't invent a precision/recall figure you haven't measured. Say: "We
haven't benchmarked against a labeled surveillance/ANPR dataset yet —
that's the right next step before a pilot, ideally against SSB's own
footage rather than a public dataset that won't match real border camera
conditions (angle, distance, weather, plate wear)."

## 6. Slide order (if you also need a deck)

1. The operational gap, in the spec's own words (one line, no chart)
2. All eight required capabilities, one line each, marked done
3. Architecture diagram (from README.md — Mermaid renders cleanly as an
   image export)
4. Live demo (this document, section 3)
5. What's real vs. what's roadmap (README sections 6 + 7, verbatim —
   judges respect this more than a polished overclaim)
6. Ask: what you need to pilot this (sample SSB CCTV footage access, one
   deployment site) — end on a concrete next step, not "thank you"

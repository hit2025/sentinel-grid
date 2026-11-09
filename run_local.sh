#!/usr/bin/env bash
# Quick local run without Docker -- what to use during the SIH demo itself
# (Docker adds startup latency you don't want on stage).
set -e

cd "$(dirname "$0")/backend"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt

echo ""
echo "Starting IBVAP backend on http://localhost:8000"
echo "Open frontend/index.html in a browser once this is running."
echo ""

PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

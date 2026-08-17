#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
echo "[1/3] Checking dependencies..."
python3 -m pip install -r requirements.txt --quiet
echo "[2/3] Opening browser..."
(sleep 2 && open http://localhost:8000 2>/dev/null || xdg-open http://localhost:8000 2>/dev/null) &
echo "[3/3] Starting BDH x FPGA Server on http://localhost:8000..."
python3 -m uvicorn server.app:app --host 127.0.0.1 --port 8000

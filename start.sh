#!/bin/bash
# Start both backend and frontend for development.
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "▶ Starting FastAPI backend on :8000"
cd "$ROOT"
.venv/bin/uvicorn backend.main:app --reload --port 8000 &
BACKEND_PID=$!

echo "▶ Starting React frontend on :5173"
cd "$ROOT/frontend"
npm run dev &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM

echo ""
echo "  Backend : http://localhost:8000"
echo "  Frontend: http://localhost:5173"
echo ""
echo "  Press Ctrl+C to stop."
echo ""

wait

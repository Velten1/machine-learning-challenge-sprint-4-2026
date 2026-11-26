#!/bin/sh
set -e
PORT_TO_USE="${ML_PORT:-${PORT:-8001}}"
exec python -m uvicorn main:app --host 0.0.0.0 --port "$PORT_TO_USE"

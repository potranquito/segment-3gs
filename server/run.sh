#!/usr/bin/env bash
#
# Boot the segmentation inference server.
#
#   ./run.sh                 # default backend (SAM_MODEL=mock unless exported)
#   SAM_MODEL=mock ./run.sh
#   SAM_MODEL=mobile_sam ./run.sh
#   SAM_MODEL=sam3 ./run.sh   # requires gated weights + hf auth login (see README)
#
set -euo pipefail

cd "$(dirname "$0")"

VENV=".venv"
PYTHON="${PYTHON:-python3}"
PORT="${PORT:-8765}"
export SAM_MODEL="${SAM_MODEL:-mock}"

# Let unsupported MPS ops fall back to CPU instead of crashing.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Keep ultralytics / matplotlib caches inside the server dir (writable).
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$PWD/Ultralytics}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.matplotlib}"

if [ ! -d "$VENV" ]; then
  echo "[run.sh] creating venv at $VENV"
  "$PYTHON" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "[run.sh] installing requirements"
pip install --upgrade pip >/dev/null
pip install -r requirements.txt

echo "[run.sh] starting uvicorn on :$PORT (SAM_MODEL=$SAM_MODEL, MPS fallback on)"
exec uvicorn app:app --host 0.0.0.0 --port "$PORT"

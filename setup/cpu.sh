#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="$PWD/.cache/uv"
uv venv --python "${PYTHON:-python3}" --allow-existing .venv
if [ -d wheelhouse ]; then
  uv pip install --python .venv/bin/python --no-index --find-links wheelhouse -r requirements-cpu.lock
else
  uv pip install --python .venv/bin/python -r requirements-cpu.lock
fi
.venv/bin/python tools/doctor.py --output setup/environment.json

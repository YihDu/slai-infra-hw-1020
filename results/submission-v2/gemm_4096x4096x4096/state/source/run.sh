#!/bin/sh
set -eu
task_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="$task_root"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-4}"
export TILELANG_DISABLE_CACHE=1
export TILELANG_AUTO_TUNING_DISABLE_CACHE=1
exec "$task_root/.venv/bin/python" -m optimizer.runner "$@"

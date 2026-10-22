#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
uv venv --python "${SLAI_PYTHON:-python3.13}" --allow-existing .venv-metal
uv pip install --python .venv-metal/bin/python -r setup/mac-numeric.lock
.venv-metal/bin/python -c 'import torch,tilelang; assert torch.backends.mps.is_available(); assert hasattr(torch.mps,"compile_shader"); print("Metal ready",torch.__version__,tilelang.__version__)'

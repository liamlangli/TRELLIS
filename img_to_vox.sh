#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "缺少本地 Python 环境，请先运行：$ROOT/setup.sh" >&2
    exit 1
fi
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa SPARSE_CONV_BACKEND=none
export CONVERT_MODE=direct PIPELINE_TYPE=512
if [[ $# -eq 0 ]]; then
    mkdir -p "$ROOT/vox"
    set -- --input_folder "$ROOT/vox" --skip
fi
# Keep the caller's working directory so relative input/output arguments work.
exec "$PYTHON" "$ROOT/img_to_vox.py" "$@"

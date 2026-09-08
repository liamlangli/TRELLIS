#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
    echo '此安装脚本需要原生 Apple Silicon macOS 环境。' >&2
    exit 1
fi
PYTHON="${PYTHON:-python3}"
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    "$PYTHON" -m venv "$ROOT/.venv"
fi
"$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements.txt" -c "$ROOT/requirements.lock.txt"
"$ROOT/.venv/bin/python" -c 'import torch; assert torch.backends.mps.is_available(), "MPS unavailable"; print("PyTorch", torch.__version__, "MPS OK")'
mkdir -p "$ROOT/vox"
echo '环境已就绪。把图片放进 vox/，运行 ./img_to_vox.sh；首次转换自动下载模型权重。'

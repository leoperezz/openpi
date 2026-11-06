#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if ! command -v uv >/dev/null 2>&1 || [[ ! -x .venv/bin/python ]]; then
    echo 'Run bash bin/install.sh first.' >&2
    exit 1
fi
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export GIT_LFS_SKIP_SMUDGE=1
exec uv run --frozen python scripts/train_pi05.py "$@"

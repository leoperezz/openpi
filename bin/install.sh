#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if ! command -v uv >/dev/null 2>&1; then
    python3 -m pip install --user uv
    export PATH="$HOME/.local/bin:$PATH"
fi
export GIT_LFS_SKIP_SMUDGE=1
git submodule update --init --recursive
uv sync --frozen
uv run --no-sync python -c 'import av, jax, yaml; import openpi; print("JAX devices:", jax.devices())'
echo 'Installation complete. Run: bash bin/train.sh NONHUMAN-RESEARCH/hanoi-rewards-triple'

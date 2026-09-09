#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

uv sync --extra dev --frozen
uv run traceguard doctor
uv run pytest
uv run traceguard experiment \
  --provider fixture \
  --n-per-cell "${N_PER_CELL:-2}" \
  --seed "${SEED:-20260710}"

echo "Fixture verification complete. It validates plumbing only, not the paper's empirical claims."


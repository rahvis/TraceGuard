#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

python3 -m json.tool src/traceguard/data/synthetic-medical-v2.json >/dev/null
# v1 stays on disk and stays valid: archived receipts embed its dataset_hash.
python3 -m json.tool src/traceguard/data/synthetic-medical-v1.json >/dev/null
python3 -m json.tool src/traceguard/data/paper-reported-unverified.json >/dev/null
python3 -m json.tool src/traceguard/data/synthetic-aml-v1.json >/dev/null

# Credential scanning lives in one place (scripts/scan_secrets.py) so this
# script, .github/workflows/ci.yml and tests/test_publication.py cannot drift
# apart. It scans git-tracked files and self-tests its patterns first.
python3 scripts/scan_secrets.py

# --frozen installs exactly what uv.lock records, which is the point: it is how a
# reader gets the analysis environment rather than whatever resolves today. If the
# lock is absent, sync without it rather than failing the whole gate.
if [ -f uv.lock ]; then
  uv sync --extra dev --frozen
else
  echo "no uv.lock; syncing against the ranges in pyproject.toml instead" >&2
  uv sync --extra dev
fi
uv run ruff check src tests scripts
uv run pytest
uv run traceguard doctor

# Docker is optional for reproducing numbers and is only used for the packaged
# deployment, so validate the compose file when docker is present and skip it
# otherwise rather than failing a machine that will never build the image.
if command -v docker >/dev/null 2>&1; then
  docker compose config >/dev/null
else
  echo "docker not present; skipping the compose validation" >&2
fi
echo "Release verification passed. Live-paper findings still require an independent API-backed run."

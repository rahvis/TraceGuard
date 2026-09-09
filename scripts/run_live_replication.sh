#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

# Live replication runs against one Azure OpenAI resource. The direct OpenAI and
# Anthropic providers were removed; archived journals naming them remain readable
# but cannot be regenerated.
if [ -z "${AZURE_OPENAI_API_KEY:-}" ]; then
  echo "AZURE_OPENAI_API_KEY is not set. Export a rotated key in your shell; never put it in this repository." >&2
  exit 2
fi
if [ -z "${AZURE_OPENAI_ENDPOINT:-}" ]; then
  echo "AZURE_OPENAI_ENDPOINT is not set. Use the resource host (https://<resource>.services.ai.azure.com), not an /api/projects/<name> Foundry project URL." >&2
  exit 2
fi

uv sync --extra dev --frozen
uv run traceguard doctor
uv run traceguard experiment \
  --provider azure \
  --n-per-cell "${N_PER_CELL:-1}" \
  --seed "${SEED:-20260710}"

echo "Live Azure OpenAI replication complete. Treat its new run manifest/results as a replication, not as the original paper run."

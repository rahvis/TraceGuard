#!/usr/bin/env bash
# Delete the TraceGuard Confidential VM and all its resources.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${HERE}/deploy.env" ]; then
  # shellcheck disable=SC1091
  set -a && . "${HERE}/deploy.env" && set +a
fi

RESOURCE_GROUP="${RESOURCE_GROUP:-traceguard-cc}"
SUBSCRIPTION_ID="${SUBSCRIPTION_ID:-}"

command -v az >/dev/null 2>&1 || { echo "ERROR: az CLI not found." >&2; exit 1; }
if [ -n "$SUBSCRIPTION_ID" ]; then az account set --subscription "$SUBSCRIPTION_ID"; fi

if ! az group show --name "$RESOURCE_GROUP" >/dev/null 2>&1; then
  echo "Resource group '$RESOURCE_GROUP' does not exist. Nothing to do."
  exit 0
fi

echo "About to DELETE resource group '$RESOURCE_GROUP' and everything in it."
read -r -p "Type the resource group name to confirm: " CONFIRM
[ "$CONFIRM" = "$RESOURCE_GROUP" ] || { echo "Confirmation did not match. Aborting."; exit 1; }

echo "==> Deleting resource group '$RESOURCE_GROUP'…"
az group delete --name "$RESOURCE_GROUP" --yes --no-wait
echo "==> Deletion started (running in the background). Verify with: az group list -o table"

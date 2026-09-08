#!/usr/bin/env bash
# TraceGuard — one-command Azure Confidential VM deployment.
#
# Provisions an AMD SEV-SNP Confidential VM (vTPM + Secure Boot + guest attestation),
# builds and runs the TraceGuard container on it, and prints the frontend URL. The
# in-guest app then produces a real, MAA-verified attestation you can check live.
#
# Usage:
#   cp deploy.env.example deploy.env && edit deploy.env
#   ./deploy.sh                      # reads deploy.env / environment
#
# All inputs are parameters — no secrets are hardcoded. See README.md.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Load config: deploy.env (if present) then environment overrides.
# ---------------------------------------------------------------------------
if [ -f "${HERE}/deploy.env" ]; then
  # shellcheck disable=SC1091
  set -a && . "${HERE}/deploy.env" && set +a
fi

SUBSCRIPTION_ID="${SUBSCRIPTION_ID:-}"
TENANT_ID="${TENANT_ID:-}"
RESOURCE_GROUP="${RESOURCE_GROUP:-traceguard-cc}"
LOCATION="${LOCATION:-eastus2}"
NAME_PREFIX="${NAME_PREFIX:-traceguard}"
VM_SIZE="${VM_SIZE:-Standard_DC4as_v5}"
ADMIN_USERNAME="${ADMIN_USERNAME:-azureuser}"
SSH_PUBLIC_KEY_FILE="${SSH_PUBLIC_KEY_FILE:-$HOME/.ssh/id_rsa.pub}"
IMAGE_OFFER="${IMAGE_OFFER:-ubuntu-24_04-lts}"
IMAGE_SKU="${IMAGE_SKU:-cvm}"
OSDISK_ENC_TYPE="${OSDISK_ENC_TYPE:-VMGuestStateOnly}"
REPO_URL="${REPO_URL:-}"
REPO_BRANCH="${REPO_BRANCH:-main}"
MAA_ENDPOINT="${MAA_ENDPOINT:-https://sharedeus2.eus2.attest.azure.net}"
SSH_SOURCE_CIDR="${SSH_SOURCE_CIDR:-*}"
APP_SOURCE_CIDR="${APP_SOURCE_CIDR:-*}"
# build = classic demo path (clone + docker build on the VM).
# ghcr  = CI/CD host prep only; GitHub Actions delivers the app from GHCR.
BOOTSTRAP_MODE="${BOOTSTRAP_MODE:-build}"
# No provider secret is ever embedded in VM custom-data. The GitHub Actions
# deploy job streams the runtime environment over SSH instead, so the key never
# lands in an Azure-visible field. Without it the VM runs the offline fixture
# provider, which is still a fully attested demo.

fail() { echo "ERROR: $*" >&2; exit 1; }

command -v az >/dev/null 2>&1 || fail "Azure CLI (az) not found. Install: https://learn.microsoft.com/cli/azure/install-azure-cli"
case "$BOOTSTRAP_MODE" in
  build)
    [ -n "$REPO_URL" ] || fail "REPO_URL is required (a git URL this VM can clone). Set it in deploy.env. See README.md for the ACR alternative if your repo is private."
    ;;
  ghcr) ;;
  *) fail "BOOTSTRAP_MODE must be 'build' or 'ghcr'" ;;
esac
[ -f "$SSH_PUBLIC_KEY_FILE" ] || fail "SSH public key not found at $SSH_PUBLIC_KEY_FILE. Generate one with: ssh-keygen -t ed25519"

# ---------------------------------------------------------------------------
# Azure login / subscription.
# ---------------------------------------------------------------------------
if ! az account show >/dev/null 2>&1; then
  echo "==> Not logged in. Launching az login…"
  if [ -n "$TENANT_ID" ]; then az login --tenant "$TENANT_ID" >/dev/null; else az login >/dev/null; fi
fi
if [ -n "$SUBSCRIPTION_ID" ]; then
  az account set --subscription "$SUBSCRIPTION_ID"
fi
ACTIVE_SUB="$(az account show --query id -o tsv)"
echo "==> Subscription: $ACTIVE_SUB"
echo "==> Resource group: $RESOURCE_GROUP  Location: $LOCATION  Size: $VM_SIZE"

# ---------------------------------------------------------------------------
# Confirm the size is a confidential (SEV-SNP) family and available in-region.
# ---------------------------------------------------------------------------
echo "==> Checking that $VM_SIZE is available in ${LOCATION}…"
if ! az vm list-skus --location "$LOCATION" --size "${VM_SIZE%%_*}" --query "[?name=='$VM_SIZE'] | [0].name" -o tsv 2>/dev/null | grep -q .; then
  echo "    WARNING: could not confirm $VM_SIZE in $LOCATION. Confidential sizes are region-limited."
  echo "    List options:  az vm list-skus --location $LOCATION --query \"[?family=='standardDCASv5Family'].name\" -o tsv"
fi

# ---------------------------------------------------------------------------
# Accept the Ubuntu CVM marketplace image terms (idempotent; ignore if none).
# ---------------------------------------------------------------------------
az vm image terms accept --publisher Canonical --offer "$IMAGE_OFFER" --plan "$IMAGE_SKU" >/dev/null 2>&1 || true

# ---------------------------------------------------------------------------
# Render cloud-init and base64-encode it.
# ---------------------------------------------------------------------------
TMP_INIT="$(mktemp)"
trap 'rm -f "$TMP_INIT" "${TMP_INIT}.b64" "${TMP_PARAMS:-}"' EXIT
if [ "$BOOTSTRAP_MODE" = "ghcr" ]; then
  # Host prep only — static file, no placeholders, no secrets in custom-data.
  echo "    CI/CD host-prep bootstrap: GitHub Actions will deliver the app from GHCR."
  cp "${HERE}/cloud-init-ghcr.yaml" "$TMP_INIT"
else
  sed -e "s|@@REPO_URL@@|${REPO_URL}|g" \
      -e "s|@@REPO_BRANCH@@|${REPO_BRANCH}|g" \
      -e "s|@@MAA_ENDPOINT@@|${MAA_ENDPOINT}|g" \
      "${HERE}/cloud-init.yaml.tmpl" > "$TMP_INIT"
fi

# base64 without newlines, portable across macOS/Linux. Read from stdin rather than
# passing the path: recent macOS base64 rejects a positional file argument, and GNU
# and BSD disagree on the line-wrap flag, so we feed stdin and strip newlines either way.
if base64 --help 2>&1 | grep -q -- '-w'; then
  CUSTOM_DATA="$(base64 -w0 < "$TMP_INIT")"
else
  CUSTOM_DATA="$(base64 < "$TMP_INIT" | tr -d '\n')"
fi

SSH_PUB="$(cat "$SSH_PUBLIC_KEY_FILE")"

# ---------------------------------------------------------------------------
# Create the resource group and deploy.
# ---------------------------------------------------------------------------
echo "==> Creating resource group…"
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" -o none

# Pass params via a file to avoid quoting the base64 blob on the command line.
TMP_PARAMS="$(mktemp)"
cat > "$TMP_PARAMS" <<JSON
{
  "\$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentParameters.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "location": { "value": "${LOCATION}" },
    "namePrefix": { "value": "${NAME_PREFIX}" },
    "vmSize": { "value": "${VM_SIZE}" },
    "adminUsername": { "value": "${ADMIN_USERNAME}" },
    "adminPublicKey": { "value": "${SSH_PUB}" },
    "imageOffer": { "value": "${IMAGE_OFFER}" },
    "imageSku": { "value": "${IMAGE_SKU}" },
    "osDiskSecurityEncryptionType": { "value": "${OSDISK_ENC_TYPE}" },
    "customData": { "value": "${CUSTOM_DATA}" },
    "sshSourceCidr": { "value": "${SSH_SOURCE_CIDR}" },
    "appSourceCidr": { "value": "${APP_SOURCE_CIDR}" }
  }
}
JSON

echo "==> Deploying the Confidential VM (this provisions hardware + boots + builds the image)…"
DEPLOY_NAME="traceguard-$(date -u +%Y%m%d%H%M%S)"
az deployment group create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$DEPLOY_NAME" \
  --template-file "${HERE}/main.bicep" \
  --parameters "@${TMP_PARAMS}" \
  -o none

# ---------------------------------------------------------------------------
# Report the endpoints.
# ---------------------------------------------------------------------------
FQDN="$(az deployment group show -g "$RESOURCE_GROUP" -n "$DEPLOY_NAME" --query properties.outputs.fqdn.value -o tsv)"
URL="$(az deployment group show -g "$RESOURCE_GROUP" -n "$DEPLOY_NAME" --query properties.outputs.frontendUrl.value -o tsv)"
SSH_CMD="$(az deployment group show -g "$RESOURCE_GROUP" -n "$DEPLOY_NAME" --query properties.outputs.sshCommand.value -o tsv)"

cat <<DONE

============================================================================
  TraceGuard Confidential VM deployed.
----------------------------------------------------------------------------
  Frontend URL : ${URL}
  FQDN         : ${FQDN}
  SSH          : ${SSH_CMD}

  Watch bootstrap progress:
      ${SSH_CMD} 'sudo tail -f /var/log/traceguard-bootstrap.log'

  Tear it all down:
      RESOURCE_GROUP=${RESOURCE_GROUP} ./teardown.sh
============================================================================
DONE

if [ "$BOOTSTRAP_MODE" = "ghcr" ]; then
  cat <<NEXT
  Next steps (CI/CD): the VM only has Docker + tpm2-tools; GitHub Actions
  delivers the app on every push to main. Wire the repository up once:

      gh secret set AZURE_VM_HOST --body "${FQDN}"
      gh secret set AZURE_VM_SSH_KEY < ~/.ssh/traceguard_deploy
      gh secret set AZURE_OPENAI_API_KEY    # paste the key when prompted
      gh secret set AZURE_OPENAI_ENDPOINT   # https://<resource>.services.ai.azure.com
      gh variable set MODEL_FAST --body gpt-4o
      gh variable set MODEL_DEEP --body gpt-4o
      gh variable set MODEL_REVIEW --body gpt-4o
      gh variable set MODEL_VISION --body gpt-4o
      gh variable set AZURE_VM_USER --body ${ADMIN_USERNAME}
      gh variable set TRACEGUARD_MAA_ENDPOINT --body ${MAA_ENDPOINT}

  Then push to main (or: gh workflow run deploy). The deploy job pulls the
  :sha image from GHCR over SSH, runs it with the vTPM mounted, fronts it
  with Caddy (automatic HTTPS for ${FQDN}), and fails unless the live MAA
  attestation reports hardware_backed=true.
NEXT
else
  cat <<NEXT
  The VM is booting and building the image (first boot: ~3-6 min). Then open:
      ${URL}/attestation
  and click "Run live verification" to see the SEV-SNP quote validated by
  Microsoft Azure Attestation.
NEXT
fi

# TraceGuard on an Azure Confidential VM (one command)

This deploys TraceGuard to a genuine **Azure Confidential VM (AMD SEV-SNP)** with
vTPM, Secure Boot, and guest attestation enabled, runs the container, and gives you
a frontend URL. The app running **inside** the VM then reads the guest vTPM, submits
hardware evidence to **Microsoft Azure Attestation (MAA)**, and shows a verified
SEV-SNP quote with its measurement values — a reviewer can trigger the check live.

> **Honest scope.** Off Azure (local Docker, CI), the same image reports an honest
> `simulated` attestation — never a fabricated hardware claim. Only a real
> MAA-verified quote flips `hardware_attestation` / `quote_available` to true, and the
> receipt verifier enforces that. This is the property that makes the demo credible.

---

## What you get

- A SEV-SNP `Standard_DC4as_v5` CVM (parameterizable), Ubuntu 24.04 CVM image.
- Memory encryption (data-in-use protection), a measured boot chain, and a vTPM that
  roots guest attestation.
- The container run with `--device /dev/tpmrm0` and the TCG event log mounted, so the
  in-guest attestation module works.
- A public IP + DNS name; the console at `http://<fqdn>/` and attestation at
  `http://<fqdn>/attestation`.

## Prerequisites

1. **Azure CLI** — <https://learn.microsoft.com/cli/azure/install-azure-cli>
2. An Azure subscription with quota for a **confidential VM** family in your region.
   Check availability:
   ```bash
   az vm list-skus --location eastus2 \
     --query "[?family=='standardDCASv5Family'].name" -o tsv
   ```
3. An **SSH key** (`ssh-keygen -t ed25519` if you don't have one).
4. A **git URL the VM can clone** (`REPO_URL`) — see *How the image is built* below.

## Run it

```bash
cd deploy/azure
cp deploy.env.example deploy.env
$EDITOR deploy.env          # set REPO_URL, LOCATION, VM_SIZE, subscription, etc.
./deploy.sh
```

`deploy.sh` logs in if needed, verifies the size/region, accepts the Ubuntu CVM image
terms, renders `cloud-init.yaml.tmpl`, creates the resource group, deploys
`main.bicep`, and prints:

```
Frontend URL : http://traceguard-xxxx.eastus2.cloudapp.azure.com
SSH          : ssh azureuser@traceguard-xxxx.eastus2.cloudapp.azure.com
```

First boot builds the image (~3–6 min). Watch progress:

```bash
ssh azureuser@<fqdn> 'sudo tail -f /var/log/traceguard-bootstrap.log'
```

Then open `http://<fqdn>/attestation` and click **Run live verification**.

## Verify it is really a Confidential VM

From your machine (no SSH needed):

```bash
# Live attestation verdict — on a real CVM: hardware_backed=true, maa_verified=true.
curl -s http://<fqdn>/api/attestation | jq '{verdict, hardware_backed, maa_verified, tee_type, issuer}'

# Runtime posture.
curl -s http://<fqdn>/api/runtime | jq '{hw: .hardware_tee, attestation}'
```

You can independently re-verify the returned MAA token: decode its `iss` (a
`https://shared*.attest.azure.net` provider), fetch `${iss}/certs`, and check the
RS256 signature over the token — exactly what `POST /api/attestation/verify` does and
reports back under `reverification`.

## CI/CD from GitHub Actions (recommended)

Provision once with `BOOTSTRAP_MODE=ghcr` — the VM gets Docker + tpm2-tools and
prepared directories, nothing else. Every push to `main` then builds the image,
publishes it to GHCR, and delivers it to the VM over SSH (`.github/workflows/deploy.yml`,
job `deploy-cvm`): pull the immutable `:sha` tag with the job's ephemeral
`GITHUB_TOKEN`, run the container with the vTPM mounted, front it with Caddy
(automatic Let's Encrypt HTTPS for the `*.cloudapp.azure.com` FQDN), and fail
the deploy unless live MAA attestation reports `hardware_backed=true`.

```bash
ssh-keygen -t ed25519 -f ~/.ssh/traceguard_deploy -N '' -C traceguard-gha
cd deploy/azure
cp deploy.env.example deploy.env      # set LOCATION etc.; SSH_PUBLIC_KEY_FILE=~/.ssh/traceguard_deploy.pub
BOOTSTRAP_MODE=ghcr ./deploy.sh       # prints the FQDN + the gh secret/variable commands
```

| Name | Kind | Meaning |
|---|---|---|
| `AZURE_VM_HOST` | secret | the VM FQDN; also the Caddy/Let's Encrypt domain |
| `AZURE_VM_SSH_KEY` | secret | private half of the dedicated deploy keypair |
| `AZURE_OPENAI_API_KEY` | secret | injected into the container at deploy time only |
| `AZURE_OPENAI_ENDPOINT` | secret | resource host, not an `/api/projects/<name>` Foundry URL |
| `MODEL_FAST/DEEP/REVIEW/VISION` | variables | model ids passed to the container |
| `AZURE_VM_USER` | variable (optional) | SSH user, default `azureuser` |
| `TRACEGUARD_MAA_ENDPOINT` | variable (optional) | regional shared MAA provider |

Keys travel only over the encrypted SSH session into `docker run -e`; nothing is
baked into the image or VM custom-data. SSH stays key-only and open to `*` because
GitHub-hosted runners deploy from ephemeral IPs.

Run experiments inside the CVM (journals land on the host bind mount):

```bash
ssh -i ~/.ssh/traceguard_deploy azureuser@<fqdn> \
  'docker exec traceguard traceguard experiment --provider azure \
     --n-per-cell 12 --seed 20260710 --journal /data/experiment-main.jsonl'
scp -i ~/.ssh/traceguard_deploy azureuser@<fqdn>:/opt/traceguard/data/experiment-main.jsonl artifacts/
```

The DigitalOcean droplet deploy that previously lived in `deploy.yml` has been
removed; the Azure Confidential VM is the only deployment path.

## Tear down

```bash
cd deploy/azure
./teardown.sh              # deletes the whole resource group (asks for confirmation)
```

## How the image is built on the VM

`cloud-init.yaml.tmpl` clones `REPO_URL@REPO_BRANCH` and runs `docker build` on the
VM, so the deployed image is byte-for-byte your source (React console + `tpm2-tools`
for attestation). Options if your repo is private:

- **Deploy-token URL:** set `REPO_URL` to an HTTPS clone URL that embeds a read-only
  token (e.g. GitHub fine-grained PAT). Simplest, but the token lands in VM
  custom-data.
- **Azure Container Registry (ACR):** build and push locally, then point the VM at
  the registry instead of building:
  ```bash
  az acr create -g <rg> -n <acr> --sku Basic
  az acr login -n <acr>
  docker build -t <acr>.azurecr.io/traceguard:cvm .
  docker push <acr>.azurecr.io/traceguard:cvm
  # Then edit cloud-init to `docker pull` + `az acr login` via a managed identity
  # instead of git-clone + build.
  ```

## Parameters (deploy.env)

| Var | Default | Meaning |
|---|---|---|
| `SUBSCRIPTION_ID` / `TENANT_ID` | — | Azure identity (optional if `az login` is set) |
| `RESOURCE_GROUP` | `traceguard-cc` | Resource group (created if absent) |
| `LOCATION` | `eastus2` | Region — must offer the confidential size |
| `VM_SIZE` | `Standard_DC4as_v5` | SEV-SNP family; use `DCesv5` for Intel TDX |
| `OSDISK_ENC_TYPE` | `VMGuestStateOnly` | or `DiskWithVMGuestState` for full OS-disk confidential encryption |
| `IMAGE_OFFER`/`IMAGE_SKU` | `ubuntu-24_04-lts`/`cvm` | Canonical CVM image |
| `ADMIN_USERNAME` | `azureuser` | SSH user |
| `SSH_PUBLIC_KEY_FILE` | `~/.ssh/id_rsa.pub` | your SSH public key |
| `REPO_URL` / `REPO_BRANCH` | — / `main` | source the VM clones + builds |
| `MAA_ENDPOINT` | `https://sharedeus2.eus2.attest.azure.net` | MAA provider |
| `SSH_SOURCE_CIDR` / `APP_SOURCE_CIDR` | `*` | NSG source ranges — lock these down |
| — | — | no provider secret is embedded in VM custom-data; Actions streams it over SSH |

## AMD SEV-SNP vs Intel TDX

The vTPM guest-attestation flow is identical across SEV-SNP and TDX on Azure (the HCL
paravisor presents the same NV indexes). To use TDX, pick a `Standard_DC*es_v5` size
in a TDX region (e.g. West Europe, Central US, East US 2) and keep everything else.
The attestation module auto-detects the TEE type; SEV-SNP is the primary, most widely
available path.

## Security notes

- **No secrets are hardcoded.** The SSH public key and (optional, opt-in) API key are
  parameters. Leaving `AZURE_OPENAI_API_KEY` empty runs the offline fixture provider —
  still a fully attested demo.
- For anything beyond a short demo, set `SSH_SOURCE_CIDR`/`APP_SOURCE_CIDR` to your IP
  and put the app behind TLS (e.g. add Caddy with the FQDN for automatic HTTPS).
- The MAA token surfaced by the app is a **signed public artifact** meant to be shared
  with a relying party; it carries measurement claims, not secrets.

## Engineering references

Azure CVM options & sizes · guest-attestation design (vTPM NV indexes `0x01400001` /
`0x01C101D0`) · MAA `attest/SevSnpVm` + claim sets (`x-ms-compliance-status`,
`x-ms-sevsnpvm-*`) · THIM VCEK endpoint · Ubuntu CVM images · `az vm create`
`--security-type ConfidentialVM --enable-vtpm --enable-secure-boot`. Exact URLs are in
`docs/GAP_ANALYSIS_AND_RESEARCH.md` and the paper's attestation appendix.

"""Hardware-backed remote attestation for Azure Confidential VMs.

This module turns the paper's *emulated* confidential substrate into a real,
verifiable one when the artifact is deployed on an Azure Confidential VM, while
staying honest everywhere else.  It follows the vTPM guest-attestation flow that
Azure actually uses (``/dev/sev-guest`` is hidden behind the paravisor on Azure
CVMs, so raw SEV-SNP ioctls do not work; evidence flows through the vTPM):

    1. read the HCL attestation report from vTPM NV index 0x01400001,
    2. extract the raw SEV-SNP (1184 B) or TDX (1024 B) hardware report,
    3. fetch the VCEK/PCK certificate chain from Azure IMDS/THIM,
    4. POST the evidence to Microsoft Azure Attestation (MAA),
    5. validate the returned RS256 JWT against MAA's JWKS and a pinned issuer,
    6. surface the measurement claims and a human-readable verdict.

Design invariants (mirrored by the receipt verifier and the /api/runtime report):

* Nothing is ever fabricated.  With no vTPM present (this dev machine, CI, a
  plain container), :func:`attest` returns a clearly-labelled ``simulated`` or
  ``unavailable`` result.  ``hardware_backed`` and ``maa_verified`` only become
  ``True`` for a genuine, MAA-validated Azure quote.
* The JWT-validation path (:func:`verify_maa_token`) is pure and offline-testable:
  it verifies an RS256 signature against a JWKS and checks the issuer allow-list,
  so the same "Verify" action a reviewer triggers on Azure also runs — honestly
  labelled — against a self-signed simulated token locally.
* Only the standard library plus ``cryptography`` (already a dependency) are used;
  no PyJWT / requests / Azure SDK is required inside the confidential container.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import pathlib
import shutil
import subprocess  # noqa: S404 - used only for local tpm2-tools evidence collection
import tempfile
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey

# --------------------------------------------------------------------------- #
# Constants grounded in Azure's guest-attestation design.
# --------------------------------------------------------------------------- #

# vTPM NV indexes reserved by the Azure HCL paravisor.
NV_INDEX_HCL_REPORT = "0x01400001"  # HCL report (embeds the HW report), 2600 B
NV_INDEX_AK_CERT = "0x01C101D0"  # Azure-CA-signed vTPM AK cert, 4096 B

# HCL report layout: 32-byte header, then the raw hardware report, then the
# paravisor's runtime data. Verified against a live Azure SEV-SNP CVM: the NV
# blob is 2600 bytes = 32 + 1184 + 1384.
HCL_HEADER_BYTES = 32
SEV_SNP_REPORT_BYTES = 1184

# Offset and length of REPORT_DATA inside a SEV-SNP attestation report.
SNP_REPORT_DATA_OFFSET = 0x50
SNP_REPORT_DATA_BYTES = 64

# The runtime data begins with five little-endian uint32s
# (total, version, report_type, hash_type, payload_length) followed by a JSON
# payload. The hardware commits to that payload: on the live CVM,
# sha256(payload) equals REPORT_DATA[:32] with the remaining 32 bytes zero.
# That is the link which makes the vTPM AK hardware-vouched rather than merely
# present, and it is what lets a receipt be bound to the attested boundary.
HCL_RUNTIME_HEADER_BYTES = 20

# The paravisor's persistent handle for the vTPM Attestation Key. Its public
# half appears in the runtime payload above under kid "HCLAkPub".
TPM_AK_HANDLE = "0x81000003"
HCL_AK_KID = "HCLAkPub"

# TPMS_ATTEST framing (TPM 2.0 Part 2). magic proves the structure was produced
# by the TPM itself rather than assembled in software.
TPM_GENERATED_MAGIC = 0xFF544347
TPM_ST_ATTEST_QUOTE = 0x8018
TDX_REPORT_BYTES = 1024

# Azure IMDS / THIM.
IMDS_THIM_VCEK = (
    "http://169.254.169.254/metadata/THIM/amd/certification"  # VCEK + ASK/ARK chain
)
IMDS_TDQUOTE = "http://169.254.169.254/acc/tdquote"  # TD report -> TD quote (TDX)

# Default regional shared MAA provider (default policy only). Overridable.
DEFAULT_MAA_ENDPOINT = "https://sharedeus.eus.attest.azure.net"
MAA_SEVSNP_PATH = "/attest/SevSnpVm?api-version=2022-08-01"
MAA_TDX_PATH = "/attest/TdxVm?api-version=2023-04-01-preview"

# Only tokens issued by a Microsoft Azure Attestation host are hardware-rooted.
TRUSTED_MAA_ISSUER_SUFFIX = ".attest.azure.net"

# Device nodes that indicate an Azure CVM vTPM (NOT /dev/sev-guest, which the
# paravisor hides on Azure). Presence is necessary but never sufficient.
VTPM_DEVICES = ("/dev/tpmrm0", "/dev/tpm0")

_HTTP_TIMEOUT_S = 8.0


# --------------------------------------------------------------------------- #
# Result type.
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class AttestationResult:
    """An honest verdict about the confidential-computing posture of this host."""

    mode: str  # azure_sevsnp | azure_tdx | simulated | unavailable
    tee_type: str  # sev-snp | tdx | none
    hardware_backed: bool
    maa_verified: bool
    verdict: str  # verified | simulated | unavailable | failed
    summary: str
    checked_at: str
    provider: str | None = None
    issuer: str | None = None
    token: str | None = None
    token_sha256: str | None = None
    claims: dict[str, Any] = field(default_factory=dict)
    measurements: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    # Proof that a specific key lives inside *this* attested TEE, and that the
    # proof is fresh. Absent unless a caller passed a key to bind.
    key_binding: dict[str, Any] | None = None

    def measurement_sha256(self) -> str | None:
        if not self.measurements:
            return None
        payload = json.dumps(self.measurements, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self, *, include_token: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": "traceguard.attestation.v1",
            "mode": self.mode,
            "tee_type": self.tee_type,
            "hardware_backed": self.hardware_backed,
            "maa_verified": self.maa_verified,
            "verdict": self.verdict,
            "summary": self.summary,
            "checked_at": self.checked_at,
            "provider": self.provider,
            "issuer": self.issuer,
            "token_sha256": self.token_sha256,
            "measurement_sha256": self.measurement_sha256(),
            "claims": self.claims,
            "measurements": self.measurements,
            "evidence": self.evidence,
            "errors": self.errors,
        }
        if include_token:
            data["token"] = self.token
        return data

    def receipt_binding(self) -> dict[str, Any] | None:
        """Compact, signable evidence binding for a Trace Receipt.

        Returns ``None`` when there is no attestation to bind (the honest
        default — the receipt then keeps ``hardware_attestation: false``).
        """

        if self.mode in {"unavailable", "disabled"}:
            return None
        return {
            "present": True,
            # verdict/mode make the binding self-describing: a consumer can
            # tell "verified on SEV-SNP" from "simulated on a plain host"
            # without re-deriving it from the flags below.
            "verdict": self.verdict,
            "mode": self.mode,
            "hardware_backed": self.hardware_backed,
            "maa_verified": self.maa_verified,
            "tee_type": self.tee_type,
            "issuer": self.issuer,
            "provider": self.provider,
            "token_sha256": self.token_sha256,
            "measurement_sha256": self.measurement_sha256(),
            # measurement_sha256 is an aggregate digest and therefore opaque.
            # The launch measurement in the clear is what lets a reader check
            # that two rows came from the same boot of the same image.
            "launch_measurement": self.measurements.get("launch_measurement"),
            # Without this a receipt proves only that *an* MAA-verified quote
            # exists for *some* VM at *some* time, plus a signature under a key
            # of unproven residency. A captured token would satisfy the
            # verifier. The binding is what closes both gaps.
            "key_binding": self.key_binding,
        }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# --------------------------------------------------------------------------- #
# Environment probing.
# --------------------------------------------------------------------------- #


def detect_vtpm_device() -> str | None:
    for path in VTPM_DEVICES:
        if os.path.exists(path):
            return path
    return None


def probe_environment() -> dict[str, Any]:
    """Read-only probe of the confidential-VM interfaces present on this host."""

    isolation = (os.getenv("TRACEGUARD_ISOLATION") or "").strip()
    return {
        "vtpm_device": detect_vtpm_device(),
        "tpm2_tools_available": shutil.which("tpm2_nvread") is not None,
        "isolation_env": isolation or None,
        "sev_guest_device_present": os.path.exists("/dev/sev-guest"),
        "tdx_guest_device_present": os.path.exists("/dev/tdx_guest"),
        "security_fs_present": os.path.isdir("/sys/kernel/security"),
    }


def resolve_mode(requested: str | None, probe: dict[str, Any]) -> str:
    """Decide which attestation path to run.

    ``auto`` (the default) uses a real Azure path only when a vTPM and the
    tpm2 tooling are both present; otherwise it falls back to ``simulated`` so
    the console and receipt flow are always exercisable — never overclaiming.
    """

    requested = (requested or "auto").strip().lower()
    if requested in {"azure_sevsnp", "azure_tdx", "simulated", "disabled"}:
        return requested
    if requested not in {"auto", ""}:
        return "simulated"
    if not (probe.get("vtpm_device") and probe.get("tpm2_tools_available")):
        return "simulated"
    if probe.get("tdx_guest_device_present"):
        return "azure_tdx"
    return "azure_sevsnp"


# --------------------------------------------------------------------------- #
# JWT / MAA token verification (pure, offline-testable).
# --------------------------------------------------------------------------- #


def _b64url_decode(segment: str) -> bytes:
    padding_needed = (-len(segment)) % 4
    return base64.urlsafe_b64decode(segment + "=" * padding_needed)


def decode_jwt(token: str) -> tuple[dict[str, Any], dict[str, Any], bytes, bytes]:
    """Split a compact JWS into (header, payload, signing_input, signature)."""

    try:
        header_b64, payload_b64, signature_b64 = token.split(".")
    except ValueError as exc:
        raise ValueError("token is not a compact JWS with three segments") from exc
    header = json.loads(_b64url_decode(header_b64))
    payload = json.loads(_b64url_decode(payload_b64))
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    signature = _b64url_decode(signature_b64)
    return header, payload, signing_input, signature


def _rsa_key_from_jwk(jwk: dict[str, Any]) -> RSAPublicKey | None:
    """Build an RSA public key from a JWK (``x5c`` cert or raw ``n``/``e``)."""

    x5c = jwk.get("x5c")
    if isinstance(x5c, list) and x5c:
        der = base64.b64decode(x5c[0])
        cert = x509.load_der_x509_certificate(der)
        key = cert.public_key()
        return key if isinstance(key, RSAPublicKey) else None
    if jwk.get("n") and jwk.get("e"):
        n = int.from_bytes(_b64url_decode(jwk["n"]), "big")
        e = int.from_bytes(_b64url_decode(jwk["e"]), "big")
        return rsa.RSAPublicNumbers(e, n).public_key()
    return None


def _select_jwk(jwks: dict[str, Any], kid: str | None) -> list[dict[str, Any]]:
    keys = [k for k in jwks.get("keys", []) if isinstance(k, dict)]
    if kid:
        matched = [k for k in keys if k.get("kid") == kid]
        if matched:
            return matched
    return keys  # RFC 7515 allows trying all keys when no kid matches.


def verify_maa_token(
    token: str,
    jwks: dict[str, Any],
    *,
    trusted_issuer_suffix: str = TRUSTED_MAA_ISSUER_SUFFIX,
    expected_issuer: str | None = None,
    now_epoch: int | None = None,
) -> dict[str, Any]:
    """Verify an RS256 MAA token against a JWKS and an issuer allow-list.

    Pure function: does no network I/O.  Returns ``{"valid", "reasons", "claims",
    "issuer", "hardware_rooted"}``.  ``hardware_rooted`` is ``True`` only when the
    signature verifies, the issuer is a ``*.attest.azure.net`` host, and the token
    is within its validity window — never for a self-signed simulated token.
    """

    reasons: list[str] = []
    try:
        header, claims, signing_input, signature = decode_jwt(token)
    except (ValueError, json.JSONDecodeError, base64.binascii.Error) as exc:
        return {
            "valid": False,
            "reasons": [f"malformed token: {type(exc).__name__}"],
            "claims": {},
            "issuer": None,
            "hardware_rooted": False,
        }

    if header.get("alg") != "RS256":
        reasons.append(f"unexpected alg {header.get('alg')!r} (require RS256)")

    signature_ok = False
    for jwk in _select_jwk(jwks, header.get("kid")):
        key = _rsa_key_from_jwk(jwk)
        if key is None:
            continue
        try:
            key.verify(signature, signing_input, padding.PKCS1v15(), hashes.SHA256())
            signature_ok = True
            break
        except InvalidSignature:
            continue
        except Exception:  # noqa: BLE001, S112 - a malformed JWK must not abort the loop
            continue
    if not signature_ok:
        reasons.append("RS256 signature did not verify against any JWKS key")

    issuer = claims.get("iss")
    issuer_host = ""
    if isinstance(issuer, str) and "://" in issuer:
        issuer_host = issuer.split("://", 1)[1].split("/", 1)[0].lower()
    if expected_issuer is not None and issuer != expected_issuer:
        reasons.append("issuer does not match the pinned provider")
    hardware_issuer = issuer_host.endswith(trusted_issuer_suffix)

    now_epoch = now_epoch if now_epoch is not None else int(datetime.now(UTC).timestamp())
    exp = claims.get("exp")
    nbf = claims.get("nbf")
    if isinstance(exp, (int, float)) and now_epoch > exp + 60:
        reasons.append("token is expired")
    if isinstance(nbf, (int, float)) and now_epoch + 60 < nbf:
        reasons.append("token is not yet valid")

    valid = not reasons
    return {
        "valid": valid,
        "reasons": reasons,
        "claims": claims,
        "issuer": issuer,
        # Hardware-rooted requires a valid signature AND a real MAA issuer.
        "hardware_rooted": bool(valid and signature_ok and hardware_issuer),
    }


def interpret_maa_claims(claims: dict[str, Any]) -> dict[str, Any]:
    """Extract the measurement/compliance claims a reviewer cares about.

    Handles both the flat SEV-SNP/TDX token shape and the nested
    ``x-ms-isolation-tee`` shape used by AzureGuest tokens.
    """

    isolation = claims.get("x-ms-isolation-tee")
    tee = isolation if isinstance(isolation, dict) else claims
    tee_type = tee.get("x-ms-attestation-type") or claims.get("x-ms-attestation-type")
    measurements: dict[str, Any] = {}
    for source_key, out_key in (
        ("x-ms-sevsnpvm-launchmeasurement", "launch_measurement"),
        ("x-ms-sevsnpvm-idkeydigest", "id_key_digest"),
        ("x-ms-sevsnpvm-guestsvn", "guest_svn"),
        ("x-ms-sevsnpvm-reportdata", "report_data"),
        ("x-ms-sevsnpvm-vmpl", "vmpl"),
        ("x-ms-sevsnpvm-bootloader-svn", "bootloader_svn"),
        ("tdx_mrtd", "tdx_mrtd"),
        ("tdx_rtmr0", "tdx_rtmr0"),
        ("tdx_rtmr1", "tdx_rtmr1"),
        ("tdx_rtmr2", "tdx_rtmr2"),
        ("tdx_rtmr3", "tdx_rtmr3"),
    ):
        value = tee.get(source_key)
        if value is not None:
            measurements[out_key] = value
    return {
        "tee_type": tee_type,
        "compliance_status": tee.get("x-ms-compliance-status")
        or claims.get("x-ms-compliance-status"),
        "secure_boot": tee.get("x-ms-runtime", {}).get("vm-configuration", {}).get("secure-boot")
        if isinstance(tee.get("x-ms-runtime"), dict)
        else claims.get("secureboot"),
        "measurements": measurements,
    }


# --------------------------------------------------------------------------- #
# Network + evidence collection (Azure-only; degrades gracefully).
# --------------------------------------------------------------------------- #


def _http_get(url: str, headers: dict[str, str] | None = None) -> bytes:
    request = urllib.request.Request(url, headers=headers or {})  # noqa: S310 - fixed hosts
    with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_S) as response:  # noqa: S310
        return response.read()


def _http_post_json(url: str, body: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310 - fixed MAA/IMDS hosts
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_S) as response:  # noqa: S310
        return json.loads(response.read())


def fetch_jwks(provider_url: str) -> dict[str, Any]:
    """Fetch the JWKS from ``{provider}/certs`` (network I/O)."""

    url = provider_url.rstrip("/") + "/certs"
    return json.loads(_http_get(url))


def _read_nv_index(index: str) -> bytes:
    """Read a vTPM NV index via tpm2-tools (Azure CVM only)."""

    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["tpm2_nvread", "-C", "o", index],  # noqa: S607
        capture_output=True,
        timeout=15,
        check=True,
    )
    return result.stdout


def collect_sevsnp_evidence() -> dict[str, Any]:
    """Collect SEV-SNP evidence from the vTPM and IMDS/THIM on an Azure CVM."""

    hcl_report = _read_nv_index(NV_INDEX_HCL_REPORT)
    if len(hcl_report) < HCL_HEADER_BYTES + SEV_SNP_REPORT_BYTES:
        raise ValueError("HCL report shorter than a SEV-SNP hardware report")
    snp_report = hcl_report[HCL_HEADER_BYTES : HCL_HEADER_BYTES + SEV_SNP_REPORT_BYTES]
    thim = json.loads(_http_get(IMDS_THIM_VCEK, headers={"Metadata": "true"}))
    vcek = thim.get("vcekCert", "")
    chain = thim.get("certificateChain", "")
    maa_report = {
        "SnpReport": base64.urlsafe_b64encode(snp_report).decode("ascii").rstrip("="),
        "VcekCertChain": base64.urlsafe_b64encode((vcek + chain).encode()).decode("ascii").rstrip(
            "="
        ),
    }
    # The paravisor's runtime payload is what REPORT_DATA commits to, and it
    # carries the vTPM AK. Surfacing it lets a verifier reach the AK from the
    # hardware report instead of taking the prover's word for it.
    try:
        payload, _ = parse_hcl_runtime_payload(hcl_report)
        runtime_data = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    except ValueError:
        runtime_data = ""
    return {
        "report": base64.urlsafe_b64encode(json.dumps(maa_report).encode())
        .decode("ascii")
        .rstrip("="),
        "runtimeData": {"data": runtime_data, "dataType": "JSON"},
        "hcl_report_len": len(hcl_report),
    }



# --------------------------------------------------------------------------- #
# Key binding: proving a signing key lives inside *this* attested TEE.
# --------------------------------------------------------------------------- #
#
# The chain, each link verified against a live Azure SEV-SNP CVM:
#
#   1. MAA validates the SEV-SNP report, so REPORT_DATA is hardware-attested.
#   2. sha256(HCL runtime payload) == REPORT_DATA[:32], so the payload is
#      hardware-committed.
#   3. The payload carries the vTPM AK public key (kid "HCLAkPub"), so that AK
#      is hardware-vouched rather than merely present.
#   4. TPM2_Quote under that AK, with qualifying data
#      sha256(signing public key || nonce), proves the quote came from a key
#      resident in this vTPM and names both our signing key and a fresh nonce.
#
# Link 4 is why a captured token is no longer sufficient, and link 3 is why the
# AK cannot simply be asserted by the prover.
#
# What this does NOT prove: that the signing key was *generated* inside the TEE
# rather than imported into it. Closing that needs a TPM-resident signing key
# (TPM2_Create + TPM2_Certify), which the receipt signer does not use.


def key_binding_digest(public_key_bytes: bytes, nonce: str) -> bytes:
    """The value the hardware signs over.

    One definition shared by the prover and the verifier, so the two can never
    disagree about what was bound -- the same reason
    ``attestation_is_hardware_bound`` is shared rather than duplicated.
    """

    return hashlib.sha256(public_key_bytes + nonce.encode("utf-8")).digest()


def parse_hcl_runtime_payload(hcl_report: bytes) -> tuple[bytes, dict[str, Any]]:
    """Return the runtime payload bytes and its parsed JSON."""

    offset = HCL_HEADER_BYTES + SEV_SNP_REPORT_BYTES
    runtime = hcl_report[offset:]
    if len(runtime) < HCL_RUNTIME_HEADER_BYTES:
        raise ValueError("HCL report carries no runtime data")
    payload_length = int.from_bytes(runtime[16:20], "little")
    payload = runtime[HCL_RUNTIME_HEADER_BYTES : HCL_RUNTIME_HEADER_BYTES + payload_length]
    if len(payload) != payload_length:
        raise ValueError("HCL runtime payload is truncated")
    return payload, json.loads(payload)


def snp_report_data(hcl_report: bytes) -> bytes:
    start = HCL_HEADER_BYTES + SNP_REPORT_DATA_OFFSET
    return hcl_report[start : start + SNP_REPORT_DATA_BYTES]


def ak_public_from_runtime(document: Mapping[str, Any]) -> RSAPublicKey:
    """Extract the vTPM AK public key from the hardware-committed payload."""

    for entry in document.get("keys", ()):
        if entry.get("kid") != HCL_AK_KID:
            continue
        key = _rsa_key_from_jwk(dict(entry))
        if key is None:
            raise ValueError("HCLAkPub is present but not a usable RSA key")
        return key
    raise ValueError(f"runtime payload does not contain a {HCL_AK_KID!r} key")


def verify_runtime_commitment(hcl_report: bytes) -> dict[str, Any]:
    """Check that the hardware committed to the runtime payload (link 2).

    A mismatch means the runtime payload -- and therefore the AK it carries --
    is not the one the hardware signed, so nothing downstream may be trusted.
    """

    payload, document = parse_hcl_runtime_payload(hcl_report)
    report_data = snp_report_data(hcl_report)
    digest = hashlib.sha256(payload).digest()
    committed = report_data[: len(digest)] == digest and not any(report_data[len(digest) :])
    return {
        "committed": committed,
        "payload_sha256": digest.hex(),
        "report_data": report_data.hex(),
        "document": document,
    }


def bind_signing_key(public_key_bytes: bytes, nonce: str) -> dict[str, Any]:
    """Produce hardware proof that ``public_key_bytes`` is used inside this TEE.

    Raises rather than degrading: a caller that asked for a binding and cannot
    have one must not receive a result that merely looks like one.
    """

    if not nonce:
        raise ValueError("a binding nonce is required for freshness")
    hcl_report = _read_nv_index(NV_INDEX_HCL_REPORT)
    commitment = verify_runtime_commitment(hcl_report)
    if not commitment["committed"]:
        raise ValueError("HCL runtime payload is not committed by the hardware report")
    ak_public_from_runtime(commitment["document"])  # fail early if the AK is unusable

    qualifying = key_binding_digest(public_key_bytes, nonce)
    with tempfile.TemporaryDirectory() as work:
        message = os.path.join(work, "quote.msg")
        signature = os.path.join(work, "quote.sig")
        pcrs = os.path.join(work, "quote.pcr")
        subprocess.run(  # noqa: S603 - fixed argv, no shell
            [  # noqa: S607
                "tpm2_quote",
                "-c", TPM_AK_HANDLE,
                "-l", "sha256:0,1,2,3,4,5,6,7",
                "-q", qualifying.hex(),
                "-m", message,
                "-s", signature,
                "-o", pcrs,
                "-g", "sha256",
            ],
            capture_output=True,
            timeout=30,
            check=True,
        )
        quote_message = pathlib.Path(message).read_bytes()
        quote_signature = pathlib.Path(signature).read_bytes()

    return {
        "scheme": "azure-vtpm-ak-quote-v1",
        "nonce": nonce,
        "signing_key_sha256": hashlib.sha256(public_key_bytes).hexdigest(),
        "qualifying_digest": qualifying.hex(),
        "quote_message": base64.b64encode(quote_message).decode("ascii"),
        "quote_signature": base64.b64encode(quote_signature).decode("ascii"),
        "hcl_runtime_payload": base64.b64encode(
            parse_hcl_runtime_payload(hcl_report)[0]
        ).decode("ascii"),
        "report_data": commitment["report_data"],
        "ak_handle": TPM_AK_HANDLE,
    }


def verify_key_binding(
    binding: Mapping[str, Any] | None,
    *,
    public_key_bytes: bytes,
    expected_nonce: str | None = None,
) -> dict[str, Any]:
    """Verify the four-link chain. Returns a verdict; never raises on bad input.

    Independent of any MAA call: it checks that the runtime payload is
    hardware-committed, that the AK comes from that payload, that the quote was
    produced by the TPM, and that it signs over exactly our key and nonce. The
    caller pairs this with MAA verification of the same REPORT_DATA.
    """

    reasons: list[str] = []
    if not isinstance(binding, Mapping):
        return {"valid": False, "reasons": ["no key binding present"]}
    if binding.get("scheme") != "azure-vtpm-ak-quote-v1":
        return {"valid": False, "reasons": [f"unknown binding scheme {binding.get('scheme')!r}"]}

    nonce = str(binding.get("nonce") or "")
    if expected_nonce is not None and nonce != expected_nonce:
        reasons.append("binding nonce does not match the challenge")

    try:
        payload = base64.b64decode(binding["hcl_runtime_payload"])
        quote_message = base64.b64decode(binding["quote_message"])
        quote_signature = base64.b64decode(binding["quote_signature"])
        report_data = bytes.fromhex(str(binding["report_data"]))
    except Exception:  # noqa: BLE001 - malformed input is simply invalid
        return {"valid": False, "reasons": ["binding fields are malformed"]}

    # Link 2: the hardware committed to this runtime payload.
    digest = hashlib.sha256(payload).digest()
    if not (report_data[: len(digest)] == digest and not any(report_data[len(digest) :])):
        reasons.append("runtime payload is not committed by REPORT_DATA")

    # Link 3: the AK comes out of that committed payload, not from the prover.
    try:
        ak_public = ak_public_from_runtime(json.loads(payload))
    except Exception as exc:  # noqa: BLE001
        return {"valid": False, "reasons": [*reasons, f"AK unusable: {exc}"]}

    # Link 4a: the quote really was produced by a TPM, over our exact digest.
    if len(quote_message) < 6:
        return {"valid": False, "reasons": [*reasons, "quote message is truncated"]}
    magic = int.from_bytes(quote_message[0:4], "big")
    attest_type = int.from_bytes(quote_message[4:6], "big")
    if magic != TPM_GENERATED_MAGIC:
        reasons.append("quote is not TPM_GENERATED")
    if attest_type != TPM_ST_ATTEST_QUOTE:
        reasons.append("attestation structure is not a quote")

    extra_data = b""
    try:
        offset = 6
        signer_length = int.from_bytes(quote_message[offset : offset + 2], "big")
        offset += 2 + signer_length
        extra_length = int.from_bytes(quote_message[offset : offset + 2], "big")
        offset += 2
        extra_data = quote_message[offset : offset + extra_length]
    except Exception:  # noqa: BLE001
        reasons.append("quote extraData could not be parsed")

    expected = key_binding_digest(public_key_bytes, nonce)
    if extra_data != expected:
        reasons.append("quote does not sign over this key and nonce")

    # Link 4b: the AK signature over the quote.
    if len(quote_signature) < 6:
        reasons.append("quote signature is truncated")
    else:
        length = int.from_bytes(quote_signature[4:6], "big")
        raw = quote_signature[6 : 6 + length]
        try:
            ak_public.verify(raw, quote_message, padding.PKCS1v15(), hashes.SHA256())
        except InvalidSignature:
            reasons.append("quote signature does not verify under the attested AK")
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"quote signature check failed: {type(exc).__name__}")

    return {
        "valid": not reasons,
        "reasons": reasons,
        "scheme": binding.get("scheme"),
        "signing_key_sha256": binding.get("signing_key_sha256"),
        "nonce": nonce,
    }


# --------------------------------------------------------------------------- #
# Simulated attester (local demo; never claims hardware).
# --------------------------------------------------------------------------- #

# One ephemeral RSA key per process so the local "Verify" action exercises the
# real RS256 verification path against a self-signed JWKS.  It is stamped as
# non-hardware and its issuer is not a Microsoft attest host, so it can never be
# mistaken for a genuine MAA token.
_SIMULATED_ISSUER = "https://simulated.traceguard.local"
_simulated_key: rsa.RSAPrivateKey | None = None


def _get_simulated_key() -> rsa.RSAPrivateKey:
    global _simulated_key
    if _simulated_key is None:
        _simulated_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return _simulated_key


def _simulated_jwks(key: rsa.RSAPrivateKey) -> dict[str, Any]:
    numbers = key.public_key().public_numbers()

    def _b64u(value: int) -> str:
        raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return {
        "keys": [
            {
                "kty": "RSA",
                "kid": "traceguard-simulated",
                "alg": "RS256",
                "n": _b64u(numbers.n),
                "e": _b64u(numbers.e),
            }
        ]
    }


def _simulated_measurements() -> dict[str, Any]:
    """Deterministic, clearly-labelled demo measurement values for the simulated path.

    These are fixed placeholders so the console shows a complete confidential-VM
    experience off Azure. They are NOT hardware measurements and never make the
    receipt's hardware_attestation flag true.
    """

    return {
        "launch_measurement": "SIMULATED-" + ("0" * 88),
        "id_key_digest": "SIMULATED-" + ("0" * 88),
        "guest_svn": "0",
        "report_data": "SIMULATED-" + ("0" * 56),
        "vmpl": "0",
    }


def _mint_simulated_token(key: rsa.RSAPrivateKey, isolation: str | None) -> str:
    now = int(datetime.now(UTC).timestamp())
    header = {"alg": "RS256", "typ": "JWT", "kid": "traceguard-simulated"}
    payload = {
        "iss": _SIMULATED_ISSUER,
        "iat": now,
        "nbf": now,
        "exp": now + 3600,
        "x-ms-attestation-type": "sevsnpvm-simulated",
        "x-ms-compliance-status": "simulated-demo",
        "traceguard-note": (
            "Self-signed by the TraceGuard app for the demo environment. This is NOT a "
            "Microsoft Azure Attestation token and carries no hardware root of trust."
        ),
        "declared_isolation": isolation,
        **{
            "x-ms-sevsnpvm-launchmeasurement": "SIMULATED-" + ("0" * 88),
            "x-ms-sevsnpvm-guestsvn": 0,
        },
    }

    def _seg(obj: dict[str, Any]) -> str:
        raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    signing_input = f"{_seg(header)}.{_seg(payload)}".encode("ascii")
    signature = key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    signature_b64 = base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")
    return f"{signing_input.decode('ascii')}.{signature_b64}"


# --------------------------------------------------------------------------- #
# Top-level entry point.
# --------------------------------------------------------------------------- #


def attest(
    *,
    mode: str | None = None,
    maa_endpoint: str | None = None,
    nonce: str | None = None,
    bind_public_key: bytes | None = None,
) -> AttestationResult:
    """Obtain and validate an attestation for the current host.

    ``mode`` defaults to ``auto`` (env ``TRACEGUARD_ATTESTATION_MODE``).  On a
    genuine Azure CVM this returns a hardware-backed, MAA-verified result; on any
    other host it returns an honest ``simulated`` (or ``unavailable``) result.
    """

    mode = mode or os.getenv("TRACEGUARD_ATTESTATION_MODE")
    maa_endpoint = (
        maa_endpoint or os.getenv("TRACEGUARD_MAA_ENDPOINT") or DEFAULT_MAA_ENDPOINT
    ).rstrip("/")
    probe = probe_environment()
    resolved = resolve_mode(mode, probe)

    if resolved == "disabled":
        return AttestationResult(
            mode="disabled",
            tee_type="none",
            hardware_backed=False,
            maa_verified=False,
            verdict="unavailable",
            summary="Attestation disabled by configuration (TRACEGUARD_ATTESTATION_MODE=disabled).",
            checked_at=_now(),
            evidence=probe,
        )

    if resolved in {"azure_sevsnp", "azure_tdx"}:
        return _attest_azure(resolved, maa_endpoint, probe, nonce, bind_public_key)

    return _attest_simulated(probe)


def _attest_simulated(probe: dict[str, Any]) -> AttestationResult:
    key = _get_simulated_key()
    isolation = probe.get("isolation_env")
    token = _mint_simulated_token(key, isolation)
    jwks = _simulated_jwks(key)
    verdict = verify_maa_token(token, jwks, expected_issuer=_SIMULATED_ISSUER)
    # The signature verifies (proving the verification pipeline works) but the
    # issuer is not a Microsoft attest host, so hardware_rooted stays False.
    return AttestationResult(
        mode="simulated",
        tee_type="sev-snp (simulated)",
        hardware_backed=False,
        maa_verified=False,
        verdict="simulated",
        summary=(
            "Demo environment: this host has no Azure confidential-VM vTPM, so a "
            "simulated SEV-SNP attestation is presented to exercise the full flow "
            "end to end. A self-signed token was minted and verified; it is NOT a "
            "Microsoft Azure Attestation token and carries no hardware root of trust. "
            "Deploy on an Azure Confidential VM (deploy/azure) for a hardware-backed, "
            "MAA-verified result."
        ),
        checked_at=_now(),
        provider=_SIMULATED_ISSUER,
        issuer=verdict.get("issuer"),
        token=token,
        token_sha256=hashlib.sha256(token.encode()).hexdigest(),
        claims={
            "x-ms-attestation-type": "sevsnpvm-simulated",
            "x-ms-compliance-status": "simulated-demo",
            "signature_pipeline_verified": verdict.get("valid", False),
        },
        measurements=_simulated_measurements(),
        evidence=probe,
        errors=[],
    )


def _attest_azure(
    resolved: str,
    maa_endpoint: str,
    probe: dict[str, Any],
    nonce: str | None,
    bind_public_key: bytes | None = None,
) -> AttestationResult:
    errors: list[str] = []
    # Bind before the MAA round-trip so a binding failure is reported as an
    # error on an otherwise-verified result, rather than silently producing a
    # result that reads as bound when it is not.
    key_binding: dict[str, Any] | None = None
    if bind_public_key is not None:
        if not nonce:
            errors.append("key binding requested without a nonce; refusing to bind")
        else:
            try:
                key_binding = bind_signing_key(bind_public_key, nonce)
            except Exception as exc:  # noqa: BLE001 - degrade honestly, never fake
                errors.append(f"key binding unavailable: {type(exc).__name__}")
    try:
        if resolved == "azure_sevsnp":
            evidence = collect_sevsnp_evidence()
            token = _http_post_json(maa_endpoint + MAA_SEVSNP_PATH, evidence).get("token")
            tee_type = "sev-snp"
        else:
            # TDX path: the TD report -> TD quote -> MAA. Collection differs but the
            # verification is identical. Kept minimal here; SEV-SNP is the primary path.
            raise NotImplementedError("TDX evidence collection is deploy-specific")
        if not token:
            raise ValueError("MAA returned no token")
        jwks = fetch_jwks(maa_endpoint)
        verdict = verify_maa_token(
            token, jwks, expected_issuer=maa_endpoint if maa_endpoint.startswith("http") else None
        )
        interpreted = interpret_maa_claims(verdict.get("claims", {}))
        hardware = bool(verdict.get("hardware_rooted"))
        return AttestationResult(
            mode=resolved,
            tee_type=interpreted.get("tee_type") or tee_type,
            hardware_backed=hardware,
            maa_verified=bool(verdict.get("valid")),
            verdict="verified" if (hardware and verdict.get("valid")) else "failed",
            summary=(
                "Hardware-backed attestation verified by Microsoft Azure Attestation: "
                f"{interpreted.get('compliance_status') or 'compliance unknown'}."
                if hardware
                else "Attestation evidence was collected but did not verify as a genuine "
                "Azure CVM quote; see errors."
            ),
            checked_at=_now(),
            provider=maa_endpoint,
            issuer=verdict.get("issuer"),
            token=token,
            token_sha256=hashlib.sha256(token.encode()).hexdigest(),
            claims={
                "x-ms-attestation-type": interpreted.get("tee_type"),
                "x-ms-compliance-status": interpreted.get("compliance_status"),
                "secure_boot": interpreted.get("secure_boot"),
            },
            measurements=interpreted.get("measurements", {}),
            evidence={**probe, "hcl_report_len": evidence.get("hcl_report_len")},
            errors=[*errors, *verdict.get("reasons", [])],
            key_binding=key_binding,
        )
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
        urllib.error.URLError,
        NotImplementedError,
    ) as exc:
        errors.append(f"{type(exc).__name__}: attestation evidence unavailable")
        return AttestationResult(
            mode="unavailable",
            tee_type="sev-snp" if resolved == "azure_sevsnp" else "tdx",
            hardware_backed=False,
            maa_verified=False,
            verdict="unavailable",
            summary=(
                "This host declares an Azure confidential mode but hardware attestation "
                "evidence could not be collected or validated. No quote is asserted."
            ),
            checked_at=_now(),
            provider=maa_endpoint,
            evidence=probe,
            errors=errors,
        )

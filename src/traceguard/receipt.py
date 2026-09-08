"""Fixed-size, Ed25519-signed Trace Receipt envelopes."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ._canonical import canonical_json
from .config import Settings
from .plans import ADAPTIVE_PLAN_TEMPLATE, resolve_plan
from .types import Condition, ObservableTrace

RECEIPT_SCHEMA = "traceguard.receipt.v1"

# Re-exported: several modules (and the verifier) import canonical_json from
# here, and the plan digest must be computed with the identical encoding.
__all__ = [
    "RECEIPT_SCHEMA",
    "ReceiptEnvelope",
    "ReceiptSigner",
    "attestation_is_hardware_bound",
    "build_receipt_body",
    "canonical_json",
    "public_key_b64",
    "sha256_hex",
]


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def public_key_b64(private_key: Ed25519PrivateKey) -> str:
    raw = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return base64.b64encode(raw).decode("ascii")


@dataclass(frozen=True, slots=True)
class ReceiptEnvelope:
    """A canonical envelope whose compact JSON encoding has a fixed byte size."""

    body: Mapping[str, Any]
    signature: str
    public_key: str
    padding: str
    envelope_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "body": dict(self.body),
            "envelope_bytes": self.envelope_bytes,
            "padding": self.padding,
            "public_key": self.public_key,
            "signature": self.signature,
        }

    def to_bytes(self) -> bytes:
        encoded = canonical_json(self.to_dict())
        if len(encoded) != self.envelope_bytes:
            raise ValueError(
                f"receipt envelope has {len(encoded)} bytes, expected {self.envelope_bytes}"
            )
        return encoded

    @property
    def receipt_hash(self) -> str:
        return sha256_hex(self.to_bytes())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReceiptEnvelope:
        body = value.get("body")
        if not isinstance(body, Mapping):
            raise ValueError("receipt body must be an object")
        try:
            envelope = cls(
                body=dict(body),
                signature=str(value["signature"]),
                public_key=str(value["public_key"]),
                padding=str(value["padding"]),
                envelope_bytes=int(value["envelope_bytes"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid receipt envelope fields") from exc
        envelope.to_bytes()
        return envelope

    @classmethod
    def from_bytes(cls, value: bytes) -> ReceiptEnvelope:
        try:
            decoded = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("receipt is not valid canonical JSON") from exc
        if not isinstance(decoded, Mapping):
            raise ValueError("receipt root must be an object")
        envelope = cls.from_dict(decoded)
        if envelope.to_bytes() != value:
            raise ValueError("receipt bytes are not in canonical encoding")
        return envelope


class ReceiptSigner:
    """Demo Ed25519 signer.

    This signs an integrity claim; it is not a hardware attestation.  Production
    deployments should pin a KMS-managed, attested public key.
    """

    def __init__(self, private_key: Ed25519PrivateKey, *, envelope_bytes: int = 8192) -> None:
        self._private_key = private_key
        self.envelope_bytes = envelope_bytes
        self.public_key = public_key_b64(private_key)
        raw_public = base64.b64decode(self.public_key)
        self.key_fingerprint = sha256_hex(raw_public)

    @classmethod
    def generate(cls, *, envelope_bytes: int = 8192) -> ReceiptSigner:
        return cls(Ed25519PrivateKey.generate(), envelope_bytes=envelope_bytes)

    @classmethod
    def from_settings(cls, settings: Settings) -> ReceiptSigner:
        if settings.signing_private_key_b64:
            raw = base64.b64decode(settings.signing_private_key_b64, validate=True)
            private_key = Ed25519PrivateKey.from_private_bytes(raw)
        elif settings.signing_key_file:
            private_key = Ed25519PrivateKey.from_private_bytes(
                _load_or_create_signing_key(settings.signing_key_file)
            )
        else:
            private_key = Ed25519PrivateKey.generate()
        return cls(private_key, envelope_bytes=settings.receipt_envelope_bytes)

    def sign(self, body: Mapping[str, Any]) -> ReceiptEnvelope:
        signed_body = dict(body)
        signed_body["signer"] = {
            "algorithm": "Ed25519",
            "key_fingerprint_sha256": self.key_fingerprint,
            "trust_root": "demo_key_not_hardware_attested",
        }
        signature = base64.b64encode(self._private_key.sign(canonical_json(signed_body))).decode(
            "ascii"
        )
        base = {
            "body": signed_body,
            "envelope_bytes": self.envelope_bytes,
            "padding": "",
            "public_key": self.public_key,
            "signature": signature,
        }
        padding_length = self.envelope_bytes - len(canonical_json(base))
        if padding_length < 0:
            raise ValueError(f"receipt body exceeds fixed {self.envelope_bytes}-byte envelope")
        envelope = ReceiptEnvelope(
            body=signed_body,
            signature=signature,
            public_key=self.public_key,
            padding="0" * padding_length,
            envelope_bytes=self.envelope_bytes,
        )
        envelope.to_bytes()
        return envelope


def build_receipt_body(
    *,
    run_id: str,
    case_id: str,
    condition: Condition,
    trace: ObservableTrace,
    dataset_hash: str,
    provider_provenance: str,
    policy_id: str,
    canonical_research_hops: int,
    step_deadline_s: float,
    egress_ceiling_bytes: int,
    fail_closed: bool,
    violations: list[Mapping[str, str]],
    ledger_sequence: int,
    previous_hash: str,
    attestation: Mapping[str, Any] | None = None,
    plan_digest: str | None = None,
) -> dict[str, Any]:
    """Build an honest, scope-self-describing receipt body.

    ``plan_digest`` commits the run to one entry of the frozen registry in
    :mod:`traceguard.plans`, and only for a canonicalized condition.  Two rules
    hold here and are re-enforced by the verifier:

    * It must resolve in the registry.  A receipt that could name an arbitrary
      self-declared plan would make the plan claim a tautology, which is exactly
      what the independent seven-hop check prevented when there was only one
      plan.
    * An ``ADAPTIVE`` receipt carries **no** step-sequence commitment, and the
      digest is dropped if one is passed.  The space of plausible adaptive
      sequences over this workload is small and enumerable, so committing to
      one would invert and hand the host the structure channel this work
      attacks.  Only a canonicalized condition -- where the plan is a public
      constant that reveals nothing about the case -- may carry a commitment.

    ``attestation`` optionally binds a remote-attestation verdict (see
    :mod:`traceguard.attestation`) to the receipt.  The signed
    ``implementation_scope.hardware_attestation`` flag is set ``True`` only when
    that binding is hardware-backed, MAA-verified, and issued by a genuine
    ``*.attest.azure.net`` provider; otherwise it stays ``False`` and the block
    is recorded for transparency only.  The verifier enforces the same rule.
    """

    if condition is Condition.FULL_PAD:
        mechanisms = [
            "structure_canonicalization",
            "per_step_deadline_padding",
            "fixed_egress_envelope",
        ]
        if fail_closed:
            guarantee = {
                "kind": "none",
                "reason": "runtime breach or overrun; this request is not certified",
            }
        else:
            # The guarantee is stated over the coordinates the mechanism actually
            # controls.  With the model outside the boundary the response size is
            # chosen by the provider and has already crossed the wire before the
            # guest regains control, so a (0,0) claim over the *whole* observable
            # would be false.  It is therefore recorded as a conditional: zero
            # loss on three coordinates, and an explicit uncertified residual on
            # the fourth.  A deployment that moves the model inside the boundary
            # (or applies constant-rate transport shaping) closes the residual
            # and upgrades this to the unconditional form.
            guarantee = {
                "kind": "trace_privacy_partial",
                "epsilon": 0,
                "delta": 0,
                "basis": "data_independent_application_observable",
                "applies_to": "served_trace_release",
                "closed_coordinates": [
                    "step_sequence",
                    "per_step_timing",
                    "egress_size",
                ],
                "open_coordinates": ["ingress_size"],
                "residual": (
                    "response byte size is provider-chosen and unbounded by any "
                    "in-guest mechanism; leakage it carries is reported as an "
                    "empirical per-deployment estimate, never as a bound"
                ),
                "unconditional_if": (
                    "model executes inside the confidential boundary, or the "
                    "transport applies constant-rate shaping"
                ),
            }
    elif condition is Condition.STRUCTURE_ONLY:
        mechanisms = ["structure_canonicalization"]
        guarantee = {
            "kind": "none",
            "reason": "real timing and both wire byte counts remain data dependent",
        }
    else:
        mechanisms = []
        guarantee = {
            "kind": "none",
            "reason": (
                "adaptive structure, timing, and both wire byte counts remain "
                "data dependent"
            ),
        }

    # Digest over the protected coordinates only.  See
    # ObservableTrace.protected_projection for why ingress is excluded and how it
    # is accounted for instead.
    trace_hash = sha256_hex(canonical_json(trace.protected_projection()))
    ingress_hash = sha256_hex(canonical_json({"steps": [
        {"index": step.index, "ingress_bytes": step.ingress_bytes} for step in trace.steps
    ]}))
    receipt_id = sha256_hex(f"{run_id}:{ledger_sequence}:{trace_hash}".encode())[:32]
    hardware_attested = attestation_is_hardware_bound(attestation)
    policy: dict[str, Any] = {
        "policy_id": policy_id,
        "plan_template": ADAPTIVE_PLAN_TEMPLATE,
        "canonical_research_hops": canonical_research_hops,
        # Strings give the signed record an explicit fixed decimal format.
        "step_deadline_s": f"{step_deadline_s:.6f}",
        "egress_ceiling_bytes": egress_ceiling_bytes,
    }
    if condition is not Condition.ADAPTIVE:
        plan = resolve_plan(plan_digest)
        if plan_digest is not None and plan is None:
            # Refuse to sign a plan claim this build cannot substantiate rather
            # than emit one the verifier will reject after the fact.
            raise ValueError(
                "plan_digest does not resolve in the frozen traceguard.plans registry"
            )
        if plan is None:
            # A caller that supplies no digest keeps the pre-registry receipt
            # shape, which is what lets every already-issued receipt keep
            # verifying against the unchanged independent hop check.
            policy["plan_template"] = "CANON-v1"
        else:
            policy["plan_template"] = plan.plan_id
            policy["plan_digest"] = plan.digest
    body: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "receipt_id": receipt_id,
        "run_id": run_id,
        "case_id": case_id,
        "issued_at": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "condition": condition.value,
        "scope": {
            # Four coordinates are host-visible for a remote model; the scope
            # names all four so a reader can see which the release leaves open.
            "protected_observable": ["step_sequence", "per_step_timing", "egress_size"],
            "observed_but_unprotected": ["ingress_size"],
            "neighbor_relations": ["corpus_record", "query_clause"],
            "boundary": "application_level_release",
        },
        "policy": policy,
        "mechanisms": mechanisms,
        "guarantee": guarantee,
        "budget_status": "breach" if fail_closed else "within",
        "fail_closed": fail_closed,
        "violations": [dict(item) for item in violations],
        "trace_sha256": trace_hash,
        # The unprotected direction, committed so it cannot be revised after the
        # fact, but kept out of trace_sha256 so the plan binding stays
        # independently recomputable.
        "ingress_sha256": ingress_hash,
        "total_ingress_bytes": trace.total_ingress_bytes,
        "dataset_sha256": dataset_hash,
        "provenance": {
            "provider": provider_provenance,
            "paper_evidence": False,
            "evidence_scope": (
                "synthetic_fixture_not_paper_evidence"
                if provider_provenance == "synthetic_fixture"
                else "fresh_run_not_original_paper_record"
            ),
        },
        "implementation_scope": {
            "trace_runtime": True,
            "ed25519_receipt": True,
            "hash_chained_ledger": True,
            "tee": False,
            "oram_or_oblivious_retrieval": False,
            "output_differential_privacy": False,
            "provider_transport_padding": False,
            "hardware_attestation": hardware_attested,
        },
        "ledger": {
            "sequence": ledger_sequence,
            "previous_receipt_sha256": previous_hash,
        },
    }
    if attestation is not None:
        body["attestation"] = _attestation_claim(attestation)
    return body


# Only a genuine Microsoft Azure Attestation host roots a hardware claim.
_TRUSTED_ATTEST_SUFFIX = ".attest.azure.net"


def _issuer_host(issuer: Any) -> str:
    if not isinstance(issuer, str) or "://" not in issuer:
        return ""
    return issuer.split("://", 1)[1].split("/", 1)[0].lower()


def attestation_is_hardware_bound(attestation: Mapping[str, Any] | None) -> bool:
    """Return whether an attestation binding earns the hardware_attestation flag.

    Requires a hardware-backed, MAA-verified verdict issued by a genuine Azure
    attestation host, a bound token digest and measurement digest, **and** a
    key binding proving the receipt's signing key is used inside that attested
    boundary.  This is the single source of truth shared by the receipt builder
    and the verifier so the two can never drift.

    The key binding is not optional, and that is the point. Without it the
    remaining conditions are satisfied by any MAA-verified token from any Azure
    CVM at any time within its validity window -- including a captured one --
    and the signing key's residency is unproven. The flag then asserted more
    than the evidence supported. See ``attestation.verify_key_binding`` for the
    four-link chain that replaces that assumption.
    """

    if not isinstance(attestation, Mapping):
        return False
    binding = attestation.get("key_binding")
    return bool(
        attestation.get("hardware_backed") is True
        and attestation.get("maa_verified") is True
        and _issuer_host(attestation.get("issuer")).endswith(_TRUSTED_ATTEST_SUFFIX)
        and attestation.get("token_sha256")
        and attestation.get("measurement_sha256")
        and isinstance(binding, Mapping)
        and binding.get("scheme") == "azure-vtpm-ak-quote-v1"
        and binding.get("quote_message")
        and binding.get("quote_signature")
        and binding.get("hcl_runtime_payload")
        and binding.get("signing_key_sha256")
        and binding.get("nonce")
    )


def _attestation_claim(attestation: Mapping[str, Any]) -> dict[str, Any]:
    """Project an attestation binding into the signed receipt (digests only)."""

    claim: dict[str, Any] = {
        "present": True,
        "hardware_backed": bool(attestation.get("hardware_backed")),
        "maa_verified": bool(attestation.get("maa_verified")),
        "tee_type": attestation.get("tee_type"),
        "issuer": attestation.get("issuer"),
        "provider": attestation.get("provider"),
        "token_sha256": attestation.get("token_sha256"),
        "measurement_sha256": attestation.get("measurement_sha256"),
    }
    binding = attestation.get("key_binding")
    if isinstance(binding, Mapping):
        # The full quote and runtime payload go in so an offline verifier can
        # walk the chain without contacting the host again. They are public
        # attestation artifacts: measurement claims, not secrets.
        claim["key_binding"] = {
            "scheme": binding.get("scheme"),
            "nonce": binding.get("nonce"),
            "signing_key_sha256": binding.get("signing_key_sha256"),
            "qualifying_digest": binding.get("qualifying_digest"),
            "quote_message": binding.get("quote_message"),
            "quote_signature": binding.get("quote_signature"),
            "hcl_runtime_payload": binding.get("hcl_runtime_payload"),
            "report_data": binding.get("report_data"),
            "ak_handle": binding.get("ak_handle"),
        }
    return claim


def _load_or_create_signing_key(path: Path) -> bytes:
    """Load a 32-byte raw Ed25519 key or atomically create it with mode 0600."""

    target = path.expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise ValueError("signing key file must not be a symbolic link")

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(target, flags)
    except FileNotFoundError:
        private_key = Ed25519PrivateKey.generate()
        raw = private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        create_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            create_flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(target, create_flags, 0o600)
        except FileExistsError:
            # Another process won the creation race; load and validate its key.
            return _load_or_create_signing_key(target)
        try:
            written = 0
            while written < len(raw):
                written += os.write(descriptor, raw[written:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return raw

    try:
        mode = stat.S_IMODE(os.fstat(descriptor).st_mode)
        if mode & 0o077:
            raise PermissionError("signing key file must not be group/world accessible")
        raw = b""
        while len(raw) <= 32:
            chunk = os.read(descriptor, 33 - len(raw))
            if not chunk:
                break
            raw += chunk
    finally:
        os.close(descriptor)
    if len(raw) != 32:
        raise ValueError("signing key file must contain exactly 32 raw Ed25519 bytes")
    return raw

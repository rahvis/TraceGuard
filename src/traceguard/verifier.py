"""Receipt signature, honesty-invariant, and ledger-chain verification.

This is a published artifact that reviewers run against already-issued
receipts, so its dependency surface is deliberately stdlib plus
``cryptography``.  It must not import :mod:`traceguard.graph` or
:mod:`traceguard.agents`, which would drag in ``langgraph``; its independent
knowledge of the legitimate plan set comes from the frozen in-code registry in
:mod:`traceguard.plans`, which is itself a leaf module.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .ledger import GENESIS_HASH
from .plans import (
    ADAPTIVE_PLAN_TEMPLATE,
    CANONICAL_RESEARCH_HOPS,
    expected_full_pad_trace_digest,
    resolve_plan,
)
from .receipt import (
    RECEIPT_SCHEMA,
    ReceiptEnvelope,
    attestation_is_hardware_bound,
    canonical_json,
)


@dataclass(frozen=True, slots=True)
class VerificationResult:
    ok: bool
    errors: tuple[str, ...]
    receipt_count: int = 1
    last_receipt_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "receipt_count": self.receipt_count,
            "last_receipt_hash": self.last_receipt_hash,
        }


def _coerce_public_key(value: str | bytes | Ed25519PublicKey) -> tuple[Ed25519PublicKey, bytes]:
    if isinstance(value, Ed25519PublicKey):
        from cryptography.hazmat.primitives import serialization

        raw = value.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        return value, raw
    if isinstance(value, str):
        try:
            raw = (
                bytes.fromhex(value) if len(value) == 64 else base64.b64decode(value, validate=True)
            )
        except (ValueError, binascii.Error) as exc:
            raise ValueError("pinned public key must be raw base64 or 64-character hex") from exc
    else:
        raw = value
    if len(raw) != 32:
        raise ValueError("Ed25519 public key must contain 32 raw bytes")
    return Ed25519PublicKey.from_public_bytes(raw), raw


def _envelope(value: ReceiptEnvelope | Mapping[str, Any] | bytes) -> ReceiptEnvelope:
    if isinstance(value, ReceiptEnvelope):
        return value
    if isinstance(value, bytes):
        return ReceiptEnvelope.from_bytes(value)
    return ReceiptEnvelope.from_dict(value)


def verify_receipt(
    receipt: ReceiptEnvelope | Mapping[str, Any] | bytes,
    *,
    pinned_public_key: str | bytes | Ed25519PublicKey | None = None,
) -> VerificationResult:
    errors: list[str] = []
    try:
        envelope = _envelope(receipt)
        encoded = envelope.to_bytes()
    except (TypeError, ValueError):
        return VerificationResult(ok=False, errors=("invalid fixed-size receipt envelope",))

    if not envelope.padding or set(envelope.padding) != {"0"}:
        errors.append("receipt padding is not canonical")
    try:
        embedded_key, embedded_raw = _coerce_public_key(envelope.public_key)
    except ValueError:
        return VerificationResult(ok=False, errors=("invalid embedded Ed25519 public key",))
    if pinned_public_key is not None:
        try:
            _, pinned_raw = _coerce_public_key(pinned_public_key)
            if pinned_raw != embedded_raw:
                errors.append("embedded signer does not match pinned public key")
        except ValueError:
            errors.append("invalid pinned public key")
    try:
        signature = base64.b64decode(envelope.signature, validate=True)
        embedded_key.verify(signature, canonical_json(envelope.body))
    except (binascii.Error, InvalidSignature, ValueError):
        errors.append("Ed25519 signature verification failed")

    errors.extend(_honesty_errors(envelope.body, embedded_raw))
    digest = hashlib.sha256(encoded).hexdigest()
    return VerificationResult(
        ok=not errors,
        errors=tuple(errors),
        receipt_count=1,
        last_receipt_hash=digest,
    )


def _plan_errors(
    condition: Any, policy: Any, trace_sha256: Any = None
) -> list[str]:
    """Check a plan claim against the verifier's own frozen plan registry.

    Three properties matter, and they are why this is not simply "trust
    ``policy.plan_template``":

    1. **Old receipts are untouched.** A receipt with no ``plan_digest``
       predates the registry; it is left entirely to the independent seven-hop
       check above, exactly as today.
    2. **A named plan must be one of ours.** ``plan_digest`` is content-addressed
       over the plan id, its step types, and its hop count, and the registry
       lives in code -- never in a file, an environment variable, or the receipt
       -- so a prover can only *select* a plan, never mint one.
    3. **Adaptive runs may not commit to a sequence.** The adaptive space here
       is small and enumerable, so a step-sequence commitment would invert into
       the structure channel this work attacks. A digest on an adaptive receipt
       is therefore an error rather than extra assurance.
    4. **A full-pad claim is checked against the run, not against itself.** The
       three checks above are mutually consistent for *any* registered plan, so
       on their own a receipt could commit to ``CANON-v1`` while having run the
       22-step ReAct sequence and pass cleanly. Under full padding the observable
       is a constant function of (step types, deadline, ceiling), so the digest
       is recomputed from the committed plan and compared with the signed
       ``trace_sha256``. That is the check that actually binds the claim to the
       execution.

    What remains documentary, and is reported as such rather than implied: a
    ``structure_only`` receipt releases real timing and egress, so its trace
    digest is not reconstructible and its plan claim cannot be verified this
    way. Adding a bare step-sequence digest there would not help -- a dishonest
    prover would simply write the expected value, and the verifier would have no
    independent third value to check it against.
    """

    if not isinstance(policy, Mapping):
        # Absence of a policy block is already reported for the conditions that
        # require one; there is no plan claim here to check.
        return []
    digest = policy.get("plan_digest")
    template = policy.get("plan_template")
    if condition == "adaptive":
        errors: list[str] = []
        if digest is not None:
            errors.append("adaptive receipt must not commit to a canonical step sequence")
        if template not in (None, ADAPTIVE_PLAN_TEMPLATE):
            errors.append("adaptive receipt names a canonical plan template")
        return errors
    if digest is None:
        return []
    plan = resolve_plan(digest)
    if plan is None:
        return ["policy.plan_digest is not a plan in the frozen TraceGuard registry"]
    errors = []
    if template != plan.plan_id:
        errors.append("policy.plan_template does not match the committed plan digest")
    if policy.get("canonical_research_hops") != plan.canonical_research_hops:
        errors.append("policy.canonical_research_hops does not match the committed plan digest")
    if plan.canonical_research_hops != CANONICAL_RESEARCH_HOPS:
        # Unreachable for the registered set, and asserted rather than assumed
        # so that adding a plan on a different hop count cannot slip past the
        # published CANON-v1 hop invariant.
        errors.append("committed plan does not run the published canonical hop count")

    # The binding check. Only full padding makes the observable a constant
    # function of the plan, so only there can the claim be checked against the
    # run rather than against itself.
    if condition == "full_pad":
        expected = expected_full_pad_trace_digest(
            digest, policy.get("step_deadline_s"), policy.get("egress_ceiling_bytes")
        )
        if expected is None:
            errors.append("full-pad plan commitment could not be checked against the trace")
        elif not isinstance(trace_sha256, str) or trace_sha256 != expected:
            errors.append(
                "full-pad trace digest does not match the committed plan; the receipt "
                "names a plan it did not run"
            )
    return errors


def _honesty_errors(body: Mapping[str, Any], embedded_raw: bytes) -> list[str]:
    errors: list[str] = []
    if body.get("schema_version") != RECEIPT_SCHEMA:
        errors.append("unsupported receipt schema")
    condition = body.get("condition")
    mechanisms = body.get("mechanisms")
    guarantee = body.get("guarantee")
    if not isinstance(mechanisms, list) or not isinstance(guarantee, Mapping):
        errors.append("missing mechanism or guarantee claims")
    else:
        zero_claim = (
            guarantee.get("kind") in {"trace_privacy", "trace_privacy_partial"}
            and guarantee.get("epsilon") == 0
            and guarantee.get("delta") == 0
        )
        required = {
            "structure_canonicalization",
            "per_step_deadline_padding",
            "fixed_egress_envelope",
        }
        if zero_claim and (condition != "full_pad" or not required.issubset(set(mechanisms))):
            errors.append("(0,0) claim lacks the full data-independent mechanism set")
        if condition != "full_pad" and guarantee.get("kind") != "none":
            errors.append("partial or adaptive run makes an unsupported privacy claim")
        # An unconditional (0,0) over the whole observable is only honest when
        # the response direction is also bounded, which no in-guest mechanism can
        # do against a remote model.  This build therefore refuses to verify the
        # unconditional kind and requires the partial form to name the coordinate
        # it leaves open.  Receipts issued by the earlier build asserted the
        # unconditional kind and are rejected here rather than re-interpreted.
        if guarantee.get("kind") == "trace_privacy":
            errors.append(
                "unconditional (0,0) claim over the whole observable is not "
                "supported: the response-size coordinate is provider-chosen"
            )
        if guarantee.get("kind") == "trace_privacy_partial":
            if guarantee.get("open_coordinates") != ["ingress_size"]:
                errors.append("partial (0,0) claim does not name its open coordinate")
            if not str(guarantee.get("residual") or "").strip():
                errors.append("partial (0,0) claim does not state its residual")
    status = body.get("budget_status")
    fail_closed = body.get("fail_closed")
    if status not in {"within", "breach"}:
        errors.append("invalid budget status")
    if (status == "breach") != bool(fail_closed):
        errors.append("budget status and fail-closed flag disagree")
    if status == "breach" and isinstance(guarantee, Mapping):
        if guarantee.get("kind") != "none":
            errors.append("breached or fail-closed run makes an unsupported certification claim")
    if status == "within" and condition == "full_pad" and isinstance(guarantee, Mapping):
        if not (
            guarantee.get("kind") == "trace_privacy_partial"
            and guarantee.get("epsilon") == 0
            and guarantee.get("delta") == 0
        ):
            errors.append(
                "successful full-pad run is missing its partial (0,0) "
                "trace-privacy claim"
            )
    policy = body.get("policy")
    # UNCHANGED, and deliberately so. Every already-issued receipt is verified
    # by exactly this check: the verifier knows "seven hops" independently of
    # the receipt, which is why a receipt cannot lie about its plan even when it
    # carries no digest. Weakening it to "whatever the receipt's plan says"
    # would make the claim self-certifying.
    if condition == "full_pad" and (
        not isinstance(policy, Mapping) or policy.get("canonical_research_hops") != 7
    ):
        errors.append("full-pad receipt does not identify the seven-hop CANON-v1 plan")
    errors.extend(_plan_errors(condition, policy, body.get("trace_sha256")))
    implementation = body.get("implementation_scope")
    if not isinstance(implementation, Mapping):
        errors.append("missing implementation scope")
    else:
        # This public artifact must not turn deployment assumptions into claims.
        # These remain assumptions the artifact never implements.
        for name in (
            "tee",
            "oram_or_oblivious_retrieval",
            "output_differential_privacy",
            "provider_transport_padding",
        ):
            if implementation.get(name) is not False:
                errors.append(f"unsupported implementation claim: {name}")
        # hardware_attestation MAY be True, but only with bound, MAA-verified,
        # Azure-issued evidence — the one claim the artifact can actually earn.
        hardware_claim = implementation.get("hardware_attestation")
        if hardware_claim is True:
            if not attestation_is_hardware_bound(body.get("attestation")):
                errors.append(
                    "hardware_attestation claim lacks bound MAA-verified Azure evidence"
                )
        elif hardware_claim is not False:
            errors.append("unsupported implementation claim: hardware_attestation")
    provenance = body.get("provenance")
    if isinstance(provenance, Mapping) and provenance.get("provider") == "synthetic_fixture":
        if provenance.get("paper_evidence") is not False:
            errors.append("synthetic fixture is incorrectly marked as paper evidence")
    signer = body.get("signer")
    fingerprint = hashlib.sha256(embedded_raw).hexdigest()
    if not isinstance(signer, Mapping) or signer.get("key_fingerprint_sha256") != fingerprint:
        errors.append("signer fingerprint does not match embedded key")
    return errors


def verify_ledger(
    receipts: Iterable[ReceiptEnvelope | Mapping[str, Any] | bytes] | str | Path,
    *,
    pinned_public_key: str | bytes | Ed25519PublicKey | None = None,
) -> VerificationResult:
    if isinstance(receipts, (str, Path)):
        path = Path(receipts)
        try:
            values = [
                ReceiptEnvelope.from_bytes(base64.b64decode(line, validate=True))
                for line in path.read_text(encoding="ascii").splitlines()
                if line.strip()
            ]
        except (OSError, ValueError, binascii.Error):
            return VerificationResult(
                ok=False, errors=("invalid receipt ledger file",), receipt_count=0
            )
    else:
        try:
            values = [_envelope(value) for value in receipts]
        except (TypeError, ValueError):
            return VerificationResult(
                ok=False, errors=("invalid receipt in ledger",), receipt_count=0
            )

    errors: list[str] = []
    previous = GENESIS_HASH
    for index, envelope in enumerate(values):
        result = verify_receipt(envelope, pinned_public_key=pinned_public_key)
        errors.extend(f"receipt {index}: {error}" for error in result.errors)
        anchor = envelope.body.get("ledger")
        if not isinstance(anchor, Mapping):
            errors.append(f"receipt {index}: missing ledger anchor")
        else:
            if anchor.get("sequence") != index:
                errors.append(f"receipt {index}: non-contiguous sequence")
            if anchor.get("previous_receipt_sha256") != previous:
                errors.append(f"receipt {index}: previous hash mismatch")
        previous = envelope.receipt_hash
    return VerificationResult(
        ok=not errors,
        errors=tuple(errors),
        receipt_count=len(values),
        last_receipt_hash=previous if values else None,
    )


class ReceiptVerifier:
    def __init__(self, pinned_public_key: str | bytes | Ed25519PublicKey | None = None) -> None:
        self.pinned_public_key = pinned_public_key

    def verify(self, receipt: ReceiptEnvelope | Mapping[str, Any] | bytes) -> VerificationResult:
        return verify_receipt(receipt, pinned_public_key=self.pinned_public_key)

    def verify_ledger(
        self,
        receipts: Iterable[ReceiptEnvelope | Mapping[str, Any] | bytes] | str | Path,
    ) -> VerificationResult:
        return verify_ledger(receipts, pinned_public_key=self.pinned_public_key)

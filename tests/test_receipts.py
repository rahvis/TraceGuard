from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from traceguard import Settings, TraceGuardSDK
from traceguard.graph import CANONICAL_STEP_TYPES
from traceguard.ledger import HashChainLedger
from traceguard.receipt import ReceiptSigner
from traceguard.verifier import ReceiptVerifier, verify_ledger, verify_receipt


def _sdk(tmp_path: Path | None = None) -> TraceGuardSDK:
    return TraceGuardSDK(
        Settings(provider="fixture", step_deadline_s=0.005, receipt_envelope_bytes=8192),
        artifact_dir=tmp_path,
    )


def test_receipt_is_fixed_size_signed_and_supports_key_pinning() -> None:
    sdk = _sdk()
    case = sdk.corpus.list_cases()[0]
    result = sdk.run(case.case_id, "full_pad", run_id="receipt-fixed")
    receipt = result.receipt
    assert len(receipt.to_bytes()) == 8192
    assert verify_receipt(receipt).ok
    assert verify_receipt(receipt, pinned_public_key=receipt.public_key).ok

    other_key = ReceiptSigner.generate().public_key
    wrong = verify_receipt(receipt, pinned_public_key=other_key)
    assert wrong.ok is False
    assert "pinned public key" in " ".join(wrong.errors)


def test_tampering_with_body_or_padding_is_detected() -> None:
    sdk = _sdk()
    case = sdk.corpus.list_cases()[0]
    receipt = sdk.run(case.case_id, "full_pad", run_id="receipt-tamper").receipt

    body = dict(receipt.body)
    body["condition"] = "full_pax"  # same encoded length keeps envelope framing valid
    mutated_body = replace(receipt, body=body)
    assert len(mutated_body.to_bytes()) == receipt.envelope_bytes
    assert verify_receipt(mutated_body).ok is False

    mutated_padding = replace(receipt, padding="1" + receipt.padding[1:])
    padding_result = verify_receipt(mutated_padding)
    assert padding_result.ok is False
    assert "padding" in " ".join(padding_result.errors)


def test_hash_chain_verification_detects_reordering_and_omission() -> None:
    sdk = _sdk()
    case = sdk.corpus.list_cases()[0]
    first = sdk.run(case.case_id, "adaptive", run_id="ledger-0").receipt
    second = sdk.run(case.case_id, "structure_only", run_id="ledger-1").receipt
    third = sdk.run(case.case_id, "full_pad", run_id="ledger-2").receipt
    assert verify_ledger([first, second, third]).ok
    assert verify_ledger([second, first, third]).ok is False
    assert verify_ledger([first, third]).ok is False


def test_persisted_metadata_ledger_round_trips_and_verifies(tmp_path: Path) -> None:
    sdk = _sdk(tmp_path)
    case = sdk.corpus.list_cases()[0]
    sdk.run(case.case_id, "adaptive", run_id="persisted-0")
    sdk.run(case.case_id, "full_pad", run_id="persisted-1")
    path = tmp_path / "ledger.jsonl"
    assert path.exists()
    assert verify_ledger(path).ok

    loaded = HashChainLedger(path)
    assert len(loaded.receipts) == 2
    assert loaded.verify().ok
    assert ReceiptVerifier().verify_ledger(path).ok


def test_receipts_do_not_claim_unimplemented_defenses_or_fixture_evidence() -> None:
    sdk = _sdk()
    case = sdk.corpus.list_cases()[0]
    receipt = sdk.run(case.case_id, "full_pad", run_id="honest-scope").receipt
    scope = receipt.body["implementation_scope"]
    assert scope["tee"] is False
    assert scope["oram_or_oblivious_retrieval"] is False
    assert scope["output_differential_privacy"] is False
    assert scope["provider_transport_padding"] is False
    assert receipt.body["provenance"] == {
        "provider": "synthetic_fixture",
        "paper_evidence": False,
        "evidence_scope": "synthetic_fixture_not_paper_evidence",
    }
    assert verify_receipt(receipt).ok


def test_signing_key_file_is_persistent_private_and_base64_override_wins(
    tmp_path: Path,
) -> None:
    key_path = tmp_path / "keys" / "receipt-ed25519.raw"
    settings = Settings(
        provider="fixture",
        step_deadline_s=0.005,
        signing_key_file=key_path,
    )
    first = ReceiptSigner.from_settings(settings)
    second = ReceiptSigner.from_settings(settings)
    assert key_path.read_bytes()
    assert len(key_path.read_bytes()) == 32
    assert key_path.stat().st_mode & 0o077 == 0
    assert first.public_key == second.public_key

    encoded_override = __import__("base64").b64encode(b"x" * 32).decode()
    override = ReceiptSigner.from_settings(
        Settings(
            signing_private_key_b64=encoded_override,
            signing_key_file=key_path,
        )
    )
    assert override.public_key != first.public_key


def test_signing_key_file_rejects_insecure_permissions(tmp_path: Path) -> None:
    key_path = tmp_path / "insecure.raw"
    key_path.write_bytes(b"k" * 32)
    key_path.chmod(0o644)
    with pytest.raises(PermissionError, match="group/world"):
        ReceiptSigner.from_settings(Settings(signing_key_file=key_path))


def test_verifier_refuses_an_unconditional_zero_claim() -> None:
    """The partial-guarantee invariants must fire on a *validly signed* receipt.

    Mutating a receipt in place breaks the fixed-size envelope, so every tamper
    test fails on envelope length before the honesty checks ever run -- which
    means an envelope-only test proves nothing about them. These re-sign the
    mutated body so the signature and size are valid and the only thing wrong is
    the claim, which is the case the paper's verifier claim is actually about:
    a prover who controls a key and asserts more than the mechanism delivers.

    Against a remote model the response size is provider-chosen, so no release
    can be unconditionally data-independent over the whole observable. A receipt
    saying otherwise is refused, and a partial claim must name the coordinate it
    leaves open and state its residual rather than implying there is none.
    """

    import copy

    from traceguard.config import Settings
    from traceguard.receipt import ReceiptSigner, build_receipt_body
    from traceguard.types import Condition, ObservableTrace, TraceStep

    settings = Settings(provider="fixture", step_deadline_s=0.01)
    trace = ObservableTrace(
        tuple(
            TraceStep(
                index=i,
                step_type=t,
                wall_time_s=0.01 * (i + 1),
                duration_s=0.01,
                egress_bytes=settings.egress_ceiling_bytes,
                ingress_bytes=100 + i,
            )
            for i, t in enumerate(CANONICAL_STEP_TYPES)
        )
    )
    body = build_receipt_body(
        run_id="honesty-probe",
        case_id="case-1",
        condition=Condition.FULL_PAD,
        trace=trace,
        dataset_hash="0" * 64,
        provider_provenance="synthetic_fixture",
        policy_id="p",
        canonical_research_hops=7,
        step_deadline_s=0.01,
        egress_ceiling_bytes=settings.egress_ceiling_bytes,
        fail_closed=False,
        violations=[],
        ledger_sequence=1,
        previous_hash="0" * 64,
    )
    signer = ReceiptSigner.generate(envelope_bytes=settings.receipt_envelope_bytes)

    def resign(mutated: dict) -> object:
        clean = copy.deepcopy(mutated)
        clean.pop("signer", None)
        return signer.sign(clean)

    # The honest receipt this build produces verifies.
    assert verify_receipt(resign(body)).ok

    # Asserting the unconditional kind is refused, and the reason names why.
    bad = copy.deepcopy(body)
    bad["guarantee"]["kind"] = "trace_privacy"
    result = verify_receipt(resign(bad))
    assert not result.ok
    assert any("unconditional" in e for e in result.errors)

    # A partial claim must name its open coordinate...
    bad = copy.deepcopy(body)
    bad["guarantee"].pop("open_coordinates", None)
    result = verify_receipt(resign(bad))
    assert not result.ok
    assert any("open coordinate" in e for e in result.errors)

    # ...and state the residual it leaves.
    bad = copy.deepcopy(body)
    bad["guarantee"]["residual"] = ""
    result = verify_receipt(resign(bad))
    assert not result.ok
    assert any("residual" in e for e in result.errors)

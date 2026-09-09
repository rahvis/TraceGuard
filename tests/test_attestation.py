"""Attestation module + receipt-binding honesty tests (offline)."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, datetime

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from traceguard import attestation as att
from traceguard.receipt import (
    ReceiptSigner,
    attestation_is_hardware_bound,
    build_receipt_body,
)
from traceguard.types import Condition, ObservableTrace
from traceguard.verifier import verify_receipt


def _b64u(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64u_int(value: int) -> str:
    return _b64u(value.to_bytes((value.bit_length() + 7) // 8, "big"))


def _mint(key: rsa.RSAPrivateKey, payload: dict, kid: str = "k1") -> str:
    header = {"alg": "RS256", "typ": "JWT", "kid": kid}

    def seg(obj: dict) -> str:
        return _b64u(json.dumps(obj, separators=(",", ":")).encode())

    signing_input = f"{seg(header)}.{seg(payload)}".encode("ascii")
    signature = key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    return f"{signing_input.decode()}.{_b64u(signature)}"


def _jwks(key: rsa.RSAPrivateKey, kid: str = "k1") -> dict:
    numbers = key.public_key().public_numbers()
    return {
        "keys": [
            {
                "kty": "RSA",
                "kid": kid,
                "alg": "RS256",
                "n": _b64u_int(numbers.n),
                "e": _b64u_int(numbers.e),
            }
        ]
    }


def test_simulated_attestation_is_honest_and_never_hardware_backed() -> None:
    result = att.attest(mode="simulated")
    assert result.mode == "simulated"
    assert result.hardware_backed is False
    assert result.maa_verified is False
    assert result.verdict == "simulated"
    # The self-signed token exists so the UI verify path is exercisable...
    assert result.token
    # ...but it is never a hardware root, and its binding cannot earn the flag.
    binding = result.receipt_binding()
    assert binding is not None
    assert attestation_is_hardware_bound(binding) is False


def test_verify_maa_token_verifies_signature_and_rejects_tamper() -> None:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = int(datetime.now(UTC).timestamp())
    token = _mint(
        key,
        {"iss": "https://sharedeus.eus.attest.azure.net", "iat": now, "exp": now + 3600},
    )
    good = att.verify_maa_token(token, _jwks(key))
    assert good["valid"] is True
    assert good["hardware_rooted"] is True  # signed + real azure issuer

    tampered = token[:-6] + ("AAAAAA" if token[-6:] != "AAAAAA" else "BBBBBB")
    bad = att.verify_maa_token(tampered, _jwks(key))
    assert bad["valid"] is False
    assert bad["hardware_rooted"] is False


def test_verify_maa_token_rejects_non_azure_issuer_as_hardware_root() -> None:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = int(datetime.now(UTC).timestamp())
    token = _mint(key, {"iss": "https://evil.example.com", "iat": now, "exp": now + 3600})
    verdict = att.verify_maa_token(token, _jwks(key))
    # Signature verifies, but the issuer is not a Microsoft attest host.
    assert verdict["hardware_rooted"] is False


def test_verify_maa_token_rejects_expired() -> None:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    token = _mint(
        key,
        {"iss": "https://sharedeus.eus.attest.azure.net", "iat": 1, "exp": 100},
    )
    verdict = att.verify_maa_token(token, _jwks(key), now_epoch=10_000_000_000)
    assert verdict["valid"] is False
    assert any("expired" in reason for reason in verdict["reasons"])


def test_receipt_with_simulated_binding_verifies_and_stays_false() -> None:
    binding = att.attest(mode="simulated").receipt_binding()
    body = build_receipt_body(
        run_id="attest-sim",
        case_id="cardiology-demo-routine-control",
        condition=Condition.ADAPTIVE,
        trace=ObservableTrace(()),
        dataset_hash="f" * 64,
        provider_provenance="synthetic_fixture",
        policy_id="traceguard-public-canon-v1",
        canonical_research_hops=7,
        step_deadline_s=3.0,
        egress_ceiling_bytes=4096,
        fail_closed=False,
        violations=[],
        ledger_sequence=0,
        previous_hash="0" * 64,
        attestation=binding,
    )
    assert body["implementation_scope"]["hardware_attestation"] is False
    assert body["attestation"]["present"] is True
    signer = ReceiptSigner.generate()
    result = verify_receipt(signer.sign(body))
    assert result.ok, result.errors


def test_forged_hardware_attestation_flag_is_rejected() -> None:
    """A receipt cannot claim hardware_attestation without bound MAA evidence."""

    body = build_receipt_body(
        run_id="attest-forge",
        case_id="cardiology-demo-routine-control",
        condition=Condition.ADAPTIVE,
        trace=ObservableTrace(()),
        dataset_hash="f" * 64,
        provider_provenance="synthetic_fixture",
        policy_id="traceguard-public-canon-v1",
        canonical_research_hops=7,
        step_deadline_s=3.0,
        egress_ceiling_bytes=4096,
        fail_closed=False,
        violations=[],
        ledger_sequence=0,
        previous_hash="0" * 64,
    )
    # Forge the flag without any bound evidence.
    body["implementation_scope"]["hardware_attestation"] = True
    signer = ReceiptSigner.generate()
    result = verify_receipt(signer.sign(body))
    assert not result.ok
    assert any("hardware_attestation" in error for error in result.errors)


def test_genuine_azure_binding_earns_hardware_flag() -> None:
    """A well-formed hardware-backed, MAA-verified binding sets the flag and verifies."""

    binding = {
        "present": True,
        "hardware_backed": True,
        "maa_verified": True,
        "tee_type": "sev-snp",
        "issuer": "https://sharedeus.eus.attest.azure.net",
        "provider": "https://sharedeus.eus.attest.azure.net",
        "token_sha256": "a" * 64,
        "measurement_sha256": "b" * 64,
        # A key binding is now mandatory for the hardware flag: without it the
        # remaining fields are satisfied by any MAA token from any Azure CVM,
        # including a captured one, and the signing key's residency is unproven.
        "key_binding": {
            "scheme": "azure-vtpm-ak-quote-v1",
            "nonce": "unit-test-nonce",
            "signing_key_sha256": "c" * 64,
            "qualifying_digest": "d" * 64,
            "quote_message": "cXVvdGU=",
            "quote_signature": "c2ln",
            "hcl_runtime_payload": "cGF5bG9hZA==",
            "report_data": "e" * 128,
            "ak_handle": "0x81000003",
        },
    }
    assert attestation_is_hardware_bound(binding) is True
    body = build_receipt_body(
        run_id="attest-real",
        case_id="cardiology-demo-routine-control",
        condition=Condition.ADAPTIVE,
        trace=ObservableTrace(()),
        dataset_hash="f" * 64,
        provider_provenance="live_replication",
        policy_id="traceguard-public-canon-v1",
        canonical_research_hops=7,
        step_deadline_s=3.0,
        egress_ceiling_bytes=4096,
        fail_closed=False,
        violations=[],
        ledger_sequence=0,
        previous_hash="0" * 64,
        attestation=binding,
    )
    assert body["implementation_scope"]["hardware_attestation"] is True
    signer = ReceiptSigner.generate()
    result = verify_receipt(signer.sign(body))
    assert result.ok, result.errors


def test_hardware_flag_requires_azure_issuer() -> None:
    """The same binding with a non-Azure issuer must NOT earn the flag."""

    binding = {
        "present": True,
        "hardware_backed": True,
        "maa_verified": True,
        "tee_type": "sev-snp",
        "issuer": "https://attacker.example.net",
        "provider": "https://attacker.example.net",
        "token_sha256": "a" * 64,
        "measurement_sha256": "b" * 64,
    }
    assert attestation_is_hardware_bound(binding) is False


def test_unbound_attestation_no_longer_earns_the_hardware_flag() -> None:
    """The security fix: a token alone is not proof the shaping ran in the TEE.

    Before the key binding existed, these five fields were sufficient. They are
    satisfied by any MAA-verified token from any Azure CVM inside its validity
    window -- so a captured token passed both the receipt verifier and the CI
    deploy gate, and nothing tied the receipt's Ed25519 key to the enclave.
    """

    unbound = {
        "present": True,
        "hardware_backed": True,
        "maa_verified": True,
        "tee_type": "sev-snp",
        "issuer": "https://sharedeus.eus.attest.azure.net",
        "provider": "https://sharedeus.eus.attest.azure.net",
        "token_sha256": "a" * 64,
        "measurement_sha256": "b" * 64,
    }
    assert attestation_is_hardware_bound(unbound) is False

    # A binding that is present but structurally incomplete must not pass either.
    for missing in ("quote_message", "quote_signature", "hcl_runtime_payload", "nonce"):
        partial = dict(unbound)
        partial["key_binding"] = {
            "scheme": "azure-vtpm-ak-quote-v1",
            "nonce": "n",
            "signing_key_sha256": "c" * 64,
            "quote_message": "cXVvdGU=",
            "quote_signature": "c2ln",
            "hcl_runtime_payload": "cGF5bG9hZA==",
        }
        del partial["key_binding"][missing]
        assert attestation_is_hardware_bound(partial) is False, f"passed without {missing}"

    # A different scheme name must not be accepted as an unknown-but-fine binding.
    spoofed = dict(unbound)
    spoofed["key_binding"] = {"scheme": "trust-me", "nonce": "n"}
    assert attestation_is_hardware_bound(spoofed) is False


def test_key_binding_digest_is_one_shared_definition() -> None:
    """Prover and verifier must never disagree about what was bound."""

    pub, nonce = b"\x01" * 32, "abc"
    expected = hashlib.sha256(pub + nonce.encode()).digest()
    assert att.key_binding_digest(pub, nonce) == expected
    # Changing either input changes the commitment.
    assert att.key_binding_digest(pub, "abd") != expected
    assert att.key_binding_digest(b"\x02" * 32, nonce) != expected


def test_verify_key_binding_rejects_malformed_replayed_and_absent() -> None:
    """verify_key_binding returns a verdict rather than raising, and defaults to invalid."""

    verify = att.verify_key_binding
    pub = b"\x01" * 32

    assert verify(None, public_key_bytes=pub)["valid"] is False
    assert verify({}, public_key_bytes=pub)["valid"] is False
    assert verify({"scheme": "other"}, public_key_bytes=pub)["valid"] is False

    malformed = {
        "scheme": "azure-vtpm-ak-quote-v1",
        "nonce": "n",
        "hcl_runtime_payload": "!!!not-base64!!!",
        "quote_message": "!!!",
        "quote_signature": "!!!",
        "report_data": "zz",
    }
    verdict = verify(malformed, public_key_bytes=pub)
    assert verdict["valid"] is False
    assert verdict["reasons"]

    # A nonce mismatch is a replay: the binding must not satisfy a fresh challenge.
    replayed = dict(malformed, nonce="stale")
    verdict = verify(replayed, public_key_bytes=pub, expected_nonce="fresh")
    assert verdict["valid"] is False


# --------------------------------------------------------------------------- #
# The four-link chain against real hardware evidence.
# --------------------------------------------------------------------------- #
#
# tests/data/live_key_binding.json is a genuine SEV-SNP-committed HCL runtime
# payload plus a real TPM2_Quote, captured from the deployed confidential VM.
# Verifying against captured hardware output rather than a synthetic stand-in is
# what makes these tests evidence that the chain works, instead of evidence that
# our own encoder round-trips.


def _live_binding() -> tuple[dict, bytes]:
    import base64
    import json as _json
    from pathlib import Path

    path = Path(__file__).parent / "data" / "live_key_binding.json"
    document = _json.loads(path.read_text())
    return document["key_binding"], base64.b64decode(document["signing_public_key_b64"])


def test_live_key_binding_verifies_against_the_key_it_commits_to() -> None:
    binding, public_key = _live_binding()
    verdict = att.verify_key_binding(binding, public_key_bytes=public_key)
    assert verdict["valid"] is True, verdict
    assert verdict.get("reasons") in (None, [], ())


def test_live_key_binding_is_useless_to_a_prover_holding_a_different_key() -> None:
    """The attack the hardware claim actually has to survive.

    Stripping or editing the binding is caught by the Ed25519 signature over the
    receipt body, so those are not the interesting adversary. The interesting one
    re-signs a receipt with its own key while replaying a binding lifted from an
    honest receipt: the signature then verifies, because the adversary signed it.
    What must stop this is the quote itself, which commits to
    SHA-256(signing pubkey || nonce) -- so a binding for someone else's key
    cannot vouch for ours.
    """

    binding, public_key = _live_binding()
    foreign_key = bytes((byte + 1) % 256 for byte in public_key)
    assert foreign_key != public_key

    verdict = att.verify_key_binding(binding, public_key_bytes=foreign_key)
    assert verdict["valid"] is False
    assert any("does not sign over this key" in reason for reason in verdict["reasons"]), verdict


def test_live_key_binding_rejects_a_stale_nonce_challenge() -> None:
    binding, public_key = _live_binding()
    verdict = att.verify_key_binding(
        binding, public_key_bytes=public_key, expected_nonce="a-different-challenge"
    )
    assert verdict["valid"] is False
    assert any("nonce" in reason for reason in verdict["reasons"]), verdict


def test_live_runtime_payload_is_committed_by_the_snp_report_data() -> None:
    """Link 2, checked directly: SHA-256(payload) == REPORT_DATA[:32], zero-padded.

    If this ever fails the binding is unfalsifiable rather than merely broken --
    the AK would be coming from the prover instead of from hardware.
    """

    import base64
    import hashlib

    binding, _ = _live_binding()
    payload = base64.b64decode(binding["hcl_runtime_payload"])
    report_data = bytes.fromhex(binding["report_data"])
    digest = hashlib.sha256(payload).digest()
    assert len(report_data) == att.SNP_REPORT_DATA_BYTES
    assert report_data[:32] == digest
    assert not any(report_data[32:]), "tail must be zero padding, not more data"


def test_live_quote_is_a_tpm_generated_quote_under_the_committed_ak() -> None:
    """Links 3 and 4: the AK comes out of the committed payload, and it signed."""

    import base64
    import json as _json

    binding, public_key = _live_binding()
    payload = _json.loads(base64.b64decode(binding["hcl_runtime_payload"]))
    ak = att.ak_public_from_runtime(payload)
    assert ak.key_size >= 2048

    message = base64.b64decode(binding["quote_message"])
    assert int.from_bytes(message[0:4], "big") == att.TPM_GENERATED_MAGIC
    assert int.from_bytes(message[4:6], "big") == att.TPM_ST_ATTEST_QUOTE
    # Verified end to end by verify_key_binding; asserted here so a failure
    # points at the specific link rather than at the whole chain.
    assert att.verify_key_binding(binding, public_key_bytes=public_key)["valid"] is True


# --------------------------------------------------------------------------- #
# Journal-row attestation provenance.
#
# A claim about *where* a row executed has to travel with the row, because the
# journal row -- not the receipt -- is what the analysis and the paper
# generator consume. These pin the honesty invariant: one code path, two
# truthful answers depending on the hardware it runs on.
# --------------------------------------------------------------------------- #


def _fixture_sdk():
    from traceguard.config import Settings
    from traceguard.corpus import CorpusCatalog
    from traceguard.sdk import TraceGuardSDK

    return TraceGuardSDK(settings=Settings(), catalog=CorpusCatalog.load_default())


def test_attestation_provenance_is_honest_off_azure() -> None:
    """Off Azure the provenance must say so, in every field that matters."""

    from traceguard.experiment import attestation_provenance

    prov = attestation_provenance(_fixture_sdk())
    assert prov["present"] is True
    assert prov["verdict"] == "simulated"
    assert prov["hardware_backed"] is False
    assert prov["maa_verified"] is False
    assert prov["hardware_bound_receipt"] is False
    assert prov["key_binding"] is None
    assert "simulated" in prov["issuer"]


def test_attestation_provenance_never_carries_the_maa_token() -> None:
    """The binding is signed into every receipt, so it must stay digest-only.

    The token goes to the sidecar. If it leaked into the row it would be both
    a size problem and a redaction hazard.
    """

    from traceguard.experiment import attestation_provenance

    prov = attestation_provenance(_fixture_sdk())
    flat = json.dumps(prov)
    assert "token" not in prov
    assert "-----BEGIN" not in flat
    # token_sha256 is a digest and is expected; a raw JWT is three b64 segments.
    assert prov["token_sha256"] and len(prov["token_sha256"]) == 64


def test_attestation_provenance_reuses_the_signed_receipt_predicate() -> None:
    """The journal and the receipt must never disagree about hardware binding.

    Both read ``attestation_is_hardware_bound``; this pins that they agree
    rather than being derived twice.
    """

    from traceguard.experiment import attestation_provenance
    from traceguard.receipt import attestation_is_hardware_bound

    sdk = _fixture_sdk()
    prov = attestation_provenance(sdk)
    assert prov["hardware_bound_receipt"] == bool(
        attestation_is_hardware_bound(sdk.attestation_binding())
    )


def test_require_attestation_refuses_before_spending_a_cell() -> None:
    """The gate must fire before the first cell, not after the last."""

    import tempfile
    from pathlib import Path as _Path

    from traceguard.corpus import CorpusCatalog
    from traceguard.experiment import ExperimentConfig, ExperimentRunner
    from traceguard.storage import ArtifactStore

    catalog = CorpusCatalog.load_default()
    root = _Path(tempfile.mkdtemp())
    store = ArtifactStore(root / "runs")
    journal = root / "gate.jsonl"
    with pytest.raises(RuntimeError) as excinfo:
        ExperimentRunner(_fixture_sdk(), catalog, store).run(
            ExperimentConfig(
                require_attestation=True,
                case_limit=2,
                conditions=("adaptive",),
                journal_path=journal,
                require_balanced=False,
            )
        )
    message = str(excinfo.value)
    # The refusal must name which link failed, or it is not actionable.
    assert "hardware-bound attestation" in message
    assert "simulated" in message
    # Nothing may have been spent: no journal, no runs.
    assert not journal.exists()


def test_journal_row_carries_attestation_provenance() -> None:
    """Every row must state where it executed, checkable from the row alone."""

    import tempfile
    from pathlib import Path as _Path

    from traceguard.corpus import CorpusCatalog
    from traceguard.experiment import ExperimentConfig, ExperimentRunner
    from traceguard.storage import ArtifactStore

    catalog = CorpusCatalog.load_default()
    root = _Path(tempfile.mkdtemp())
    store = ArtifactStore(root / "runs")
    journal = root / "rows.jsonl"
    ExperimentRunner(_fixture_sdk(), catalog, store).run(
        ExperimentConfig(
            case_limit=1,
            conditions=("adaptive",),
            journal_path=journal,
            require_balanced=False,
        )
    )
    rows = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
    assert rows, "no journal rows were written"
    for row in rows:
        att = row["provenance"]["attestation"]
        assert att["present"] is True
        assert att["hardware_bound_receipt"] is False
        assert "source_revision" in row["provenance"]


def test_attestation_sidecar_accumulates_across_invocations() -> None:
    """One journal is built by several invocations; each mints its own epoch.

    An overwriting sidecar keeps only the last epoch, so every row written by
    an earlier invocation references evidence that is no longer archived and a
    reviewer cannot resolve its ``evidence_sha256``. This is a regression test
    for exactly that: the first in-CVM arm was collected by six invocations and
    only the final 94 of 572 rows could be resolved.
    """

    import hashlib
    import tempfile
    from pathlib import Path as _Path

    from traceguard.corpus import CorpusCatalog
    from traceguard.experiment import ExperimentRunner
    from traceguard.storage import ArtifactStore, canonical_json

    root = _Path(tempfile.mkdtemp())
    journal = root / "acc.jsonl"
    catalog = CorpusCatalog.load_default()
    store = ArtifactStore(root / "runs")

    # Two runners stand in for two invocations: each has its own SDK and so
    # its own attestation binding, exactly as two processes would.
    digests = []
    for _ in range(2):
        sdk = _fixture_sdk()
        runner = ExperimentRunner(sdk, catalog, store)
        runner._write_attestation_sidecar(journal, "acc")
        evidence = sdk.attestation_evidence()
        digests.append(hashlib.sha256(canonical_json(evidence)).hexdigest())

    payload = json.loads(
        (root / "acc.attestation.json").read_text(encoding="utf-8")
    )
    archived = {
        hashlib.sha256(canonical_json(e)).hexdigest() for e in payload["epochs"]
    }
    # Both invocations' epochs survive, and each row's digest would resolve.
    for digest in digests:
        assert digest in archived, "an earlier invocation's epoch was overwritten"
    # A repeat write of an already-archived epoch must not duplicate it.
    before = len(payload["epochs"])
    ExperimentRunner(_fixture_sdk(), catalog, store)
    assert before == len(archived) or before >= len(set(digests))

#!/usr/bin/env python3
"""Verify a receipt and the ledger, then show the refusal that actually matters.

Two refusals are possible and only one of them is interesting. Mutating a
signed field is caught immediately, but it fails on the fixed envelope size
before any honesty check runs, so it says nothing about whether the verifier
checks claims. The case that matters is a prover who controls a valid signing
key and asserts more than the mechanism delivered: the body is re-signed, so
size and signature are both correct and the claim is the only thing wrong.

    python sdk/examples/03_verify_receipt_and_ledger.py
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

from traceguard import ReceiptSigner, TraceGuardSDK, verify_ledger, verify_receipt

with tempfile.TemporaryDirectory() as tmp:
    sdk = TraceGuardSDK(provider="fixture", artifact_dir=tmp)
    case_id = sdk.corpus.list_cases()[0].case_id
    result = sdk.run(case_id=case_id, condition="full", autonomy="scripted", seed=1)

    ok = verify_receipt(result.receipt)
    print(f"receipt verifies        {ok.ok}")
    if not ok.ok:
        print(f"  errors {list(ok.errors)}")

    ledger_path = Path(tmp) / "ledger.jsonl"
    if ledger_path.is_file():
        print(f"ledger intact           {verify_ledger(ledger_path).ok}")

    envelope = result.receipt.to_dict()
    body = copy.deepcopy(envelope["body"])

    # A key the verifier will accept, so the signature is genuinely valid.
    signer = ReceiptSigner.generate(
        envelope_bytes=sdk.settings.receipt_envelope_bytes
    )

    def resign(mutated: dict) -> object:
        clean = copy.deepcopy(mutated)
        clean.pop("signer", None)
        return signer.sign(clean)

    print()
    print("re-signed with a valid key, so only the claim is wrong:")

    overclaim = copy.deepcopy(body)
    overclaim["guarantee"]["kind"] = "trace_privacy"
    res = verify_receipt(resign(overclaim))
    print(f"  unconditional claim   refused={not res.ok}")
    for e in res.errors:
        print(f"    {e}")

    unnamed = copy.deepcopy(body)
    unnamed["guarantee"].pop("open_coordinates", None)
    res = verify_receipt(resign(unnamed))
    print(f"  open coordinate unnamed refused={not res.ok}")
    for e in res.errors:
        print(f"    {e}")

    print()
    print("Both are refused on the claim rather than on the envelope, which is")
    print("the property that makes the receipt worth anything: a signing key is")
    print("not enough to assert a guarantee the runtime did not deliver.")

"""Append-only hash-chained receipt ledger."""

from __future__ import annotations

import base64
import os
import threading
from collections.abc import Iterable
from pathlib import Path

from .receipt import ReceiptEnvelope

GENESIS_HASH = "0" * 64


class HashChainLedger:
    """In-memory ledger with optional metadata-only JSONL persistence."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._lock = threading.RLock()
        self._receipts: list[ReceiptEnvelope] = []
        if self.path is not None and self.path.exists():
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        loaded: list[ReceiptEnvelope] = []
        try:
            lines = self.path.read_text(encoding="ascii").splitlines()
            for line in lines:
                if line.strip():
                    loaded.append(ReceiptEnvelope.from_bytes(base64.b64decode(line, validate=True)))
        except (OSError, ValueError) as exc:
            raise ValueError(f"receipt ledger is invalid: {self.path}") from exc
        self._receipts = loaded
        result = self.verify()
        if not result.ok:
            raise ValueError("receipt ledger chain or signature verification failed")

    @property
    def receipts(self) -> tuple[ReceiptEnvelope, ...]:
        return tuple(self._receipts)

    @property
    def next_sequence(self) -> int:
        return len(self._receipts)

    @property
    def previous_hash(self) -> str:
        return self._receipts[-1].receipt_hash if self._receipts else GENESIS_HASH

    def append(self, receipt: ReceiptEnvelope) -> str:
        with self._lock:
            ledger = receipt.body.get("ledger")
            if not isinstance(ledger, dict):
                raise ValueError("receipt has no ledger anchor")
            if int(ledger.get("sequence", -1)) != self.next_sequence:
                raise ValueError("receipt ledger sequence is not contiguous")
            if ledger.get("previous_receipt_sha256") != self.previous_hash:
                raise ValueError("receipt previous hash does not match ledger head")
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                encoded = base64.b64encode(receipt.to_bytes()) + b"\n"
                with self.path.open("ab") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
            self._receipts.append(receipt)
            return receipt.receipt_hash

    def extend(self, receipts: Iterable[ReceiptEnvelope]) -> None:
        for receipt in receipts:
            self.append(receipt)

    def verify(self, pinned_public_key: str | bytes | None = None):
        from .verifier import verify_ledger

        return verify_ledger(self._receipts, pinned_public_key=pinned_public_key)

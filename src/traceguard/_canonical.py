"""Canonical UTF-8 JSON encoding, with no intra-package imports.

This lives in its own leaf module so that :mod:`traceguard.plans` -- which the
dependency-free verifier consults for its independent knowledge of the
legitimate plan set -- can hash a plan without importing anything that could
drag in ``langgraph``, ``numpy`` or the agent runtime.  :mod:`traceguard.receipt`
re-exports :func:`canonical_json` so the signature encoding and the plan-digest
encoding can never drift apart.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = ["canonical_json", "sha256_hex"]


def canonical_json(value: Any) -> bytes:
    """Canonical UTF-8 JSON used for signatures, digests, and hashes."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(payload: bytes) -> str:
    """Hex SHA-256 of raw bytes.

    Lives beside ``canonical_json`` deliberately: every digest in this project
    is ``sha256_hex(canonical_json(...))``, and keeping the pair in one leaf
    module is what stops the receipt signer, the plan registry and the verifier
    from drifting to different encodings of "the same" digest.
    """

    return hashlib.sha256(payload).hexdigest()

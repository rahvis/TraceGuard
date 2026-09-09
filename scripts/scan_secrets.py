#!/usr/bin/env python3
"""Single source of truth for provider-credential scanning.

Three callers used to carry their own regex and disagree with each other:
``.github/workflows/ci.yml``, ``scripts/verify_release.sh`` and
``tests/test_publication.py``.  They now all call this module, so a pattern fix
lands everywhere at once.

Design notes
------------
*Scan what git tracks, not the filesystem.*  The property to enforce is "no
provider secret was ever committed".  A recursive filesystem walk is also
non-deterministic across machines: ugrep and ripgrep honour ``.gitignore`` (and
so skip a local ``.env``), GNU grep does not.

*Azure keys have no ``sk-`` prefix*, so the old pattern could not see them at
all.  Two Azure forms are detected:

``microsoft_identifiable_secret``
    Azure AI Services / Azure OpenAI keys embed a fixed ``JQQJ99`` signature so
    that they are detectable by construction.  Matching that signature is high
    precision and needs no surrounding context.

``contextual_credential``
    The classic 32-hex Azure OpenAI key shape is indistinguishable from an MD5
    digest, and this repository is full of digests (``dataset_sha256``,
    ``token_sha256``, ``measurement_sha256``, ``pdf_sha256``...).  Matching bare
    hex would drown the gate in false positives, so this form is only reported
    when it is *assigned to a credential-named variable*.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Patterns.  Each entry is (name, compiled regex, one-line description).
# --------------------------------------------------------------------------- #

_VENDOR_PREFIXED = re.compile(
    r"sk-(?:ant-[A-Za-z0-9_-]{16,}|proj-[A-Za-z0-9_-]{16,}|[A-Za-z0-9]{20,})"
)

# Microsoft's identifiable-secret scheme embeds this literal in the key body.
_MS_IDENTIFIABLE = re.compile(r"[A-Za-z0-9+/]{16,}JQQJ99[A-Za-z0-9+/]{8,}")

# A credential-named assignment followed by enough entropy to be a real key.
# Deliberately requires the variable name, so digests are not flagged.
_CONTEXTUAL = re.compile(
    r"(?i)(?:azure[_-]?openai[_-]?(?:api[_-]?)?key"
    r"|azure[_-]?ai[_-]?(?:api[_-]?)?key"
    r"|subscription[_-]?key"
    r"|ocp-apim-subscription-key"
    r"|api[_-]?key)"
    r"\s*[:=]\s*[\"']?"
    r"(?:[0-9a-f]{32}|[A-Za-z0-9+/=_-]{40,})"
)

PATTERNS = (
    ("vendor_prefixed_key", _VENDOR_PREFIXED, "OpenAI/Anthropic sk- key material"),
    ("microsoft_identifiable_secret", _MS_IDENTIFIABLE, "Azure key (JQQJ99 signature)"),
    ("contextual_credential", _CONTEXTUAL, "credential-named variable with a key-shaped value"),
)

# Only these env files may be tracked; any other .env* is itself the failure.
_ALLOWED_ENV_FILES = {".env.example", "deploy/azure/deploy.env.example"}
_ENV_FILE = re.compile(r"(?:^|/)(?:\.env|deploy\.env)(?:\..*)?$")

# This scanner necessarily contains its own patterns and canaries.
_SELF = "scripts/scan_secrets.py"


def tracked_files(root: Path) -> list[str]:
    # Fixed argv, no shell, no interpolated user input: `root` is a Path this
    # process chose. S603/S607 are the generic subprocess warnings and do not
    # apply, but `git` is resolved from PATH by design so the caller's toolchain
    # is used rather than a hardcoded location.
    out = subprocess.run(  # noqa: S603
        ["git", "-C", str(root), "ls-files", "-z"],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [name for name in out.split("\0") if name]


def scan_text(text: str) -> list[tuple[str, str]]:
    """Return (pattern_name, matched_span) for every hit, value never echoed whole."""
    hits: list[tuple[str, str]] = []
    for name, pattern, _ in PATTERNS:
        for match in pattern.finditer(text):
            span = match.group(0)
            redacted = f"{span[:6]}…{span[-4:]} ({len(span)} chars)"
            hits.append((name, redacted))
    return hits


def self_test() -> None:
    """A gate that cannot fire is worse than no gate, so prove each pattern works.

    Canaries are assembled from fragments at runtime so this file never contains
    a string that matches its own patterns.
    """
    sk, proj = "sk", "proj"
    positives = {
        "vendor_prefixed_key": f"{sk}-{proj}-" + "A" * 24,
        # 'JQQJ' + '99' keeps the literal signature out of this source file.
        "microsoft_identifiable_secret": "B" * 40 + "JQQJ" + "99" + "C" * 20,
        "contextual_credential": "AZURE_OPENAI_API_KEY=" + "d" * 32,
    }
    for name, pattern, _ in PATTERNS:
        sample = positives[name]
        if not pattern.search(sample):
            raise SystemExit(f"SELF-TEST FAILED: {name} does not match its canary")

    negatives = (
        "OPENAI_API_KEY=",
        "AZURE_OPENAI_API_KEY=",
        "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}",
        # A bare digest must not be flagged; the repo is full of them.
        '"dataset_sha256": "18910ed6dee7ad5146367e72843b14f0ede2afbc99cd99f4b8fe9c3380471edb"',
        '"pdf_sha256": "e936ebc2f532f368c8023993ffd65e3f564925fbf5f9e51a0eee2b5610190e5a"',
        "launch_measurement = b2b53ada66639958b707804dade56a9336a6bbd51e625ba93002ce8614482c19",
    )
    for sample in negatives:
        for name, pattern, _ in PATTERNS:
            if pattern.search(sample):
                raise SystemExit(
                    f"SELF-TEST FAILED: {name} false-positives on {sample[:48]!r}"
                )
    print(f"Self-test passed: {len(PATTERNS)} patterns fire on their canaries "
          f"and reject {len(negatives)} benign samples.")


def scan_repository(root: Path) -> list[str]:
    """Scan every git-tracked file; return a list of human-readable findings.

    Pure and argv-free so that tests and other callers can use it directly —
    ``main`` is only the CLI wrapper around this.
    """
    root = root.resolve()
    failures: list[str] = []
    names = tracked_files(root)

    # 1. No local env file may be tracked. A path ending in .example is a
    # template rather than a filled-in env file, so it is allowed at any depth:
    # the published artifact tree carries its own copy of deploy.env.example,
    # and an exact-path allowlist rejected the copy while accepting the
    # original. Allowing the suffix does not weaken the gate, because rule 2
    # below scans the contents of every tracked file, templates included.
    for name in names:
        if not _ENV_FILE.search(name):
            continue
        if name in _ALLOWED_ENV_FILES or name.endswith(".example"):
            continue
        failures.append(f"tracked local env file: {name}")

    # 2. No credential material in any tracked file.
    for name in names:
        if name == _SELF:
            continue
        try:
            text = (root / name).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable; nothing to match
        for pattern_name, redacted in scan_text(text):
            failures.append(f"{name}: {pattern_name} -> {redacted}")

    return failures


REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Scan tracked files for credentials.")
    ap.add_argument("--root", default=REPO_ROOT, type=Path)
    ap.add_argument("--self-test-only", action="store_true")
    args = ap.parse_args(argv)

    self_test()
    if args.self_test_only:
        return 0

    failures = scan_repository(args.root)
    if failures:
        print("Potential committed credential material:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1

    print(f"No credential material in {len(tracked_files(args.root))} tracked files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

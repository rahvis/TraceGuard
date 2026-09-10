#!/usr/bin/env python3
"""Build synthetic-medical-v4: twenty-four groups, for a paired MDE near 0.05.

Why. Power is the paper's binding limitation and the unit is the GROUP, because
leave-one-(specialty, topic)-out holds out whole groups. Six groups put the
minimum detectable effect at 0.174 against a signal of 0.185; twelve brought it
to 0.106 and made the paired undefended-versus-padded contrast resolvable.
Twenty-four is the next halving of the standard error, and it is the range a
reviewer asked for on the grounds that the comparative claims are what the group
count actually limits.

Additive, for the same reason v3 was: ``dataset_hash`` covers the whole corpus
file, so editing an existing corpus invalidates every archived journal and signed
receipt generated against it. The twelve v3 topics are carried across
byte-identical and twelve more are appended, so a v3 run stays reproducible on
its own file and a v4 run is a strict superset.

The six new specialties are again chosen to be independent adjudication domains
rather than more topics inside existing ones, because the protocol holds out whole
groups and correlated groups do not buy the power they appear to.

A provenance note that belongs here rather than in the paper's prose: v4 is six
hand-authored domains and eighteen AI-drafted ones, so the provenance split of
Appendix~\\ref{app:provenance} becomes more load-bearing, not less. The
length-channel audit is run per half there and must be re-run for this corpus
before any v4 result is quoted.

Run:  python3 scripts/gen_pa_corpus_v4.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_v2 = _load("gen_v2", ROOT / "scripts" / "gen_pa_corpus.py")
_v3 = _load("gen_v3", ROOT / "scripts" / "gen_pa_corpus_v3.py")
_v4 = _load("gen_v4_topics", ROOT / "scripts" / "_v4_topics.py")

OUT = ROOT / "src" / "traceguard" / "data" / "synthetic-medical-v4.json"


def build_v4() -> dict:
    """Twenty-four topics: v2's six, v3's six, and twelve more.

    ``build()`` reads the module-level TOPICS and CANARY_DETAILS at call time, so
    extending them before the call is enough and nothing changes on disk.
    """

    _v2.TOPICS.extend(_v3.NEW_TOPICS)
    _v2.CANARY_DETAILS.extend(_v3.NEW_CANARIES)
    _v2.TOPICS.extend(_v4.V4_TOPICS)
    _v2.CANARY_DETAILS.extend(_v4.V4_CANARIES)
    data = _v2.build()
    version = data.get("version")
    if isinstance(version, str):
        data["version"] = version.replace("v2", "v4").replace("2.0.0", "4.0.0")
    data["notes"] = (
        "Twenty-four topics across twelve specialties. Additive extension of "
        "synthetic-medical-v3, itself additive over v2: the first twelve topics are "
        "carried across unchanged so a v2 or v3 run remains reproducible against its "
        "own file. Built to take the number of independent leave-one-group-out units "
        "from twelve to twenty-four. Provenance: six domains hand-authored, eighteen "
        "AI-drafted; the length-channel audit must be run per half before quoting a "
        "v4 result."
    )
    return data


def main() -> int:
    data = build_v4()
    report = _v2._assert_length_band(data)
    text = json.dumps(data, indent=2, ensure_ascii=True) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    topics = report.get("per_topic", [])
    print(f"wrote {OUT} ({len(text)} bytes, sha256 {digest[:16]}...)")
    print(f"  topics: {len(topics)}  (v2 had 6, v3 had 12)")
    specialties = sorted({entry["specialty"] for entry in topics})
    print(f"  specialties ({len(specialties)}): {', '.join(specialties)}")
    print(f"  cases: {len(data.get('cases', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

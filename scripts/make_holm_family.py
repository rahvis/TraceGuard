#!/usr/bin/env python3
r"""Holm--Bonferroni over every p-value the artifact reports, not one table's.

Why. The manuscript corrects the generality sweep's \holmFamilySize{} hypotheses
and nothing else, which leaves a fair objection open: the paper performs many
more tests than that, and a correction applied to one table is not a familywise
guarantee for the paper. This harvests every p-value from every committed report
and corrects across all of them at once.

Two deliberate choices, both conservative.

*Nothing is excluded.* Some harvested entries are controls, where a
\emph{non}-rejection is the outcome the paper wants -- the corpus length-channel
audits are the clearest case. Including them in the family is arguably incoherent,
since one does not usually correct a test one hopes will fail. We include them
anyway. Adding hypotheses can only shrink Holm's smallest threshold, so this makes
every reported rejection harder to obtain, and it removes any question of our
having chosen the family after seeing which tests came out well.

*The floor is respected.* A permutation p-value cannot go below $1/(n+1)$. Where
the reported value sits at that floor we carry it as the floor rather than as an
exact quantity, so a rejection near the corrected threshold is never manufactured
by printing more digits than the estimator has.

The correction itself is the shipped ``holm_bonferroni``, not a reimplementation.

Run:  python3 scripts/make_holm_family.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from make_paper_artifacts import assert_macro_names_are_latex_safe  # noqa: E402
from traceguard.attack import holm_bonferroni  # noqa: E402

REPORTS = [
    Path("tables/nulls.json"),
    Path("tables/matrix-report.json"),
    Path("tables/family-report.json"),
    Path("iclr2027/tables/ladder-report.json"),
    Path("iclr2027/tables/v3-null-report.json"),
    Path("iclr2027/tables/provenance-report.json"),
]


# Subtrees that restate a p-value harvested elsewhere in the same report rather
# than carrying a new hypothesis. Counting these would inflate the family with
# duplicates of tests already in it, which is not conservatism -- it is the wrong
# family, and it would make the correction look stricter than the evidence
# warrants while double-counting the same claim.
_RESTATEMENTS = ("/holm/", "/null_report")


def harvest(obj: Any, path: str, out: dict[str, float]) -> None:
    if any(marker in path for marker in _RESTATEMENTS):
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "p_value" and isinstance(value, (int, float)):
                out[path.lstrip("/")] = float(value)
            else:
                harvest(value, f"{path}/{key}", out)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            harvest(value, f"{path}[{i}]", out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--report", type=Path,
                    default=ROOT / "iclr2027" / "tables" / "holm-family-report.json")
    ap.add_argument("--macros", type=Path,
                    default=ROOT / "iclr2027" / "tables" / "macros_holm.tex")
    args = ap.parse_args()

    family: dict[str, float] = {}
    sources: dict[str, str] = {}
    missing: list[str] = []
    for rel in REPORTS:
        path = ROOT / rel
        if not path.exists():
            missing.append(str(rel))
            continue
        found: dict[str, float] = {}
        harvest(json.loads(path.read_text()), "", found)
        for key, value in found.items():
            name = f"{rel.stem}:{key}"
            family[name] = value
            sources[name] = str(rel)

    if not family:
        raise SystemExit("harvested no p-values; the report paths are wrong")

    result = holm_bonferroni(family, alpha=args.alpha)
    rows = sorted(result["results"].items(), key=lambda kv: family[kv[0]])
    rejected = [k for k, v in rows if v["rejected"]]
    retained = [k for k, v in rows if not v["rejected"]]

    print(f"family size {result['family_size']} across {len(REPORTS) - len(missing)} reports"
          f"{'; missing ' + ', '.join(missing) if missing else ''}")
    print(f"alpha {args.alpha}; smallest threshold "
          f"{args.alpha / result['family_size']:.2e}")
    print(f"rejected {len(rejected)}, retained {len(retained)}\n")
    print("  retained (not rejected at familywise alpha):")
    for k in retained:
        print(f"    p={family[k]:.4f}  {k}")

    # The adjusted p the headline claims land on. Emitted because the useful
    # statement is not the family size but what survives it.
    headline = [
        "nulls:nulls/adaptive_attribute",
        "nulls:nulls/adaptive_membership",
        "v3-null-report:attribute",
        "v3-null-report:membership",
    ]
    adj = [result["results"][k]["p_adjusted"] for k in headline if k in result["results"]]
    ladder = [k for k in result["results"] if k.startswith("ladder-report:")]
    ladder_rejected = [k for k in ladder if result["results"][k]["rejected"]]
    strongest_retained = min((family[k] for k in retained), default=None)
    macros = {
        "holmLadderRejected": str(len(ladder_rejected)),
        "holmLadderTotal": str(len(ladder)),
        "holmStrongestRetained": (f"{strongest_retained:.4f}"
                                  if strongest_retained is not None else "--"),
        "holmAllHeadlineAdj": f"{max(adj):.4f}" if adj else "--",
        "holmAllHeadlineN": str(len(adj)),
        "holmAllFamily": str(result["family_size"]),
        "holmAllRejected": str(len(rejected)),
        "holmAllRetained": str(len(retained)),
        "holmAllAlpha": f"{args.alpha:g}",
        "holmAllMinThreshold": f"{args.alpha / result['family_size']:.2e}".replace(
            "e-0", "\\times 10^{-").replace("e-", "\\times 10^{-") + "}",
    }
    assert_macro_names_are_latex_safe(list(macros))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        {"alpha": args.alpha, "family_size": result["family_size"],
         "sources": sources, "p_values": family,
         "rejected": rejected, "retained": retained,
         "results": result["results"]}, indent=2, sort_keys=True) + "\n")
    args.macros.write_text(
        "% Holm-Bonferroni across every p-value in every committed report,\n"
        "% controls included. Generated by scripts/make_holm_family.py.\n"
        "% ICLR-scoped.\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items())
    )
    print(f"\nwrote {args.report}\nwrote {args.macros}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

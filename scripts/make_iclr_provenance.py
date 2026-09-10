#!/usr/bin/env python3
r"""Split the twelve-group replication by corpus provenance, and audit the split.

Why. Six of the twelve adjudication domains were authored by hand and carried
across from the original corpus; six were drafted with AI assistance to raise the
number of independent leave-one-group-out units. If the generator correlated the
sensitivity gradation with some surface property of the text, an attack reading
byte counts would recover that correlation and we would report it as leakage. The
direction such an artifact would push is the direction the data actually goes on
the attribute, so the split has to be reported rather than pooled, and reported
with a control rather than a reassurance.

Two things are computed here. The per-half attack AUCs, so a reader can see the
asymmetry and its sign on both secrets. And, per half, the length-channel audit
the corpus builder already ships: a nearest-centroid classifier that sees
\emph{only} per-role character counts and tries to recover the rung, scored
against a within-group permutation null. That audit is the one that speaks to the
hypothesis, because a generator artifact of this kind has to show up as
length-recoverable structure. If the AI-drafted half is no more length-recoverable
than the manual half, the attribute asymmetry is not explained by text volume, and
the remaining explanations are about the domains rather than about the generator.

Run:  python3 scripts/make_iclr_provenance.py
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

from gen_pa_corpus import length_channel_recoverability  # noqa: E402
from make_paper_artifacts import (  # noqa: E402
    _records_for,
    assert_macro_names_are_latex_safe,
)
from traceguard.attack import (  # noqa: E402
    evaluate_attack,
    observability_transform,
)

# The three specialties carried across from the hand-authored corpus, and the
# three added by the AI-assisted extension. Named explicitly rather than derived,
# so that adding a specialty forces a decision here instead of silently landing
# in whichever bucket a heuristic picks.
MANUAL = ("cardiology", "oncology", "psychiatry")
DRAFTED = ("endocrinology", "nephrology", "rheumatology")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", type=Path, default=ROOT / "artifacts" / "cvm-v3.jsonl")
    ap.add_argument("--corpus", type=Path,
                    default=ROOT / "src" / "traceguard" / "data" / "synthetic-medical-v3.json")
    ap.add_argument("--condition", default="adaptive")
    ap.add_argument("--seed", type=int, default=20260710)
    ap.add_argument("--permutations", type=int, default=4000)
    ap.add_argument("--report", type=Path,
                    default=ROOT / "iclr2027" / "tables" / "provenance-report.json")
    ap.add_argument("--macros", type=Path,
                    default=ROOT / "iclr2027" / "tables" / "macros_provenance.tex")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.journal.read_text().splitlines() if l.strip()]
    records = _records_for(rows, args.condition)
    if not records:
        raise SystemExit(f"no {args.condition} records in {args.journal}")

    def half(names: tuple[str, ...]) -> list[dict[str, Any]]:
        return [
            r for r in records
            if ((r.get("trace") or {}).get("service") or r.get("service")) in names
        ]

    corpus = json.loads(args.corpus.read_text())

    def corpus_half(names: tuple[str, ...]) -> dict[str, Any]:
        sub = dict(corpus)
        sub["topics"] = [t for t in corpus["topics"] if t.get("specialty") in names]
        return sub

    report: dict[str, Any] = {"journal": str(args.journal), "corpus": str(args.corpus),
                              "condition": args.condition, "seed": args.seed,
                              "manual": list(MANUAL), "drafted": list(DRAFTED),
                              "halves": {}}
    macros: dict[str, str] = {}
    print(f"{'half':10} {'n':>4} {'groups':>7} {'attr':>7} {'memb':>7} "
          f"{'len%':>7} {'null%':>7} {'p':>9}")
    for label, names, tag in (("manual", MANUAL, "Manual"),
                              ("drafted", DRAFTED, "Drafted")):
        sub = half(names)
        groups = {((r.get("trace") or {}).get("service"),
                   (r.get("trace") or {}).get("topic")) for r in sub}
        cell: dict[str, Any] = {"n": len(sub), "n_groups": len(groups),
                                "specialties": list(names)}
        for target, ttag in (("attribute", "Attr"), ("membership", "Memb")):
            rep = evaluate_attack(sub, target=target, n_bootstrap=0, seed=args.seed)
            cell[target] = {"auc": rep["auc"], "pooled": rep["pooled_auc"],
                            "selected": rep["selected_attacker"]}
            macros[f"prov{tag}{ttag}"] = f"{rep['auc']:.3f}"
        # The share of the attribute the defense could remove, per half. This is
        # what decides whether the provenance asymmetry matters: extra leakage on
        # the closeable coordinates is extra leakage the mechanism closes.
        ing = evaluate_attack(
            sub, target="attribute", n_bootstrap=0, seed=args.seed,
            feature_transform=observability_transform("ingress_only"),
        )["auc"]
        cell["attribute_ingress_only"] = ing
        cell["attribute_closeable_share"] = cell["attribute"]["auc"] - ing
        macros[f"prov{tag}AttrIngress"] = f"{ing:.3f}"
        macros[f"prov{tag}AttrCloseable"] = f"{cell['attribute_closeable_share']:.3f}"

        audit = length_channel_recoverability(
            corpus_half(names), n_perm=args.permutations, seed=args.seed)
        cell["length_audit"] = audit
        macros[f"prov{tag}Len"] = f"{audit['recoverability'] * 100:.1f}"
        macros[f"prov{tag}LenNull"] = f"{audit['null_mean'] * 100:.1f}"
        macros[f"prov{tag}LenP"] = f"{audit['p_value']:.3f}"
        macros[f"prov{tag}N"] = str(len(sub))
        report["halves"][label] = cell
        print(f"{label:10} {len(sub):4} {len(groups):7} "
              f"{cell['attribute']['auc']:7.3f} {cell['membership']['auc']:7.3f} "
              f"{audit['recoverability']*100:7.1f} {audit['null_mean']*100:7.1f} "
              f"{audit['p_value']:9.3f}")

    a = report["halves"]
    gap_attr = a["drafted"]["attribute"]["auc"] - a["manual"]["attribute"]["auc"]
    gap_memb = a["drafted"]["membership"]["auc"] - a["manual"]["membership"]["auc"]
    gap_len = (a["drafted"]["length_audit"]["recoverability"]
               - a["manual"]["length_audit"]["recoverability"])
    report["gaps"] = {"attribute": gap_attr, "membership": gap_memb,
                      "length_recoverability": gap_len}
    macros["provAttrGap"] = f"{gap_attr:+.3f}"
    macros["provMembGap"] = f"{gap_memb:+.3f}"
    macros["provLenGap"] = f"{gap_len * 100:+.1f}"
    macros["provPerms"] = str(args.permutations)
    macros["provChance"] = f"{a['manual']['length_audit']['chance'] * 100:.0f}"
    assert_macro_names_are_latex_safe(list(macros))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.macros.write_text(
        "% Twelve-group replication split by corpus provenance, with the\n"
        "% length-channel audit run per half. Generated by\n"
        "% scripts/make_iclr_provenance.py. ICLR-scoped.\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items())
    )
    print(f"\n  attribute gap (drafted - manual): {gap_attr:+.3f}")
    print(f"  membership gap:                   {gap_memb:+.3f}")
    print(f"  length-recoverability gap:        {gap_len*100:+.1f} points")
    print(f"\nwrote {args.report}\nwrote {args.macros}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

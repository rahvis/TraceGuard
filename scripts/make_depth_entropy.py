#!/usr/bin/env python3
r"""Attack AUC against realized depth entropy: is the locus a law or an anecdote?

Why this exists. The manuscript's most quotable claim is that the control-flow
channel closes on its own as agents improve, so the residue migrates to the
coordinate no in-guest mechanism can reach. As shipped, that claim rested on an
observation across three crew deployments with non-commensurable instrumentation:
saturation at 67\% against 95--98\%. An observation across three points is an
anecdote, and the claim deserves better because it is the paper's most
consequential one.

So we make saturation a knob instead. The adaptive stopping rule is
``(hop >= min and sufficient) or hop >= max``, where ``sufficient`` is a live
model decision on retrieved content. On the shipped configuration ``max`` is 7
and \hopSaturatedPct{}\% of runs clip at it, so realized depth is nearly
constant and the depth channel carries almost nothing. Raising ``max`` unbinds
the rule and lets the content-dependent decision express itself; at ``max=13``
realized trace length spans 13 to 22 steps where at 7 it was uniformly 12. Depth
entropy therefore becomes the independent variable of a sweep, with everything
else -- corpus, crew, provider, seed, instrumentation, and all four observable
coordinates -- held fixed.

The prediction the taxonomy makes is specific and falsifiable. Membership is
carried by control flow, so its recoverability should rise with depth entropy.
The attribute is carried by request and response size, so it should stay roughly
flat. If both rise together the taxonomy is wrong and the paper should say so;
if membership rises while the attribute does not, the claim becomes a measured
relationship with a mechanism rather than a story about three crews.

Entropy is the Shannon entropy of the realized extraction-pass distribution
within an arm, in bits, which is the natural scale for "how much does depth vary"
and is comparable across arms with different caps.

Run:  python3 scripts/make_depth_entropy.py --journal-dir artifacts/depth
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from make_paper_artifacts import (  # noqa: E402
    _records_for,
    assert_macro_names_are_latex_safe,
)
from traceguard.attack import (  # noqa: E402
    evaluate_attack,
    observability_transform,
    permutation_null,
)


def shannon_bits(counts: collections.Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h


def depth_of(record: dict[str, Any]) -> int:
    """Realized extraction passes, taken from the trace rather than the config.

    The config records the cap that was permitted; only the trace records what the
    agent actually did, and the difference between those two is the whole subject
    of this sweep.
    """
    trace = record.get("trace") or {}
    steps = trace.get("steps") or []
    n = sum(1 for s in steps if (s.get("step_type") or s.get("name") or "")
            == "clinical_extraction")
    if n:
        return n
    if isinstance(trace.get("research_hops"), int):
        return int(trace["research_hops"])
    return len(steps)


def censor_to_budget(record: dict[str, Any], budget: int) -> dict[str, Any]:
    """Truncate a run to its first ``budget`` extraction passes.

    Equalising the number of observations across arms is what separates a change
    in the channel from a change in how many times the attacker gets to look at
    it. Arms with a larger cap take more passes, and each pass is one more
    boundary crossing the host observes, so an attacker aggregating per-run
    features has strictly more samples in the deeper arms whether or not the
    per-observation leakage changed.
    """

    import copy

    out = copy.deepcopy(record)
    steps = (out.get("trace") or {}).get("steps") or []
    kept: list[Any] = []
    seen = 0
    for step in steps:
        kept.append(step)
        if (step.get("step_type") or "") == "clinical_extraction":
            seen += 1
            if seen >= budget:
                break
    out["trace"]["steps"] = kept
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal-dir", type=Path, default=ROOT / "artifacts" / "depth")
    ap.add_argument("--glob", default="depth-h*.jsonl")
    ap.add_argument("--condition", default="adaptive")
    ap.add_argument("--seed", type=int, default=20260710)
    ap.add_argument("--permutations", type=int, default=2000)
    ap.add_argument("--report", type=Path,
                    default=ROOT / "iclr2027" / "tables" / "depth-entropy-report.json")
    ap.add_argument("--macros", type=Path,
                    default=ROOT / "iclr2027" / "tables" / "macros_depth.tex")
    ap.add_argument("--prefix", default="depth",
                    help="macro-name prefix, so the announced and prompt-invariant "
                         "sweeps can be emitted side by side")
    ap.add_argument("--table", type=Path,
                    default=ROOT / "iclr2027" / "tables" / "tab_depth.tex")
    ap.add_argument("--censor-budget", type=int, default=7,
                    help="equalise every arm to this many extraction passes and "
                         "re-score, to separate a channel change from an "
                         "observation-count change")
    args = ap.parse_args()

    paths = sorted(args.journal_dir.glob(args.glob),
                   key=lambda p: int(re.search(r"h(\d+)", p.name).group(1)))
    if not paths:
        raise SystemExit(f"no journals matching {args.glob} in {args.journal_dir}")

    arms: list[dict[str, Any]] = []
    print(f"{'cap':>4} {'n':>4} {'depths':>22} {'H(bits)':>8} "
          f"{'attr':>7} {'a-null':>7} {'memb':>7} {'m-null':>7} {'memb|struct':>12}")
    for path in paths:
        cap = int(re.search(r"h(\d+)", path.name).group(1))
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        records = _records_for(rows, args.condition)
        if len(records) < 40:
            print(f"{cap:>4}  only {len(records)} records; skipped")
            continue
        depths = collections.Counter(depth_of(r) for r in records)
        h = shannon_bits(depths)

        cell: dict[str, Any] = {
            "cap": cap, "n": len(records), "journal": str(path),
            "depth_histogram": {str(k): v for k, v in sorted(depths.items())},
            "depth_entropy_bits": h,
            "depth_modal_share": max(depths.values()) / len(records),
        }
        for target in ("attribute", "membership"):
            rep = evaluate_attack(records, target=target, n_bootstrap=0, seed=args.seed)
            null = permutation_null(records, target=target, n_perm=args.permutations,
                                    within_group=True, seed=args.seed)
            cell[target] = {
                "auc": rep["auc"], "null_mean": null["null_mean"],
                "signal": rep["auc"] - null["null_mean"],
                "p_value": null["p_value"], "selected": rep["selected_attacker"],
            }
        # The decisive test is per coordinate, not on the full vantage. Raising the
        # cap lengthens the trace, so the pooled classifier gets more of
        # everything and BOTH secrets rise on the full vantage; reading only that
        # column would say the taxonomy failed. The taxonomy's claim is about
        # which coordinate carries which secret, so we score every coordinate
        # alone and let the named carriers answer it.
        cell["by_vantage"] = {}
        for target in ("attribute", "membership"):
            per: dict[str, float | None] = {}
            for vantage in ("structure_only", "timing_only", "ingress_only", "size_only"):
                try:
                    per[vantage] = evaluate_attack(
                        records, target=target, n_bootstrap=0, seed=args.seed,
                        feature_transform=observability_transform(vantage))["auc"]
                except ValueError:
                    per[vantage] = None
            cell["by_vantage"][target] = per
        # Named carriers, per the taxonomy: membership on timing, attribute on the
        # provider-chosen response size.
        cell["membership_structure_only"] = cell["by_vantage"]["membership"]["structure_only"]
        cell["membership_timing_only"] = cell["by_vantage"]["membership"]["timing_only"]
        cell["attribute_ingress_only"] = cell["by_vantage"]["attribute"]["ingress_only"]
        arms.append(cell)
        hist = ",".join(f"{k}:{v}" for k, v in sorted(depths.items()))
        print(f"{cap:>4} {len(records):>4} {hist[:22]:>22} {h:>8.3f} "
              f"{cell['attribute']['auc']:>7.3f} {cell['attribute']['null_mean']:>7.3f} "
              f"{cell['membership']['auc']:>7.3f} {cell['membership']['null_mean']:>7.3f} "
              f"{str(cell['membership_structure_only'])[:12]:>12}")

    if len(arms) < 2:
        raise SystemExit("need at least two arms to report a relationship")

    # Censored view: every arm equalised to the same observation budget.
    for cell in arms:
        rows = [json.loads(l) for l in Path(cell["journal"]).read_text().splitlines()
                if l.strip()]
        records = [censor_to_budget(r, args.censor_budget)
                   for r in _records_for(rows, args.condition)]
        depths = collections.Counter(depth_of(r) for r in records)
        cell["censored"] = {
            "budget": args.censor_budget,
            "depth_entropy_bits": shannon_bits(depths),
            "steps_per_run": sum(len((r.get("trace") or {}).get("steps") or [])
                                 for r in records) / len(records),
        }
        for target, vantage, key in (("membership", "timing_only", "membership_timing_only"),
                                     ("attribute", "ingress_only", "attribute_ingress_only")):
            try:
                cell["censored"][key] = evaluate_attack(
                    records, target=target, n_bootstrap=0, seed=args.seed,
                    feature_transform=observability_transform(vantage))["auc"]
            except ValueError:
                cell["censored"][key] = None

    # Slope of signal-above-null against entropy, by least squares. Reported with
    # the point count because four points is a line through few data, and we would
    # rather show the arms than lean on the fit.
    def slope(ys: list[float]) -> float:
        xs = [a["depth_entropy_bits"] for a in arms]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        den = sum((x - mx) ** 2 for x in xs)
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0

    memb_slope = slope([a["membership"]["signal"] for a in arms])
    attr_slope = slope([a["attribute"]["signal"] for a in arms])
    memb_timing_slope = slope([a["membership_timing_only"] for a in arms])
    attr_ingress_slope = slope([a["attribute_ingress_only"] for a in arms])
    report = {"condition": args.condition, "seed": args.seed,
              "permutations": args.permutations, "arms": arms,
              "membership_signal_slope_per_bit": memb_slope,
              "attribute_signal_slope_per_bit": attr_slope,
              "membership_timing_slope_per_bit": memb_timing_slope,
              "attribute_ingress_slope_per_bit": attr_ingress_slope}

    # Entropy is NOT monotone in the cap -- the middle arm came out highest -- so
    # order by the independent variable, and expose the peak explicitly. Quoting
    # the two cap-extremes alone would have hidden the middle arm and made a
    # non-monotone carrier look flat.
    by_ent = sorted(arms, key=lambda a: a["depth_entropy_bits"])
    lo, hi = by_ent[0], by_ent[-1]
    peak_attr_ing = max(arms, key=lambda a: a["attribute_ingress_only"])
    peak_memb_tim = max(arms, key=lambda a: a["membership_timing_only"])
    macros = {
        "depthArms": str(len(arms)),
        "depthCapLo": str(lo["cap"]), "depthCapHi": str(hi["cap"]),
        "depthEntLo": f"{lo['depth_entropy_bits']:.2f}",
        "depthEntHi": f"{hi['depth_entropy_bits']:.2f}",
        "depthModalLo": f"{lo['depth_modal_share'] * 100:.0f}",
        "depthModalHi": f"{hi['depth_modal_share'] * 100:.0f}",
        "depthMembLo": f"{lo['membership']['auc']:.3f}",
        "depthMembHi": f"{hi['membership']['auc']:.3f}",
        "depthAttrLo": f"{lo['attribute']['auc']:.3f}",
        "depthAttrHi": f"{hi['attribute']['auc']:.3f}",
        "depthMembSigLo": f"{lo['membership']['signal']:+.3f}",
        "depthMembSigHi": f"{hi['membership']['signal']:+.3f}",
        "depthAttrSigLo": f"{lo['attribute']['signal']:+.3f}",
        "depthAttrSigHi": f"{hi['attribute']['signal']:+.3f}",
        "depthMembSlope": f"{memb_slope:+.3f}",
        "depthAttrSlope": f"{attr_slope:+.3f}",
        "depthMembTimingLo": f"{lo['membership_timing_only']:.3f}",
        "depthMembTimingHi": f"{hi['membership_timing_only']:.3f}",
        "depthAttrIngressLo": f"{lo['attribute_ingress_only']:.3f}",
        "depthAttrIngressHi": f"{hi['attribute_ingress_only']:.3f}",
        "depthMembStructLo": f"{lo['membership_structure_only']:.3f}",
        "depthMembStructHi": f"{hi['membership_structure_only']:.3f}",
        "depthMembTimingSlope": f"{memb_timing_slope:+.3f}",
        "depthAttrIngressSlope": f"{attr_ingress_slope:+.3f}",
        "depthEntPeak": f"{hi['depth_entropy_bits']:.2f}",
        "depthCapPeak": str(hi["cap"]),
        "depthAttrIngressPeak": f"{peak_attr_ing['attribute_ingress_only']:.3f}",
        "depthAttrIngressPeakCap": str(peak_attr_ing["cap"]),
        "depthMembTimingPeak": f"{peak_memb_tim['membership_timing_only']:.3f}",
        "depthMembTimingPeakCap": str(peak_memb_tim["cap"]),
        "depthCensorBudget": str(args.censor_budget),
        "depthCensorEntLo": f"{min(a['censored']['depth_entropy_bits'] for a in arms):.2f}",
        "depthCensorEntHi": f"{max(a['censored']['depth_entropy_bits'] for a in arms):.2f}",
        "depthMembTimingSpread": f"{max(a['membership_timing_only'] for a in arms) - min(a['membership_timing_only'] for a in arms):+.3f}",
        "depthAttrIngressSpread": f"{max(a['attribute_ingress_only'] for a in arms) - min(a['attribute_ingress_only'] for a in arms):+.3f}",
        "depthCensorMembSpread": f"{max(a['censored']['membership_timing_only'] for a in arms) - min(a['censored']['membership_timing_only'] for a in arms):+.3f}",
        "depthCensorAttrSpread": f"{max(a['censored']['attribute_ingress_only'] for a in arms) - min(a['censored']['attribute_ingress_only'] for a in arms):+.3f}",
        "depthPerms": str(args.permutations),
        "depthNPerArm": str(lo["n"]),
    }
    if args.prefix != "depth":
        macros = {args.prefix + k[5:] if k.startswith("depth") else args.prefix + k: v
                  for k, v in macros.items()}
    assert_macro_names_are_latex_safe(list(macros))

    rows_tex = []
    for a in arms:
        ms = a["membership_structure_only"]
        mt, ai = a["membership_timing_only"], a["attribute_ingress_only"]
        rows_tex.append(
            "%d & %d & %.2f & %.0f\\%% & %.3f & %.3f & %.3f & %.3f \\\\"
            % (a["cap"], a["n"], a["depth_entropy_bits"], a["depth_modal_share"] * 100,
               a["membership"]["auc"], mt if isinstance(mt, float) else float("nan"),
               a["attribute"]["auc"], ai if isinstance(ai, float) else float("nan")))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.macros.write_text(
        "% Depth-entropy sweep: attack AUC against realized depth entropy, with\n"
        "% the pass cap as the knob. Generated by scripts/make_depth_entropy.py.\n"
        "% ICLR-scoped.\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
    args.table.write_text(
        "% Generated by scripts/make_depth_entropy.py.\n"
        "{\\setlength{\\tabcolsep}{4pt}\\renewcommand{\\arraystretch}{1.15}\n"
        "\\begin{tabular}{@{}rrrrcccc@{}}\n\\toprule\n"
        "cap & $n$ & $H$ (bits) & modal & Memb.\\ all & Memb.\\ timing "
        "& Attr.\\ all & Attr.\\ ingress \\\\\n\\midrule\n"
        + "\n".join(rows_tex) + "\n\\bottomrule\n\\end{tabular}}\n")
    print(f"\n  membership signal slope: {memb_slope:+.3f} per bit")
    print(f"  attribute  signal slope: {attr_slope:+.3f} per bit")
    print(f"\nwrote {args.report}\nwrote {args.macros}\nwrote {args.table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

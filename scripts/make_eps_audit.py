#!/usr/bin/env python3
r"""Empirical $\eps$ lower bounds for trace-privacy, from the attack's own ROC.

Why. The paper defines $(\eps,\delta)$-trace-privacy and then reports every
measurement as an AUC against a calibrated null, which is a proxy for a privacy
statement rather than one. The hypothesis-testing characterisation of
differential privacy closes that gap without a single new experiment: an
adversary distinguishing a neighbour pair with true-positive rate $T$ and
false-positive rate $F$ certifies

    $\eps \ge \max\{\log((T-\delta)/F),\ \log((1-F-\delta)/(1-T))\}$,

because an $(\eps,\delta)$-private mechanism admits no such test. This is the
DP-auditing construction (Jagielski et al., 2020; Nasr et al., 2021, 2023), and
it turns the ROC curves the paper already has into lower bounds on the budget
the release actually spends.

Three choices keep the bound honest rather than flattering.

*It is a high-confidence bound, not a point estimate.* Plugging in empirical
rates and maximising over thresholds is optimistically biased twice over. We
take one-sided Clopper--Pearson bounds on each rate -- a lower bound on $T$, an
upper bound on $F$ -- and Bonferroni-correct the confidence level over every
candidate threshold and both forms, so the reported $\eps$ holds jointly.

*The scores are out-of-fold.* Thresholds are selected on leave-one-group-out
predictions from the manuscript's own estimator, reached through
``_evaluate_features``, so no threshold is chosen on data the attacker fitted.

*The direction of every remaining approximation is conservative.* The attacker
is a deliberately weak learner, and the neighbour pairs are the ones this corpus
instantiates rather than the worst case the definition quantifies over. A
stronger attacker or an adversarial pair can only raise $\eps$, so what we report
is a floor on the floor. It is emphatically not an upper bound and must not be
read as one: nothing here certifies that any coordinate is private.

The interesting cell is the padded arm restricted to the ingress vantage. The
closed coordinates carry a proved $(0,0)$, so an audit of them should return no
evidence, and it does -- which is the sanity check that this estimator is not
manufacturing signal. The provider-chosen coordinate is where the paper has only
ever had an AUC, and where a budget in the notion's own units is what a reader
asked the definition for.

Run:  python3 scripts/make_eps_audit.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta as _beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from make_paper_artifacts import (  # noqa: E402
    _records_for,
    assert_macro_names_are_latex_safe,
)
from traceguard import attack as A  # noqa: E402


def cp_lower(k: int, n: int, alpha: float) -> float:
    """One-sided Clopper--Pearson lower confidence bound on a binomial rate."""
    if n == 0:
        return 0.0
    if k == 0:
        return 0.0
    return float(_beta.ppf(alpha, k, n - k + 1))


def cp_upper(k: int, n: int, alpha: float) -> float:
    """One-sided Clopper--Pearson upper confidence bound on a binomial rate."""
    if n == 0:
        return 1.0
    if k == n:
        return 1.0
    return float(_beta.ppf(1 - alpha, k + 1, n - k))


def out_of_fold_scores(records: list[dict[str, Any]], target: str, seed: int,
                       vantage: str | None) -> tuple[np.ndarray, np.ndarray, str]:
    """Leave-one-group-out scores from the manuscript's estimator, not a copy."""
    labels = [A._label(r, target) for r in records]
    groups = [A._group(r) for r in records]
    a1 = [A._record_features(r, False) for r in records]
    a2 = [A._record_features(r, True) for r in records]
    if vantage:
        transform = A.observability_transform(vantage)
        a1, a2 = transform(a1), transform(a2)
    ev = A._evaluate_features(a1, a2, labels, groups, seed=seed)
    return np.asarray(ev["labels"], dtype=int), np.asarray(ev["scores"], dtype=float), \
        str(ev["selected_attacker"])


def eps_at(y: np.ndarray, s: np.ndarray, *, delta: float, alpha: float | None,
           n_claims: int) -> dict[str, Any]:
    """One accounting of the bound. ``alpha=None`` gives the raw point estimate."""
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return {"eps": 0.0, "reason": "one class absent"}
    per = None if alpha is None else alpha / max(1, n_claims)
    best = {"eps": 0.0, "threshold": None, "form": None, "orientation": None}
    for orientation in (1.0, -1.0):
        v = orientation * s
        for thr in np.unique(v):
            pred = v >= thr
            tp = int(np.sum(pred & (y == 1)))
            fp = int(np.sum(pred & (y == 0)))
            if per is None:
                tpr, fpr = tp / n_pos, fp / n_neg
            else:
                tpr, fpr = cp_lower(tp, n_pos, per), cp_upper(fp, n_neg, per)
            cands = []
            if fpr > 0 and tpr - delta > 0:
                cands.append(("ratio", math.log((tpr - delta) / fpr)))
            if tpr < 1 and (1 - fpr - delta) > 0:
                cands.append(("complement", math.log((1 - fpr - delta) / (1 - tpr))))
            for form, value in cands:
                if value > best["eps"]:
                    best = {"eps": value, "threshold": float(thr), "form": form,
                            "orientation": "ascending" if orientation > 0 else "descending",
                            "tp": tp, "fp": fp, "rate_tpr": tpr, "rate_fpr": fpr}
    best.update({"n_pos": n_pos, "n_neg": n_neg, "alpha_total": alpha,
                 "alpha_per_claim": per, "n_claims": n_claims})
    return best


def eps_lower_bound(y: np.ndarray, s: np.ndarray, *, delta: float,
                    confidence: float) -> dict[str, Any]:
    """Bonferroni-corrected, Clopper--Pearson $\\eps$ lower bound over thresholds.

    Both orientations of the score are tried, because the manuscript's AUC is
    inversion-safe: a systematically inverted classifier is a successful
    distinguisher and the audit has to grant it the same standing.
    """
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return {"eps_lower": 0.0, "reason": "one class absent"}
    thresholds = np.unique(s)
    # Two orientations x two inequality forms x |thresholds| simultaneous claims.
    alpha = (1.0 - confidence) / max(1, 2 * 2 * thresholds.size)
    best = {"eps_lower": 0.0, "threshold": None, "form": None, "orientation": None,
            "tpr": None, "fpr": None, "tpr_lo": None, "fpr_hi": None}
    for orientation in (1.0, -1.0):
        v = orientation * s
        for thr in np.unique(v):
            pred = v >= thr
            tp = int(np.sum(pred & (y == 1)))
            fp = int(np.sum(pred & (y == 0)))
            tpr_lo = cp_lower(tp, n_pos, alpha)
            fpr_hi = cp_upper(fp, n_neg, alpha)
            cands = []
            if fpr_hi > 0 and tpr_lo - delta > 0:
                cands.append(("ratio", math.log((tpr_lo - delta) / fpr_hi)))
            if tpr_lo < 1 and (1 - fpr_hi - delta) > 0:
                cands.append(("complement", math.log((1 - fpr_hi - delta) / (1 - tpr_lo))))
            for form, value in cands:
                if value > best["eps_lower"]:
                    best = {
                        "eps_lower": value, "threshold": float(thr), "form": form,
                        "orientation": ("ascending" if orientation > 0 else "descending"),
                        "tpr": tp / n_pos, "fpr": fp / n_neg,
                        "tpr_lo": tpr_lo, "fpr_hi": fpr_hi,
                    }
    best.update({"n_pos": n_pos, "n_neg": n_neg, "delta": delta,
                 "confidence": confidence, "n_thresholds": int(thresholds.size),
                 "alpha_per_claim": alpha})
    return best


CELLS = [
    # (key, journal-role, arm, vantage, short table label, description)
    ("undefTwelve", "extended", "adaptive", None,
     "undefended, all coordinates (12 groups)",
     "undefended release, full host vantage, twelve-group corpus"),
    ("undefSix", "headline", "adaptive", None,
     "undefended, all coordinates (6 groups)",
     "undefended release, full host vantage, six-group corpus"),
    ("padIngressSix", "headline", "full", "enforced_remote",
     "padded, response volume only (6 groups)",
     "padded release, restricted to the coordinate no in-guest mechanism can "
     "hold constant"),
    ("padAllSix", "headline", "full", None,
     "padded, all coordinates (6 groups)",
     "padded release, full host vantage"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headline-journal", type=Path,
                    default=ROOT / "artifacts" / "cvm-honest.jsonl")
    ap.add_argument("--extended-journal", type=Path,
                    default=ROOT / "artifacts" / "cvm-v3.jsonl")
    ap.add_argument("--report", type=Path,
                    default=ROOT / "iclr2027" / "tables" / "eps-audit-report.json")
    ap.add_argument("--macros", type=Path,
                    default=ROOT / "iclr2027" / "tables" / "macros_eps.tex")
    ap.add_argument("--table", type=Path,
                    default=ROOT / "iclr2027" / "tables" / "tab_eps.tex")
    ap.add_argument("--seed", type=int, default=20260710)
    ap.add_argument("--delta", type=float, default=0.0)
    ap.add_argument("--confidence", type=float, default=0.95)
    args = ap.parse_args()

    journals = {}
    for role, path in (("headline", args.headline_journal),
                       ("extended", args.extended_journal)):
        journals[role] = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    report: dict[str, Any] = {
        "seed": args.seed, "delta": args.delta, "confidence": args.confidence,
        "headline_journal": str(args.headline_journal),
        "extended_journal": str(args.extended_journal),
        "estimator": ("hypothesis-testing eps lower bound, maximised over out-of-fold "
                      "thresholds and both score orientations; reported at three "
                      "multiplicity accountings so the cost of threshold selection "
                      "is visible"),
        "cells": {},
    }
    macros: dict[str, str] = {}
    table_rows: list[str] = []
    print(f"{'cell':<15} {'target':<11} {'n+':>4} {'n-':>4} "
          f"{'point':>7} {'CP':>7} {'CPjoint':>8}")
    for key, role, arm, vantage, short, desc in CELLS:
        records = _records_for(journals[role], arm)
        if not records:
            print(f"  {key}: no {arm} records in the {role} journal; skipped")
            continue
        for target, tag in (("attribute", "Attr"), ("membership", "Memb")):
            y, s_, selected = out_of_fold_scores(records, target, args.seed, vantage)
            n_thr = int(np.unique(s_).size)
            claims = 2 * 2 * n_thr  # two orientations, two forms, every threshold
            point = eps_at(y, s_, delta=args.delta, alpha=None, n_claims=1)
            single = eps_at(y, s_, delta=args.delta,
                            alpha=1 - args.confidence, n_claims=1)
            joint = eps_at(y, s_, delta=args.delta,
                           alpha=1 - args.confidence, n_claims=claims)
            cell = {"arm": arm, "vantage": vantage, "description": desc,
                    "journal_role": role, "selected_attacker": selected,
                    "target": target, "n_thresholds": n_thr,
                    "point_estimate": point, "confident_single_threshold": single,
                    "confident_joint": joint}
            report["cells"][f"{key}:{target}"] = cell
            macros[f"eps{key}{tag}"] = f"{joint['eps']:.2f}"
            macros[f"eps{key}{tag}Point"] = f"{point['eps']:.2f}"
            print(f"{key:<15} {target:<11} {joint.get('n_pos',0):>4} "
                  f"{joint.get('n_neg',0):>4} {point['eps']:>7.3f} "
                  f"{single['eps']:>7.3f} {joint['eps']:>8.3f}")
            table_rows.append(
                "%s & %s & %d/%d & %.2f & %.2f & %.2f \\\\"
                % (short, target, joint.get("n_pos", 0), joint.get("n_neg", 0),
                   point["eps"], single["eps"], joint["eps"])
            )
    macros["epsConf"] = f"{int(round(args.confidence * 100))}"
    macros["epsDelta"] = f"{args.delta:g}"
    assert_macro_names_are_latex_safe(list(macros))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.macros.write_text(
        "% Empirical eps lower bounds via the hypothesis-testing characterisation\n"
        "% of DP, generated by scripts/make_eps_audit.py from the committed journals.\n"
        "% One-sided Clopper-Pearson; the headline figure is corrected jointly over\n"
        "% every threshold and both score orientations. These are LOWER bounds and\n"
        "% certify no coordinate as private. ICLR-scoped.\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items())
    )
    args.table.write_text(
        "% Generated by scripts/make_eps_audit.py. Lower bounds only.\n"
        "{\\setlength{\\tabcolsep}{5pt}\\renewcommand{\\arraystretch}{1.15}\n"
        "\\begin{tabular}{@{}llcccc@{}}\n\\toprule\n"
        "Release and vantage & Secret & $n_+/n_-$ & point & one thr. & joint \\\\\n"
        "\\midrule\n"
        + "\n".join(table_rows) + "\n"
        "\\midrule\n"
        "\\multicolumn{6}{@{}p{0.95\\linewidth}@{}}{\\footnotesize Every entry is a "
        "\\emph{lower} bound on $\\eps$ at $\\delta=\\epsDelta{}$ and certifies "
        "nothing as private. \\textbf{point} maximises the empirical rates over "
        "thresholds and corrects for nothing; \\textbf{one thr.} applies a "
        "\\epsConf{}\\% one-sided Clopper--Pearson bound to each rate at a single "
        "threshold; \\textbf{joint} corrects that confidence over every threshold and "
        "both score orientations, which is the only column that supports a claim. The "
        "distance between the first and last columns is the price of choosing a "
        "threshold post hoc.}\\\\\n"
        "\\bottomrule\n\\end{tabular}}\n"
    )
    print(f"\nwrote {args.report}\nwrote {args.macros}\nwrote {args.table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

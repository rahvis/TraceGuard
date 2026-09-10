#!/usr/bin/env python3
r"""Paired power analysis for the twelve-group v3 replication.

Why this script exists. The manuscript's binding limitation was never the number
of runs; it was the number of *independent units*. The leave-one-(specialty,
topic)-out protocol holds out whole groups, so the effective n for the
undefended-versus-padded contrast is the group count. At six groups the minimum
detectable effect is \attrMDE = 0.174 against an observed attribute signal of
0.185: the design could barely have detected the whole effect, so "padding does
not reduce the attribute leak" was a failure to detect and the paper says so.
The v3 corpus doubles the groups to twelve, and this script recomputes the
paired contrast and the minimum detectable effect on that corpus.

What it does NOT do. It runs no experiment and touches no journal. It reads the
committed v3 journal and re-uses two pieces of the shipped analysis verbatim:
``_records_for`` for the row projection, so exactly the fields the manuscript's
attack consumes reach this one, and ``paired_group_difference`` for the folds,
so the per-group AUCs are the published estimator's and not a reimplementation.
That second point is not incidental: an earlier attacker reimplementation here
diverged from the published fitter and produced 0.825 where the manuscript
reports 0.784, and it was caught only because it failed to reproduce a number
the paper already had. Delegating removes the whole failure mode.

The minimum detectable effect is the same statistic the six-group figure uses,
  MDE = (t_{0.975, n-1} + t_{0.80, n-1}) * sd(paired differences) / sqrt(n),
the smallest true paired difference this design would reject at two-sided 0.05
with 80% power. The only change is that the t quantiles are taken at the actual
degrees of freedom rather than hard-coded, because n is no longer six.

Run:  python3 scripts/make_v3_power.py --journal artifacts/cvm-v3.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
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
from traceguard.attack import evaluate_attack, paired_group_difference  # noqa: E402

try:
    from scipy.stats import t as _t  # type: ignore

    def _tq(p: float, df: int) -> float:
        return float(_t.ppf(p, df))

except ImportError:  # pragma: no cover - scipy is a hard dep of the analysis env
    # A short table keyed by df, so the script still runs where scipy is absent
    # rather than silently falling back to a normal quantile (which would
    # understate the interval at small df and overstate the power).
    _T975 = {5: 2.5706, 11: 2.2010}
    _T80 = {5: 0.9195, 11: 0.8755}

    def _tq(p: float, df: int) -> float:
        table = _T975 if abs(p - 0.975) < 1e-9 else _T80
        if df not in table:
            raise SystemExit(f"no t quantile for p={p} df={df}; install scipy")
        return table[df]


def assert_padded_arm_matches_ceiling(rows: list[dict[str, Any]], expected: int) -> None:
    r"""Refuse a padded arm that was not run at the manuscript's egress ceiling.

    This guard exists because the absence of it cost a full 384-run arm. The
    deployed image carries ``TRACEGUARD_STEP_EGRESS_BYTES=4096`` in its
    environment and ``Settings.from_env`` falls back to 4096 as well, while the
    manuscript reports \egressCeiling = 8192 and the headline arm was recorded
    at 8192. A padded arm inherits whatever the container was given, and nothing
    in the journal announces the mismatch.

    It is not a cosmetic difference. A tighter ceiling refuses more steps
    outright, and a refused step gets no response, which zeroes the ingress
    coordinate the attribute leak actually rides on. Measured: 6.6% of steps at
    4096 against 2.6% at 8192. The paired attribute difference came out at
    -0.226 with an interval excluding zero -- an apparent reduction under
    padding, which would have contradicted this paper's own impossibility
    result while being nothing but a misconfigured ceiling. Comparing arms
    provisioned differently is not a comparison, so this refuses instead.
    """

    seen: set[int] = set()
    for row in rows:
        if row.get("condition") != "full" or row.get("status") not in (None, "completed"):
            continue
        for step in (row.get("trace") or {}).get("steps") or []:
            value = step.get("egress_bytes")
            if isinstance(value, int):
                seen.add(value)
    if not seen:
        raise SystemExit("the journal holds no padded-arm steps; nothing to compare")
    if seen != {expected}:
        raise SystemExit(
            "the padded arm was not run at the manuscript's egress ceiling: "
            f"expected every step at {expected}, found {sorted(seen)}. A tighter "
            "ceiling refuses steps and zeroes the ingress coordinate, so the "
            "paired difference would measure the provisioning and not the "
            "mechanism. Re-run the padded arm with "
            "TRACEGUARD_EGRESS_CEILING_BYTES set, or pass --expect-ceiling to "
            "analyse a deliberately different setting."
        )


def load(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def analyse(rows: list[dict[str, Any]], target: str, seed: int,
            n_bootstrap: int) -> dict[str, Any]:
    adaptive = _records_for(rows, "adaptive")
    full = _records_for(rows, "full")
    if not adaptive or not full:
        raise SystemExit(
            f"need both arms; got adaptive={len(adaptive)} full={len(full)}"
        )
    rep = paired_group_difference(
        adaptive, full, target=target, n_bootstrap=n_bootstrap, seed=seed
    )
    diffs = [rep["per_group_difference"][g] for g in rep["groups"]]
    n = len(diffs)
    if n < 3:
        raise SystemExit(f"{target}: only {n} shared groups; no paired estimate")
    sd = statistics.stdev(diffs)
    mean = statistics.mean(diffs)
    df = n - 1
    t975, t80 = _tq(0.975, df), _tq(0.80, df)
    half = t975 * sd / math.sqrt(n)
    rep.update(
        {
            "n_adaptive": len(adaptive),
            "n_full": len(full),
            "sd_difference": sd,
            "df": df,
            "t975": t975,
            "t80": t80,
            "ci95_t": [mean - half, mean + half],
            "mde": (t975 + t80) * sd / math.sqrt(n),
            "mde_formula": "(t_{0.975,n-1} + t_{0.80,n-1}) * sd / sqrt(n)",
        }
    )
    return rep


def carriage(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    r"""Where each secret is carried, per arm, per corpus.

    This exists because the paired result is corpus-dependent in a way that
    needs explaining rather than reporting. Padding closes structure, timing and
    egress and cannot touch ingress, so the benefit it buys is bounded by how
    much of the secret was on the closeable coordinates in the first place --
    and that split is a property of the workload, not of the mechanism. The
    decomposition makes the dependence visible instead of leaving two corpora
    disagreeing.
    """

    from traceguard.attack import OBSERVABILITY, observability_transform

    out: dict[str, Any] = {}
    for cond in ("adaptive", "full"):
        records = _records_for(rows, cond)
        if not records:
            continue
        for target in ("attribute", "membership"):
            cell: dict[str, float] = {}
            for vantage in ("full", "structure_only", "timing_only", "ingress_only"):
                if vantage not in OBSERVABILITY:
                    continue
                try:
                    rep = evaluate_attack(
                        records, target=target, n_bootstrap=0, seed=seed,
                        feature_transform=observability_transform(vantage),
                    )
                except ValueError:
                    continue
                cell[vantage] = rep["auc"]
            out[f"{cond}:{target}"] = cell
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", type=Path, default=ROOT / "artifacts" / "cvm-v3.jsonl")
    ap.add_argument("--report", type=Path,
                    default=ROOT / "iclr2027" / "tables" / "v3-power-report.json")
    ap.add_argument("--macros", type=Path,
                    default=ROOT / "iclr2027" / "tables" / "macros_power.tex")
    ap.add_argument("--prefix", default="pow",
                    help="macro-name prefix, so a second corpus can be emitted "
                         "alongside the first without colliding")
    ap.add_argument("--seed", type=int, default=20260710)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--baseline-journal", type=Path,
                    default=ROOT / "artifacts" / "cvm-honest.jsonl",
                    help="the six-group headline journal, for the corpus contrast")
    ap.add_argument("--expect-ceiling", type=int, default=8192,
                    help="egress ceiling every padded step must carry (default: "
                         "8192, the value the manuscript reports)")
    args = ap.parse_args()

    rows = load(args.journal)
    assert_padded_arm_matches_ceiling(rows, args.expect_ceiling)
    baseline_rows = load(args.baseline_journal) if args.baseline_journal.exists() else []
    report: dict[str, Any] = {"journal": str(args.journal), "seed": args.seed}
    for target in ("attribute", "membership"):
        report[target] = analyse(rows, target, args.seed, args.bootstrap)

    report["carriage"] = carriage(rows, args.seed)
    if baseline_rows:
        report["carriage_baseline"] = carriage(baseline_rows, args.seed)

    a, m = report["attribute"], report["membership"]
    car, carb = report["carriage"], report.get("carriage_baseline", {})

    def _c(store, key, vantage):
        v = (store.get(key) or {}).get(vantage)
        return f"{v:.3f}" if isinstance(v, (int, float)) else "--"

    macros = {
        "powAdaptAttr": _c(car, "adaptive:attribute", "full"),
        "powFullAttr": _c(car, "full:attribute", "full"),
        "powAdaptAttrIngress": _c(car, "adaptive:attribute", "ingress_only"),
        "powFullAttrIngress": _c(car, "full:attribute", "ingress_only"),
        "powAdaptMemb": _c(car, "adaptive:membership", "full"),
        # Membership on the provider-chosen coordinate alone, both arms. These
        # are what show that reshaping degrades the coordinate no mechanism
        # bounds -- the effect that makes the benefit exceed the closeable share.
        "powBaseIngressMembAdaptive": _c(car, "adaptive:membership", "ingress_only"),
        "powBaseIngressMembFull": _c(car, "full:membership", "ingress_only"),
        "powFullMemb": _c(car, "full:membership", "full"),
        "powBaseAdaptAttr": _c(carb, "adaptive:attribute", "full"),
        "powBaseFullAttr": _c(carb, "full:attribute", "full"),
        "powBaseAdaptAttrIngress": _c(carb, "adaptive:attribute", "ingress_only"),
        "powBaseFullAttrIngress": _c(carb, "full:attribute", "ingress_only"),
        "powGroups": str(a["n_groups"]),
        "powNAdaptive": str(a["n_adaptive"]),
        "powNFull": str(a["n_full"]),
        "powAttrDelta": f"{a['mean_difference']:+.3f}",
        "powAttrCI": f"{a['ci95_t'][0]:+.3f}, {a['ci95_t'][1]:+.3f}",
        "powAttrCIboot": f"{a['ci95'][0]:+.3f}, {a['ci95'][1]:+.3f}",
        "powAttrSd": f"{a['sd_difference']:.3f}",
        "powAttrMDE": f"{a['mde']:.3f}",
        "powMembDelta": f"{m['mean_difference']:+.3f}",
        "powMembCI": f"{m['ci95_t'][0]:+.3f}, {m['ci95_t'][1]:+.3f}",
        "powMembCIboot": f"{m['ci95'][0]:+.3f}, {m['ci95'][1]:+.3f}",
        "powMembSd": f"{m['sd_difference']:.3f}",
        "powMembMDE": f"{m['mde']:.3f}",
    }
    if args.prefix != "pow":
        macros = {args.prefix + k[3:] if k.startswith("pow") else args.prefix + k: v
                  for k, v in macros.items()}
    assert_macro_names_are_latex_safe(list(macros))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    header = (
        "% Paired power analysis on the twelve-group v3 corpus. Generated by\n"
        "% scripts/make_v3_power.py from artifacts/cvm-v3.jsonl. ICLR-scoped: the\n"
        "% shared tables/macros.tex is NOT touched, so \\attrMDE keeps naming the\n"
        "% six-group figure the other manuscripts report.\n"
    )
    args.macros.write_text(
        header + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items())
    )

    print(f"wrote {args.report}")
    print(f"wrote {args.macros}")
    for name, rep in (("attribute", a), ("membership", m)):
        print(f"  {name}: n_groups={rep['n_groups']} "
              f"delta={rep['mean_difference']:+.3f} "
              f"sd={rep['sd_difference']:.3f} MDE={rep['mde']:.3f}")
        print(f"    t-CI  [{rep['ci95_t'][0]:+.3f}, {rep['ci95_t'][1]:+.3f}]  "
              f"boot-CI [{rep['ci95'][0]:+.3f}, {rep['ci95'][1]:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

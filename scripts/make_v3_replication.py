#!/usr/bin/env python3
r"""Regenerate the twelve-group replication macros from the v3 journal.

Why this script exists. The Reproducibility Statement claims that every reported
number is regenerated from a released journal by a released script rather than
transcribed. That was true of every table in the shared pipeline and *not* true
of ``iclr2027/tables/macros_replication.tex``, which a one-off analysis produced.
A reviewer checking the artifact against the claim would have found the gap, and
the claim is the more important of the two things to keep honest, so here is the
script. It reproduces the committed values exactly; ``--check`` asserts that
rather than asking anyone to take it on faith.

What it reads. The v3 corpus arm named by ``--condition``, defaulting to the
undefended ``adaptive`` arm, which is the one the replication reports. This arm
is deliberately independent of the egress ceiling that the padded arm is
provisioned with (see ``make_v3_power.py``): the undefended arm pads nothing, so
its numbers do not move when that setting does.

Every AUC is the leave-one-(specialty, topic)-out, attacker-favorable estimate
from the shipped ``evaluate_attack``, and every null is the within-group
permutation null of the same arm from the shipped ``permutation_null``. Neither
is reimplemented here. Comparing an observed AUC against 0.5 would overstate
leakage, because selecting the better of A1/A2 and reporting max(auc, 1-auc)
puts this design's floor near 0.6, which is why the null is calibrated per arm.

Run:  python3 scripts/make_v3_replication.py            # regenerate
      python3 scripts/make_v3_replication.py --check    # verify, write nothing
"""

from __future__ import annotations

import argparse
import json
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
from traceguard.attack import evaluate_attack, permutation_null  # noqa: E402


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


def baseline_signal(macros: Path) -> str:
    r"""Read the six-group signal from the shared macros rather than retyping it.

    ``\repBaseAttrSignal`` exists only so the replication can be read against the
    figure it improves on. Copying the number here would let the two drift apart
    silently, which is the whole failure mode this file is closing.
    """

    match = re.search(
        r"\\newcommand\{\\attrSignalAboveNull\}\{([^}]*)\}", macros.read_text()
    )
    if not match:
        raise SystemExit(f"no \\attrSignalAboveNull in {macros}")
    return match.group(1)


def cluster_interval(rows: list[dict[str, Any]], condition: str, target: str,
                     seed: int, n_boot: int, min_n: int) -> dict[str, Any]:
    r"""Per-specialty spread as a logit-scale cluster bootstrap.

    Why not the shared pipeline's estimator. That one takes a normal-approximation
    $t$ interval over the raw per-specialty AUCs, and on three clusters it returns
    $[0.445, 1.024]$. An upper limit above $1$ is outside the parameter space, so
    the interval is not merely wide, it is invalid, and a reader is right to read
    it as evidence that the variance estimate cannot be trusted at this cluster
    count. Anything resting on it -- including the Ethics statement's claim that
    leakage is worst for psychiatric content -- inherits that.

    Two changes. The specialty is the cluster and the bootstrap resamples
    specialties, which is the unit the leave-one-group-out protocol actually
    holds out. And the average is taken on the logit scale and mapped back, so
    the interval cannot leave $(0,1)$ by construction. Six specialties is still
    few; a percentile interval over six clusters is reported as the honest width
    it is rather than dressed up, and the point of the fix is validity, not a
    tighter number.
    """

    import math
    import random

    groups: dict[str, list[dict[str, Any]]] = {}
    for record in _records_for(rows, condition):
        key = (record.get("trace") or {}).get("service") or record.get("service") or "?"
        groups.setdefault(str(key), []).append(record)
    per = {
        name: evaluate_attack(members, target=target, n_bootstrap=0, seed=seed)["pooled_auc"]
        for name, members in groups.items()
        if len(members) > min_n
    }
    if len(per) < 3:
        raise SystemExit(f"{target}: {len(per)} usable specialty clusters; need 3")

    # A pooled AUC of exactly 0 or 1 has no finite logit. Clamp at half the
    # smallest resolvable step for the smallest cluster and record that we did,
    # rather than dropping the cluster and quietly changing the estimand.
    smallest = min(len(v) for v in groups.values() if len(v) > min_n)
    eps = 1.0 / (4.0 * smallest)
    clamped = sorted(k for k, v in per.items() if v <= eps or v >= 1 - eps)

    def logit(a: float) -> float:
        a = min(max(a, eps), 1 - eps)
        return math.log(a / (1 - a))

    def expit(z: float) -> float:
        return 1.0 / (1.0 + math.exp(-z))

    names = sorted(per)
    values = [per[n] for n in names]
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        pick = [rng.choice(values) for _ in values]
        draws.append(sum(logit(v) for v in pick) / len(pick))
    draws.sort()
    lo = draws[int(0.025 * (n_boot - 1))]
    hi = draws[int(0.975 * (n_boot - 1))]
    centre = sum(logit(v) for v in values) / len(values)
    return {
        "target": target,
        "per_specialty": per,
        "n_clusters": len(per),
        "point": expit(centre),
        "ci95": [expit(lo), expit(hi)],
        "weakest": min(per, key=per.get),
        "weakest_auc": per[min(per, key=per.get)],
        "strongest": max(per, key=per.get),
        "strongest_auc": per[max(per, key=per.get)],
        "clamped": clamped,
        "n_boot": n_boot,
        "estimator": (
            "mean of per-specialty pooled AUC on the logit scale, percentile "
            "cluster bootstrap over specialties, mapped back through the logistic"
        ),
    }


def analyse(rows: list[dict[str, Any]], condition: str, target: str,
            perms: int, seed: int) -> dict[str, Any]:
    records = _records_for(rows, condition)
    if not records:
        raise SystemExit(f"no {condition} records in the journal")
    report = evaluate_attack(records, target=target, n_bootstrap=0, seed=seed)
    model = report["models"][report["selected_attacker"]]
    null = permutation_null(
        records, target=target, n_perm=perms, within_group=True, seed=seed
    )
    return {
        "auc": report["auc"],
        "fold_std": model["fold_std_auc"],
        "null_mean": null["null_mean"],
        "null_q95": null["null_q95"],
        "p_value": null["p_value"],
        "signal": report["auc"] - null["null_mean"],
        "n_draws": null["n_perm"],
        "pooled": report["pooled_auc"],
        "selected": report["selected_attacker"],
        "n_records": report["n"],
        "n_groups": report["n_groups"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", type=Path, default=ROOT / "artifacts" / "cvm-v3.jsonl")
    ap.add_argument("--condition", default="adaptive")
    ap.add_argument("--permutations", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260710)
    ap.add_argument("--shared-macros", type=Path, default=ROOT / "tables" / "macros.tex")
    ap.add_argument("--report", type=Path,
                    default=ROOT / "iclr2027" / "tables" / "v3-null-report.json")
    ap.add_argument("--macros", type=Path,
                    default=ROOT / "iclr2027" / "tables" / "macros_replication.tex")
    ap.add_argument("--check", action="store_true",
                    help="compare against the committed files and write nothing")
    ap.add_argument("--clusters-only", action="store_true",
                    help="emit only the per-specialty cluster interval; skips the "
                         "permutation nulls, so it runs in seconds")
    ap.add_argument("--cluster-macros", type=Path,
                    default=ROOT / "iclr2027" / "tables" / "macros_clusters.tex")
    ap.add_argument("--cluster-report", type=Path,
                    default=ROOT / "iclr2027" / "tables" / "v3-cluster-report.json")
    ap.add_argument("--cluster-boot", type=int, default=20000)
    ap.add_argument("--cluster-min-n", type=int, default=20)
    args = ap.parse_args()

    rows = load(args.journal)

    if args.clusters_only:
        attr_c = cluster_interval(rows, args.condition, "attribute", args.seed,
                                  args.cluster_boot, args.cluster_min_n)
        memb_c = cluster_interval(rows, args.condition, "membership", args.seed,
                                  args.cluster_boot, args.cluster_min_n)
        cm = {
            "clusSpecialties": str(attr_c["n_clusters"]),
            "clusBoot": str(args.cluster_boot),
            "clusAttrPoint": f"{attr_c['point']:.3f}",
            "clusAttrCI": f"{attr_c['ci95'][0]:.3f}, {attr_c['ci95'][1]:.3f}",
            "clusAttrWeakest": attr_c["weakest"],
            "clusAttrWeakestAUC": f"{attr_c['weakest_auc']:.3f}",
            "clusAttrStrongest": attr_c["strongest"],
            "clusAttrStrongestAUC": f"{attr_c['strongest_auc']:.3f}",
            "clusMembPoint": f"{memb_c['point']:.3f}",
            "clusMembCI": f"{memb_c['ci95'][0]:.3f}, {memb_c['ci95'][1]:.3f}",
        }
        assert_macro_names_are_latex_safe(list(cm))
        args.cluster_macros.parent.mkdir(parents=True, exist_ok=True)
        args.cluster_macros.write_text(
            "% Per-specialty cluster interval on the logit scale, generated by\n"
            "% scripts/make_v3_replication.py --clusters-only. Replaces the shared\n"
            "% pipeline's normal-approximation interval, which returned an upper\n"
            "% limit above 1 on three clusters. ICLR-scoped.\n"
            + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in cm.items())
        )
        args.cluster_report.write_text(
            json.dumps({"attribute": attr_c, "membership": memb_c}, indent=2,
                       sort_keys=True) + "\n"
        )
        print(f"wrote {args.cluster_macros}\nwrote {args.cluster_report}")
        for name, rep in (("attribute", attr_c), ("membership", memb_c)):
            print(f"  {name}: {rep['n_clusters']} clusters  point={rep['point']:.3f}  "
                  f"CI=[{rep['ci95'][0]:.3f}, {rep['ci95'][1]:.3f}]")
            print(f"    weakest={rep['weakest']} {rep['weakest_auc']:.3f}  "
                  f"strongest={rep['strongest']} {rep['strongest_auc']:.3f}")
            if rep["clamped"]:
                print(f"    clamped for the logit: {', '.join(rep['clamped'])}")
            print("    per specialty: " + ", ".join(
                f"{k} {v:.3f}" for k, v in sorted(rep["per_specialty"].items())))
        return 0
    attr = analyse(rows, args.condition, "attribute", args.permutations, args.seed)
    memb = analyse(rows, args.condition, "membership", args.permutations, args.seed)

    specialties = sorted(
        {
            (row.get("case") or {}).get("specialty")
            for row in rows
            if row.get("condition") == args.condition
            and row.get("status") in (None, "completed")
        }
        - {None}
    )

    def p(value: float) -> str:
        # A p-value at the resolution floor is reported as a bound, not as a
        # point: with 2000 draws the smallest attainable value is 1/(n+1) and
        # printing it as if it were measured exactly would overclaim.
        floor = 1.0 / (args.permutations + 1)
        return f"\\le {floor:.4f}" if value <= floor + 1e-12 else f"= {value:.4f}"

    macros = {
        "repN": str(attr["n_records"]),
        "repGroups": str(attr["n_groups"]),
        "repSpecialties": str(len(specialties)),
        "repPerms": str(args.permutations),
        "repAttr": f"{attr['auc']:.3f}",
        "repAttrSd": f"{attr['fold_std']:.3f}",
        "repAttrNull": f"{attr['null_mean']:.3f}",
        "repAttrQupper": f"{attr['null_q95']:.3f}",
        "repAttrP": p(attr["p_value"]),
        "repAttrSignal": f"{attr['signal']:.3f}",
        "repMemb": f"{memb['auc']:.3f}",
        "repMembSd": f"{memb['fold_std']:.3f}",
        "repMembNull": f"{memb['null_mean']:.3f}",
        "repMembP": p(memb["p_value"]),
        "repMembSignal": f"{memb['signal']:.3f}",
        "repBaseAttrSignal": baseline_signal(args.shared_macros),
    }
    assert_macro_names_are_latex_safe(list(macros))

    header = (
        "% Twelve-group replication on the v3 corpus. Generated from\n"
        f"% {args.journal.name} by scripts/make_v3_replication.py; the report beside\n"
        "% it carries the full precision. ICLR-scoped: the shared tables/macros.tex\n"
        "% is NOT touched.\n"
    )
    body = "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items())
    report_text = json.dumps(
        {
            "n_records": attr["n_records"],
            "groups": attr["n_groups"],
            "permutations": args.permutations,
            "condition": args.condition,
            "specialties": specialties,
            "attribute": attr,
            "membership": memb,
        },
        indent=2,
        sort_keys=True,
    ) + "\n"

    if args.check:
        drift = []
        existing = args.macros.read_text() if args.macros.exists() else ""
        for name, value in macros.items():
            found = re.search(r"\\newcommand\{\\" + name + r"\}\{([^}]*)\}", existing)
            if found is None:
                drift.append(f"  {name}: absent from the committed file, now {value}")
            elif found.group(1) != value:
                drift.append(f"  {name}: committed {found.group(1)!r}, regenerated {value!r}")
        if drift:
            print("REGENERATION DIFFERS from the committed macros:")
            print("\n".join(drift))
            return 1
        print(f"ok: all {len(macros)} macros in {args.macros.name} reproduce exactly")
        return 0

    args.macros.parent.mkdir(parents=True, exist_ok=True)
    args.macros.write_text(header + body)
    args.report.write_text(report_text)
    print(f"wrote {args.macros}\nwrote {args.report}")
    for name, rep in (("attribute", attr), ("membership", memb)):
        print(f"  {name}: auc={rep['auc']:.4f} null={rep['null_mean']:.4f} "
              f"signal={rep['signal']:.4f} p={rep['p_value']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

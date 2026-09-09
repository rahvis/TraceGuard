#!/usr/bin/env python3
"""Deadline provisioning: what the $(0,0)$-on-$C$ release costs at each deadline.

The fully-padded release's utility cost is mediated entirely by one event -- a step
that misses the public per-step deadline is released empty and the crew continues
with a missing input. An earlier version of this work reported that cost at a
single deadline and described the resulting rate as "a deadline-provisioning
property of the deployment", which is true but untested: the obvious question is
whether *any* deadline drives the rate to zero while leaving the guarantee intact,
and that was never measured.

This reads the per-deadline journals and answers it directly:

* the fail-closed rate per deadline, with a Wilson interval, because the rate is a
  proportion over a few dozen runs and a naive interval would overstate precision;
* which step type the overruns land on, since a rate concentrated on one node is a
  provisioning problem and a rate spread across all of them is not;
* the resulting latency, because the whole point is that buying a lower rate costs
  wall-clock: a padded run takes exactly (plan length) x (deadline) by
  construction, so the trade is explicit rather than incidental.

Emits tables/tab_deadline.tex plus macros, in the same style as the other
generators, so the manuscript reads the measurement rather than a claim about it.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Used rather than the normal approximation because these are small n with
    proportions that can sit near 0, where the naive interval runs below zero and
    claims a precision the sample does not support.
    """

    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_condition(path: Path, conditions: tuple[str, ...]) -> list[dict[str, Any]]:
    """Completed rows of the given conditions, with their traces."""

    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("status") != "completed" or not (row.get("trace") or {}).get("steps"):
            continue
        if row.get("condition") in conditions:
            rows.append(row)
    return rows


def load(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("status") != "completed" or not (row.get("trace") or {}).get("steps"):
            continue
        # Only padded rows carry a deadline to miss. Filtering here rather than at
        # the call site lets the headline journal -- which also holds the adaptive
        # and structure arms -- be passed as a sweep point in its own right, at
        # the largest n of any setting.
        if row.get("condition") not in ("full", "full_pad"):
            continue
        rows.append(row)
    return rows


def receipt_fail_closed(run_dir: Path) -> dict[str, dict[str, Any]] | None:
    """Fail-closed state per run, read from the signed receipts.

    Journals written before the fail-closed field was added carry no
    ``violations`` key at all, so ``row.get("violations")`` returns a missing
    key and an analysis over such a journal silently reports a 0% rate. The
    receipts always carried the fact -- ``budget_status`` is ``breach`` and
    ``fail_closed`` is true -- so for those journals the receipts are the
    authoritative source and this reads them.
    """

    if not run_dir.is_dir():
        return None
    out: dict[str, dict[str, Any]] = {}
    # Match at any depth. The archived store is <root>/runs/<run-id>/receipts.jsonl
    # while a store pointed at one level in is <root>/<run-id>/receipts.jsonl, and
    # a single-level glob silently finds nothing in the first layout. That made
    # the deadline stage refuse to report a rate it could in fact source, which
    # is the right refusal for the wrong reason: the receipts were there.
    for rp in run_dir.rglob("receipts.jsonl"):
        for line in rp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                body = json.loads(line)
            except ValueError:
                continue
            body = body.get("body", body)
            rid = body.get("run_id")
            if not rid:
                continue
            out[str(rid)] = {
                "fail_closed": bool(body.get("fail_closed")),
                "violations": list(body.get("violations") or []),
            }
    return out or None


def summarize(
    rows: list[dict[str, Any]],
    deadline_s: float,
    receipts: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    n = len(rows)
    failed, viol_steps, total_steps = 0, 0, 0
    nodes: Counter[str] = Counter()
    codes: Counter[str] = Counter()
    latencies, ingress = [], []
    for row in rows:
        steps = row["trace"]["steps"]
        total_steps += len(steps)
        # Prefer the journal's own field; fall back to the receipt for journals
        # written before that field existed. Never treat an absent key as zero.
        rec = (receipts or {}).get(str(row.get("run_id"))) if receipts else None
        if "violations" in row or "fail_closed" in row:
            vs = list(row.get("violations") or [])
            if not vs and row.get("fail_closed"):
                vs = [{"step_type": "?", "code": "fail_closed"}]
        elif rec is not None:
            vs = list(rec["violations"]) or (
                [{"step_type": "?", "code": "fail_closed"}] if rec["fail_closed"] else []
            )
        else:
            raise SystemExit(
                f"journal rows carry no fail-closed field and no receipts were found "
                f"for {deadline_s:g}s; refusing to report a rate of zero that is "
                f"really an absent measurement"
            )
        if vs:
            failed += 1
            viol_steps += len(vs)
            for v in vs:
                nodes[str(v.get("step_type", "?"))] += 1
                codes[str(v.get("code", "?"))] += 1
        latencies.append(sum(float(s.get("duration_s") or 0.0) for s in steps))
        ingress.append(sum(int(s.get("ingress_bytes") or 0) for s in steps))
    lo, hi = wilson(failed, n)
    _tot = sum(codes.values())
    return {
        "viol_total": _tot,
        "viol_deadline_pct": (100.0 * codes.get("deadline_overrun", 0) / _tot) if _tot else 0.0,
        "deadline_s": deadline_s,
        "runs": n,
        "fail_closed": failed,
        "fail_closed_pct": 100.0 * failed / n if n else 0.0,
        "ci_lo_pct": 100.0 * lo,
        "ci_hi_pct": 100.0 * hi,
        "viol_step_pct": 100.0 * viol_steps / total_steps if total_steps else 0.0,
        "worst_node": nodes.most_common(1)[0][0] if nodes else "--",
        "worst_node_share": (
            100.0 * nodes.most_common(1)[0][1] / viol_steps if viol_steps else 0.0
        ),
        "mean_latency_s": statistics.mean(latencies) if latencies else 0.0,
        "mean_ingress_b": statistics.mean(ingress) if ingress else 0.0,
    }


def _baseline_latency(path: Path) -> float:
    """Mean adaptive wall-clock in the same journal, for the padding increment."""

    rows = load_condition(path, ("adaptive",))
    if not rows:
        return 0.0
    return statistics.mean(
        sum(float(s.get("duration_s") or 0.0) for s in r["trace"]["steps"]) for r in rows
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--journal",
        action="append",
        required=True,
        metavar="DEADLINE_MS=PATH",
        help="one per swept deadline, e.g. 15000=artifacts/cvm-dl15000.jsonl",
    )
    ap.add_argument("--tables", default="tables")
    ap.add_argument(
        "--runs",
        type=Path,
        default=None,
        help=(
            "the immutable run store, used to recover fail-closed state from the "
            "signed receipts for journals written before the journal itself "
            "carried that field"
        ),
    )
    ap.add_argument(
        "--headline",
        metavar="DEADLINE_MS=PATH",
        help=(
            "the headline arm's journal, whose full-pad rows supply the pad-cost "
            "macros. These were previously hand-written into tables/macros.tex, a "
            "file whose own header says it is machine-generated and must not be "
            "edited by hand -- so the appendix's rate sat outside the "
            "regeneration path that every other number goes through."
        ),
    )
    args = ap.parse_args()

    recs = receipt_fail_closed(args.runs) if args.runs else None
    if recs:
        print(f"  recovered fail-closed state for {len(recs)} runs from receipts")
    rows = []
    for spec in args.journal:
        ms, _, path = spec.partition("=")
        p = Path(path)
        if not p.exists():
            print(f"  skipping missing {p}")
            continue
        rows.append(summarize(load(p), float(ms) / 1000.0, recs))
    rows.sort(key=lambda r: r["deadline_s"])
    if not rows:
        print("no journals found; nothing written")
        return 1

    out = Path(args.tables)
    body = "\n".join(
        f"{r['deadline_s']:.0f} & {r['runs']} & "
        f"{r['fail_closed_pct']:.1f} & [{r['ci_lo_pct']:.1f}, {r['ci_hi_pct']:.1f}] & "
        f"{r['viol_step_pct']:.2f} & {r['mean_latency_s']:.1f} \\\\"
        for r in rows
    )
    (out / "tab_deadline.tex").write_text(
        "% Generated by scripts/make_deadline_artifacts.py from the per-deadline journals.\n"
        "{\\setlength{\\tabcolsep}{4pt}\\renewcommand{\\arraystretch}{1.15}\n"
        "\\begin{tabular}{@{}rrrlrr@{}}\n\\toprule\n"
        "deadline & runs & fail-closed & 95\\% CI & steps lost & latency \\\\\n"
        "(s) & & (\\%) & (\\%) & (\\%) & (s) \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}}}\n"
    )

    # Pad-cost macros for the headline arm, generated rather than transcribed.
    head = {}
    if args.headline:
        ms, _, path = args.headline.partition("=")
        hp = Path(path)
        if hp.exists():
            hrows = load(hp)
            if hrows:
                h = summarize(hrows, float(ms) / 1000.0, recs)
                head = {
                    "padRunsN": str(h["runs"]),
                    "padFailClosedN": str(h["fail_closed"]),
                    "padFailClosedPct": f"{h['fail_closed_pct']:.1f}",
                    "padFailClosedCI": f"{h['ci_lo_pct']:.1f}--{h['ci_hi_pct']:.1f}",
                    "padViolStepPct": f"{h['viol_step_pct']:.2f}",
                    # Violation totals and the split by code, because "a lower
                    # refusal rate is bought in wall-clock" is only true for the
                    # deadline component; a provider error or an over-ceiling
                    # request also releases empty and neither is
                    # deadline-sensitive. Stating the split lets the reader see
                    # whether the carve-out bites on this deployment.
                    "padViolTotal": str(h["viol_total"]),
                    "padViolDeadlinePct": f"{h['viol_deadline_pct']:.0f}",
                    # padExtraS is deliberately NOT emitted here:
                    # make_paper_artifacts.py already owns it and the two agree to
                    # 0.3 s, so emitting both only risks a duplicate definition.
                }
                print(
                    f"  headline arm ({h['deadline_s']:.0f}s): "
                    f"{h['fail_closed']}/{h['runs']} runs fail closed "
                    f"= {h['fail_closed_pct']:.1f}% "
                    f"[{h['ci_lo_pct']:.1f}, {h['ci_hi_pct']:.1f}]"
                )

    zero = [r for r in rows if r["fail_closed"] == 0]
    macros = {
        "dlDeadlines": ", ".join(f"{r['deadline_s']:.0f}" for r in rows),
        "dlCount": str(len(rows)),
        "dlLoDeadline": f"{rows[0]['deadline_s']:.0f}",
        "dlLoPct": f"{rows[0]['fail_closed_pct']:.1f}",
        "dlHiDeadline": f"{rows[-1]['deadline_s']:.0f}",
        "dlHiPct": f"{rows[-1]['fail_closed_pct']:.1f}",
        "dlHiLatency": f"{rows[-1]['mean_latency_s']:.1f}",
        "dlWorstNode": rows[0]["worst_node"].replace("_", "\\_"),
        "dlZeroDeadline": (f"{zero[0]['deadline_s']:.0f}" if zero else "none swept"),
        "dlZeroFound": "yes" if zero else "no",
    }
    macros.update(head)
    (out / "macros_deadline.tex").write_text(
        "% Generated by scripts/make_deadline_artifacts.py -- do not edit by hand.\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items())
    )
    for r in rows:
        print(
            f"  deadline {r['deadline_s']:5.0f}s  n={r['runs']:3}  "
            f"fail-closed {r['fail_closed_pct']:5.1f}% "
            f"[{r['ci_lo_pct']:.1f},{r['ci_hi_pct']:.1f}]  "
            f"latency {r['mean_latency_s']:6.1f}s  worst={r['worst_node']}"
        )
    print(f"\nwrote {out/'tab_deadline.tex'} and {out/'macros_deadline.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

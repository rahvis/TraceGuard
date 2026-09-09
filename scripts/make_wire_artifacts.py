#!/usr/bin/env python3
r"""Re-run the attack on packet-level observables instead of the guest's counters.

The paper's threat model is a host reading the guest's virtual NIC; its
instrumentation is the guest's own runtime counting JSON body bytes. Those are
not the same quantity -- TLS record framing, header overhead and packet
coalescing sit between them -- so the reported AUCs were a claim about
application-level counters and only an argument about the wire.

This closes that gap. It reads a header-only capture taken at the guest's NIC
during a sequential run set, attributes packets to steps, and recomputes both
the per-coordinate single-feature AUCs and the group-aware attack on
wire-derived features. Two numbers matter: how closely wire bytes track the
recorded bytes, and whether the attack survives the substitution.

Attribution requires absolute time, and the journal's ``wall_time_s`` is
relative to run start. The run's receipt carries ``issued_at``, so
``run_start = issued_at - (last step's wall_time_s + duration_s)`` and the run's
timeline is then partitioned at the step boundaries. That is why the capture
must be sequential: with concurrent runs the runs' timelines interleave and no
packet can be attributed to a step at all.

Input is the text form of the capture, produced on the capture host by

    tcpdump -r wire.pcap -nn -tt -q

whose lines look like

    1757353456.123456 IP 10.0.0.4.54321 > 20.1.2.3.443: tcp 1234

so direction comes from which side carries port 443 and the trailing integer is
the TCP payload length. Passing the text rather than the pcap keeps the payload
bytes -- which were never captured -- out of the pipeline entirely.

Usage:
    python3 scripts/make_wire_artifacts.py \
        --journal artifacts/wire.jsonl --runs artifacts/wire-runs \
        --packets artifacts/wire-packets.txt --out-dir tables
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# "<epoch> IP <a>.<port> > <b>.<port>: tcp <len>"; ip6 lines use the same shape.
_PKT = re.compile(
    r"^(\d+\.\d+)\s+IP6?\s+(\S+?)\.(\d+)\s+>\s+(\S+?)\.(\d+):\s+tcp\s+(\d+)"
)


def peer_totals(path: Path) -> dict[str, tuple[int, int]]:
    """:443 peer -> (packets, bytes). Used to pick the model endpoint."""
    tot: dict[str, list[int]] = {}
    for line in path.read_text(errors="replace").splitlines():
        m = _PKT.match(line.strip())
        if not m:
            continue
        _, src, sport, dst, dport, length = m.groups()
        peer = dst if dport == "443" else src if sport == "443" else None
        if peer is None:
            continue
        cur = tot.setdefault(peer, [0, 0])
        cur[0] += 1
        cur[1] += int(length)
    return {k: (v[0], v[1]) for k, v in tot.items()}


def read_packets(path: Path, peer: str | None = None) -> list[tuple[float, str, int]]:
    """(timestamp, 'out'|'in', tcp_payload_len), dropping pure-ACK segments.

    ``peer`` restricts the capture to one remote address and is not optional in
    practice. A capture filtered only on ``tcp port 443`` picks up every other
    HTTPS flow on the machine -- the host's own monitoring agents, the service
    container's health checks, package updates -- and on our first run one
    unrelated peer contributed 227 kB, which alone produced a 248 kB ingress
    outlier and inflated the response-direction variance by an order of
    magnitude. The model endpoint is the peer carrying essentially all the
    traffic; ``peer_totals`` reports the distribution so the choice is made from
    the data rather than assumed.

    A zero-length segment carries no volume and is excluded; it remains visible
    to a host as timing, which is measured from the step boundaries instead.
    """
    out: list[tuple[float, str, int]] = []
    for line in path.read_text(errors="replace").splitlines():
        m = _PKT.match(line.strip())
        if not m:
            continue
        ts, src, sport, dst, dport, length = m.groups()
        n = int(length)
        if n == 0:
            continue
        if dport == "443":
            if peer is None or dst == peer:
                out.append((float(ts), "out", n))
        elif sport == "443":
            if peer is None or src == peer:
                out.append((float(ts), "in", n))
    out.sort()
    return out


def _issued_at(path: Path) -> dict[str, float]:
    """run_id -> receipt issue time, as an epoch second."""
    stamps: dict[str, float] = {}
    for receipt in path.rglob("receipts.jsonl"):
        for line in receipt.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                body = json.loads(line)
            except ValueError:
                continue
            body = body.get("body", body)
            run_id, when = body.get("run_id"), body.get("issued_at")
            if not run_id or not when:
                continue
            stamps[str(run_id)] = (
                datetime.fromisoformat(str(when).replace("Z", "+00:00")).timestamp()
            )
    return stamps


def load_runs(journal: Path, condition: str) -> list[dict[str, Any]]:
    rows = []
    for line in journal.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "completed" and row.get("condition") == condition:
            rows.append(row)
    return rows


def attribute(
    rows: list[dict[str, Any]], packets: list[tuple[float, str, int]],
    stamps: dict[str, float],
) -> list[dict[str, Any]]:
    """Attach wire byte counts to each step by PARTITIONING the run timeline.

    Step i owns ``[wall_i, wall_{i+1})``, and the last step owns the remainder
    up to the receipt. Every packet inside a run therefore belongs to exactly
    one step.

    The first version widened each step to ``[wall_i - s, wall_i + dur_i + 2s]``
    to catch packets arriving just outside the guest's own stamps. Those windows
    overlap -- on a real trace step 0 ran to 5.71s while step 1 began at 4.75s --
    so every packet in the overlap was counted twice and the wire/app ratio came
    out at 6.4x on ingress, which is not a plausible TLS overhead. Partitioning
    needs no slack parameter and cannot double count, which is why it replaced
    the window: a step's call is issued at ``wall_i``, so provider traffic
    arriving before the next call is that step's by construction.
    """
    attributed = []
    for row in rows:
        run_id = str(row.get("run_id"))
        end = stamps.get(run_id)
        steps = (row.get("trace") or {}).get("steps") or []
        if end is None or not steps:
            continue
        last = steps[-1]
        span = float(last["wall_time_s"]) + float(last["duration_s"])
        start = end - span
        bounds = [start + float(s["wall_time_s"]) for s in steps] + [end]
        per_step = []
        for i, step in enumerate(steps):
            a, b = bounds[i], bounds[i + 1]
            wout = sum(n for ts, d, n in packets if a <= ts < b and d == "out")
            win = sum(n for ts, d, n in packets if a <= ts < b and d == "in")
            per_step.append({
                "index": step["index"],
                "step_type": step["step_type"],
                "app_egress": int(step["egress_bytes"]),
                "app_ingress": int(step.get("ingress_bytes") or 0),
                "wire_egress": wout,
                "wire_ingress": win,
                "duration_s": float(step["duration_s"]),
            })
        attributed.append({
            "run_id": run_id,
            "case": row.get("case", {}),
            "steps": per_step,
            "matched": sum(1 for s in per_step if s["wire_egress"] > 0),
            "run_wire_egress": sum(s["wire_egress"] for s in per_step),
            "run_wire_ingress": sum(s["wire_ingress"] for s in per_step),
            "run_app_egress": sum(s["app_egress"] for s in per_step),
            "run_app_ingress": sum(s["app_ingress"] for s in per_step),
        })
    return attributed


def _auc(values: list[float], labels: list[int]) -> float:
    pos = [v for v, y in zip(values, labels, strict=True) if y]
    neg = [v for v, y in zip(values, labels, strict=True) if not y]
    if not pos or not neg:
        return 0.5
    s = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
    raw = s / (len(pos) * len(neg))
    return max(raw, 1 - raw)


def _perm_null(
    values: list[float], labels: list[int], groups: list[str], draws: int, seed: int
) -> tuple[float, float, float]:
    """Observed AUC, the calibrated null mean, and a one-sided p.

    An AUC is uninterpretable here without this. The reported statistic is
    ``max(AUC, 1-AUC)``, which is attacker-favourable and pushes the null of the
    design well above 0.5, so a wire-level AUC of 0.60 against a null of 0.60 is
    no signal at all rather than a weak one. Labels are permuted WITHIN each
    (service, topic) group so the group-by-label balance the statistic is
    defined against is preserved -- the same protocol the paper uses.
    """
    import random

    # Seeded Mersenne Twister on purpose: the permutation null must be
    # reproducible from the seed, which a CSPRNG would not give, and no
    # security property rests on the draw.
    rng = random.Random(seed)  # noqa: S311
    observed = _auc(values, labels)
    by_group: dict[str, list[int]] = {}
    for g, y in zip(groups, labels, strict=True):
        by_group.setdefault(g, []).append(y)
    idx: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        idx.setdefault(g, []).append(i)

    at_least = 0
    total = 0.0
    for _ in range(draws):
        shuffled = [0] * len(labels)
        for g, positions in idx.items():
            pool = by_group[g][:]
            rng.shuffle(pool)
            for pos, y in zip(positions, pool, strict=True):
                shuffled[pos] = y
        drawn = _auc(values, shuffled)
        total += drawn
        if drawn >= observed:
            at_least += 1
    # add-one estimator: with `draws` permutations the floor is 1/(draws+1)
    return observed, total / draws, (at_least + 1) / (draws + 1)


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / (sx * sy)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--journal", required=True, type=Path)
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--packets", required=True, type=Path)
    ap.add_argument("--condition", default="adaptive")
    ap.add_argument("--out-dir", type=Path, default=Path("tables"))
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--peer", default=None,
                    help="model endpoint address; defaults to the busiest :443 peer")
    ap.add_argument("--permutations", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260710)
    args = ap.parse_args()

    totals = peer_totals(args.packets)
    peer = args.peer
    if peer is None and totals:
        peer = max(totals, key=lambda k: totals[k][1])
    print("peers on :443 (bytes):")
    for k in sorted(totals, key=lambda k: -totals[k][1]):
        mark = "  <- model endpoint" if k == peer else "  (excluded)"
        print(f"  {k:24s} {totals[k][0]:6d} pkts {totals[k][1]:10d} B{mark}")
    packets = read_packets(args.packets, peer)
    rows = load_runs(args.journal, args.condition)
    stamps = _issued_at(args.runs)
    print(f"{len(packets)} non-empty segments, {len(rows)} completed runs, "
          f"{len(stamps)} receipts")
    if not packets or not rows:
        raise SystemExit("nothing to correlate")

    runs = attribute(rows, packets, stamps)
    steps = [s for r in runs for s in r["steps"]]
    matched = [s for s in steps if s["wire_egress"] > 0]
    print(f"{len(runs)} runs attributed; {len(matched)}/{len(steps)} steps carry packets")
    if not matched:
        raise SystemExit("no step window matched any packet; check clock alignment")

    # (1) Does the wire track the guest's counters?
    r_out = _pearson([s["app_egress"] for s in matched],
                     [s["wire_egress"] for s in matched])
    r_in = _pearson([s["app_ingress"] for s in matched],
                    [s["wire_ingress"] for s in matched])
    infl_out = sum(s["wire_egress"] for s in matched) / max(
        1, sum(s["app_egress"] for s in matched))
    infl_in = sum(s["wire_ingress"] for s in matched) / max(
        1, sum(s["app_ingress"] for s in matched))
    print(f"  step-level  egress : r={r_out:+.3f}  wire/app={infl_out:.2f}x")
    print(f"  step-level  ingress: r={r_in:+.3f}  wire/app={infl_in:.2f}x")

    # Run level: immune to step-boundary error, so a run-level ratio near the
    # protocol's real overhead alongside a noisy step-level one localises the
    # discrepancy to attribution rather than to the wire.
    rr_out = _pearson([float(r["run_app_egress"]) for r in runs],
                      [float(r["run_wire_egress"]) for r in runs])
    rr_in = _pearson([float(r["run_app_ingress"]) for r in runs],
                     [float(r["run_wire_ingress"]) for r in runs])
    ri_out = sum(r["run_wire_egress"] for r in runs) / max(
        1, sum(r["run_app_egress"] for r in runs))
    ri_in = sum(r["run_wire_ingress"] for r in runs) / max(
        1, sum(r["run_app_ingress"] for r in runs))
    print(f"  run-level   egress : r={rr_out:+.3f}  wire/app={ri_out:.2f}x")
    print(f"  run-level   ingress: r={rr_in:+.3f}  wire/app={ri_in:.2f}x")

    # (2) Does the attack survive on wire features? Same statistic per
    # coordinate as the paper's attribution table: the per-run max.
    def label(run: dict[str, Any], which: str) -> int:
        case = run["case"]
        if which == "attribute":
            return int(case.get("attribute_label", int(bool(case.get("sensitive")))))
        return int(case.get("membership_label", int(bool(case.get("canary_member")))))

    feats = {
        "max wire egress": lambda r: max(s["wire_egress"] for s in r["steps"]),
        "max wire ingress": lambda r: max(s["wire_ingress"] for s in r["steps"]),
        "max app egress": lambda r: max(s["app_egress"] for s in r["steps"]),
        "max app ingress": lambda r: max(s["app_ingress"] for s in r["steps"]),
        "total wire egress": lambda r: r["run_wire_egress"],
        "total wire ingress": lambda r: r["run_wire_ingress"],
        "step count": lambda r: len(r["steps"]),
        "max step timing": lambda r: max(s["duration_s"] for s in r["steps"]),
    }
    usable = [r for r in runs if r["matched"] > 0]
    groups = [
        f'{r["case"].get("specialty", r["case"].get("service"))}/{r["case"].get("topic")}'
        for r in usable
    ]
    table: dict[str, dict[str, float]] = {}
    print(f"\nsingle-feature AUC over {len(usable)} runs, each against its own "
          f"within-group permutation null ({args.permutations} draws):")
    print(f"  {'feature':20s} {'attr':>6s} {'null':>6s} {'p':>7s}    "
          f"{'memb':>6s} {'null':>6s} {'p':>7s}")
    for name, fn in feats.items():
        vals = [float(fn(r)) for r in usable]
        row: dict[str, float] = {}
        cells = []
        for target in ("attribute", "membership"):
            ys = [label(r, target) for r in usable]
            obs, null, pval = _perm_null(
                vals, ys, groups, args.permutations, args.seed
            )
            row[target] = obs
            row[f"{target}_null"] = null
            row[f"{target}_p"] = pval
            flag = "*" if pval < 0.05 else " "
            cells.append(f"{obs:6.3f} {null:6.3f} {pval:7.4f}{flag}")
        table[name] = row
        print(f"  {name:20s} {cells[0]}   {cells[1]}")

    report = {
        "condition": args.condition,
        "peer": peer,
        "peers_seen": {k: {"packets": v[0], "bytes": v[1]} for k, v in totals.items()},
        "runs": len(usable),
        "steps_total": len(steps),
        "steps_with_packets": len(matched),
        "segments": len(packets),
        "correlation": {
            "step_egress_r": r_out, "step_ingress_r": r_in,
            "run_egress_r": rr_out, "run_ingress_r": rr_in,
        },
        "inflation": {
            "step_egress": infl_out, "step_ingress": infl_in,
            "run_egress": ri_out, "run_ingress": ri_in,
        },
        "single_feature_auc": table,
    }
    out = args.report or (args.out_dir / "wire-report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    # Generated, like every other number the manuscript quotes.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_paper_artifacts import merge_macros

    def p_str(value: float) -> str:
        floor = 1.0 / (args.permutations + 1)
        return f"\\le {floor:.4f}" if value <= floor else f"={value:.3f}"

    macros = {
        "wireRuns": str(len(usable)),
        "wireSegments": str(len(packets)),
        "wireStepsMatched": str(len(matched)),
        "wireStepsTotal": str(len(steps)),
        "wirePerms": str(args.permutations),
        "wireEgressR": f"{rr_out:.3f}",
        "wireIngressR": f"{rr_in:.3f}",
        "wireEgressOverhead": f"{ri_out:.2f}",
        "wireIngressOverhead": f"{ri_in:.2f}",
        "wireIngressAttr": f"{table['max wire ingress']['attribute']:.3f}",
        "wireIngressAttrNull": f"{table['max wire ingress']['attribute_null']:.3f}",
        "wireIngressAttrP": p_str(table["max wire ingress"]["attribute_p"]),
        "wireEgressAttr": f"{table['max wire egress']['attribute']:.3f}",
        "wireEgressAttrNull": f"{table['max wire egress']['attribute_null']:.3f}",
        "wireEgressAttrP": p_str(table["max wire egress"]["attribute_p"]),
        "wireAppIngressAttr": f"{table['max app ingress']['attribute']:.3f}",
        "wireAppIngressAttrP": p_str(table["max app ingress"]["attribute_p"]),
        "wireAppEgressAttr": f"{table['max app egress']['attribute']:.3f}",
        "wireAppEgressAttrP": p_str(table["max app egress"]["attribute_p"]),
    }
    mpath = args.out_dir / "macros_wire.tex"
    merge_macros(
        mpath, macros,
        "% Generated by scripts/make_wire_artifacts.py -- do not edit by hand.\n",
    )
    print(f"\nwrote {out} and {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

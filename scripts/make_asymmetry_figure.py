#!/usr/bin/env python3
"""The two results this revision turns on, as one two-panel monochrome figure.

Panel (a) is the paper's central asymmetry. Each observable coordinate is scored
alone against both secrets, and the point is not the magnitudes but *which* secret
sits on *which* coordinate: membership on timing and structure, which the guest
composes and the defense closes; the attribute on the response size, which the
provider chooses and Proposition 2 puts out of reach. Read that way the panel
explains, rather than merely reports, why full padding closes one secret and not
the other.

Panel (b) is the deadline provisioning curve. The padded release's whole utility
cost is mediated by one event -- a step that misses the public deadline is
released empty -- so the rate of that event against the deadline is the number a
deployer needs. Wilson intervals, because the shorter settings are a few dozen
runs and a naive interval would claim precision the sample does not support.

Monochrome throughout, like every other figure here: category is carried by hatch
and fill level rather than hue, so the figure survives greyscale print and a
colour-blind reader.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.compression"] = 6
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.hashsalt"] = "traceguard"
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = 8

import matplotlib.pyplot as plt  # noqa: E402

_INK = "#111111"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _auc(values: list[float], labels: list[int]) -> float:
    pos = [v for v, lab in zip(values, labels, strict=True) if lab]
    neg = [v for v, lab in zip(values, labels, strict=True) if not lab]
    if not pos or not neg:
        return 0.5
    s = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
    return max(s / (len(pos) * len(neg)), 1 - s / (len(pos) * len(neg)))


def load(path: Path, condition: str) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") == "completed" and r.get("condition") == condition:
            out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", required=True, type=Path)
    ap.add_argument("--deadline-journal", action="append", default=[],
                    metavar="MS=PATH")
    ap.add_argument("--runs", type=Path, default=None)
    ap.add_argument("--out", default="figures/fig_asymmetry.pdf")
    args = ap.parse_args()

    rows = load(args.journal, "adaptive")
    attr = [int(r["trace"]["attribute_label"]) for r in rows]
    memb = [int(r["trace"]["membership_label"]) for r in rows]

    # One statistic per coordinate, each computed the same way, so the panel
    # compares coordinates and not estimators.
    # Labels carry no "max" prefix: every bar is that coordinate's strongest
    # single statistic and the caption says so, so repeating it four times only
    # crowds the axis.
    coords = [
        ("step count\n(structure)", lambda r: len(r["trace"]["steps"])),
        ("step timing\n(per step)", lambda r: max(s["duration_s"] for s in r["trace"]["steps"])),
        ("request\n(egress)", lambda r: max(s["egress_bytes"] for s in r["trace"]["steps"])),
        ("response\n(ingress)", lambda r: max(s["ingress_bytes"] for s in r["trace"]["steps"])),
    ]
    names = [c[0] for c in coords]
    a_auc = [_auc([f(r) for r in rows], attr) for _, f in coords]
    m_auc = [_auc([f(r) for r in rows], memb) for _, f in coords]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.35, 4.0), dpi=200)

    # ---- panel (a): which coordinate carries which secret -------------------
    x = range(len(names))
    w = 0.38
    ax1.bar([i - w / 2 for i in x], a_auc, w, label="attribute",
            facecolor="#4a4a4a", edgecolor=_INK, linewidth=0.6, zorder=3)
    ax1.bar([i + w / 2 for i in x], m_auc, w, label="membership",
            facecolor="white", edgecolor=_INK, linewidth=0.6, hatch="////", zorder=3)
    ax1.axhline(0.5, color=_INK, linewidth=0.6, linestyle="-", zorder=2)
    ax1.axhline(0.60, color=_INK, linewidth=0.9, linestyle="--", zorder=4)
    ax1.text(-0.42, 0.603, "calibrated null", ha="left", va="bottom",
             fontsize=6.5, color=_INK)
    # The two right-hand coordinates are the ones the mechanism cannot close.
    ax1.axvspan(2.5, 3.5, color="#000000", alpha=0.055, zorder=1)
    ax1.text(3.0, 0.838, "provider-chosen:\nnot paddable", ha="center",
             va="top", fontsize=6.5, color=_INK)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(names, fontsize=6)
    ax1.set_ylim(0.48, 0.90)
    ax1.set_ylabel("single-coordinate ROC-AUC", color=_INK)
    ax1.set_title("(a) which coordinate carries which secret", fontsize=7.5, color=_INK)
    ax1.legend(frameon=False, fontsize=6.5, loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.04))
    ax1.grid(axis="y", color="#d8d8d8", linewidth=0.5, zorder=0)
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False)

    # ---- panel (b): the deadline provisioning curve -------------------------
    pts = []
    unresolved: list[tuple[float, int, str]] = []
    recs = None
    if args.runs and args.runs.is_dir():
        recs = {}
        # Receipts sit at <runs>/runs/<run-id>/receipts.jsonl in the archived
        # layout and at <runs>/<run-id>/receipts.jsonl when pointed one level in,
        # so match at any depth rather than assuming one.
        for rp in args.runs.rglob("receipts.jsonl"):
            for line in rp.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    b = json.loads(line)
                except ValueError:
                    continue
                b = b.get("body", b)
                if b.get("run_id"):
                    recs[str(b["run_id"])] = bool(b.get("fail_closed"))

    def rate(path: Path) -> tuple[int, int]:
        """Fail-closed count and n, or (-1, n) when neither source can supply it.

        A journal row written before the runtime recorded ``fail_closed`` has no
        such key, and ``bool(r.get("fail_closed"))`` on it is False -- which is
        indistinguishable from a run that met every deadline. That mistake once
        put a 0% refusal rate in this paper. So a row must be resolvable from the
        journal field or from its archived receipt, and if any row is not, the
        setting is refused rather than reported low.
        """
        rs = load(path, "full")
        n = len(rs)
        k = 0
        for r in rs:
            if "fail_closed" in r:
                k += bool(r["fail_closed"])
                continue
            rid = str(r.get("run_id"))
            if recs is not None and rid in recs:
                k += recs[rid]
                continue
            return -1, n
        return k, n

    for spec in args.deadline_journal:
        ms, _, path = spec.partition("=")
        p = Path(path)
        if not p.exists():
            continue
        k, n = rate(p)
        if n and k < 0:
            unresolved.append((float(ms) / 1000.0, n, str(p)))
        elif n:
            pts.append((float(ms) / 1000.0, k, n))
    k, n = rate(args.journal)
    if n and k < 0:
        unresolved.append((10.0, n, str(args.journal)))
    elif n:
        pts.append((10.0, k, n))
    pts.sort()

    if unresolved:
        for d, n, src in unresolved:
            print(f"  ERROR {d:.1f}s: {n} padded runs carry neither a fail_closed "
                  f"field nor an archived receipt ({src})", file=sys.stderr)
        print("refusing to draw panel (b): a missing field is not a zero rate",
              file=sys.stderr)
        return 1

    if pts:
        xs = [p[0] for p in pts]
        ys = [100.0 * p[1] / p[2] for p in pts]
        lo = [100.0 * wilson(p[1], p[2])[0] for p in pts]
        hi = [100.0 * wilson(p[1], p[2])[1] for p in pts]
        ax2.errorbar(
            xs, ys,
            yerr=[[y - a for y, a in zip(ys, lo, strict=True)],
                  [h - y for y, h in zip(ys, hi, strict=True)]],
            fmt="o-", color=_INK, markerfacecolor="white", markeredgecolor=_INK,
            linewidth=1.0, markersize=5, capsize=2.5, zorder=3,
        )
        for xx, yy, pp in zip(xs, ys, pts, strict=True):
            last = xx == max(xs)
            dx = -22 if last else 6
            # Above the marker normally; below it where the curve or the panel
            # title would otherwise run through the label.
            dy = -13 if (yy > 88 or last) else 9
            ax2.annotate(f"n={pp[2]}", (xx, yy), textcoords="offset points",
                         xytext=(dx, dy), fontsize=6.5, color=_INK)
        ax2.set_xticks(xs)
    ax2.set_xlabel("public per-step deadline $\\tau^\\star$ (s)", color=_INK)
    ax2.set_ylabel("runs losing a step (%)", color=_INK)
    ax2.set_title("(b) what the $(0,0)$-on-$C$ release costs", fontsize=7.5, color=_INK)
    ax2.set_ylim(-6, 112)
    ax2.grid(axis="y", color="#d8d8d8", linewidth=0.5, zorder=0)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    # Latency is the other half of the trade and is exact by construction.
    ax2b = ax2.twinx()
    if pts:
        ax2b.plot(xs, [12 * v for v in xs], ":", color=_INK, linewidth=0.9, zorder=2)
        ax2b.set_ylim(-6 * 1.2, 112 * 1.2)
    ax2b.set_ylabel("latency (s), $12\\times\\tau^\\star$", color=_INK, fontsize=7)
    ax2b.spines["top"].set_visible(False)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", metadata={"CreationDate": None})
    print(f"wrote {out}")
    print("  panel (a) coordinate AUCs:")
    for nm, a, m in zip(names, a_auc, m_auc, strict=True):
        print(f"    {nm.replace(chr(10), ' '):26} attr={a:.3f} memb={m:.3f}")
    print("  panel (b) deadline points:")
    for d, k, n in pts:
        print(f"    {d:5.1f}s  {k}/{n} = {100*k/n:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

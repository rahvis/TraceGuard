#!/usr/bin/env python3
"""The wire-level cross-check as one two-panel monochrome figure.

Panel (a) is fidelity: per-run bytes on the wire against the bytes the guest's
runtime recorded, one series per direction. It answers "is the quantity the
attack uses the quantity a host sees?" and the answer differs by direction only
in slope, not in shape -- both track closely, which is what licenses reading the
paper's application-level numbers as being about a real observable.

Panel (b) is what survives. The attribute AUC at both instrumentation levels for
both directions, each against its own calibrated permutation null. The point is
the crossing: the response direction -- the coordinate Proposition 2 says cannot
be padded -- clears its null at the wire, while the request direction, which the
defense *can* close, does not, because MTU-scale segmentation quantizes a
per-step maximum. A defense designer reading only application-level numbers
would get that backwards.

Monochrome like every other figure here: direction is carried by marker and
hatch, instrumentation level by fill, so the figure survives greyscale print and
a colour-blind reader.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.compression"] = 6
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.hashsalt"] = "traceguard"
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = 8

import matplotlib.pyplot as plt  # noqa: E402

_INK = "#111111"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--journal", required=True, type=Path)
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--packets", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--peer", default=None)
    ap.add_argument("--out", default="figures/fig_wire.pdf")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_wire_artifacts import (
        _issued_at,
        attribute,
        load_runs,
        peer_totals,
        read_packets,
    )

    report = json.loads(args.report.read_text())
    peer = args.peer or report.get("peer")
    if peer is None:
        totals = peer_totals(args.packets)
        peer = max(totals, key=lambda k: totals[k][1])

    packets = read_packets(args.packets, peer)
    runs = attribute(
        load_runs(args.journal, report.get("condition", "adaptive")),
        packets,
        _issued_at(args.runs),
    )
    runs = [r for r in runs if r["matched"] > 0]
    if not runs:
        raise SystemExit("no attributed runs; cannot draw")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.35, 4.0), dpi=200)

    # ---- panel (a): does the wire track the guest's counters? ---------------
    series = [
        ("request (egress)", "run_app_egress", "run_wire_egress", "o", "white",
         report["correlation"]["run_egress_r"]),
        ("response (ingress)", "run_app_ingress", "run_wire_ingress", "s", "#4a4a4a",
         report["correlation"]["run_ingress_r"]),
    ]
    for label, ka, kw, marker, face, r in series:
        xs = [r_["".join(ka)] / 1000.0 for r_ in runs]
        ys = [r_[kw] / 1000.0 for r_ in runs]
        ax1.scatter(xs, ys, s=13, marker=marker, facecolor=face, edgecolor=_INK,
                    linewidth=0.5, zorder=3, label=f"{label}, $r={r:.3f}$")
    lim = max(max(r_["run_wire_egress"], r_["run_wire_ingress"]) for r_ in runs) / 1000.0
    ax1.plot([0, lim * 1.05], [0, lim * 1.05], ":", color=_INK, linewidth=0.8,
             zorder=2, label="$y=x$ (zero overhead)")
    ax1.set_xlabel("bytes recorded by the guest (kB / run)", color=_INK)
    ax1.set_ylabel("bytes on the wire (kB / run)", color=_INK)
    ax1.set_title("(a) the wire tracks the recorded volume", fontsize=7.5, color=_INK)
    # Lower right: the two clusters occupy the upper left and the middle, and a
    # legend over the response cluster hid the very points it labels.
    ax1.legend(frameon=False, fontsize=6, loc="lower right")
    ax1.grid(color="#d8d8d8", linewidth=0.5, zorder=0)
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False)

    # ---- panel (b): which direction survives the substitution? -------------
    auc = report["single_feature_auc"]
    groups = [
        ("response\n(ingress)", "max app ingress", "max wire ingress"),
        ("request\n(egress)", "max app egress", "max wire egress"),
    ]
    width = 0.34
    # Computed before the loop: the annotation placement below depends on it,
    # and accumulating it inside the loop left it unbound on the first pass.
    null = sum(
        auc[key]["attribute_null"]
        for _, app_key, wire_key in groups
        for key in (app_key, wire_key)
    ) / (2 * len(groups))
    for i, (_, app_key, wire_key) in enumerate(groups):
        a = auc[app_key]["attribute"]
        w = auc[wire_key]["attribute"]
        ax2.bar(i - width / 2, a, width, facecolor="#4a4a4a", edgecolor=_INK,
                linewidth=0.6, zorder=3, label="guest counters" if i == 0 else None)
        ax2.bar(i + width / 2, w, width, facecolor="white", edgecolor=_INK,
                linewidth=0.6, hatch="////", zorder=3,
                label="wire (packets)" if i == 0 else None)
        for x, v, key in ((i - width / 2, a, app_key), (i + width / 2, w, wire_key)):
            sig = auc[key]["attribute_p"] < 0.05
            mark = "*" if sig else "n.s."
            # A bar that stops below the null would put its label on the dashed
            # line, so label it inside the bar instead.
            inside = v < null + 0.03
            ax2.annotate(mark, (x, v), textcoords="offset points",
                         xytext=(0, -8 if inside else 2),
                         va="top" if inside else "baseline",
                         ha="center", fontsize=6.5, color=_INK)
    # The null goes in the legend rather than as floating text: the bars fill
    # the plot from the baseline up, so there is nowhere on the line itself that
    # a label does not sit on top of a bar.
    ax2.axhline(null, color=_INK, linewidth=0.9, linestyle="--", zorder=4,
                label=f"calibrated null ($\\approx{null:.2f}$)")
    ax2.set_xticks(range(len(groups)))
    ax2.set_xticklabels([g[0] for g in groups], fontsize=6.5)
    ax2.set_ylim(0.48, 0.88)
    ax2.set_ylabel("attribute ROC-AUC", color=_INK)
    ax2.set_title("(b) the un-paddable coordinate survives", fontsize=7.5, color=_INK)
    ax2.legend(frameon=False, fontsize=6.0, loc="upper right", ncol=1,
               handlelength=1.6)
    ax2.grid(axis="y", color="#d8d8d8", linewidth=0.5, zorder=0)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", metadata={"CreationDate": None})
    print(f"wrote {out} over {len(runs)} runs (peer {peer})")
    for _, ka, kw, _, _, r in series:
        print(f"  {ka} -> {kw}: r={r:+.3f}")
    for name, app_key, wire_key in groups:
        one = name.replace(chr(10), " ")
        print(f"  {one}: app={auc[app_key]['attribute']:.3f} "
              f"(p={auc[app_key]['attribute_p']:.4f})  "
              f"wire={auc[wire_key]['attribute']:.3f} "
              f"(p={auc[wire_key]['attribute_p']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

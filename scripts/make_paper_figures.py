#!/usr/bin/env python3
"""Generate the manuscript's figures from an experiment journal.

Both figures used to be hand-produced binaries that no script regenerated, and
they had drifted: ``fig_hops.pdf`` plotted 7.00/6.83/4.11 mean research hops from
a pre-publication run while the shipped journal and every macro said 2.93/4.74,
and ``fig_frontier.pdf`` carried a y-axis floor of 0.40 that only makes sense for
a superseded run whose membership AUC was 0.411. Nothing could detect that,
because the tables were gated by CI and the figures were not.

They are now derived from the journal like the tables, so the same
reproducibility gate covers them: a figure is a set of numbers, and every number
in the manuscript has to come from committed evidence.

Determinism: PDF output is byte-stable because the metadata date is pinned and
the font type is fixed, so CI can diff a regenerated figure against the
committed one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_paper_artifacts import build  # noqa: E402

# Byte-stable output: without this matplotlib stamps a creation date.
matplotlib.rcParams["pdf.compression"] = 6
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.hashsalt"] = "traceguard"
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = 9

_INK = "#1a1a1a"
_ROUTINE = "#5b8fa8"
_SENSITIVE = "#a84b3c"
_NULL = "#8a8a8a"


def _style(ax: Any) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_INK)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=_INK, width=0.6, labelsize=8)
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def figure_hops(result: dict[str, Any], out: Path) -> None:
    """Hop-count distribution per framing.

    Plotted as a distribution rather than a mean because the sensitive arm is
    bimodal: its mass sits at the floor and at the public cap with nothing
    between, so the mean names a depth no individual run takes.
    """

    dist = result["hop_distributions"]
    depths = sorted({int(k) for entry in dist.values() for k in entry["histogram"]})
    fig, ax = plt.subplots(figsize=(3.4, 2.2), dpi=200)
    width = 0.34
    # A true numeric x-axis, not categorical. With categories the 4->7 gap
    # collapses to one tick and the mean markers land in visually wrong places;
    # on a numeric axis the empty span between the sensitive arm's two modes is
    # visible, which is the whole point of the figure.
    peak = 0
    for offset, (name, colour) in enumerate(
        (("routine", _ROUTINE), ("sensitive", _SENSITIVE))
    ):
        entry = dist[name]
        counts = [entry["histogram"].get(str(d), 0) for d in depths]
        peak = max(peak, max(counts) if counts else 0)
        ax.bar([d + (offset - 0.5) * width for d in depths], counts, width=width,
               color=colour, zorder=3, label=f"{name} (mean {entry['mean']:.1f})")
        if entry["mean"] is not None:
            ax.axvline(entry["mean"], color=colour, linestyle=":", linewidth=1.0, zorder=4)
    ax.set_xticks(depths)
    ax.set_xticklabels([str(d) for d in depths])
    ax.set_xlim(min(depths) - 0.7, max(depths) + 0.7)
    # Headroom so the legend never sits on a bar or a mean line.
    ax.set_ylim(0, peak * 1.32)
    ax.set_xlabel("clinical-extraction passes", color=_INK)
    ax.set_ylabel("runs", color=_INK)
    ax.legend(frameon=False, fontsize=7, loc="upper left", handlelength=1.2)
    _style(ax)
    fig.tight_layout(pad=0.4)
    fig.savefig(out, format="pdf", metadata={"CreationDate": None})
    plt.close(fig)


def figure_frontier(result: dict[str, Any], out: Path) -> None:
    """Attack AUC against the per-request budget.

    The reference line is the *calibrated* null for the undefended arm, not 0.5:
    the protocol's floor is about 0.57, so drawing a 0.5 "chance" line would
    invite the reader to over-read every point above it.
    """

    rows = result["frontier"]["attribute"]
    eps = [row["epsilon"] for row in rows]
    attr = [row["auc_mean"] for row in rows]
    memb_rows = {
        round(row["epsilon"], 4): row["auc_mean"] for row in result["frontier"]["membership"]
    }
    memb = [memb_rows.get(round(e, 4)) for e in eps]

    fig, ax = plt.subplots(figsize=(3.5, 2.2), dpi=200)
    x = range(len(eps))
    ax.plot(x, attr, marker="o", markersize=3.2, color=_SENSITIVE, linewidth=1.2,
            zorder=3, label="attribute")
    if all(value is not None for value in memb):
        ax.plot(x, memb, marker="s", markersize=3.0, color=_ROUTINE, linewidth=1.2,
                zorder=3, label="membership")
    null = result.get("nulls", {}).get("adaptive_attribute", {}).get("null_mean")
    if null:
        ax.axhline(null, color=_NULL, linestyle="--", linewidth=0.9, zorder=2,
                   label=f"calibrated null ({null:.2f})")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{e:g}" for e in eps])
    ax.set_xlabel(r"per-request budget $\epsilon$ (smaller = more private)", color=_INK)
    ax.set_ylabel("attack ROC-AUC", color=_INK)
    # Lower right: the curves rise left to right and all of them sit above
    # 0.57 across the right half, so this is the one corner of the panel
    # with no data in it. "center left" put the legend under the steepest
    # part of both curves.
    ax.legend(frameon=False, fontsize=7, loc="lower right",
              handlelength=1.4, handletextpad=0.5, labelspacing=0.35,
              borderaxespad=0.3)
    ax.set_xlim(-0.4, len(eps) - 0.6)
    _style(ax)
    fig.tight_layout(pad=0.5)
    fig.savefig(out, format="pdf", metadata={"CreationDate": None})
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--journal", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    ap.add_argument("--seed", type=int, default=20260710)
    ap.add_argument("--nulls", type=Path, default=Path("tables/nulls.json"))
    args = ap.parse_args(argv)

    result = build(args.journal, args.seed, bootstrap=0, n_perm=0, nulls_path=args.nulls)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure_hops(result, args.out_dir / "fig_hops.pdf")
    figure_frontier(result, args.out_dir / "fig_frontier.pdf")
    print(f"wrote fig_hops.pdf, fig_frontier.pdf -> {args.out_dir}")
    print(f"provenance tier: {result['tier']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

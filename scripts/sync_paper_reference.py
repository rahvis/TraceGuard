#!/usr/bin/env python3
"""Sync the numeric half of paper-reported-unverified.json from the built artifacts.

That file is the machine-readable record of what the manuscript claims, and it
is hash-anchored to the PDF and LaTeX. Its numbers were hand-transcribed, which
is a step that can be got wrong silently and has already cost four re-anchors.
Reading them from the same ``tables/macros.tex`` and ``tables/nulls.json`` the
PDF is typeset from removes the transcription entirely: the anchor still detects
source/PDF drift, which is what it is for, and the numbers can no longer
disagree with the paper by accident.

Prose fields -- ``evidence_status``, ``known_limitations``, the ``*_history``
supersession reasons -- stay hand-written, because they are judgements and not
transcriptions. This tool never touches them.

Usage:
    uv run python scripts/sync_paper_reference.py            # report the drift
    uv run python scripts/sync_paper_reference.py --write    # apply it
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "src/traceguard/data/paper-reported-unverified.json"
_MACRO = re.compile(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}$")


def read_macros(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _MACRO.match(line.strip())
        if match:
            out[match.group(1)] = match.group(2)
    return out


def _num(macros: dict[str, str], name: str) -> float | None:
    """A macro as a number, or None. Macros carry things like '=0.521' and '+9.8'."""

    raw = macros.get(name)
    if raw is None:
        return None
    cleaned = raw.lstrip("=<>+").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


_UTIL_ROW = re.compile(
    r"^(Baseline \(adaptive\)|Enforced \(canonical\)|Enforced \+ full pad)\s*&"
    r"\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*\\\\"
)
_UTIL_KEY = {
    "Baseline (adaptive)": "adaptive",
    "Enforced (canonical)": "structure_only",
    "Enforced + full pad": "full_pad",
}


_UTIL_DIFF = re.compile(r"difference ([\d.]+) \((?:not )?significant")
_UTIL_FULL_DIFF = re.compile(r"Full pad vs baseline: difference ([\d.]+)")


def read_utility_diffs(path: Path) -> dict[str, float]:
    """The differences the table prints, not a subtraction of its rounded rows.

    ``0.660 - 0.636`` is 0.024; the table, which subtracts before rounding,
    prints 0.023. Recomputing here would put a number in the reference file that
    disagrees with the paper by a digit for no reason.
    """

    out: dict[str, float] = {}
    if not path.is_file():
        return out
    text = path.read_text(encoding="utf-8")
    full = _UTIL_FULL_DIFF.search(text)
    if full:
        out["full_pad"] = float(full.group(1))
    for match in _UTIL_DIFF.finditer(text):
        value = float(match.group(1))
        if value != out.get("full_pad"):
            out.setdefault("structure_only", value)
    return out


def read_utility_table(path: Path) -> dict[str, dict[str, float]]:
    """Per-condition scores from the generated utility table.

    The table is the only place the three-decimal values exist -- the \\util*
    macros round to two, which is the precision the prose quotes. Reading the
    table keeps the reference file at the table's precision without a second
    hand-transcription of the same run.
    """

    out: dict[str, dict[str, float]] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _UTIL_ROW.match(line.strip())
        if match:
            out[_UTIL_KEY[match.group(1)]] = {
                "overall": float(match.group(2)),
                "faithfulness": float(match.group(3)),
                "completeness": float(match.group(4)),
            }
    return out


_ATTR_ROW = re.compile(r"^([A-Za-z][A-Za-z ()]*?)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*\\\\")


def read_single_feature_table(path: Path) -> dict[str, dict[str, float]]:
    r"""Per-coordinate single-feature AUCs from the generated attribution table.

    This block was hand-maintained and drifted furthest of anything in the
    reference file: it still named ``total_timing`` and ``total_egress_size``,
    features the analysis deliberately stopped computing -- carrying a sum
    beside a mean hands the attacker ``sum/mean == n`` -- so it recorded
    quantities that no longer exist, at values from a superseded journal.
    Deriving it from the table the paper typesets makes that impossible.
    """

    out: dict[str, dict[str, float]] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _ATTR_ROW.match(line.strip())
        if not match:
            continue
        label = match.group(1).strip()
        if label.lower().startswith("single trace feature"):
            continue
        slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        out[slug] = {
            "label": label,
            "membership": float(match.group(2)),
            "attribute": float(match.group(3)),
        }
    return out


def build_numeric(
    macros: dict[str, str],
    nulls: dict[str, Any],
    utility: dict[str, dict[str, float]] | None = None,
    diffs: dict[str, float] | None = None,
    single_feature: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    def n(name: str) -> float | None:
        return _num(macros, name)

    section: dict[str, Any] = {
        "design": {
            "topics_per_service": 2,
            "documents_per_case": 6,
            "cases": int(n("nCases") or 0),
            "topic_groups": int(n("nTopicGroups") or 0),
            "sensitivity_levels": int(n("nFramings") or 0),
            "baseline_runs_reported": int(n("nBaseline") or 0),
            "shielded_runs_reported": int(n("nShielded") or 0),
            "receipts_reported": int(n("nReceipts") or 0),
        },
        "headline": {
            "adaptive": {
                "membership_auc": n("memberAUC"),
                "attribute_auc": n("attrAUC"),
            },
            "structure_only": {
                "membership_auc": n("residMemberAUC"),
                "attribute_auc": n("residAttrAUC"),
            },
            "full_pad": {
                "membership_auc": n("enfMemberAUC"),
                "attribute_auc": n("enfAttrAUC"),
            },
            "controls": {
                "permutation_auc": n("permAUC"),
                "shape_fixed_auc": n("shapeFixedAUC"),
            },
        },
        # The same quantity as `headline` plus its spread. It was hand-kept and
        # went stale independently of the block it duplicates, which is the
        # argument for generating it rather than the argument for deleting it:
        # the artifact checks against these fields by name.
        "multi_split": {
            "membership_auc_mean": n("memberAUC"),
            "membership_auc_std": n("memberMSstd"),
            "attribute_auc_mean": n("attrAUC"),
            "attribute_auc_std": n("attrMSstd"),
            "membership_auc_pooled": n("memberAUCPooled"),
            "attribute_auc_pooled": n("attrAUCPooled"),
        },
        # Packet-level cross-check of the observable, which the body's AUCs are
        # NOT: those come from the guest runtime's own byte counters. Recorded
        # separately for exactly that reason.
        "wire_measurement": {
            "runs": int(n("wireRuns") or 0),
            "segments": int(n("wireSegments") or 0),
            "steps_with_packets": int(n("wireStepsMatched") or 0),
            "steps_total": int(n("wireStepsTotal") or 0),
            "run_level_correlation": {
                "egress_r": n("wireEgressR"), "ingress_r": n("wireIngressR"),
            },
            "wire_over_app_bytes": {
                "egress": n("wireEgressOverhead"), "ingress": n("wireIngressOverhead"),
            },
            "attribute_auc": {
                "wire_ingress": n("wireIngressAttr"),
                "wire_ingress_null": n("wireIngressAttrNull"),
                "wire_ingress_p": macros.get("wireIngressAttrP"),
                "wire_egress": n("wireEgressAttr"),
                "wire_egress_null": n("wireEgressAttrNull"),
                "wire_egress_p": macros.get("wireEgressAttrP"),
                "app_ingress": n("wireAppIngressAttr"),
                "app_egress": n("wireAppEgressAttr"),
            },
        },
        "latency_seconds": {
            "adaptive_observed": n("latBase"),
            "full_pad_observed": n("latFull"),
            "full_pad_multiplier": n("padRatio"),
            "step_deadline_s": n("stepDeadline"),
            "structure_pct_change": n("latStructPct"),
        },
        "membership_mechanism": {
            "egress_delta_bytes": n("memberEgressDelta"),
            "egress_delta_pct": n("memberEgressPct"),
            "wall_delta_s": n("memberWallDelta"),
            "depth_delta_passes": n("memberDepthDelta"),
            "phi_with_attribute": n("memberAttrPhi"),
            "size_only_auc": n("memberFeatSize"),
            "timing_only_auc": n("memberFeatTiming"),
            "structure_only_auc": n("memberFeatStructure"),
        },
        "depth_channel": {
            "cap": int(n("hopCap") or 0),
            "share_at_cap_pct": n("hopSaturatedPct"),
            "shape": macros.get("hopShapeShort"),
        },
        "frontier": {
            "regime": macros.get("frontierRegime"),
            "top_band_share_pct": n("frontierTopBandShare"),
            "bands_occupied": int(n("frontierBandsOccupied") or 0),
        },
        "crew": {
            "models": macros.get("crewModel"),
            "platform": macros.get("crewApi"),
            "judge": macros.get("judgeModel"),
        },
        "historical_comparison": {
            key[4].lower() + key[5:]: macros[key]
            for key in sorted(macros)
            if key.startswith("hist")
        },
    }
    if single_feature:
        section["single_feature_auc"] = single_feature

    # The utility block was hand-maintained and went stale twice: it still
    # carried a superseded run's structure-only scores, and its prose still said
    # the full-pad arm was unmeasured after we had measured it. The numbers are
    # generated here for the same reason every other number in this file is; the
    # `interpretation` prose is a judgement and stays hand-written.
    if utility:
        util: dict[str, Any] = {
            "pairs": int(n("utilPairs") or 0),
            "judge": macros.get("judgeModel"),
        }
        util.update(utility)
        if "structure_only" in (diffs or {}):
            util["difference"] = diffs["structure_only"]
        util["reported_p_value"] = macros.get("utilP")
        if "full_pad" in utility:
            util["full_pad_pairs"] = int(n("utilFullPairs") or 0)
            if "full_pad" in (diffs or {}):
                util["full_pad_difference"] = diffs["full_pad"]
            util["full_pad_reported_p_value"] = macros.get("utilFullP")
            util["full_pad_fail_closed_pct"] = n("utilFullFailClosedPct")
            util["full_pad_fail_closed_ci_pct"] = macros.get("utilFullFailClosedCI")
            util["full_pad_overall_no_step_lost"] = n("utilFullIntact")
            util["full_pad_overall_one_step_lost"] = n("utilFullLost")
        section["utility"] = util

    if nulls:
        section["calibrated_nulls_measured"] = {
            arm: {
                # The cache calls it observed_auc; reading "auc" here silently
                # recorded null for every arm.
                "observed": report.get("observed_auc"),
                "null_mean": report.get("null_mean"),
                "p_value": report.get("p_value"),
            }
            for arm, report in sorted(nulls.items())
        }
    return section


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--macros", type=Path, default=ROOT / "tables/macros.tex")
    ap.add_argument("--nulls", type=Path, default=ROOT / "tables/nulls.json")
    ap.add_argument("--deadline-macros", type=Path,
                    default=ROOT / "tables/macros_deadline.tex")
    ap.add_argument("--wire-macros", type=Path, default=ROOT / "tables/macros_wire.tex")
    ap.add_argument("--utility-table", type=Path, default=ROOT / "tables/tab_utility.tex")
    ap.add_argument(
        "--attribution-table", type=Path, default=ROOT / "tables/tab_attribution.tex"
    )
    ap.add_argument("--reference", type=Path, default=REFERENCE)
    ap.add_argument("--write", action="store_true", help="apply the update")
    ap.add_argument(
        "--reanchor",
        action="store_true",
        help="also re-anchor the PDF/LaTeX digests (requires --reason)",
    )
    ap.add_argument("--reason", default=None, help="why the previous anchor was superseded")
    ap.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help=(
            "directory the source.pdf/source.tex paths resolve against "
            "(defaults to the repository root; exists so the anchoring logic is "
            "testable without a built PDF present)"
        ),
    )
    args = ap.parse_args(argv)

    macros = read_macros(args.macros)
    # The wire-level cross-check writes its own macro file; the reference must
    # record it too, since it is a distinct claim class (packet-level rather
    # than application-level observables) that a reader will want to check.
    for extra in (args.deadline_macros, args.wire_macros):
        if extra and extra.is_file():
            macros.update(read_macros(extra))
    utility = read_utility_table(args.utility_table)
    diffs = read_utility_diffs(args.utility_table)
    single_feature = read_single_feature_table(args.attribution_table)
    nulls_payload = (
        json.loads(args.nulls.read_text(encoding="utf-8")) if args.nulls.is_file() else {}
    )
    document = json.loads(args.reference.read_text(encoding="utf-8"))
    numeric = build_numeric(
        macros, nulls_payload.get("nulls", {}), utility, diffs, single_feature
    )

    # Prose lives inside some of these sections (e.g. /utility/interpretation).
    # The assignment below replaces a section wholesale, so carry any
    # hand-written key the generated section does not define -- otherwise this
    # tool would delete exactly the judgements it promises not to touch.
    PROSE_KEYS = ("interpretation", "note", "caveat", "known_gap")
    changed: list[str] = []
    for key, value in numeric.items():
        existing = document.get(key)
        if isinstance(value, dict) and isinstance(existing, dict):
            for prose in PROSE_KEYS:
                if prose in existing and prose not in value:
                    value[prose] = existing[prose]
        if existing != value:
            changed.append(key)
        document[key] = value

    if args.reanchor:
        if not args.reason:
            raise SystemExit("--reanchor requires --reason: a supersession must say why")
        source = document["source"]
        for kind in ("pdf", "tex"):
            path = args.root / source[kind]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != source[f"{kind}_sha256"]:
                source.setdefault(f"{kind}_sha256_history", []).append(
                    {
                        "sha256": source[f"{kind}_sha256"],
                        "superseded_because": args.reason
                        if kind == "pdf"
                        else "Same revision as the corresponding pdf_sha256_history entry.",
                    }
                )
                source[f"{kind}_sha256"] = digest
                changed.append(f"source.{kind}_sha256")

    if not changed:
        print("paper reference is already in sync with the built artifacts")
        return 0
    print(f"{'updating' if args.write else 'would update'}: {', '.join(changed)}")
    if not args.write:
        print("(re-run with --write to apply)")
        return 1
    args.reference.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.reference}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

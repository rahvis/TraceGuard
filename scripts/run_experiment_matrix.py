#!/usr/bin/env python3
"""Run the reviewer-requested experimental matrix as a star design.

Crossing every factor would be thousands of cells. A generality claim does not
need a full factorial: it needs to know whether the channel survives when one
thing at a time moves away from the configuration that was measured. So this is
a *star* design -- one baseline arm, and one spoke per factor varying only that
factor, every spoke sharing the baseline's corpus and seed so each comparison
is against a measured reference rather than against another spoke.

Ordering is deliberate. The baseline runs first because everything is compared
to it, and the crew-model spoke runs last because it is the most expensive and
therefore the one to trim if the budget runs short. Each spoke journals
separately, so a trimmed run still yields complete arms rather than a partial
matrix.

Cost control: --dry-run prints the plan and the cell count without spending
anything, and every arm resumes from its journal, so an interrupted run
continues rather than restarting.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traceguard.config import Settings  # noqa: E402
from traceguard.corpus import CorpusCatalog  # noqa: E402
from traceguard.experiment import (  # noqa: E402
    Arm,
    ExperimentConfig,
    ExperimentRunner,
    catalog_cases,
)
from traceguard.sdk import TraceGuardSDK  # noqa: E402
from traceguard.storage import ArtifactStore  # noqa: E402

# The judge deployment is reserved: it scores the crew's answers and must never
# be one of the models under test, or the "independent, stronger judge" claim
# inverts. scripts/judge_utility.py enforces this at run time too.
JUDGE_DEPLOYMENT = "gpt-5.6-terra"
CREW_DEPLOYMENTS = ("gpt-4o", "gpt-5.6-luna", "gpt-5.6-sol")

BASELINE = Arm(condition="adaptive", autonomy="bounded", architecture="pipeline_crew")


def spokes() -> dict[str, tuple[Arm, ...]]:
    """The star: a baseline arm plus one spoke per factor."""

    return {
        # Answers the graded-sensitivity ask, and re-anchors every existing
        # number on the current provider. Includes the defended conditions
        # because the defense claim has to be re-measured too.
        "baseline": (
            BASELINE,
            Arm(condition="structure", autonomy="scripted"),
            Arm(condition="full", autonomy="scripted"),
        ),
        # Leakage versus how much control flow depends on the data. Only the
        # rungs the baseline does not already contain are listed: the scripted
        # rung is the structure arm and the bounded rung IS the baseline, so
        # running them again would pay twice for the same cells. Comparisons are
        # made against the baseline journal, not against a re-run.
        "autonomy": (Arm(condition="adaptive", autonomy="free"),),
        # Leakage versus crew topology.
        "architecture": tuple(
            Arm(condition="adaptive", autonomy="bounded", architecture=name)
            for name in ("hierarchical_supervisor", "react_single_agent")
        ),
        # Leakage versus model. Three points cannot support a scaling law and
        # the paper says so; this establishes an ordering, not a law. Each arm
        # pins all three roles to one deployment so the axis is the model and
        # not the role mix.
        "model": tuple(
            Arm(
                condition="adaptive",
                autonomy="bounded",
                models=(("fast", name), ("deep", name), ("review", name)),
            )
            for name in CREW_DEPLOYMENTS
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--seed", type=int, default=20260710)
    ap.add_argument("--reps", type=int, default=2, help="repetitions per cell")
    ap.add_argument("--baseline-reps", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--provider", default="azure")
    ap.add_argument(
        "--only",
        action="append",
        choices=sorted(spokes()),
        help="run only these spokes (repeatable); default runs all, baseline first",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the plan, spend nothing")
    args = ap.parse_args(argv)

    settings = Settings.from_env()
    if args.provider == "azure" and JUDGE_DEPLOYMENT in {
        settings.model_fast,
        settings.model_deep,
        settings.model_review,
    }:
        raise SystemExit(
            f"{JUDGE_DEPLOYMENT} is reserved for the independent utility judge and must "
            "not be a crew deployment; repoint MODEL_FAST/DEEP/REVIEW"
        )

    catalog = CorpusCatalog.load_default()
    n_cases = len(catalog_cases(catalog))
    plan = spokes()
    selected = args.only or ["baseline", "autonomy", "architecture", "model"]

    # Every paid cell should be paid for once. An arm appearing in two spokes
    # would be re-run under a second journal, so overlap is an error rather
    # than a redundancy to tolerate.
    seen: dict[str, str] = {}
    for name, arms in plan.items():
        for arm in arms:
            if arm.slug in seen and seen[arm.slug] != name:
                raise SystemExit(
                    f"arm {arm.slug} appears in both {seen[arm.slug]!r} and {name!r}; "
                    "spokes must not overlap or the same cells are paid for twice"
                )
            seen[arm.slug] = name

    total = 0
    print(f"corpus: {n_cases} cases\n")
    for name in selected:
        arms = plan[name]
        reps = args.baseline_reps if name == "baseline" else args.reps
        cells = n_cases * len(arms) * reps
        total += cells
        print(f"{name:14s} arms={len(arms)} reps={reps:2d} cells={cells:5d}")
        for arm in arms:
            print(f"                 {arm.slug}")
    print(f"\ntotal cells: {total}")
    if args.dry_run:
        print("\n--dry-run: nothing was executed and nothing was spent.")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {}
    for name in selected:
        arms = plan[name]
        reps = args.baseline_reps if name == "baseline" else args.reps
        journal = args.out_dir / f"matrix-{name}.jsonl"
        print(f"\n=== {name}: {len(arms)} arms x {n_cases} cases x {reps} reps -> {journal.name}")
        store = ArtifactStore(args.out_dir / "runs")

        # An arm that names crew models needs its OWN runner: the SDK holds one
        # Settings for its lifetime, so a single runner cannot sweep models.
        # Grouping by the arm's model tuple and building one runner per group is
        # what makes the model axis real -- without it every arm silently ran
        # the ambient crew, which is exactly what happened on the first attempt.
        groups: dict[tuple[tuple[str, str], ...], list[Arm]] = {}
        for arm in arms:
            groups.setdefault(arm.models, []).append(arm)

        totals = {"planned": 0, "executed": 0, "resumed": 0, "failed": 0}
        for models, group in groups.items():
            arm_settings = settings
            if models:
                overrides = dict(models)
                arm_settings = replace(
                    settings,
                    model_fast=overrides.get("fast", settings.model_fast),
                    model_deep=overrides.get("deep", settings.model_deep),
                    model_review=overrides.get("review", settings.model_review),
                )
                print(
                    f"    crew override: fast={arm_settings.model_fast} "
                    f"deep={arm_settings.model_deep} review={arm_settings.model_review}"
                )
            sdk = TraceGuardSDK(arm_settings, provider=args.provider, catalog=catalog)
            runner = ExperimentRunner(sdk, catalog, store)
            config = ExperimentConfig(
                repetitions=reps,
                arms=tuple(group),
                seed=args.seed,
                experiment_id=f"matrix-{name}",
                journal_path=journal,
                resume=True,
                # The graded corpus is not the 3x2x2x2 shape the original
                # validator asserted, and the validator now derives the expected
                # shape from the data, so this stays on.
                require_balanced=True,
                provider=args.provider,
                workers=args.workers,
                provenance={"matrix_spoke": name, "judge_reserved": JUDGE_DEPLOYMENT},
            )
            report = runner.run(config)
            for key, field in (
                ("planned", "planned_runs"), ("executed", "executed_runs"),
                ("resumed", "resumed_runs"), ("failed", "failed_runs"),
            ):
                totals[key] += int(report.get(field) or 0)

        summary[name] = {"journal": str(journal), **totals}
        print(json.dumps(summary[name], indent=2))

    (args.out_dir / "matrix-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    failed = sum(int(v.get("failed") or 0) for v in summary.values())  # type: ignore[union-attr]
    if failed:
        print(f"\n{failed} cells failed; re-run to retry only those.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

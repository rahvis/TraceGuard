#!/usr/bin/env python3
"""Judge answer utility for the paper's utility table and \\util* macros.

For each catalog case x repetition x condition (adaptive vs structure-only) the
crew is run in-process and its determination is scored immediately by an LLM
judge against the case's query and source documents.  Determinations and
document text exist only in process memory: the scores JSONL and the generated
LaTeX carry numeric scores and identifiers, never answer or document text
(the same boundary :mod:`traceguard.storage` enforces for run artifacts).

Usage:
    uv run python scripts/judge_utility.py \\
        --scores artifacts/utility-scores.jsonl --journal artifacts/<run>.jsonl

The scores file is append-only and resumable: already-scored
(case, condition, rep) cells are skipped on rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from make_paper_artifacts import (  # noqa: E402
    _mc,
    _wrap,
    condition_latency_means,
    merge_macros,
)

from traceguard.config import Settings  # noqa: E402
from traceguard.corpus import CorpusCatalog  # noqa: E402
from traceguard.experiment import (  # noqa: E402
    DEFAULT_AUTONOMY_FOR_CONDITION,
    Arm,
    ExperimentRunner,
    catalog_cases,
    load_experiment_records,
    normalize_condition,
)
from traceguard.sdk import TraceGuardSDK  # noqa: E402
from traceguard.types import MedicalCase  # noqa: E402

CONDITIONS = ("adaptive", "structure_only", "full_pad")
# The arms the headline contrast cannot do without. full_pad is optional so a
# scores file written before it existed still analyses.
REQUIRED_CONDITIONS = ("adaptive", "structure_only")
CONDITION_LABELS = {
    "adaptive": "Baseline (adaptive)",
    "structure_only": "Enforced (canonical)",
    "full_pad": "Enforced + full pad",
}
SCORE_KEYS = ("overall", "faithful", "complete")
_ID_SAFE = re.compile(r"[^A-Za-z0-9._-]+")

# One judge prompt per task family, length-matched so the judge's own input
# size does not vary by family. The rubric sentence is byte-identical across
# families; only the noun for the artifact being graded changes.
#
# Honest scope, and the paper says so: utility is comparable *within* a family
# (adaptive vs enforced) and NOT across families -- different judge prompt,
# different task. Only the direction and significance of the within-family
# contrast transfer.
_JUDGE_SYSTEM_BY_DOMAIN = {
    "prior-authorization": (
        "You grade a synthetic prior-authorization determination draft against its "
        "case query and source documents. Respond with a JSON object only: "
        '{"overall": 1-3, "faithful": 1-3, "complete": 1-3}, where 3 means fully '
        "useful/faithful/complete, 2 minor issues, and 1 major issues."
    ),
    "aml-alert-triage": (
        "You grade a synthetic monitoring-alert disposition draft against its "
        "case query and source documents. Respond with a JSON object only: "
        '{"overall": 1-3, "faithful": 1-3, "complete": 1-3}, where 3 means fully '
        "useful/faithful/complete, 2 minor issues, and 1 major issues."
    ),
}
_JUDGE_SYSTEM = _JUDGE_SYSTEM_BY_DOMAIN["prior-authorization"]


def _judge_system_for(case: MedicalCase) -> str:
    return _JUDGE_SYSTEM_BY_DOMAIN.get(
        getattr(case, "domain", None), _JUDGE_SYSTEM
    )


def _judge_prompt(case: MedicalCase, answer: str) -> str:
    documents = "\n\n".join(
        f"[{document.document_id}] {document.title}\n{document.text}"
        for document in case.documents
    )
    return (
        f"QUERY:\n{case.query}\n\nSOURCE DOCUMENTS:\n{documents}\n\n"
        f"CANDIDATE ANSWER:\n{answer}\n"
    )


def parse_judge_scores(text: str, *, finish_reason: str | None = None) -> dict[str, int]:
    """Parse the judge reply strictly, and say *why* when it cannot be parsed.

    The reply itself is never echoed into an error message -- it is derived from
    case documents. But the previous message, "judge returned non-JSON output",
    was unactionable, and it hid a specific and expensive failure: at
    max_completion_tokens=64 this deployment returns finish_reason='length' with
    *empty* content, having spent the whole budget before emitting any. Fifteen
    paid crew runs were burned before that was diagnosed. Truncation now names
    itself.
    """

    if finish_reason == "length":
        raise ValueError(
            "judge reply was truncated (finish_reason='length'); it returned "
            f"{len(text)} characters. Raise max_completion_tokens -- an empty "
            "content field here means the budget was exhausted before any "
            "output, not that the model declined to answer"
        )
    if not text.strip():
        raise ValueError(
            f"judge returned empty content (finish_reason={finish_reason!r})"
        )
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"judge returned non-JSON output ({len(text)} chars, "
            f"finish_reason={finish_reason!r})"
        ) from exc
    if not isinstance(value, Mapping):
        raise ValueError("judge returned a non-object JSON payload")
    scores: dict[str, int] = {}
    for key in SCORE_KEYS:
        item = value.get(key)
        valid = (
            not isinstance(item, bool)
            and isinstance(item, (int, float))
            and float(item).is_integer()
            and 1 <= item <= 3
        )
        if not valid:
            raise ValueError(f"judge score {key!r} is missing or out of range")
        scores[key] = int(item)
    return scores


class AzureOpenAIJudge:
    """JSON-only utility judge over one Azure OpenAI deployment.

    ``model`` is an Azure deployment name. The judge is deliberately a separate
    channel from the crew's models so the two can never be accidentally shared;
    ``assert_judge_independence`` enforces that.
    """

    def __init__(self, settings: Settings, model: str) -> None:
        if not settings.azure_openai_api_key:
            raise SystemExit("AZURE_OPENAI_API_KEY is required unless --dry-run is set")
        if not settings.azure_openai_endpoint:
            raise SystemExit("AZURE_OPENAI_ENDPOINT is required unless --dry-run is set")
        from openai import AzureOpenAI

        # The key is held only by the official client and never re-exposed.
        # max_retries=0 matches the crew adapter: a retry would distort the
        # judged cell's timing, which is recorded alongside its score.
        self._client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
            timeout=settings.provider_timeout_s,
            max_retries=0,
        )
        self.model = model

    def __call__(self, case: MedicalCase, answer: str, seed: int) -> dict[str, int]:
        completion = self._client.chat.completions.create(
            model=self.model,
            # Measured, then measured again. This deployment spends ~145
            # completion tokens on a ~2.6k-character judging prompt, and at 64
            # it returns finish_reason='length' with empty content. 512 still
            # truncated 7 of the first 155 cells, because a longer
            # determination costs proportionally more, so the budget is 1024 --
            # the reply is three integers either way, and the cost of being
            # generous here is nothing against the cost of a lost paid cell.
            max_completion_tokens=1024,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _judge_system_for(case)},
                {"role": "user", "content": _judge_prompt(case, answer)},
            ],
            seed=seed,
        )
        choices = getattr(completion, "choices", None) or []
        choice = choices[0] if choices else None
        message = getattr(choice, "message", None) if choice else None
        return parse_judge_scores(
            str(getattr(message, "content", "") or ""),
            finish_reason=getattr(choice, "finish_reason", None),
        )


def assert_judge_independence(judge_model: str, settings: Settings) -> None:
    """Refuse to score a crew with one of its own models.

    The manuscript claims utility is scored by "an independent, stronger judge
    model ... distinct from the crew's", but nothing in the code enforced it. A
    stale judge default silently inverts that claim the moment the crew is
    pointed at a newer deployment, and the resulting numbers look normal. So it
    is checked rather than documented.
    """

    crew = {
        name
        for name in (settings.model_fast, settings.model_deep, settings.model_review)
        if name
    }
    if judge_model in crew:
        raise SystemExit(
            f"judge model {judge_model!r} is also a crew model ({', '.join(sorted(crew))}); "
            "the utility comparison requires an independent judge. Pass a different "
            "--judge-model or repoint the crew deployments."
        )


def stub_judge(case: MedicalCase, answer: str, seed: int) -> dict[str, int]:
    """Deterministic offline judge used by --dry-run and the tests."""

    digest = hashlib.sha256(f"stub-judge|{seed}".encode()).digest()
    return {key: 2 + digest[index] % 2 for index, key in enumerate(SCORE_KEYS)}


def load_scored_cells(path: Path) -> set[tuple[str, str, int]]:
    done: set[tuple[str, str, int]] = set()
    if not path.is_file():
        return done
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid scores JSONL at line {line_number}") from exc
            done.add((str(row["case_id"]), str(row["condition"]), int(row["rep"])))
    return done


def _normalized(score: int | float) -> float:
    # The 1-3 judge scale mapped onto [0, 1] as reported by the paper's table.
    return (float(score) - 1.0) / 2.0


def _fmt_p(p: float) -> str:
    """Format a p-value without inventing or destroying precision.

    ``f"{p:.2f}"`` renders a real 0.003 as "0.00", which reads as an exact zero
    the test never produced. Two decimals are right for the non-significant
    arm, where the reader only needs "nowhere near 0.05"; a small p needs enough
    digits to be a number, and below the third decimal an inequality is the
    honest form.
    """
    if p >= 0.01:
        return f"{p:.2f}"
    if p >= 0.001:
        return f"{p:.3f}"
    return "<0.001"


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-condition normalized means plus a paired Wilcoxon test on overall."""

    by_condition: dict[str, dict[tuple[str, int], Mapping[str, Any]]] = {
        condition: {} for condition in CONDITIONS
    }
    for record in records:
        condition = str(record.get("condition"))
        if condition in by_condition:
            key = (str(record["case_id"]), int(record["rep"]))
            by_condition[condition][key] = record["scores"]
    means: dict[str, dict[str, float]] = {}
    for condition, cells in by_condition.items():
        if not cells:
            # The full-pad arm was added after the first judged study, so a
            # scores file predating it is still valid input: the arm degrades
            # to absent rather than crashing the paper build. The two arms the
            # headline contrast needs are still required.
            if condition in REQUIRED_CONDITIONS:
                raise SystemExit(f"no scores recorded for condition {condition!r}")
            continue
        means[condition] = {
            key: sum(_normalized(scores[key]) for scores in cells.values()) / len(cells)
            for key in SCORE_KEYS
        }
    paired = sorted(set(by_condition["adaptive"]) & set(by_condition["structure_only"]))
    if not paired:
        raise SystemExit("no paired (case, rep) cells across conditions")
    base = [_normalized(by_condition["adaptive"][key]["overall"]) for key in paired]
    shield = [_normalized(by_condition["structure_only"][key]["overall"]) for key in paired]
    if base == shield:
        # All paired differences are zero; Wilcoxon is undefined and there is
        # no evidence of a difference.
        p_value = 1.0
    else:
        from scipy.stats import wilcoxon

        p_value = float(wilcoxon(base, shield).pvalue)
    out = {
        "means": means,
        "difference": means["adaptive"]["overall"] - means["structure_only"]["overall"],
        "p_overall": p_value,
        "pairs": len(paired),
    }

    # The full-pad arm is the one the (0,0) claim rests on, so its utility cost
    # gets its own paired contrast against the same baseline. Padding a step
    # that returns within the deadline is bit-for-bit invisible to the answer
    # -- the runtime sleeps and discards the frame -- so the entire cost is
    # mediated by the fail-closed rate, and both are reported.
    paired_full = sorted(set(by_condition["adaptive"]) & set(by_condition["full_pad"]))
    if "full_pad" not in means:
        paired_full = []
    if paired_full:
        base_f = [
            _normalized(by_condition["adaptive"][key]["overall"]) for key in paired_full
        ]
        full_f = [
            _normalized(by_condition["full_pad"][key]["overall"]) for key in paired_full
        ]
        if base_f == full_f:
            p_full = 1.0
        else:
            from scipy.stats import wilcoxon

            p_full = float(wilcoxon(base_f, full_f).pvalue)
        out["difference_full"] = (
            means["adaptive"]["overall"] - means["full_pad"]["overall"]
        )
        out["p_full"] = p_full
        out["pairs_full"] = len(paired_full)
        fc = [
            bool(r.get("fail_closed"))
            for r in records
            if str(r.get("condition")) == "full_pad"
        ]
        if fc:
            out["full_fail_closed_rate"] = sum(fc) / len(fc)
            out["full_fail_closed_n"] = sum(fc)
            out["full_n"] = len(fc)
            # The conditional loss: mean utility among runs that lost a step
            # versus runs that did not. This is the quantity the rate
            # multiplies, and reporting the product alone would hide it.
            intact, lost = [], []
            for r in records:
                if str(r.get("condition")) != "full_pad":
                    continue
                value = _normalized(r["scores"]["overall"])
                (lost if r.get("fail_closed") else intact).append(value)
            if intact:
                out["full_overall_intact"] = sum(intact) / len(intact)
            if lost:
                out["full_overall_lost"] = sum(lost) / len(lost)
    return out


def _utility_header(tier: str) -> str:
    # Same provenance-header convention as make_paper_artifacts._header.
    if tier == "live_replication":
        note = "Generated by scripts/judge_utility.py from a live_replication judged run."
    else:
        note = (
            "Generated by scripts/judge_utility.py from a synthetic_fixture dry-run "
            "(fixture crew, stub judge). DIAGNOSTIC ONLY: these numbers test the "
            "pipeline and are NOT judged utility evidence."
        )
    return f"% {note}\n"


def latency_footnote(journal: Path) -> str:
    """Latency footnote line measured from the main experiment journal."""

    lat = condition_latency_means(load_experiment_records(journal))
    if lat.get("adaptive") is None:
        raise SystemExit("journal has no completed adaptive rows for the latency footnote")
    text = f"Latency: baseline {lat['adaptive']:.1f}s"
    if lat.get("structure") is not None and lat["adaptive"]:
        pct = (lat["structure"] - lat["adaptive"]) / lat["adaptive"] * 100.0
        text += f", struct.\\ canon.\\ {lat['structure']:.1f}s ({pct:+.0f}\\%)"
    if lat.get("full") is not None:
        text += f", full pad {lat['full']:.1f}s"
    return _mc(text)


def existing_latency_footnote(table_path: Path) -> str:
    """Keep the committed latency line when no journal is supplied."""

    if table_path.is_file():
        for line in table_path.read_text(encoding="utf-8").splitlines():
            if "Latency:" in line:
                return line
    return _mc("Latency: not measured (rerun with --journal <main experiment journal>)")


def render_utility_table(agg: dict[str, Any], judge_model: str, latency_line: str) -> str:
    def row(condition: str) -> str:
        label = CONDITION_LABELS[condition]
        means = agg["means"][condition]
        cells = " & ".join(f"{means[key]:.3f}" for key in SCORE_KEYS)
        return f"{label:<21} & {cells} \\\\"

    significance = "not significant" if agg["p_overall"] >= 0.05 else "significant"
    judge_line = _mc(
        f"Judge: {judge_model}; difference {agg['difference']:.3f} "
        f"({significance}, $p\\approx{_fmt_p(agg['p_overall'])}$)"
    )
    header = "Condition & Overall & Faithful. & Complete. \\\\\n"
    rows = "\n".join(
        row(condition) for condition in CONDITIONS if condition in agg["means"]
    )
    extra = ""
    if agg.get("pairs_full"):
        sig_full = "not significant" if agg["p_full"] >= 0.05 else "significant"
        extra = _mc(
            f"Full pad vs baseline: difference {agg['difference_full']:.3f} "
            f"({sig_full}, $p\\approx{_fmt_p(agg['p_full'])}$); "
            f"{100 * agg.get('full_fail_closed_rate', 0.0):.0f}\\% of padded runs "
            "released at least one empty step"
        ) + "\n"
    body = rows + "\n\\midrule\n" + judge_line + "\n" + extra + latency_line
    return _wrap(5, "lccc", header, body)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-per-cell", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260710)
    ap.add_argument("--judge-model", default="gpt-5.6-terra")
    ap.add_argument("--scores", required=True, type=Path, help="scores JSONL (append/resume)")
    ap.add_argument("--tables-dir", type=Path, default=Path("tables"))
    ap.add_argument(
        "--journal",
        type=Path,
        default=None,
        help="main experiment journal used for the latency footnote",
    )
    ap.add_argument("--case-limit", type=int, default=None)
    ap.add_argument(
        "--dry-run", action="store_true", help="fixture crew + stub judge; no network calls"
    )
    ap.add_argument(
        "--analyze-only",
        action="store_true",
        help="analyse an existing scores file; run no cells and need no credentials",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="concurrent crew+judge cells; each cell stays sequential internally",
    )
    args = ap.parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be positive")

    settings = Settings.from_env()
    catalog = (
        CorpusCatalog.load(settings.dataset_path)
        if settings.dataset_path is not None
        else CorpusCatalog.load_default()
    )
    if args.analyze_only:
        # Pure analysis of a scores file that already exists. No crew, no judge,
        # no credentials and no cell execution. This is the path a reader uses to
        # regenerate the utility table from the committed judged run, and it is
        # deliberately incapable of producing a score: a cell absent from the
        # scores file stays absent and is reported as a shortfall, because a
        # stub-filled cell in a utility table is worse than a smaller n.
        if not args.scores.is_file():
            raise SystemExit(f"--analyze-only needs an existing scores file: {args.scores}")
        sdk = None
        judge = None
    else:
        try:
            sdk = TraceGuardSDK(
                settings, provider="fixture" if args.dry_run else "azure", catalog=catalog
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if not args.dry_run:
            assert_judge_independence(args.judge_model, settings)
        judge = stub_judge if args.dry_run else AzureOpenAIJudge(settings, args.judge_model)

    cases = catalog_cases(catalog)
    if args.case_limit:
        cases = cases[: args.case_limit]
    done = load_scored_cells(args.scores)
    # Cells that raised. Collected rather than fatal: see the worker loop below.
    failures: list[tuple[tuple, Exception]] = []
    crew_models = {
        "fast": settings.model_fast,
        "deep": settings.model_deep,
        "review": settings.model_review,
    }

    resumed = 0
    pending: list[tuple[str, str, int]] = []
    for condition in CONDITIONS:
        for case_meta in cases:
            case_id = case_meta["case_id"]
            for rep in range(args.n_per_cell):
                if (case_id, condition, rep) in done:
                    resumed += 1
                    continue
                pending.append((condition, case_id, rep))

    if args.analyze_only and pending:
        print(
            f"analyze-only: {len(pending)} of {resumed + len(pending)} grid cells are absent "
            f"from {args.scores} and are not being run; the analysis below uses the "
            f"{resumed} cells that were actually scored.",
            file=sys.stderr,
        )
        pending = []

    executed = 0
    write_lock = threading.Lock()
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    with args.scores.open("a", encoding="utf-8") as handle:

        def run_cell(condition: str, case_id: str, rep: int) -> None:
            nonlocal executed
            # Same sha256 cell-seed scheme as ExperimentRunner. The judge sweeps
            # conditions at each condition's historical autonomy, so the arm is
            # derived rather than passed; using the same Arm type keeps the two
            # seed derivations identical instead of merely similar.
            short = normalize_condition(condition)
            arm = Arm(condition=short, autonomy=DEFAULT_AUTONOMY_FOR_CONDITION[short])
            seed = ExperimentRunner._cell_seed(args.seed, arm, case_id, rep)
            run_id = _ID_SAFE.sub("-", f"judge-{condition}-{case_id}-r{rep}")[:127]
            result = sdk.run(case_id, condition, run_id=run_id, seed=seed)
            # The determination stays in memory; only its scores go out.
            scores = judge(catalog.get_case(case_id), result.answer, seed)
            # Record the release outcome, not just the score. Under full
            # padding a step that overruns the deadline is released as an empty
            # payload, so the crew continues with a missing input. A 1/1/1 from
            # a lost step is a different fact about the mechanism than a 1/1/1
            # from a weak answer, and without this field the two are
            # indistinguishable in the scores file.
            metrics = getattr(result, "metrics", None)
            record = {
                "case_id": case_id,
                "condition": condition,
                "rep": rep,
                "seed": seed,
                "scores": scores,
                "judge_model": args.judge_model,
                "crew_models": crew_models,
                "fail_closed": bool(getattr(metrics, "fail_closed", False)),
                "overrun_count": int(getattr(metrics, "overrun_count", 0) or 0),
                "answer_chars": len(result.answer or ""),
            }
            with write_lock:
                handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                executed += 1

        # Cells are I/O-bound on the provider API; a thread pool multiplies
        # throughput while each cell's crew steps stay sequential internally.
        if args.workers == 1:
            for cell in pending:
                run_cell(*cell)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(run_cell, *cell): cell for cell in pending}
                for future, cell in futures.items():
                    try:
                        future.result()
                    except Exception as exc:  # noqa: BLE001 - one cell must not end the study
                        # This is a paid run. Losing the whole study to a single
                        # malformed judge response would waste every completed
                        # cell, so record the failure and continue; the missing
                        # cells are visible as a shortfall in the score file and
                        # are re-attempted on the next resume.
                        failures.append((cell, exc))
                        print(
                            f"cell failed ({type(exc).__name__}: {exc}); "
                            f"continuing: {cell[:2]}",
                            file=sys.stderr,
                        )

    if failures:
        print(
            f"\n{len(failures)} of {len(pending)} cells failed and were skipped; "
            "re-run with the same --scores to retry only those.",
            file=sys.stderr,
        )
        for cell, exc in failures[:5]:
            print(f"  {cell[:2]}: {type(exc).__name__}: {exc}", file=sys.stderr)

    records = [
        json.loads(line)
        for line in args.scores.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    agg = aggregate(records)

    table_path = args.tables_dir / "tab_utility.tex"
    latency_line = (
        latency_footnote(args.journal) if args.journal else existing_latency_footnote(table_path)
    )
    tier = "synthetic_fixture" if args.dry_run else "live_replication"
    header = _utility_header(tier)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    table_path.write_text(
        header + render_utility_table(agg, args.judge_model, latency_line), encoding="utf-8"
    )
    # \judgeModel was hand-maintained in macros.tex and this script never wrote
    # it, so it stayed at the archived run's gpt-4o while the judge became
    # gpt-5.6-terra. That is not a cosmetic staleness: the manuscript claims the
    # judge is "distinct from the crew's", and gpt-4o is one of the crew
    # deployments the model axis sweeps -- the stale macro asserted exactly the
    # independence violation assert_judge_independence exists to prevent. It is
    # generated now, for the same reason \crewModel is.
    merge_macros(
        args.tables_dir / "macros.tex",
        {
            "utilBase": f"{agg['means']['adaptive']['overall']:.2f}",
            "utilShield": f"{agg['means']['structure_only']['overall']:.2f}",
            "utilP": _fmt_p(agg["p_overall"]),
            "judgeModel": args.judge_model,
            "utilPairs": str(agg["pairs"]),
            **(
                {
                    "utilFull": f"{agg['means']['full_pad']['overall']:.2f}",
                    "utilFullP": _fmt_p(agg["p_full"]),
                    "utilFullDelta": f"{agg['difference_full']:.3f}",
                    "utilFullPairs": str(agg["pairs_full"]),
                    "utilFullFailClosedPct": (
                        f"{100 * agg.get('full_fail_closed_rate', 0.0):.0f}"
                    ),
                    "utilFullIntact": (
                        f"{agg['full_overall_intact']:.2f}"
                        if agg.get("full_overall_intact") is not None
                        else "--"
                    ),
                    "utilFullLost": (
                        f"{agg['full_overall_lost']:.2f}"
                        if agg.get("full_overall_lost") is not None
                        else "--"
                    ),
                }
                if agg.get("pairs_full")
                else {}
            ),
        },
        header,
    )

    print(f"provenance tier: {tier}")
    print(f"scored cells: executed {executed}, resumed {resumed}, paired {agg['pairs']}")
    print(f"wrote: tab_utility.tex, macros.tex -> {args.tables_dir}")
    if tier != "live_replication":
        print("WARNING: dry-run output is diagnostic only and is NOT scientific evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

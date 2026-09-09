"""Offline tests for the utility-judge pipeline (scripts/judge_utility.py).

No network is used: the crew runs on the fixture provider (--dry-run) or a fake
SDK, and the judge is the deterministic stub.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from scipy.stats import wilcoxon

from traceguard.corpus import CorpusCatalog
from traceguard.experiment import (
    DEFAULT_AUTONOMY_FOR_CONDITION,
    Arm,
    ExperimentRunner,
    catalog_cases,
    normalize_condition,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import judge_utility  # noqa: E402

SENTINEL = "SENTINEL-DETERMINATION-TEXT-9F1B"


def _record(case_id, condition, rep, overall=3, faithful=3, complete=3):
    return {
        "case_id": case_id,
        "condition": condition,
        "rep": rep,
        "seed": 1,
        "scores": {"overall": overall, "faithful": faithful, "complete": complete},
        "judge_model": "stub",
        "crew_models": {"fast": "f", "deep": "d", "review": "r"},
    }


def test_aggregate_means_and_wilcoxon_pairing() -> None:
    records = [
        _record("c1", "adaptive", 0, overall=3, faithful=3, complete=2),
        _record("c1", "adaptive", 1, overall=2, faithful=3, complete=3),
        _record("c2", "adaptive", 0, overall=3, faithful=2, complete=3),
        # unpaired adaptive cell: counted in the mean, excluded from the pairing
        _record("c9", "adaptive", 0, overall=1),
        _record("c1", "structure_only", 0, overall=2),
        _record("c1", "structure_only", 1, overall=3),
        _record("c2", "structure_only", 0, overall=2),
    ]
    agg = judge_utility.aggregate(records)
    # scores normalize as (score - 1) / 2
    assert agg["means"]["adaptive"]["overall"] == pytest.approx((1.0 + 0.5 + 1.0 + 0.0) / 4)
    assert agg["means"]["adaptive"]["faithful"] == pytest.approx((1.0 + 1.0 + 0.5 + 1.0) / 4)
    assert agg["means"]["structure_only"]["overall"] == pytest.approx((0.5 + 1.0 + 0.5) / 3)
    assert agg["pairs"] == 3
    expected = float(wilcoxon([1.0, 0.5, 1.0], [0.5, 1.0, 0.5]).pvalue)
    assert agg["p_overall"] == pytest.approx(expected)
    assert agg["difference"] == pytest.approx(
        agg["means"]["adaptive"]["overall"] - agg["means"]["structure_only"]["overall"]
    )


def test_aggregate_identical_conditions_yield_p_one() -> None:
    records = []
    for case_id in ("c1", "c2"):
        for condition in ("adaptive", "structure_only"):
            records.append(_record(case_id, condition, 0, overall=2))
    agg = judge_utility.aggregate(records)
    assert agg["p_overall"] == 1.0
    assert agg["difference"] == pytest.approx(0.0)


def test_dry_run_scores_resume_and_preserve_latency_line(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TRACEGUARD_PROVIDER", "fixture")
    first_case = catalog_cases(CorpusCatalog.load_default())[0]["case_id"]
    scores = tmp_path / "scores.jsonl"
    prewritten = _record(first_case, "adaptive", 0)
    scores.write_text(json.dumps(prewritten) + "\n", encoding="utf-8")

    tables = tmp_path / "tables"
    tables.mkdir()
    latency_line = (
        "\\multicolumn{4}{@{}l@{}}{\\footnotesize Latency: baseline 99.9s, "
        "struct.\\ canon.\\ 88.8s (-11\\%), full pad 77.7s}\\\\"
    )
    (tables / "tab_utility.tex").write_text(latency_line + "\n", encoding="utf-8")

    argv = [
        "--dry-run",
        "--case-limit", "1",
        "--n-per-cell", "2",
        "--scores", str(scores),
        "--tables-dir", str(tables),
    ]
    assert judge_utility.main(argv) == 0
    rows = [json.loads(line) for line in scores.read_text().splitlines()]
    # 1 case x 2 reps x 3 conditions, with the prewritten cell resumed (skipped).
    # The third arm is full_pad: the (0,0) claim's utility cost is measured
    # rather than assumed, so it is swept alongside the other two.
    assert len(rows) == 6
    assert sum(1 for row in rows if row["judge_model"] == "gpt-5.6-terra") == 5
    assert {row["condition"] for row in rows} == {
        "adaptive",
        "structure_only",
        "full_pad",
    }
    # Every cell this run wrote records its release outcome, so a low score
    # caused by a step released empty under padding is distinguishable from a
    # weak answer. rows[0] is the prewritten resumed cell and predates the
    # field, which is exactly why the analysis must tolerate its absence.
    for row in rows[1:]:
        assert "fail_closed" in row and isinstance(row["fail_closed"], bool)
        assert "overrun_count" in row
    for row in rows[1:]:
        # The seed is derived from the full arm coordinate, not the condition
        # alone, so two arms can never share a cell key.
        short = normalize_condition(row["condition"])
        arm = Arm(condition=short, autonomy=DEFAULT_AUTONOMY_FOR_CONDITION[short])
        assert row["seed"] == ExperimentRunner._cell_seed(
            20260710, arm, row["case_id"], row["rep"]
        )
        assert set(row["scores"]) == {"overall", "faithful", "complete"}

    # journal-less regeneration keeps the committed latency footnote verbatim
    table = (tables / "tab_utility.tex").read_text()
    assert "Latency: baseline 99.9s" in table
    assert "% " in table.splitlines()[0]  # provenance header
    assert "synthetic_fixture" in table.splitlines()[0]

    # a rerun resumes every cell and appends nothing
    assert judge_utility.main(argv) == 0
    assert len(scores.read_text().splitlines()) == 6


class _FakeSDK:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def run(self, case_id, condition, run_id=None, event_callback=None, seed=0):
        return SimpleNamespace(answer=f"{SENTINEL} {case_id} {condition} {seed}")


def test_sentinel_answer_never_reaches_scores_or_tables(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TRACEGUARD_PROVIDER", "fixture")
    monkeypatch.setattr(judge_utility, "TraceGuardSDK", _FakeSDK)
    scores = tmp_path / "scores.jsonl"
    tables = tmp_path / "tables"
    tables.mkdir()
    (tables / "macros.tex").write_text(
        "\\newcommand{\\crewModel}{gpt-4o}\n\\newcommand{\\utilBase}{0.11}\n",
        encoding="utf-8",
    )
    argv = [
        "--dry-run",
        "--case-limit", "2",
        "--n-per-cell", "1",
        "--judge-model", "stub-judge",
        "--scores", str(scores),
        "--tables-dir", str(tables),
    ]
    assert judge_utility.main(argv) == 0

    table = (tables / "tab_utility.tex").read_text()
    macros = (tables / "macros.tex").read_text()
    emitted = scores.read_text() + table + macros
    assert SENTINEL not in emitted

    assert "Baseline (adaptive)" in table
    assert "Enforced (canonical)" in table
    assert "Judge: stub-judge" in table
    # merge-preserve: unrelated macros survive, util macros are regenerated
    assert "\\newcommand{\\crewModel}{gpt-4o}" in macros
    assert "\\newcommand{\\utilBase}{0.11}" not in macros
    for name in ("utilBase", "utilShield", "utilP"):
        assert f"\\newcommand{{\\{name}}}" in macros


def test_latency_footnote_formats_journal_means(tmp_path) -> None:
    def row(condition, wall):
        return {
            "status": "completed",
            "condition": condition,
            "trace": {"steps": [{"wall_time_s": wall, "duration_s": wall}]},
        }

    journal = tmp_path / "journal.jsonl"
    rows = [
        row("adaptive", 10.0),
        row("adaptive", 20.0),
        row("structure", 12.0),
        row("full", 30.0),
    ]
    journal.write_text("\n".join(json.dumps(item) for item in rows) + "\n", encoding="utf-8")
    line = judge_utility.latency_footnote(journal)

    # The footnote may span several \multicolumn rows: a single row cannot break
    # a line, so a long footnote is split at a character budget rather than left
    # to run into the margin. Assert on the joined content, plus the property
    # that split exists to guarantee -- every emitted row fits the budget.
    import re as _re

    from make_paper_artifacts import _MC_CHARS

    rows = _re.findall(r"\\multicolumn\{4\}\{@\{\}l@\{\}\}\{\\footnotesize (.*?)\}\\\\", line)
    assert rows, line
    joined = " ".join(rows)
    assert "Latency: baseline 15.0s" in joined
    assert "struct.\\ canon.\\ 12.0s (-20\\%)" in joined
    assert "full pad 30.0s" in joined
    for row_text in rows:
        visible = _re.sub(r"\\[a-zA-Z]+\s?|[{}$]", "", row_text)
        assert len(visible) <= _MC_CHARS, (len(visible), row_text)


def test_judge_score_parsing_is_strict() -> None:
    assert judge_utility.parse_judge_scores(
        '{"overall": 3, "faithful": 2, "complete": 1}'
    ) == {"overall": 3, "faithful": 2, "complete": 1}
    for bad in ("not json", "[1,2,3]", '{"overall": 4, "faithful": 2, "complete": 1}',
                '{"overall": 2.5, "faithful": 2, "complete": 1}', '{"overall": 3}'):
        with pytest.raises(ValueError):
            judge_utility.parse_judge_scores(bad)


def test_truncated_judge_reply_names_itself() -> None:
    """A truncated reply must not read as "the model returned non-JSON".

    At max_completion_tokens=64 this deployment returns finish_reason='length'
    with an *empty* content field, having spent the whole budget before emitting
    any output. The old message ("judge returned non-JSON output") was
    indistinguishable from a malformed reply, and 15 paid crew runs were burned
    before the cause was found. The distinction is worth a test.
    """

    import pytest
    from judge_utility import parse_judge_scores

    with pytest.raises(ValueError, match="truncated"):
        parse_judge_scores("", finish_reason="length")
    with pytest.raises(ValueError, match="truncated"):
        parse_judge_scores('{"overall":3,"faith', finish_reason="length")

    # Empty for another reason is reported as empty, not as truncated.
    with pytest.raises(ValueError, match="empty content"):
        parse_judge_scores("", finish_reason="stop")

    # A genuinely malformed reply still says so, and now says how much of it
    # there was, without echoing it.
    with pytest.raises(ValueError, match="non-JSON output"):
        parse_judge_scores("I would rate this a 3 out of 3.", finish_reason="stop")

    # The happy path is unaffected.
    assert parse_judge_scores(
        '{"overall":3,"faithful":2,"complete":1}', finish_reason="stop"
    ) == {"overall": 3, "faithful": 2, "complete": 1}


def test_judge_macro_never_names_a_crew_model() -> None:
    """The manuscript's independence claim must hold in the *macros*, not just the run.

    \\judgeModel was hand-maintained and this script never wrote it, so it stayed
    at the archived run's gpt-4o while the judge became gpt-5.6-terra. gpt-4o is
    one of the crew deployments the model axis sweeps, so the stale macro
    asserted precisely the independence violation assert_judge_independence
    prevents at run time. Both halves now have to agree.
    """

    import re
    from pathlib import Path

    macros = Path(__file__).resolve().parents[1] / "tables/macros.tex"
    text = macros.read_text()
    found = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", text))
    judge = found.get("judgeModel")
    crew = found.get("crewModel")
    assert judge, "\\judgeModel is not defined"
    assert crew, "\\crewModel is not defined"
    # crewModel may name several deployments ("a and b", "a, b, and c").
    crew_names = {n.strip() for n in re.split(r",| and ", crew) if n.strip()}
    # The judge must differ from every deployment used as a crew *anywhere* in
    # the study, not only from the headline crew: the model axis sweeps three,
    # and a judge shared with any of them breaks the claim for that arm.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_experiment_matrix import CREW_DEPLOYMENTS

    crew_names |= set(CREW_DEPLOYMENTS)
    assert judge not in crew_names, (
        f"\\judgeModel{{{judge}}} is also used as a crew deployment "
        f"({sorted(crew_names)}); the manuscript's independent-judge claim "
        "would be false for at least one arm"
    )

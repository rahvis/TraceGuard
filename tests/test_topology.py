"""Crew-topology parameterization and the frozen plan registry."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from traceguard import Settings, TraceGuardSDK, verify_receipt
from traceguard.graph import ARCHITECTURES, CANONICAL_STEP_TYPES, architecture_spec
from traceguard.plans import (
    LEGACY_CREW_STEP_TYPES,
    PIPELINE_CREW_STEP_TYPES,
    PLAN_REGISTRY,
    plan_digest,
    plan_for_architecture,
    plan_for_step_types,
    resolve_plan,
)
from traceguard.receipt import build_receipt_body
from traceguard.types import Autonomy, Condition


@pytest.fixture
def fast_settings() -> Settings:
    return Settings(provider="fixture", step_deadline_s=0.005, receipt_envelope_bytes=8192)


# --------------------------------------------------------------------------
# The pinned released plan
# --------------------------------------------------------------------------


def test_pipeline_crew_step_types_are_pinned_element_for_element() -> None:
    """Archived receipts commit to this sequence through their trace_sha256.

    Parameterizing topology must not reinterpret a single archived run, so the
    released plan is asserted against a literal here as well as in the module.
    """

    assert PIPELINE_CREW_STEP_TYPES == (
        "intake",
        "clinical_extraction",
        "clinical_extraction",
        "clinical_extraction",
        "clinical_extraction",
        "clinical_extraction",
        "clinical_extraction",
        "clinical_extraction",
        "coverage_assessment",
        "criteria_check",
        "necessity_review",
        "determination",
    )
    # The name every pre-existing call site imports must still be this plan.
    assert CANONICAL_STEP_TYPES == PIPELINE_CREW_STEP_TYPES
    assert plan_for_architecture("pipeline_crew").plan_id == "CANON-v1"
    assert architecture_spec(None).name == "pipeline_crew"


# --------------------------------------------------------------------------
# Three architectures, three step sequences
# --------------------------------------------------------------------------


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_each_architecture_emits_its_own_canonical_plan(
    architecture: str,
    fast_settings: Settings,
) -> None:
    sdk = TraceGuardSDK(fast_settings)
    case = sdk.corpus.list_cases()[0]
    spec = architecture_spec(architecture)
    result = sdk.run(
        case.case_id,
        Condition.FULL_PAD,
        run_id=f"canon-{architecture}",
        architecture=architecture,
    )
    assert result.trace.step_types == spec.canonical_step_types
    assert result.runtime["architecture"] == architecture
    assert result.runtime["canonical_plan"] == spec.plan.plan_id
    assert verify_receipt(result.receipt).ok


def test_the_three_architectures_produce_distinct_step_sequences(
    fast_settings: Settings,
) -> None:
    """The point of the axis: same work, three different observable plans."""

    sdk = TraceGuardSDK(fast_settings)
    case = next(case for case in sdk.corpus.list_cases() if case.sensitive)
    scripted = {
        architecture: sdk.run(
            case.case_id,
            Condition.FULL_PAD,
            run_id=f"distinct-canon-{architecture}",
            architecture=architecture,
        ).trace.step_types
        for architecture in ARCHITECTURES
    }
    assert len(set(scripted.values())) == len(ARCHITECTURES)
    # The routing agent is observable, so its plan is one step longer.
    assert len(scripted["hierarchical_supervisor"]) == len(scripted["pipeline_crew"]) + 1
    # Every tool execution is preceded by an observable tool-choice step.
    assert scripted["react_single_agent"].count("react_agent") == 11

    adaptive = {
        architecture: sdk.run(
            case.case_id,
            Condition.ADAPTIVE,
            run_id=f"distinct-adaptive-{architecture}",
            architecture=architecture,
            autonomy=Autonomy.BOUNDED,
        ).trace.step_types
        for architecture in ARCHITECTURES
    }
    assert len(set(adaptive.values())) == len(ARCHITECTURES)


def test_hierarchical_supervisor_actually_uses_the_route_it_parses(
    fast_settings: Settings,
) -> None:
    """The hook the released topology built and then discarded.

    ``intake`` already received ``{route, service_type, complete}``; the edge
    out of it fired unconditionally. If the parsed route were still being
    thrown away, the supervisor plan would be indistinguishable from the
    pipeline plan apart from the node's name.
    """

    sdk = TraceGuardSDK(fast_settings)
    case = sdk.corpus.list_cases()[0]
    result = sdk.run(
        case.case_id,
        Condition.FULL_PAD,
        run_id="supervisor-routes",
        architecture="hierarchical_supervisor",
    )
    # Two routing decisions: pick the extractor, then pick the assessor.
    assert result.trace.step_types.count("supervisor") == 2
    assert result.trace.step_types[0] == "supervisor"


def test_react_agent_dispatches_through_the_tool_registry(
    fast_settings: Settings,
) -> None:
    from traceguard.agents import REACT_TOOLS

    assert set(REACT_TOOLS) == {
        "extract_clinical_evidence",
        "assess_coverage",
        "check_criteria",
        "review_necessity",
        "issue_determination",
    }
    sdk = TraceGuardSDK(fast_settings)
    case = sdk.corpus.list_cases()[0]
    result = sdk.run(
        case.case_id,
        Condition.FULL_PAD,
        run_id="react-tools",
        architecture="react_single_agent",
    )
    # Each tool-choice step is followed by the tool it chose, and the last tool
    # is the determination.
    assert result.trace.step_types[-1] == "determination"
    assert result.trace.step_types[-2] == "react_agent"
    assert result.answer


def test_unknown_architecture_is_rejected_by_name(fast_settings: Settings) -> None:
    sdk = TraceGuardSDK(fast_settings)
    case = sdk.corpus.list_cases()[0]
    with pytest.raises(ValueError, match="unknown crew architecture"):
        sdk.run(case.case_id, Condition.ADAPTIVE, architecture="swarm")


# --------------------------------------------------------------------------
# The plan registry and what the verifier will accept
# --------------------------------------------------------------------------


def test_legacy_and_current_twelve_step_plans_are_separately_nameable() -> None:
    """The archive's real weakness, made addressable.

    All ten archived full_pad receipts reproduce their signed trace_sha256
    under the legacy vocabulary and none under the current one, yet both shapes
    are twelve steps with seven hops -- so the hop count alone cannot tell them
    apart, and the receipts that declare "CANON-v1" are in fact the legacy
    shape. Registering the legacy plan separately makes the two distinguishable
    by digest and resolvable by observed step sequence.
    """

    current = plan_for_step_types(PIPELINE_CREW_STEP_TYPES)
    legacy = plan_for_step_types(LEGACY_CREW_STEP_TYPES)
    assert current is not None and legacy is not None
    assert current.plan_id == "CANON-v1"
    assert legacy.plan_id == "CANON-v0-legacy"
    # Same length, same hop count, different plan.
    assert len(current.step_types) == len(legacy.step_types) == 12
    assert current.canonical_research_hops == legacy.canonical_research_hops == 7
    assert current.digest != legacy.digest


def test_plan_digest_is_content_addressed_over_the_claim_it_makes() -> None:
    plan = plan_for_architecture("pipeline_crew")
    assert plan.digest in PLAN_REGISTRY
    assert resolve_plan(plan.digest) is plan
    # Renaming the plan, changing its hop count, or changing a single step type
    # all move the digest, so a receipt cannot restate any of them.
    for mutated in (
        plan_digest(
            plan_id="CANON-v2",
            step_types=plan.step_types,
            canonical_research_hops=7,
        ),
        plan_digest(
            plan_id=plan.plan_id,
            step_types=plan.step_types,
            canonical_research_hops=6,
        ),
        plan_digest(
            plan_id=plan.plan_id,
            step_types=("intake", *plan.step_types[1:]) + ("extra",),
            canonical_research_hops=7,
        ),
    ):
        assert mutated != plan.digest
        assert resolve_plan(mutated) is None


def _body(**overrides) -> dict:
    from traceguard.types import ObservableTrace, TraceStep

    trace = ObservableTrace(
        tuple(
            TraceStep(
                index=index,
                step_type=step,
                wall_time_s=0.001 * (index + 1),
                duration_s=0.001,
                egress_bytes=4096,
                ingress_bytes=1234,
            )
            for index, step in enumerate(PIPELINE_CREW_STEP_TYPES)
        )
    )
    values = {
        "run_id": "plan-check",
        "case_id": "case-1",
        "condition": Condition.FULL_PAD,
        "trace": trace,
        "dataset_hash": "0" * 64,
        "provider_provenance": "synthetic_fixture",
        "policy_id": "traceguard-public-canon-v1",
        "canonical_research_hops": 7,
        "step_deadline_s": 0.001,
        "egress_ceiling_bytes": 4096,
        "fail_closed": False,
        "violations": [],
        "ledger_sequence": 0,
        "previous_hash": "0" * 64,
    }
    values.update(overrides)
    return build_receipt_body(**values)


def test_a_receipt_cannot_name_a_plan_outside_the_frozen_registry() -> None:
    from traceguard.verifier import _plan_errors

    plan = plan_for_architecture("react_single_agent")
    # _body builds a 12-step pipeline trace. Committing it to the 22-step ReAct
    # plan is a receipt naming a plan it did not run, and must be REJECTED.
    # An earlier version of this test asserted the opposite, which is how the
    # hole survived: the three consistency checks below are all mutually
    # satisfiable by any registered plan, so nothing compared claim to run.
    body = _body(plan_digest=plan.digest)
    assert body["policy"]["plan_template"] == "CANON-REACT-v1"
    assert body["policy"]["plan_digest"] == plan.digest
    assert _plan_errors("full_pad", body["policy"], body["trace_sha256"]) == [
        "full-pad trace digest does not match the committed plan; the receipt "
        "names a plan it did not run"
    ]

    # The honest counterpart: commit the pipeline trace to the pipeline plan.
    pipeline = plan_for_architecture("pipeline_crew")
    honest = _body(plan_digest=pipeline.digest)
    assert not _plan_errors("full_pad", honest["policy"], honest["trace_sha256"])

    # A self-declared plan does not.
    forged = plan_digest(
        plan_id="CANON-MINE-v1",
        step_types=("intake", "determination"),
        canonical_research_hops=7,
    )
    errors = _plan_errors(
        "full_pad",
        {"plan_template": "CANON-MINE-v1", "plan_digest": forged, "canonical_research_hops": 7},
    )
    assert errors == ["policy.plan_digest is not a plan in the frozen TraceGuard registry"]

    # Nor does relabelling or re-stating the hop count of a registered plan.
    # These policies carry no deadline or ceiling, so the trace-binding check
    # also reports that it could not run -- which is the honest answer, and why
    # these assert containment rather than an exact single-error list.
    assert "policy.plan_template does not match the committed plan digest" in _plan_errors(
        "full_pad",
        {"plan_template": "CANON-v1", "plan_digest": plan.digest, "canonical_research_hops": 7},
    )
    assert (
        "policy.canonical_research_hops does not match the committed plan digest"
        in _plan_errors(
            "full_pad",
            {
                "plan_template": plan.plan_id,
                "plan_digest": plan.digest,
                "canonical_research_hops": 4,
            },
        )
    )

    # An unusable profile must be reported as "not checked", never silently
    # treated as checked-and-fine.
    assert "full-pad plan commitment could not be checked against the trace" in _plan_errors(
        "full_pad",
        {
            "plan_template": plan.plan_id,
            "plan_digest": plan.digest,
            "canonical_research_hops": plan.canonical_research_hops,
        },
    )

    # And a build cannot sign one at all.
    with pytest.raises(ValueError, match="frozen traceguard.plans registry"):
        _body(plan_digest=forged)


def test_a_pre_registry_receipt_still_verifies_through_the_unchanged_check() -> None:
    """Route old receipts to the independent seven-hop check, unweakened."""

    from traceguard.verifier import _honesty_errors, _plan_errors

    body = _body()  # no plan_digest, exactly today's shape
    assert "plan_digest" not in body["policy"]
    assert body["policy"]["plan_template"] == "CANON-v1"
    assert not _plan_errors("full_pad", body["policy"])

    # The independent check is still what rejects a wrong hop count, with or
    # without a digest -- so a receipt cannot escape it by omitting one.
    stale = dict(body)
    stale["policy"] = {**body["policy"], "canonical_research_hops": 4}
    assert any("seven-hop" in error for error in _honesty_errors(stale, b"\x00" * 32))


def test_adaptive_receipts_carry_no_step_sequence_commitment() -> None:
    """A digest here would hand the host the channel this work attacks."""

    from traceguard.verifier import _plan_errors

    plan = plan_for_architecture("hierarchical_supervisor")
    body = _body(condition=Condition.ADAPTIVE, plan_digest=plan.digest)
    assert "plan_digest" not in body["policy"]
    assert body["policy"]["plan_template"] == "ADAPTIVE"
    assert not _plan_errors("adaptive", body["policy"])

    # An adaptive receipt that does carry one is an error, not extra assurance.
    assert _plan_errors(
        "adaptive",
        {"plan_template": "ADAPTIVE", "plan_digest": plan.digest},
    ) == ["adaptive receipt must not commit to a canonical step sequence"]
    assert _plan_errors("adaptive", {"plan_template": "CANON-v1"}) == [
        "adaptive receipt names a canonical plan template"
    ]


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_end_to_end_receipts_commit_only_under_a_canonicalized_condition(
    architecture: str,
    fast_settings: Settings,
) -> None:
    sdk = TraceGuardSDK(fast_settings)
    case = sdk.corpus.list_cases()[0]
    plan = architecture_spec(architecture).plan

    canonical = sdk.run(
        case.case_id,
        Condition.FULL_PAD,
        run_id=f"commit-full-{architecture}",
        architecture=architecture,
    ).receipt
    assert canonical.body["policy"]["plan_digest"] == plan.digest
    assert canonical.body["policy"]["plan_template"] == plan.plan_id
    assert verify_receipt(canonical).ok

    adaptive = sdk.run(
        case.case_id,
        Condition.ADAPTIVE,
        run_id=f"commit-adaptive-{architecture}",
        architecture=architecture,
    ).receipt
    assert "plan_digest" not in adaptive.body["policy"]
    assert adaptive.body["policy"]["plan_template"] == "ADAPTIVE"
    assert verify_receipt(adaptive).ok


# --------------------------------------------------------------------------
# The verifier's dependency surface
# --------------------------------------------------------------------------


def test_verifier_imports_no_agent_runtime_or_scientific_stack() -> None:
    """The verifier is published for reviewers; keep it stdlib + cryptography.

    Reaching the plan registry through graph.py or agents.py would make running
    the verifier require langgraph, so the registry lives in a leaf module and
    this asserts that it stayed one.
    """

    blocked = (
        "langgraph",
        "langchain",
        "langchain_core",
        "numpy",
        "sklearn",
        "scipy",
        "pandas",
        "openai",
    )
    program = f"""
import json, sys

BLOCKED = {blocked!r}


class _Blocker:
    def find_module(self, name, path=None):
        return self.find_spec(name, path)

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"blocked for this test: {{name}}")
        return None


sys.meta_path.insert(0, _Blocker())
import traceguard.verifier as v
from traceguard.plans import plan_for_architecture

leaked = sorted(n for n in sys.modules if n.split(".")[0] in BLOCKED)
print(json.dumps({{
    "leaked": leaked,
    "registry_reachable": plan_for_architecture("react_single_agent").plan_id,
    "verifier_rejects_a_forged_plan": bool(
        v._plan_errors("full_pad", {{"plan_digest": "0" * 64}})
    ),
}}))
"""
    # The argv is this file's own literal program text run under the current
    # interpreter; a subprocess is required because the property under test is
    # about a *fresh* interpreter's sys.modules.
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(completed.stdout)
    # The verifier and its plan registry load and work with every one of those
    # packages made unimportable.
    assert report["leaked"] == []
    assert report["registry_reachable"] == "CANON-REACT-v1"
    assert report["verifier_rejects_a_forged_plan"] is True


def test_the_plan_registry_stays_a_leaf_module() -> None:
    """Static guard on the import that would break the above at a distance.

    ``sys.modules`` cannot catch this on its own: ``graph`` and ``agents`` are
    already loaded by ``traceguard/__init__``, and neither imports ``langgraph``
    at module scope. So the property worth pinning is structural -- the
    verifier's intra-package import closure must exclude them.
    """

    import ast
    from pathlib import Path

    package = Path(traceguard_file()).parent

    def relative_imports(module: str) -> set[str]:
        tree = ast.parse((package / f"{module}.py").read_text(encoding="utf-8"))
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                found.add(node.module.split(".")[0])
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("traceguard"), alias.name
        return found

    closure: set[str] = set()
    frontier = {"verifier"}
    while frontier:
        module = frontier.pop()
        if module in closure:
            continue
        closure.add(module)
        frontier |= relative_imports(module)

    assert "graph" not in closure
    assert "agents" not in closure
    assert "plans" in closure
    # plans itself must stay a leaf but for the pure canonical-json helper.
    assert relative_imports("plans") == {"_canonical"}
    assert relative_imports("_canonical") == set()


def traceguard_file() -> str:
    import traceguard

    assert traceguard.__file__ is not None
    return traceguard.__file__

"""LangGraph assembly for the released crew, parameterized by topology.

Three crew architectures are selectable by name.  They exist so the leakage
question can be asked of an *architecture* and not only of an autonomy level:
each one emits a different observable step sequence for the same work, which is
the coordinate the paper attacks.

``pipeline_crew``
    The released six-node sequential plan.  Byte-for-byte the topology that
    every archived run used; its step types are pinned in
    :mod:`traceguard.plans` so archived receipts keep their meaning.
``hierarchical_supervisor``
    A routing agent selects the next specialist.  The three existing routers
    (``after_assessment``, ``after_criteria``, ``after_necessity``) are pure
    state readers and are reused unchanged; only the edge out of triage becomes
    supervisor-driven.
``react_single_agent``
    One agent choosing its own tool each step, dispatched through
    :data:`traceguard.agents.REACT_TOOLS`.

Each architecture owns its canonical (scripted) plan, and the plan registry --
not this module -- is the authority on what that plan is, so the verifier can
know the legitimate plan set without importing ``langgraph``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .agents import CrewState, TraceWriterCrew
from .plans import (
    CANONICAL_RESEARCH_HOPS,
    PLANS_BY_ARCHITECTURE,
    PlanSpec,
    plan_for_architecture,
)

# Re-exported for the many call sites that predate parameterized topology.
# CANONICAL_STEP_TYPES remains the pipeline_crew plan, unchanged.
CANONICAL_STEP_TYPES: tuple[str, ...] = PLANS_BY_ARCHITECTURE["pipeline_crew"].step_types

DEFAULT_ARCHITECTURE = "pipeline_crew"
ARCHITECTURES: tuple[str, ...] = (
    "pipeline_crew",
    "hierarchical_supervisor",
    "react_single_agent",
)

# The crew architecture registry and the experiment's ARCHITECTURES tuple are
# two vocabularies for one set; keeping them in step here rather than by comment
# means a new architecture cannot be half-added.
if set(ARCHITECTURES) != set(PLANS_BY_ARCHITECTURE):  # pragma: no cover - import guard
    raise RuntimeError("graph architectures and the plan registry disagree")


# --------------------------------------------------------------------------
# Node and edge descriptors
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NodeSpec:
    """One graph node, described well enough for the console to render it."""

    id: str
    label: str
    model_role: str
    purpose: str
    observable: str


@dataclass(frozen=True, slots=True)
class EdgeSpec:
    source: str
    target: str
    kind: str = "fixed"
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "source": self.source,
            "target": self.target,
            "kind": self.kind,
        }
        if self.label is not None:
            result["label"] = self.label
        return result


NODES: Mapping[str, NodeSpec] = {
    "intake": NodeSpec(
        id="intake",
        label="Intake & Compliance",
        model_role="fast",
        purpose=(
            "Validates the request packet (patient, provider, diagnosis/procedure "
            "codes, notes), classifies the service, and routes to clinical review."
        ),
        observable="1 model step",
    ),
    "supervisor": NodeSpec(
        id="supervisor",
        label="Supervisor (routing agent)",
        model_role="fast",
        purpose=(
            "Runs the same intake and compliance triage, then selects which "
            "specialist runs next. Its route is honoured whenever the state "
            "admits it, so control flow depends on the routing agent's output."
        ),
        observable="1 model step per routing decision (canonical: 2)",
    ),
    "react_agent": NodeSpec(
        id="react_agent",
        label="ReAct agent (tool choice)",
        model_role="fast",
        purpose=(
            "A single agent proposes the next tool from the registry. The "
            "proposal is executed when the state admits it, so every tool "
            "execution is preceded by an observable tool-choice step."
        ),
        observable="1 model step before every tool execution",
    ),
    "clinical_extraction": NodeSpec(
        id="clinical_extraction",
        label="Clinical Reviewer",
        model_role="fast",
        purpose=(
            "Extracts the clinical profile from one synthetic referral source per "
            "pass and decides whether the evidence is sufficient to assess coverage."
        ),
        observable="N passes (adaptive: data-dependent; canonical: exactly 7)",
    ),
    "coverage_assessment": NodeSpec(
        id="coverage_assessment",
        label="Coverage Assessment",
        model_role="deep",
        purpose=(
            "Maps the extracted evidence to each payer policy criterion "
            "(MET / NOT_MET / INSUFFICIENT) with cited source ids."
        ),
        observable="1+ model steps (repair/revision loops re-enter)",
    ),
    "criteria_check": NodeSpec(
        id="criteria_check",
        label="Criteria Verification",
        model_role="fast",
        purpose="Verifies that every policy criterion carries a supporting source id.",
        observable="1-2 model steps",
    ),
    "necessity_review": NodeSpec(
        id="necessity_review",
        label="Medical-Necessity Review",
        model_role="review",
        purpose=(
            "Clinician-style sign-off on completeness, criterion coverage, and uncertainty."
        ),
        observable="1-2 model steps",
    ),
    "determination": NodeSpec(
        id="determination",
        label="Synthesis & Determination",
        model_role="deep",
        purpose=(
            "Applies the gate rubric and issues the APPROVE / PEND determination "
            "with rationale."
        ),
        observable="1 model step",
    ),
}


@dataclass(frozen=True, slots=True)
class ArchitectureSpec:
    """Everything a caller needs to build, describe, or check one topology."""

    name: str
    label: str
    summary: str
    node_ids: tuple[str, ...]
    edges: tuple[EdgeSpec, ...]
    builder: Callable[[TraceWriterCrew], Any] = field(repr=False)

    @property
    def plan(self) -> PlanSpec:
        return plan_for_architecture(self.name)

    @property
    def canonical_step_types(self) -> tuple[str, ...]:
        return self.plan.step_types

    @property
    def nodes(self) -> tuple[NodeSpec, ...]:
        return tuple(NODES[node_id] for node_id in self.node_ids)

    def to_dict(self) -> dict[str, Any]:
        plan = self.plan
        return {
            "architecture": self.name,
            "label": self.label,
            "summary": self.summary,
            "engine": "langgraph",
            "canonical_plan": plan.plan_id,
            "plan_digest": plan.digest,
            "canonical_research_hops": plan.canonical_research_hops,
            "canonical_step_types": list(plan.step_types),
            "canonical_step_count": plan.step_count,
            "nodes": [
                {
                    "id": node.id,
                    "label": node.label,
                    "model_role": node.model_role,
                    "purpose": node.purpose,
                    "observable": node.observable,
                }
                for node in self.nodes
            ],
            "edges": [edge.to_dict() for edge in self.edges],
        }


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------


def _state_graph(crew: TraceWriterCrew) -> Any:
    try:
        from langgraph.graph import StateGraph
    except ImportError as exc:  # pragma: no cover - dependency is declared in pyproject.toml
        raise RuntimeError("install the declared 'langgraph' dependency to run the crew") from exc
    return StateGraph(CrewState)


def _terminals() -> tuple[Any, Any]:
    from langgraph.graph import END, START

    return START, END


def _add_assessment_loops(graph: Any, crew: TraceWriterCrew) -> None:
    """Wire the three pure state-reading routers, identically in every topology.

    ``after_assessment``, ``after_criteria`` and ``after_necessity`` read only
    state keys the nodes already write, so they are reusable across
    architectures without modification.
    """

    graph.add_conditional_edges(
        "coverage_assessment",
        crew.after_assessment,
        {"criteria_check": "criteria_check", "necessity_review": "necessity_review"},
    )
    graph.add_conditional_edges(
        "criteria_check",
        crew.after_criteria,
        {"coverage_assessment": "coverage_assessment", "necessity_review": "necessity_review"},
    )
    graph.add_conditional_edges(
        "necessity_review",
        crew.after_necessity,
        {"coverage_assessment": "coverage_assessment", "determination": "determination"},
    )


def _build_pipeline_crew(crew: TraceWriterCrew) -> Any:
    """The released topology. Unchanged, and it must stay that way."""

    start, end = _terminals()
    graph = _state_graph(crew)
    graph.add_node("intake", crew.intake)
    graph.add_node("clinical_extraction", crew.clinical_extraction)
    graph.add_node("coverage_assessment", crew.coverage_assessment)
    graph.add_node("criteria_check", crew.criteria_check)
    graph.add_node("necessity_review", crew.necessity_review)
    graph.add_node("determination", crew.determination)

    graph.add_edge(start, "intake")
    graph.add_edge("intake", "clinical_extraction")
    graph.add_conditional_edges(
        "clinical_extraction",
        crew.after_extraction,
        {
            "clinical_extraction": "clinical_extraction",
            "coverage_assessment": "coverage_assessment",
        },
    )
    _add_assessment_loops(graph, crew)
    graph.add_edge("determination", end)
    return graph.compile()


def _build_hierarchical_supervisor(crew: TraceWriterCrew) -> Any:
    """A routing agent picks the next specialist.

    The hook for this was already present and dead: triage prompted for and
    received ``{route, service_type, complete}`` and the response was discarded
    while the edge out of triage fired unconditionally.  Here the route is
    parsed into ``supervisor_next`` and that edge is conditional on it.
    ``after_extraction`` is reused verbatim -- only the node its
    ``coverage_assessment`` return is *mapped to* changes, so control comes back
    to the supervisor to pick the next specialist instead of jumping straight in.
    """

    start, end = _terminals()
    graph = _state_graph(crew)
    graph.add_node("supervisor", crew.supervisor)
    graph.add_node("clinical_extraction", crew.clinical_extraction)
    graph.add_node("coverage_assessment", crew.coverage_assessment)
    graph.add_node("criteria_check", crew.criteria_check)
    graph.add_node("necessity_review", crew.necessity_review)
    graph.add_node("determination", crew.determination)

    graph.add_edge(start, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        crew.after_supervisor,
        {
            "clinical_extraction": "clinical_extraction",
            "coverage_assessment": "coverage_assessment",
            "determination": "determination",
        },
    )
    graph.add_conditional_edges(
        "clinical_extraction",
        crew.after_extraction,
        {
            "clinical_extraction": "clinical_extraction",
            # Extraction finished: hand control back to the routing agent.
            "coverage_assessment": "supervisor",
        },
    )
    _add_assessment_loops(graph, crew)
    graph.add_edge("determination", end)
    return graph.compile()


def _build_react_single_agent(crew: TraceWriterCrew) -> Any:
    """One agent, one tool registry, one dispatch loop.

    Every super-step is a tool-choice model call followed by the dispatch of the
    chosen tool, so the observable plan is twice as long as the work it does.
    """

    start, end = _terminals()
    graph = _state_graph(crew)
    graph.add_node("react_agent", crew.react_step)
    graph.add_edge(start, "react_agent")
    graph.add_conditional_edges(
        "react_agent",
        crew.after_react,
        {"react_agent": "react_agent", "done": end},
    )
    return graph.compile()


ARCHITECTURE_SPECS: Mapping[str, ArchitectureSpec] = {
    "pipeline_crew": ArchitectureSpec(
        name="pipeline_crew",
        label="Pipeline crew (released topology)",
        summary=(
            "The released sequential six-node prior-authorization crew with "
            "adaptive extraction depth and repair/revision loops."
        ),
        node_ids=(
            "intake",
            "clinical_extraction",
            "coverage_assessment",
            "criteria_check",
            "necessity_review",
            "determination",
        ),
        edges=(
            EdgeSpec("__start__", "intake"),
            EdgeSpec("intake", "clinical_extraction"),
            EdgeSpec(
                "clinical_extraction",
                "clinical_extraction",
                "conditional",
                "insufficient evidence",
            ),
            EdgeSpec(
                "clinical_extraction",
                "coverage_assessment",
                "conditional",
                "evidence sufficient / canonical passes reached",
            ),
            EdgeSpec("coverage_assessment", "criteria_check", "conditional", "initial"),
            EdgeSpec("coverage_assessment", "necessity_review", "conditional", "after repair"),
            EdgeSpec(
                "criteria_check",
                "coverage_assessment",
                "conditional",
                "needs repair (adaptive only)",
            ),
            EdgeSpec("criteria_check", "necessity_review", "conditional", "criteria supported"),
            EdgeSpec(
                "necessity_review",
                "coverage_assessment",
                "conditional",
                "needs revision (adaptive only)",
            ),
            EdgeSpec("necessity_review", "determination", "conditional", "necessity confirmed"),
            EdgeSpec("determination", "__end__"),
        ),
        builder=_build_pipeline_crew,
    ),
    "hierarchical_supervisor": ArchitectureSpec(
        name="hierarchical_supervisor",
        label="Hierarchical supervisor",
        summary=(
            "A routing agent runs triage and then selects the next specialist; "
            "control returns to it once extraction completes."
        ),
        node_ids=(
            "supervisor",
            "clinical_extraction",
            "coverage_assessment",
            "criteria_check",
            "necessity_review",
            "determination",
        ),
        edges=(
            EdgeSpec("__start__", "supervisor"),
            EdgeSpec(
                "supervisor",
                "clinical_extraction",
                "conditional",
                "route=clinical_extraction",
            ),
            EdgeSpec(
                "supervisor",
                "coverage_assessment",
                "conditional",
                "route=coverage_assessment",
            ),
            EdgeSpec("supervisor", "determination", "conditional", "route=determination"),
            EdgeSpec(
                "clinical_extraction",
                "clinical_extraction",
                "conditional",
                "insufficient evidence",
            ),
            EdgeSpec(
                "clinical_extraction",
                "supervisor",
                "conditional",
                "evidence sufficient / canonical passes reached",
            ),
            EdgeSpec("coverage_assessment", "criteria_check", "conditional", "initial"),
            EdgeSpec("coverage_assessment", "necessity_review", "conditional", "after repair"),
            EdgeSpec(
                "criteria_check",
                "coverage_assessment",
                "conditional",
                "needs repair (adaptive only)",
            ),
            EdgeSpec("criteria_check", "necessity_review", "conditional", "criteria supported"),
            EdgeSpec(
                "necessity_review",
                "coverage_assessment",
                "conditional",
                "needs revision (adaptive only)",
            ),
            EdgeSpec("necessity_review", "determination", "conditional", "necessity confirmed"),
            EdgeSpec("determination", "__end__"),
        ),
        builder=_build_hierarchical_supervisor,
    ),
    "react_single_agent": ArchitectureSpec(
        name="react_single_agent",
        label="ReAct single agent",
        summary=(
            "One agent selects its own tool from the registry each step and "
            "dispatches it; the specialists become tools rather than nodes."
        ),
        node_ids=(
            "react_agent",
            "clinical_extraction",
            "coverage_assessment",
            "criteria_check",
            "necessity_review",
            "determination",
        ),
        edges=(
            EdgeSpec("__start__", "react_agent"),
            EdgeSpec(
                "react_agent",
                "clinical_extraction",
                "tool",
                "tool=extract_clinical_evidence",
            ),
            EdgeSpec("react_agent", "coverage_assessment", "tool", "tool=assess_coverage"),
            EdgeSpec("react_agent", "criteria_check", "tool", "tool=check_criteria"),
            EdgeSpec("react_agent", "necessity_review", "tool", "tool=review_necessity"),
            EdgeSpec("react_agent", "determination", "tool", "tool=issue_determination"),
            EdgeSpec("react_agent", "react_agent", "conditional", "determination not yet issued"),
            EdgeSpec("react_agent", "__end__", "conditional", "determination issued"),
        ),
        builder=_build_react_single_agent,
    ),
}


def architecture_spec(architecture: str | None = None) -> ArchitectureSpec:
    """Look up one crew architecture by name."""

    name = DEFAULT_ARCHITECTURE if architecture is None else str(architecture).strip().lower()
    try:
        return ARCHITECTURE_SPECS[name]
    except KeyError as exc:
        allowed = ", ".join(ARCHITECTURES)
        raise ValueError(
            f"unknown crew architecture {architecture!r}; expected one of: {allowed}"
        ) from exc


def canonical_step_types(architecture: str | None = None) -> tuple[str, ...]:
    """The scripted plan one architecture must emit under a fixed structure."""

    return architecture_spec(architecture).canonical_step_types


def build_crew_graph(crew: TraceWriterCrew, architecture: str | None = None) -> Any:
    """Compile the named crew topology.

    Defaults to ``pipeline_crew``, which is the released topology, so every
    existing caller and archived configuration keeps its behaviour.
    """

    return architecture_spec(architecture).builder(crew)


def topology_manifest(architectures: Sequence[str] | None = None) -> dict[str, Any]:
    """Public per-architecture topology description (safe for the console/API)."""

    names = tuple(architectures) if architectures else ARCHITECTURES
    return {
        "engine": "langgraph",
        "default_architecture": DEFAULT_ARCHITECTURE,
        "architectures": {name: architecture_spec(name).to_dict() for name in names},
    }


__all__ = [
    "ARCHITECTURES",
    "ARCHITECTURE_SPECS",
    "CANONICAL_RESEARCH_HOPS",
    "CANONICAL_STEP_TYPES",
    "DEFAULT_ARCHITECTURE",
    "NODES",
    "ArchitectureSpec",
    "EdgeSpec",
    "NodeSpec",
    "architecture_spec",
    "build_crew_graph",
    "canonical_step_types",
    "topology_manifest",
]

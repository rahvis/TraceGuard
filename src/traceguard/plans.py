"""The frozen registry of legitimate canonical plans.

Why this module exists
----------------------

Before crew topology was parameterizable there was exactly one canonical plan,
and :mod:`traceguard.verifier` knew it *independently of the receipt*: it simply
required ``policy.canonical_research_hops == 7``.  That independence is the
whole reason a receipt cannot lie about its plan.  A receipt that could name an
arbitrary self-declared plan and thereby pass would turn the honesty invariant
into a tautology.

Once plans become plural, the verifier needs the same independent knowledge of
the *set* of legitimate plans.  This module is that knowledge, frozen in code
and keyed by a content-addressed digest over the plan itself.  The digest is
committed inside the **signed** receipt body, so a prover can only select from
this registry; it cannot mint a plan.

Deliberate constraints
----------------------

* The registry is **in code**, never read from a file, an environment variable,
  or a receipt field.  A prover-influenced source would reintroduce exactly the
  weakness the digest is meant to close.
* This module imports nothing from the package except the pure canonical-JSON
  helper in :mod:`traceguard._canonical`.  The verifier is a published,
  dependency-free artifact that reviewers run against already-issued receipts;
  it must never acquire a transitive dependency on ``langgraph`` (via
  :mod:`traceguard.graph` or :mod:`traceguard.agents`) or on the scientific
  stack.
* Only canonicalized conditions carry a plan commitment.  ``ADAPTIVE`` receipts
  deliberately carry none -- see :data:`ADAPTIVE_PLAN_TEMPLATE`.

The legacy plan
---------------

:data:`LEGACY_CREW_STEP_TYPES` is registered as a plan in its own right because
the archive proves the two shapes were being conflated.  All ten archived
``full_pad`` receipts reproduce their signed ``trace_sha256`` under the legacy
vocabulary and none under the current one, yet every one of them declares
``plan_template = "CANON-v1"`` with seven hops -- and today's verifier accepts
both, because seven hops and twelve steps is all it can see.  Registering the
legacy shape separately makes the two *nameable and distinguishable* instead of
silently equal.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ._canonical import canonical_json, sha256_hex

__all__ = [
    "ADAPTIVE_PLAN_TEMPLATE",
    "CANONICAL_RESEARCH_HOPS",
    "HIERARCHICAL_SUPERVISOR_STEP_TYPES",
    "LEGACY_CREW_STEP_TYPES",
    "PIPELINE_CREW_STEP_TYPES",
    "PLANS_BY_ARCHITECTURE",
    "PLANS_BY_ID",
    "PLAN_REGISTRY",
    "PLAN_SCHEMA",
    "REACT_SINGLE_AGENT_STEP_TYPES",
    "PlanSpec",
    "plan_ids",
    "plan_digest",
    "plan_for_architecture",
    "plan_for_step_types",
    "resolve_plan",
    "step_types_match",
]

# Bumping this string re-keys every digest, so it is part of the plan identity.
PLAN_SCHEMA = "traceguard.plan.v1"

# The published CANON-v1 hop count. Every registered canonical plan runs exactly
# this many clinical-extraction passes; config.py and the verifier both pin it.
CANONICAL_RESEARCH_HOPS = 7

# ADAPTIVE runs commit to no step sequence, and that omission is load-bearing
# rather than an oversight. The space of plausible adaptive sequences over this
# workload is tiny and enumerable, so a digest over one would invert trivially
# and hand the host exactly the structure channel the paper attacks.
ADAPTIVE_PLAN_TEMPLATE = "ADAPTIVE"


# ---------------------------------------------------------------------------
# Step-type vocabularies
# ---------------------------------------------------------------------------

# PINNED LITERAL. Archived receipts commit to this exact sequence through their
# signed trace_sha256, so it is written out element by element rather than
# generated, and checked against the generative form below. Changing any
# element silently reinterprets every archived run; adding an architecture must
# never touch it.
PIPELINE_CREW_STEP_TYPES: tuple[str, ...] = (
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

if PIPELINE_CREW_STEP_TYPES != (
    "intake",
    *("clinical_extraction" for _ in range(CANONICAL_RESEARCH_HOPS)),
    "coverage_assessment",
    "criteria_check",
    "necessity_review",
    "determination",
):  # pragma: no cover - a module-import guard, not a branch
    raise RuntimeError(
        "PIPELINE_CREW_STEP_TYPES drifted from the pinned twelve-step CANON-v1 plan"
    )

# The pre-rename vocabulary that the archived full_pad receipts actually hash
# under. Registered so the two twelve-step, seven-hop shapes can be told apart
# by name instead of being conflated by the hop count alone.
LEGACY_CREW_STEP_TYPES: tuple[str, ...] = (
    "orchestrator",
    "research",
    "research",
    "research",
    "research",
    "research",
    "research",
    "research",
    "draft",
    "citation",
    "review",
    "editor",
)

# A routing agent picks the next specialist, so it is observed once before
# extraction and once after it, and the plan is thirteen steps rather than
# twelve.
HIERARCHICAL_SUPERVISOR_STEP_TYPES: tuple[str, ...] = (
    "supervisor",
    *("clinical_extraction" for _ in range(CANONICAL_RESEARCH_HOPS)),
    "supervisor",
    "coverage_assessment",
    "criteria_check",
    "necessity_review",
    "determination",
)

# One agent choosing its own tool each step: every tool execution is preceded by
# an observable tool-choice step, so the plan is twice as long as the work it
# does.
REACT_SINGLE_AGENT_STEP_TYPES: tuple[str, ...] = (
    *(
        step
        for _ in range(CANONICAL_RESEARCH_HOPS)
        for step in ("react_agent", "clinical_extraction")
    ),
    "react_agent",
    "coverage_assessment",
    "react_agent",
    "criteria_check",
    "react_agent",
    "necessity_review",
    "react_agent",
    "determination",
)


# ---------------------------------------------------------------------------
# Plan specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlanSpec:
    """One legitimate canonical plan.

    ``digest`` is content-addressed over exactly the fields a receipt declares
    about its plan, which is what stops a receipt from renaming a plan or
    misstating its hop count while still resolving.
    """

    plan_id: str
    step_types: tuple[str, ...]
    canonical_research_hops: int
    architecture: str | None
    description: str

    @property
    def digest(self) -> str:
        return plan_digest(
            plan_id=self.plan_id,
            step_types=self.step_types,
            canonical_research_hops=self.canonical_research_hops,
        )

    @property
    def step_count(self) -> int:
        return len(self.step_types)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "plan_digest": self.digest,
            "architecture": self.architecture,
            "canonical_research_hops": self.canonical_research_hops,
            "step_count": self.step_count,
            "step_types": list(self.step_types),
            "description": self.description,
        }


def plan_digest(
    *,
    plan_id: str,
    step_types: Sequence[str],
    canonical_research_hops: int,
) -> str:
    """Content-addressed digest committed inside the signed receipt body."""

    return hashlib.sha256(
        canonical_json(
            {
                "schema": PLAN_SCHEMA,
                "plan_id": str(plan_id),
                "canonical_research_hops": int(canonical_research_hops),
                "step_types": [str(step) for step in step_types],
            }
        )
    ).hexdigest()


_SPECS: tuple[PlanSpec, ...] = (
    PlanSpec(
        plan_id="CANON-v1",
        step_types=PIPELINE_CREW_STEP_TYPES,
        canonical_research_hops=CANONICAL_RESEARCH_HOPS,
        architecture="pipeline_crew",
        description=(
            "The released sequential six-node prior-authorization crew: intake, "
            "seven clinical-extraction passes, coverage assessment, criteria "
            "check, necessity review, determination."
        ),
    ),
    PlanSpec(
        plan_id="CANON-v0-legacy",
        step_types=LEGACY_CREW_STEP_TYPES,
        canonical_research_hops=CANONICAL_RESEARCH_HOPS,
        architecture=None,
        description=(
            "The pre-rename twelve-step vocabulary that the archived full_pad "
            "receipts hash under. Registered so it is distinguishable from "
            "CANON-v1 rather than conflated with it by hop count."
        ),
    ),
    PlanSpec(
        plan_id="CANON-HIER-v1",
        step_types=HIERARCHICAL_SUPERVISOR_STEP_TYPES,
        canonical_research_hops=CANONICAL_RESEARCH_HOPS,
        architecture="hierarchical_supervisor",
        description=(
            "A routing agent selects the next specialist; observed once before "
            "the extraction passes and once after them."
        ),
    ),
    PlanSpec(
        plan_id="CANON-REACT-v1",
        step_types=REACT_SINGLE_AGENT_STEP_TYPES,
        canonical_research_hops=CANONICAL_RESEARCH_HOPS,
        architecture="react_single_agent",
        description=(
            "A single agent selects its own tool each step, so every tool "
            "execution is preceded by an observable tool-choice step."
        ),
    ),
)

#: Digest -> plan. The verifier's independent knowledge of the legitimate set.
PLAN_REGISTRY: Mapping[str, PlanSpec] = {spec.digest: spec for spec in _SPECS}

#: Plan id -> plan.
PLANS_BY_ID: Mapping[str, PlanSpec] = {spec.plan_id: spec for spec in _SPECS}

#: Architecture name -> plan, for the architectures that have one.
PLANS_BY_ARCHITECTURE: Mapping[str, PlanSpec] = {
    spec.architecture: spec for spec in _SPECS if spec.architecture is not None
}

if len(PLAN_REGISTRY) != len(_SPECS) or len(PLANS_BY_ID) != len(_SPECS):
    # Two plans sharing a digest or an id would make resolution ambiguous, which
    # is the failure this registry exists to prevent.
    raise RuntimeError("the frozen plan registry contains a duplicate digest or id")


def plan_ids() -> tuple[str, ...]:
    """Every registered plan id, sorted."""

    return tuple(sorted(PLANS_BY_ID))


def resolve_plan(digest: Any) -> PlanSpec | None:
    """Resolve a receipt-declared plan digest, or ``None`` if it is not ours."""

    if not isinstance(digest, str):
        return None
    return PLAN_REGISTRY.get(digest.strip().lower())


def plan_for_architecture(architecture: str) -> PlanSpec:
    """The canonical plan a crew architecture emits under scripted autonomy."""

    try:
        return PLANS_BY_ARCHITECTURE[architecture]
    except KeyError as exc:
        allowed = ", ".join(sorted(PLANS_BY_ARCHITECTURE))
        raise ValueError(
            f"unknown crew architecture {architecture!r}; expected one of: {allowed}"
        ) from exc


def plan_for_step_types(step_types: Sequence[str]) -> PlanSpec | None:
    """Name the plan an observed step sequence actually is, if any.

    This is how a reviewer holding a trace resolves the archive's ambiguity: a
    twelve-step, seven-hop sequence is only ``CANON-v1`` if it really carries
    the current vocabulary, and ``CANON-v0-legacy`` if it carries the old one.
    """

    observed = tuple(str(step) for step in step_types)
    for spec in _SPECS:
        if spec.step_types == observed:
            return spec
    return None


def step_types_match(digest: Any, step_types: Sequence[str]) -> bool:
    """Whether an observed step sequence is the one a plan digest commits to."""

    spec = resolve_plan(digest)
    if spec is None:
        return False
    return spec.step_types == tuple(str(step) for step in step_types)


def expected_full_pad_trace_digest(
    digest: Any, step_deadline_s: Any, egress_ceiling_bytes: Any
) -> str | None:
    """The one trace digest a full-pad run of this plan can possibly produce.

    Under full padding the observable is a constant function of the plan and its
    public profile: every step releases the deadline as its duration and the
    ceiling as its egress size, and ``wall_time_s`` is ``(index + 1) * deadline``.
    ``TraceStep.to_dict`` is a closed allowlist, so the whole digest is
    determined by (step types, deadline, ceiling) and nothing else.

    That is what lets a verifier *recompute* the digest instead of trusting the
    prover's plan claim. Without this, a receipt could commit to any registered
    plan while having run a different one -- every consistency check would still
    pass, because they only compare the claim against itself.

    Returns ``None`` when the plan or profile is unusable, so the caller can
    report "not checkable" rather than "checked and fine".
    """

    spec = resolve_plan(digest)
    if spec is None:
        return None
    try:
        deadline = float(step_deadline_s)
        ceiling = int(egress_ceiling_bytes)
    except (TypeError, ValueError):
        return None
    steps = [
        {
            "index": index,
            "step_type": step_type,
            "wall_time_s": round((index + 1) * deadline, 6),
            "duration_s": round(deadline, 6),
            "egress_bytes": ceiling,
        }
        for index, step_type in enumerate(spec.step_types)
    ]
    return sha256_hex(canonical_json({"steps": steps}))


def registry_manifest() -> dict[str, Any]:
    """Public description of the frozen plan set (safe for the console/API)."""

    return {
        "schema": PLAN_SCHEMA,
        "adaptive_plan_template": ADAPTIVE_PLAN_TEMPLATE,
        "adaptive_commits_to_a_step_sequence": False,
        "plans": [spec.to_dict() for spec in _SPECS],
    }

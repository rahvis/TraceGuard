"""Prior-authorization crew nodes used by the LangGraph runtime.

The crew replicates the agent roles of Microsoft's Prior-Authorization
Multi-Agent Solution Accelerator (intake/compliance, clinical review, coverage
assessment, and a synthesis decision) as a sequential, adaptive LangGraph plan.
The adaptivity (how many extraction passes, whether criteria are repaired, and
whether a medical-necessity review triggers a revision) is decided at runtime by
the LLM over the private request and referral sources; that data-dependent
control flow is exactly the host-observable trace the paper studies.

The same node implementations serve all three crew architectures assembled in
:mod:`traceguard.graph`.  Beyond the six released nodes this module adds:

* :meth:`TraceWriterCrew.supervisor` -- the routing agent used by
  ``hierarchical_supervisor``.  It runs the same triage prompt the released
  ``intake`` node runs and then *uses* the ``route`` field that node discarded.
* :data:`REACT_TOOLS` and :meth:`TraceWriterCrew.react_step` -- the tool
  registry and dispatch loop used by ``react_single_agent``.

Prompts, referral sources, assessments, and determinations remain transient in
graph state.  Only :class:`TraceRecorder` receives boundary metadata.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, TypedDict

from .config import Settings
from .corpus import CorpusCatalog
from .domains import DomainProfile, profile_for
from .instrumentation import TraceRecorder
from .providers import LLMProvider
from .types import (
    Autonomy,
    Condition,
    MedicalCase,
    MedicalDocument,
    ProviderRequest,
    ProviderResponse,
    fixture_target_hops,
)


class TokenBudgetExceeded(RuntimeError):
    """Soft per-matter cost guardrail reached (contains no private payload)."""


class CrewState(TypedDict, total=False):
    case: MedicalCase
    condition: Condition
    # Orthogonal to `condition`: how much of the control flow may depend on the
    # data. `condition` governs what the release looks like.
    autonomy: Autonomy
    seed: int
    research_hops: int  # extraction passes (kept for config/metrics compatibility)
    retrieved: list[MedicalDocument]
    extraction_notes: list[str]
    extraction_done: bool
    assessment: str
    assessment_mode: str
    assessment_feedback: str
    assessment_next: str
    criteria_checks: int
    criteria_repairs: int
    criteria_next: str
    necessity_checks: int
    necessity_revisions: int
    necessity_next: str
    # hierarchical_supervisor: the specialist the routing agent selected, and
    # how many routing decisions it has made. `supervisor_next` is the parsed
    # `route` field that the released intake node computed and threw away.
    supervisor_next: str
    supervisor_visits: int
    # react_single_agent: which crew node the last dispatched tool ran, the tool
    # name the agent picked, and whether the determination has been issued.
    react_last_node: str
    react_tool: str
    react_done: bool
    answer: str


def _json_object(text: str) -> Mapping[str, Any]:
    """Best-effort JSON extraction without reflecting malformed private text."""

    try:
        value = json.loads(text)
        return value if isinstance(value, Mapping) else {}
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(text[start : end + 1])
                return value if isinstance(value, Mapping) else {}
            except json.JSONDecodeError:
                return {}
        return {}


def _unique_documents(documents: list[MedicalDocument]) -> list[MedicalDocument]:
    result: list[MedicalDocument] = []
    seen: set[str] = set()
    for document in documents:
        if document.document_id not in seen:
            seen.add(document.document_id)
            result.append(document)
    return result


def _source_packet(documents: list[MedicalDocument], *, max_chars_each: int = 1800) -> str:
    return "\n\n".join(
        f"SOURCE {document.document_id} — {document.title}\n{document.text[:max_chars_each]}"
        for document in _unique_documents(documents)
    )


class TraceWriterCrew:
    """Node implementations for the six-node prior-authorization graph.

    Prompts, source documents, assessments, and determinations remain transient
    in graph state.  Only :class:`TraceRecorder` receives boundary metadata.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        provider: LLMProvider,
        corpus: CorpusCatalog,
        recorder: TraceRecorder,
        profile: DomainProfile | None = None,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.corpus = corpus
        self.recorder = recorder
        # Only prompt text is domain-coupled; node ids, routers, tool
        # vocabulary and step types are role slots shared by every domain, so
        # a second task family cannot become a second architecture.
        self.profile = profile or profile_for(getattr(corpus, "domain", None))

    def _call(
        self,
        *,
        step_type: str,
        role: str,
        model: str,
        max_tokens: int,
        system_prompt: str,
        user_prompt: str,
        state: CrewState,
        fixture_context: Mapping[str, Any] | None = None,
    ) -> ProviderResponse:
        if self.recorder.token_budget_used >= self.settings.max_tokens_per_matter:
            raise TokenBudgetExceeded("per-matter token guardrail reached")
        # Ingress-shaping probe. Unlike the egress padding, which rides in an inert
        # transport field and provably cannot change the completion, an ingress
        # request has to be an instruction inside the prompt: the size of the
        # response is the provider's to choose, so the only lever the guest has is
        # to ask. That asymmetry is exactly Prop. 2, and measuring how well the ask
        # works is what this knob is for. Off by default.
        fixed = self.settings.ingress_response_chars
        if fixed:
            system_prompt = (
                f"{system_prompt}\n\nOUTPUT LENGTH REQUIREMENT: your entire reply must "
                f"be exactly {fixed} characters long. If your content is shorter, pad it "
                f"with trailing spaces until it is exactly {fixed} characters. If longer, "
                f"truncate it to exactly {fixed} characters. This requirement overrides "
                f"brevity but never correctness of the required fields."
            )
        request = ProviderRequest(
            role=role,
            model=model,
            max_tokens=min(max_tokens, self.settings.max_tokens_per_matter),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            seed=int(state.get("seed", 0)),
            timeout_s=(
                min(self.settings.provider_timeout_s, self.settings.step_deadline_s)
                if self.recorder.shield.full_padding
                else self.settings.provider_timeout_s
            ),
            fixture_context=fixture_context or {},
            # Bound the request direction when the condition calls for it.  The
            # provider does the padding, because that is where the body is
            # serialized and therefore where its size is decided.
            pad_request_to_bytes=self.recorder.shield.pad_request_to_bytes,
        )
        return self.recorder.observe(step_type, lambda: self.provider.complete(request))

    # -- triage ------------------------------------------------------------- #


    @staticmethod
    def _initial_case_state() -> CrewState:
        """The counters every architecture must start a matter with."""

        return {
            "research_hops": 0,
            "retrieved": [],
            "extraction_notes": [],
            "criteria_checks": 0,
            "criteria_repairs": 0,
            "necessity_checks": 0,
            "necessity_revisions": 0,
            "assessment_mode": "initial",
            "assessment_next": "criteria_check",
        }

    def intake(self, state: CrewState) -> CrewState:
        """Intake & compliance triage (orchestrator + Compliance Agent).

        The triage response carries ``{route, service_type, complete}``, and in
        this released ``pipeline_crew`` topology it is deliberately discarded:
        the edge out of intake fires unconditionally, so the plan does not
        branch here.  ``hierarchical_supervisor`` parses the same response --
        see :meth:`supervisor`.
        """

        case = state["case"]
        self._call(
            step_type="intake",
            role="intake",
            model=self.settings.model_fast,
            max_tokens=self.settings.max_tokens_fast,
            system_prompt=self.profile.triage_system,
            user_prompt=f"{self.profile.request_label} (do not repeat):\n{case.query}",
            state=state,
            fixture_context={"case": case},
        )
        return self._initial_case_state()

    # -- hierarchical supervisor -------------------------------------------- #

    #: Specialists the routing agent is allowed to name, by node id.
    SUPERVISOR_SPECIALISTS: tuple[str, ...] = (
        "clinical_extraction",
        "coverage_assessment",
        "determination",
    )

    @staticmethod
    def _supervisor_admissible(state: CrewState) -> tuple[str, ...]:
        """Specialists whose preconditions the current state satisfies.

        The routing agent's choice is honoured whenever it lands in this set.
        The set exists because an LLM route is not a termination proof: a
        supervisor that could re-enter extraction after it had finished, or
        jump to determination before any assessment existed, would either not
        terminate or would emit a trace that is not comparable with the other
        architectures.
        """

        if not state.get("extraction_done", False):
            return ("clinical_extraction",)
        if not str(state.get("assessment", "")).strip():
            return ("coverage_assessment",)
        return ("determination",)

    def supervisor(self, state: CrewState) -> CrewState:
        """Routing agent: triage, then select which specialist runs next.

        This is the node the released topology left dead. ``intake`` already
        prompted for and received a ``route`` field; here it is parsed into
        ``supervisor_next`` and the outgoing edge is conditional on it, so the
        selection of the next specialist is a model output rather than a
        hardcoded edge.
        """

        case = state["case"]
        visits = int(state.get("supervisor_visits", 0))
        admissible = self._supervisor_admissible(state)
        response = self._call(
            step_type="supervisor",
            role="intake",
            model=self.settings.model_fast,
            max_tokens=self.settings.max_tokens_fast,
            system_prompt=self.profile.triage_system,
            user_prompt=(
                f"{self.profile.request_label} (do not repeat):\n{case.query}\n\n"
                f"ROUTING DECISION {visits + 1}. Specialists available now: "
                f"{', '.join(admissible)}."
            ),
            state=state,
            fixture_context={
                "case": case,
                "supervisor_visits": visits,
                "admissible": admissible,
            },
        )
        decision = _json_object(response.text)
        route = str(decision.get("route", "") or "").strip().lower()
        target = route if route in admissible else admissible[0]
        updates: CrewState = {"supervisor_next": target, "supervisor_visits": visits + 1}
        if visits == 0:
            updates.update(self._initial_case_state())
        return updates

    def clinical_extraction(self, state: CrewState) -> CrewState:
        """Clinical Reviewer Agent: extract clinical evidence over adaptive passes."""

        case = state["case"]
        condition = state["condition"]
        hop = int(state.get("research_hops", 0)) + 1
        retrieved = list(state.get("retrieved", []))
        document = self.corpus.retrieve(case, case.query, hop, limit=1)[0]
        retrieved.append(document)
        notes = list(state.get("extraction_notes", []))
        packet = _source_packet(_unique_documents(retrieved), max_chars_each=1200)
        response = self._call(
            step_type="clinical_extraction",
            role="clinical_extraction",
            model=self.settings.model_fast,
            max_tokens=self.settings.max_tokens_fast,
            system_prompt=self.profile.extraction_system,
            user_prompt=(
                f"{self.profile.request_label}:\n{case.query}\n\n"
                f"{self.profile.sources_label}:\n{packet}\n\n"
                + (
                    f"This is {self.profile.extraction_pass_label} {hop} of at most "
                    f"{self.settings.max_research_hops}."
                    if self.settings.announce_pass_budget
                    else f"This is {self.profile.extraction_pass_label} {hop}."
                )
            ),
            state=state,
            fixture_context={
                "case": case,
                "hop": hop,
                "target_hops": int(
                    case.fixture.get(
                        "research_hops", fixture_target_hops(case.sensitivity_level)
                    )
                ),
                "documents": (document,),
            },
        )
        decision = _json_object(response.text)
        sufficient = bool(decision.get("sufficient", False))
        note = decision.get("notes")
        if isinstance(note, str) and note:
            notes.append(note[:1000])
        autonomy = state.get("autonomy") or Autonomy.default_for(condition)
        if autonomy is Autonomy.SCRIPTED:
            # Data-independent depth: always the full canonical plan.
            done = hop >= self.settings.canonical_research_hops
        else:
            done = (
                hop >= self.settings.min_research_hops and sufficient
            ) or hop >= self.settings.max_research_hops
        return {
            "research_hops": hop,
            "retrieved": retrieved,
            "extraction_notes": notes,
            "extraction_done": done,
        }

    def coverage_assessment(self, state: CrewState) -> CrewState:
        """Coverage Assessment Agent: map evidence to payer policy criteria."""

        case = state["case"]
        documents = _unique_documents(list(state.get("retrieved", [])))
        mode = str(state.get("assessment_mode", "initial"))
        prior = str(state.get("assessment", ""))
        feedback = str(state.get("assessment_feedback", ""))
        packet = _source_packet(documents)
        response = self._call(
            step_type="coverage_assessment",
            role="coverage_assessment",
            model=self.settings.model_deep,
            max_tokens=self.settings.max_tokens_deep,
            system_prompt=self.profile.assessment_system,
            user_prompt=(
                f"MODE: {mode}\n{self.profile.request_label}:\n{case.query}\n\n"
                f"SOURCES:\n{packet}\n\n"
                f"PRIOR ASSESSMENT (may be empty):\n{prior}\n\nREVIEW FEEDBACK:\n{feedback}"
            ),
            state=state,
            fixture_context={
                "case": case,
                "documents": tuple(documents),
                "assessment_mode": mode,
                "prior_assessment": prior,
                "feedback": feedback,
            },
        )
        assessment = response.text.strip() or prior
        return {"assessment": assessment, "assessment_mode": "initial", "assessment_feedback": ""}

    def criteria_check(self, state: CrewState) -> CrewState:
        """Compliance/criteria verification: is each criterion source-supported?"""

        case = state["case"]
        documents = _unique_documents(list(state.get("retrieved", [])))
        checks = int(state.get("criteria_checks", 0))
        response = self._call(
            step_type="criteria_check",
            role="criteria_check",
            model=self.settings.model_fast,
            max_tokens=self.settings.max_tokens_fast,
            system_prompt=self.profile.criteria_system,
            user_prompt=(
                f"ASSESSMENT:\n{state.get('assessment', '')}\n\n"
                f"SOURCES:\n{_source_packet(documents)}"
            ),
            state=state,
            fixture_context={
                "case": case,
                "assessment": state.get("assessment", ""),
                "documents": tuple(documents),
                "criteria_checks": checks,
            },
        )
        decision = _json_object(response.text)
        repairs = int(state.get("criteria_repairs", 0))
        autonomy = state.get("autonomy") or Autonomy.default_for(state["condition"])
        budget = self.settings.max_criteria_repairs * autonomy.loop_budget_multiplier
        needs_repair = bool(decision.get("needs_repair", False)) and repairs < budget
        updates: CrewState = {"criteria_checks": checks + 1}
        if needs_repair:
            updates.update(
                {
                    "criteria_repairs": repairs + 1,
                    "criteria_next": "coverage_assessment",
                    "assessment_mode": "criteria_repair",
                    "assessment_feedback": str(decision.get("feedback", "repair criteria"))[:1000],
                    "assessment_next": "criteria_check",
                }
            )
        else:
            updates["criteria_next"] = "necessity_review"
        return updates

    def necessity_review(self, state: CrewState) -> CrewState:
        """Medical-necessity review (licensed-clinician sign-off / revision)."""

        case = state["case"]
        checks = int(state.get("necessity_checks", 0))
        if self.settings.model_review is None:
            response = self.recorder.observe(
                "necessity_review",
                lambda: ProviderResponse(
                    text='{"needs_revision":false,"feedback":"review pass disabled"}',
                    response_bytes=0,
                    request_bytes=0,
                    stop_reason="disabled",
                ),
            )
        else:
            response = self._call(
                step_type="necessity_review",
                role="necessity_review",
                model=self.settings.model_review,
                max_tokens=self.settings.max_tokens_fast,
                system_prompt=self.profile.necessity_system,
                user_prompt=(
                    f"{self.profile.request_label}:\n{case.query}\n\n"
                    f"ASSESSMENT:\n{state.get('assessment', '')}"
                ),
                state=state,
                fixture_context={
                    "case": case,
                    "assessment": state.get("assessment", ""),
                    "necessity_checks": checks,
                },
            )
        decision = _json_object(response.text)
        revisions = int(state.get("necessity_revisions", 0))
        autonomy = state.get("autonomy") or Autonomy.default_for(state["condition"])
        budget = self.settings.max_necessity_revisions * autonomy.loop_budget_multiplier
        needs_revision = bool(decision.get("needs_revision", False)) and revisions < budget
        updates: CrewState = {"necessity_checks": checks + 1}
        if needs_revision:
            updates.update(
                {
                    "necessity_revisions": revisions + 1,
                    "necessity_next": "coverage_assessment",
                    "assessment_mode": "necessity_revision",
                    "assessment_feedback": str(
                        decision.get("feedback", "revise assessment")
                    )[:1000],
                    "assessment_next": "necessity_review",
                }
            )
        else:
            updates["necessity_next"] = "determination"
        return updates

    def determination(self, state: CrewState) -> CrewState:
        """Synthesis Decision Agent: issue the APPROVE / PEND determination."""

        case = state["case"]
        response = self._call(
            step_type="determination",
            role="determination",
            model=self.settings.model_deep,
            max_tokens=self.settings.max_tokens_deep,
            system_prompt=self.profile.determination_system,
            user_prompt=(
                f"{self.profile.request_label}:\n{case.query}\n\n"
                f"{self.profile.approved_label}:\n{state.get('assessment', '')}"
            ),
            state=state,
            fixture_context={"case": case, "assessment": state.get("assessment", "")},
        )
        answer = response.text.strip()
        if not answer and self.recorder.fail_closed:
            answer = "Run failed closed; no synthetic determination was released."
        return {"answer": answer}

    # -- ReAct single agent -------------------------------------------------- #

    def react_step(self, state: CrewState) -> CrewState:
        """One ReAct super-step: choose a tool, then dispatch it.

        Unlike the other two architectures there is no per-specialist node here.
        A single agent is asked which tool to call, and :data:`REACT_TOOLS`
        dispatches the answer.  The tool-choice call is itself an observable
        step, which is why this architecture's canonical plan is twice as long
        as the work it performs.

        The agent's proposal is executed whenever it is admissible.  Admissible
        means "a tool whose preconditions the state satisfies", computed by
        :meth:`_react_admissible` from the same pure state readers the other
        topologies route on -- so the tool order is not hardcoded here either.
        """

        admissible = self._react_admissible(state)
        if not admissible:
            return {"react_done": True}
        case = state["case"]
        response = self._call(
            step_type="react_agent",
            role="react_agent",
            model=self.settings.model_fast,
            max_tokens=self.settings.max_tokens_fast,
            system_prompt=(
                self.profile.react_system_prefix
                + ", ".join(sorted(REACT_TOOLS))
                + self.profile.react_system_suffix
            ),
            user_prompt=(
                f"{self.profile.request_label}:\n{case.query}\n\n"
                f"{self.profile.passes_label}: {int(state.get('research_hops', 0))} "
                f"(minimum {self.settings.min_research_hops}, "
                f"maximum {self.settings.max_research_hops})\n"
                f"ASSESSMENT DRAFTED: {bool(str(state.get('assessment', '')).strip())}\n"
                f"CRITERIA CHECKS: {int(state.get('criteria_checks', 0))}\n"
                f"NECESSITY CHECKS: {int(state.get('necessity_checks', 0))}\n"
                f"TOOLS AVAILABLE NOW: {', '.join(admissible)}"
            ),
            state=state,
            fixture_context={
                "case": case,
                "admissible": admissible,
                "research_hops": int(state.get("research_hops", 0)),
            },
        )
        decision = _json_object(response.text)
        proposed = str(decision.get("tool", "") or "").strip().lower()
        tool_name = proposed if proposed in admissible else admissible[0]
        node, tool = REACT_TOOLS[tool_name]
        updates: CrewState = dict(tool(self, state))  # type: ignore[assignment]
        if node != "clinical_extraction":
            # Leaving the retrieval loop is itself the agent's decision to stop
            # extracting; record it so the loop cannot be re-entered.
            updates.setdefault("extraction_done", True)
        updates["react_last_node"] = node
        updates["react_tool"] = tool_name
        updates["react_done"] = node == "determination"
        return updates

    def _react_admissible(self, state: CrewState) -> tuple[str, ...]:
        """Tool names the state admits next, in the agent's preference order.

        Derived from the pipeline's own routers rather than from a second copy
        of the plan, so the two architectures cannot drift apart.
        """

        planned = self._react_planned_node(state)
        if planned is None:
            return ()
        names = [_REACT_TOOL_FOR_NODE[planned]]
        autonomy = state.get("autonomy") or Autonomy.default_for(state["condition"])
        if (
            planned == "clinical_extraction"
            and autonomy is not Autonomy.SCRIPTED
            and int(state.get("research_hops", 0)) >= self.settings.min_research_hops
        ):
            # A genuinely autonomous ReAct agent may stop retrieving early and
            # go straight to assessment. Under SCRIPTED autonomy it may not:
            # the plan has to be a data-independent constant.
            names.append("assess_coverage")
        return tuple(names)

    @staticmethod
    def _react_planned_node(state: CrewState) -> str | None:
        """The crew node the pipeline routers would run next, or None if done."""

        last = state.get("react_last_node")
        if last is None:
            return "clinical_extraction"
        if last == "clinical_extraction":
            return TraceWriterCrew.after_extraction(state)
        if last == "coverage_assessment":
            return TraceWriterCrew.after_assessment(state)
        if last == "criteria_check":
            return TraceWriterCrew.after_criteria(state)
        if last == "necessity_review":
            return TraceWriterCrew.after_necessity(state)
        return None

    # -- routers ------------------------------------------------------------- #
    #
    # after_assessment, after_criteria and after_necessity are pure state
    # readers: they touch no settings, no provider, and no case payload, which
    # is why all three architectures reuse them unchanged.

    @staticmethod
    def after_extraction(state: CrewState) -> str:
        if state.get("extraction_done", False):
            return "coverage_assessment"
        return "clinical_extraction"

    @staticmethod
    def after_assessment(state: CrewState) -> str:
        return str(state.get("assessment_next", "criteria_check"))

    @staticmethod
    def after_criteria(state: CrewState) -> str:
        return str(state.get("criteria_next", "necessity_review"))

    @staticmethod
    def after_necessity(state: CrewState) -> str:
        return str(state.get("necessity_next", "determination"))

    @staticmethod
    def after_supervisor(state: CrewState) -> str:
        return str(state.get("supervisor_next", "clinical_extraction"))

    @staticmethod
    def after_react(state: CrewState) -> str:
        return "done" if state.get("react_done", False) else "react_agent"


# --------------------------------------------------------------------------
# The ReAct tool registry
# --------------------------------------------------------------------------
#
# There was no tool registry anywhere in this package: retrieval was a direct
# ``corpus.retrieve`` call inside ``clinical_extraction`` and every other
# specialist was reachable only as a graph node. A single agent choosing its own
# tool needs those capabilities addressable by name, so this maps a tool name to
# (the crew node it corresponds to, the callable that performs it).
#
# The callables are the existing node implementations, unmodified. That is
# deliberate: the ReAct architecture must do the same work as the pipeline, or a
# step-sequence comparison between the two measures the work rather than the
# topology.

ReactTool = tuple[str, Any]

REACT_TOOLS: Mapping[str, ReactTool] = {
    "extract_clinical_evidence": (
        "clinical_extraction",
        TraceWriterCrew.clinical_extraction,
    ),
    "assess_coverage": ("coverage_assessment", TraceWriterCrew.coverage_assessment),
    "check_criteria": ("criteria_check", TraceWriterCrew.criteria_check),
    "review_necessity": ("necessity_review", TraceWriterCrew.necessity_review),
    "issue_determination": ("determination", TraceWriterCrew.determination),
}

_REACT_TOOL_FOR_NODE: Mapping[str, str] = {
    node: name for name, (node, _) in REACT_TOOLS.items()
}

if len(_REACT_TOOL_FOR_NODE) != len(REACT_TOOLS):  # pragma: no cover - import guard
    raise RuntimeError("two ReAct tools claim the same crew node")

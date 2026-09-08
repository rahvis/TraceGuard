"""Public, payload-conscious types used by the TraceGuard SDK.

The observable trace types in this module are intentionally narrow.  A trace
contains no query, retrieved document, prompt, model output, label, or provider
credential.  Those values may exist transiently inside the crew state, but they
must never be added to :class:`ObservableTrace`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Condition(StrEnum):
    """Observable-shaping conditions supported by the released experiment.

    Note this axis is about what the *release* looks like (is the plan constant,
    are timing and size padded), not about how the agent decides. See
    :class:`Autonomy` for the latter.
    """

    ADAPTIVE = "adaptive"
    STRUCTURE_ONLY = "structure_only"
    FULL_PAD = "full_pad"

    @classmethod
    def parse(cls, value: str | Condition) -> Condition:
        if isinstance(value, cls):
            return value
        aliases = {"structure": cls.STRUCTURE_ONLY, "full": cls.FULL_PAD}
        normalized = str(value).strip().lower()
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(f"unknown condition {value!r}; expected one of: {allowed}") from exc


# FREE keeps the same graph but lifts the loop budget well above anything the
# workload reaches, so the agent stops when it is satisfied rather than when a
# cap intervenes. It is a multiplier rather than "unbounded" because the graph
# has a recursion limit and a run that never terminates is not a data point.
_FREE_LOOP_MULTIPLIER = 8


class Autonomy(StrEnum):
    """How much of the agent's control flow is allowed to depend on the data.

    This is deliberately *orthogonal* to :class:`Condition`. Until now the two
    were fused: ``STRUCTURE_ONLY`` and ``FULL_PAD`` each meant both "emit a
    data-independent plan" and "pad timing and size", so there was no way to
    vary how autonomous the agent is while holding the defense fixed -- and
    therefore no way to ask how leakage scales with autonomy, which is the
    question the mechanism actually raises.

    The ladder is monotone in *how much the data can steer control flow*, which
    is the quantity the leakage question is about:

    ``SCRIPTED``
        Nothing branches on the data. Fixed depth, no repair or revision loops,
        so the plan is a constant.
    ``BOUNDED``
        Depth is the agent's own sufficiency decision within the published
        ``[min, max]`` hop range, and the repair/revision loops run up to their
        configured budgets.
    ``FREE``
        As ``BOUNDED`` for depth, but the loop budgets are effectively removed,
        so the agent may iterate until it is satisfied.

    A note on what ``FREE`` deliberately does *not* mean: an earlier version of
    this enum defined it as "skip the sufficiency check and always run to the
    maximum depth". That is not more autonomous, it is *less* -- it makes depth a
    constant and so removes a data-dependent coordinate. Measured on the fixture
    corpus it produced 12 steps on both framings where ``BOUNDED`` produced 9 and
    16, i.e. it leaked strictly less. Autonomy has to be defined as data
    influence over control flow, not as absence of limits, or the ladder is not
    monotone and the sweep measures nothing coherent.
    """

    SCRIPTED = "scripted"
    BOUNDED = "bounded"
    FREE = "free"

    @classmethod
    def parse(cls, value: str | Autonomy) -> Autonomy:
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(
                f"unknown autonomy level {value!r}; expected one of: {allowed}"
            ) from exc

    @classmethod
    def default_for(cls, condition: Condition) -> Autonomy:
        """The autonomy a condition implied before the two axes were separated.

        Preserves the historical meaning of every archived run: the defended
        conditions were scripted, the adaptive one was bounded. New experiments
        set autonomy explicitly instead of inheriting it.
        """

        return cls.SCRIPTED if condition is not Condition.ADAPTIVE else cls.BOUNDED

    @property
    def loop_budget_multiplier(self) -> int:
        """How many times the configured repair/revision budget may be spent."""

        if self is Autonomy.SCRIPTED:
            return 0
        return 1 if self is Autonomy.BOUNDED else _FREE_LOOP_MULTIPLIER


# --------------------------------------------------------------------------
# The sensitivity ladder
# --------------------------------------------------------------------------
# The corpus used to carry a binary attribute: a case was "routine" or
# "sensitive". That supports one contrast and no dose-response question, so v2
# of the corpus grades the same axis into four rungs. The names are ordered by
# ascending sensitivity and the first and last are v1's two framings, which is
# what lets the v1 file keep its exact meaning: its "sensitive" cases are the
# top rung, not the second one.
SENSITIVITY_FRAMINGS: tuple[str, ...] = ("routine", "guarded", "elevated", "sensitive")

# Fixture target research-hop depth per level.
#
# This table is the single definition of what used to be the expression
# ``7 if case.sensitive else 4``, written out independently in corpus.py,
# agents.py and providers.py. Three copies of a ladder is three chances for the
# corpus, the agent and the fixture provider to disagree about how deep a case
# should go, which would show up as an unexplained step-count difference rather
# than as an error.
FIXTURE_TARGET_HOPS: tuple[int, ...] = (4, 5, 6, 7)


def sensitivity_levels() -> int:
    """How many rungs the ladder has."""

    return len(SENSITIVITY_FRAMINGS)


def sensitivity_level_for_framing(framing: str, *, position: int | None = None) -> int:
    """Resolve a framing name to its rung.

    Named framings win over list position so that a corpus file listing only
    ``["routine", "sensitive"]`` -- the v1 layout -- still places its sensitive
    cases at the top of the ladder, matching the seven-hop depth and the
    repair/revision fixture flags that file actually carries. An unrecognized
    name falls back to its position in the file's own ordered list.
    """

    normalized = str(framing).strip().lower()
    if normalized in SENSITIVITY_FRAMINGS:
        return SENSITIVITY_FRAMINGS.index(normalized)
    if position is None:
        raise ValueError(
            f"unknown sensitivity framing {framing!r}; expected one of "
            f"{', '.join(SENSITIVITY_FRAMINGS)}"
        )
    return position


def framing_for_sensitivity_level(level: int) -> str:
    """The framing name of one rung."""

    return SENSITIVITY_FRAMINGS[_checked_level(level)]


def fixture_target_hops(level: int) -> int:
    """Fixture target research depth for one rung of the ladder."""

    return FIXTURE_TARGET_HOPS[_checked_level(level)]


def _checked_level(level: int) -> int:
    value = int(level)
    if not 0 <= value < len(SENSITIVITY_FRAMINGS):
        raise ValueError(
            f"sensitivity_level must be between 0 and {len(SENSITIVITY_FRAMINGS) - 1}; "
            f"got {level!r}"
        )
    return value


@dataclass(frozen=True, slots=True)
class MedicalDocument:
    """One synthetic document held inside the local corpus boundary."""

    document_id: str
    title: str
    text: str = field(repr=False)
    tags: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, fallback_id: str) -> MedicalDocument:
        document_id = str(value.get("id") or value.get("document_id") or fallback_id)
        title = str(value.get("title") or document_id)
        text = value.get("text", value.get("content", value.get("body", "")))
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"document {document_id!r} has no non-empty text/content/body")
        raw_tags = value.get("tags", ())
        tags = (
            tuple(str(tag) for tag in raw_tags)
            if isinstance(raw_tags, Sequence) and not isinstance(raw_tags, str)
            else ()
        )
        return cls(document_id=document_id, title=title, text=text, tags=tags)

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "document_id": self.document_id,
            "title": self.title,
            "tags": list(self.tags),
        }
        if include_text:
            result["text"] = self.text
        return result


@dataclass(frozen=True, slots=True)
class MedicalCase:
    """A normalized case.

    ``query`` and document text are private crew inputs.  The default
    :meth:`to_dict` view consequently omits both.

    ``sensitivity_level`` replaced a boolean ``sensitive`` field when the corpus
    grew a graded ladder.  :attr:`sensitive` survives as a derived property
    because roughly a dozen call sites -- the fixture provider, the console
    demos, the attack's binary attribute label -- legitimately want the coarse
    "is this case above the floor?" question, and because every archived record
    and journal is keyed on it.
    """

    case_id: str
    specialty: str
    topic: str
    sensitivity_level: int
    canary_member: bool
    query: str = field(repr=False)
    documents: tuple[MedicalDocument, ...] = field(repr=False)
    fixture: Mapping[str, Any] = field(default_factory=dict, repr=False)
    # The task family this case belongs to. Defaulted so every archived case
    # and journal row keeps its exact shape, and so the prior-authorization
    # corpus file needs no edit (its bytes are pinned by archived receipts).
    domain: str = "prior-authorization"

    @property
    def sensitive(self) -> bool:
        """True for every rung above the routine floor."""

        return self.sensitivity_level > 0

    @property
    def framing(self) -> str:
        """The ladder name of this case's rung."""

        return framing_for_sensitivity_level(self.sensitivity_level)

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must not be empty")
        _checked_level(self.sensitivity_level)
        if not self.query.strip():
            raise ValueError(f"case {self.case_id!r} has an empty query")
        if len(self.documents) != 6:
            raise ValueError(
                f"case {self.case_id!r} must contain exactly six documents; "
                f"found {len(self.documents)}"
            )
        ids = [document.document_id for document in self.documents]
        if len(ids) != len(set(ids)):
            raise ValueError(f"case {self.case_id!r} contains duplicate document ids")

    def to_dict(self, *, include_private: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "case_id": self.case_id,
            "specialty": self.specialty,
            "topic": self.topic,
            "sensitivity_level": self.sensitivity_level,
            "framing": self.framing,
            # Retained alongside the ordinal grade: the binary attribute label
            # is what archived journals and the attack evaluator are keyed on.
            "sensitive": self.sensitive,
            "canary_member": self.canary_member,
            "document_count": len(self.documents),
            # The task family, so a journal row states which family produced it
            # rather than leaving the reader to resolve an opaque dataset hash.
            "domain": self.domain,
        }
        if include_private:
            result["query"] = self.query
            result["documents"] = [
                document.to_dict(include_text=True) for document in self.documents
            ]
            result["fixture"] = dict(self.fixture)
        return result


@dataclass(frozen=True, slots=True)
class ProviderRequest:
    """An ephemeral request passed to an LLM provider.

    Private request fields are excluded from ``repr`` so an accidental exception
    or debug print does not disclose them.
    """

    role: str
    model: str
    max_tokens: int
    system_prompt: str = field(repr=False)
    user_prompt: str = field(repr=False)
    seed: int = 0
    timeout_s: float | None = None
    fixture_context: Mapping[str, Any] = field(default_factory=dict, repr=False)
    # When set, the provider pads the transmitted request to exactly this many
    # bytes, so the host-observable egress size is a public constant.  The
    # padding rides in an inert transport field and never enters the prompt, so
    # it cannot change the completion (asserted in tests).
    pad_request_to_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    """Provider response retained only in the confidential in-process state.

    Two byte counts, because the model is *outside* the confidential boundary and
    a host watching the guest's NIC sees both directions of every round-trip:

    ``request_bytes``
        The serialized prompt the guest transmits.  This is the true **egress**:
        bytes leaving the CVM.  The guest composes it, so a defense can bound it.
    ``response_bytes``
        The completion the provider returns.  This is **ingress**: bytes entering
        the CVM.  Its size is chosen by the provider, so no in-guest mechanism can
        bound it (see ``shield_runtime`` and Prop. 2 in the paper).

    Earlier revisions recorded only ``response_bytes`` and named it ``egress``,
    which inverted the direction of the one coordinate the defense claims to pad.
    """

    text: str = field(repr=False)
    response_bytes: int
    request_bytes: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None


@dataclass(frozen=True, slots=True)
class TraceStep:
    """One host-observable boundary step and nothing more.

    ``egress_bytes`` is the request direction (guest -> provider) and
    ``ingress_bytes`` the response direction (provider -> guest).  Both are
    host-visible for a remote model; only the first is guest-controlled.
    """

    index: int
    step_type: str
    wall_time_s: float
    duration_s: float
    egress_bytes: int
    ingress_bytes: int = 0

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("trace index must be non-negative")
        if not self.step_type:
            raise ValueError("step_type must not be empty")
        if self.wall_time_s < 0 or self.duration_s < 0:
            raise ValueError("trace timing values must be non-negative")
        if self.egress_bytes < 0 or self.ingress_bytes < 0:
            raise ValueError("trace byte counts must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        # Keep this exact allowlist.  Payloads and in-enclave retrieval metadata
        # do not belong on the observable surface.
        return {
            "index": self.index,
            "step_type": self.step_type,
            "wall_time_s": round(self.wall_time_s, 6),
            "duration_s": round(self.duration_s, 6),
            "egress_bytes": self.egress_bytes,
            "ingress_bytes": self.ingress_bytes,
        }


@dataclass(frozen=True, slots=True)
class ObservableTrace:
    """Ordered host-visible execution metadata."""

    steps: tuple[TraceStep, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"steps": [step.to_dict() for step in self.steps]}

    def protected_projection(self) -> dict[str, Any]:
        """The coordinates a defense can hold constant, and only those.

        The receipt's plan binding is a digest over *this*, not over
        ``to_dict()``.  Under full padding these fields are a constant function
        of (step types, deadline, ceiling), so a verifier can recompute the
        digest and catch a receipt that commits to one plan while having run
        another.  ``ingress_bytes`` is deliberately excluded: it is
        provider-chosen, so including it would make the digest unreconstructible
        and destroy that independent check -- and it is reported separately
        rather than dropped, so nothing is hidden.
        """

        return {
            "steps": [
                {
                    "index": step.index,
                    "step_type": step.step_type,
                    "wall_time_s": round(step.wall_time_s, 6),
                    "duration_s": round(step.duration_s, 6),
                    "egress_bytes": step.egress_bytes,
                }
                for step in self.steps
            ]
        }

    @property
    def step_types(self) -> tuple[str, ...]:
        return tuple(step.step_type for step in self.steps)

    @property
    def total_duration_s(self) -> float:
        return self.steps[-1].wall_time_s if self.steps else 0.0

    @property
    def total_egress_bytes(self) -> int:
        return sum(step.egress_bytes for step in self.steps)

    @property
    def total_ingress_bytes(self) -> int:
        return sum(step.ingress_bytes for step in self.steps)


EventCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True, slots=True)
class RunMetrics:
    step_count: int
    research_hops: int
    observable_duration_s: float
    total_egress_bytes: int
    fail_closed: bool
    overrun_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_count": self.step_count,
            "research_hops": self.research_hops,
            "observable_duration_s": round(self.observable_duration_s, 6),
            "total_egress_bytes": self.total_egress_bytes,
            "fail_closed": self.fail_closed,
            "overrun_count": self.overrun_count,
        }


@dataclass(frozen=True, slots=True)
class RunResult:
    """Result of one SDK invocation.

    The answer is an intended tenant-facing output and is separate from the
    metadata-only trace.  The receipt type is kept structural here to avoid a
    dependency cycle; concrete receipts implement ``to_dict`` and ``to_bytes``.
    """

    run_id: str
    case_id: str
    condition: Condition
    answer: str
    trace: ObservableTrace
    receipt: Any
    metrics: RunMetrics
    provider_provenance: str
    dataset_hash: str
    runtime: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        receipt = self.receipt.to_dict() if hasattr(self.receipt, "to_dict") else self.receipt
        return {
            "run_id": self.run_id,
            "case_id": self.case_id,
            "condition": self.condition.value,
            "answer": self.answer,
            "trace": self.trace.to_dict(),
            "receipt": receipt,
            "metrics": self.metrics.to_dict(),
            "provider_provenance": self.provider_provenance,
            "dataset_hash": self.dataset_hash,
            "runtime": dict(self.runtime),
        }

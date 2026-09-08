"""LLM provider adapters.

Live execution goes through one Azure OpenAI resource via the official
``AzureOpenAI`` client (Chat Completions).  Requests are deliberately not
retained, logged, or included in raised error messages.

The direct OpenAI and Anthropic adapters were removed when the project
consolidated onto Azure; see ``HISTORICAL_PROVIDERS`` for why their names still
appear in archived journals.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .config import Settings
from .types import (
    MedicalCase,
    MedicalDocument,
    ProviderRequest,
    ProviderResponse,
    fixture_target_hops,
    sensitivity_levels,
)


class EgressCeilingExceeded(RuntimeError):
    """A request body cannot be padded down to the public egress ceiling.

    Raised rather than silently transmitting an over-ceiling request, because a
    single over-ceiling step would make the egress coordinate data-dependent and
    void the constancy the guarantee rests on.
    """


class ProviderCallError(RuntimeError):
    """Payload-free provider failure exposed to callers."""


class LLMProvider(ABC):
    """Small provider surface used by all six agents."""

    provenance: str

    @abstractmethod
    def complete(self, request: ProviderRequest) -> ProviderResponse:
        raise NotImplementedError


def _request_body_bytes(kwargs: Mapping[str, Any]) -> int:
    """Serialized size of the request body the SDK will transmit.

    The host observes TLS record sizes, not this number, but the two differ by a
    fixed framing overhead plus whatever the transport's content coding does.
    We measure the JSON body because it is the quantity the guest controls and
    can hold constant; Section 3 of the paper states what that does and does not
    bound about the wire.  ``timeout`` is a client-side SDK argument and is not
    part of the body, so it is excluded.
    """

    body = {key: value for key, value in kwargs.items() if key != "timeout"}
    return len(json.dumps(body, separators=(",", ":"), sort_keys=True).encode("utf-8"))


def _truncate_chars(text: str, max_tokens: int) -> str:
    # Fixture-only deterministic approximation; live providers enforce max_tokens.
    return text[: max(1, max_tokens) * 4]


# --------------------------------------------------------------------------
# Fixture behaviour per rung of the sensitivity ladder
# --------------------------------------------------------------------------
# The fixture provider used to branch on the boolean ``case.sensitive`` in five
# places, each with its own ad-hoc two-way choice. With a graded corpus those
# branches would collapse L1..L3 into one behaviour and the ladder would be
# invisible offline, so each is now a table indexed by the level. Every table is
# monotone by construction, which is the property the ordinal claim rests on:
# a higher rung is never cheaper to adjudicate than a lower one.

# Intake's coarse service classification.
_SERVICE_TYPE = ("standard", "standard_review", "specialty_review", "specialty")

# How many times the extraction note repeats its cross-checking rationale. This
# is the fixture's stand-in for a longer chain of thought at higher rungs.
_EXTRACTION_EMPHASIS = (1, 1, 2, 2)

# How many synthetic sources the coverage assessment cites. Capped by the
# six-document invariant every case satisfies.
_CITED_SOURCES = (3, 4, 5, 6)

# The criterion-scoring sentence. MET on the two lower rungs, INSUFFICIENT on
# the two upper ones, which is what makes _determination pend exactly there.
# Fixture-only domain vocabulary. Live runs get their wording from the crew's
# DomainProfile prompts; these exist so an offline fixture determination does
# not name the wrong task family.
_DOMAIN_TAG = {
    "prior-authorization": "prior-auth",
    "aml-alert-triage": "alert-triage",
}
_DOMAIN_DISPOSITION = {
    None: (
        "PEND (needs additional review)",
        "APPROVE",
        "clinician sign-off",
        "not clinical advice",
    ),
    "prior-authorization": (
        "PEND (needs additional review)",
        "APPROVE",
        "clinician sign-off",
        "not clinical advice",
    ),
    "aml-alert-triage": (
        "ESCALATE (needs additional review)",
        "CLOSE",
        "investigator sign-off",
        "not a reporting decision",
    ),
}

_ASSESSMENT_QUALIFIER = (
    "This standard synthetic request maps cleanly to the policy criteria, which are MET.",
    "This synthetic request maps to the policy criteria with one narrative outcome, which "
    "are MET.",
    "This synthetic request leaves a material step-therapy gap unreconciled, so one "
    "criterion is INSUFFICIENT.",
    "This complex synthetic request requires reconciling interacting criteria and leaves "
    "documentation gaps, so several criteria are INSUFFICIENT.",
)

for _table in (_SERVICE_TYPE, _EXTRACTION_EMPHASIS, _CITED_SOURCES, _ASSESSMENT_QUALIFIER):
    if len(_table) != sensitivity_levels():  # pragma: no cover - import-time invariant
        raise RuntimeError("fixture per-level tables must cover every rung of the ladder")


def _level(case: MedicalCase | None) -> int:
    """The rung a case sits on, defaulting to the routine floor when unknown."""

    return case.sensitivity_level if case is not None else 0


class FixtureProvider(LLMProvider):
    """Offline deterministic provider for smoke tests and demonstrations.

    It is permitted to use synthetic labels to produce predictable golden paths.
    The resulting provenance is always ``synthetic_fixture`` and SDK receipts
    explicitly state that fixture runs are not evidence for paper measurements.
    """

    provenance = "synthetic_fixture"

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        context = request.fixture_context
        role = request.role
        if role == "intake":
            case = self._case(context)
            text = json.dumps(
                {
                    # The released pipeline_crew topology discards this route.
                    # hierarchical_supervisor parses it, and passes the
                    # currently admissible specialists in fixture_context so
                    # this deterministic stand-in names one of them rather than
                    # always naming the first stage of the plan.
                    "route": self._route(context),
                    "service_type": _SERVICE_TYPE[_level(case)],
                    "complete": True,
                },
                separators=(",", ":"),
            )
        elif role == "react_agent":
            text = self._react_tool_choice(context)
        elif role == "clinical_extraction":
            text = self._clinical_extraction(context)
        elif role == "coverage_assessment":
            text = self._coverage_assessment(context)
        elif role == "criteria_check":
            text = self._criteria_check(context)
        elif role == "necessity_review":
            text = self._necessity_review(context)
        elif role == "determination":
            text = self._determination(context)
        else:
            text = ""
        text = _truncate_chars(text, request.max_tokens)
        # Mirror the live provider's two directions so the offline gate exercises
        # the same observable surface.  The egress is measured with the *same*
        # function the Azure path uses, over an equivalent body, so a fixture
        # request size is directly comparable with a live one -- which is what
        # lets the offline gate be used to provision the public ceiling.
        request_bytes = _request_body_bytes(
            {
                "model": request.model,
                "max_completion_tokens": request.max_tokens,
                "messages": [
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.user_prompt},
                ],
                "seed": request.seed,
            }
        )
        target = request.pad_request_to_bytes
        if target is not None:
            if request_bytes > target:
                raise EgressCeilingExceeded(
                    "request exceeds the public egress ceiling"
                )
            request_bytes = target
        return ProviderResponse(
            text=text,
            response_bytes=len(text.encode("utf-8")),
            request_bytes=request_bytes,
            input_tokens=0,
            output_tokens=max(1, (len(text) + 3) // 4) if text else 0,
            stop_reason="end_turn",
        )

    @staticmethod
    def _case(context: Mapping[str, Any]) -> MedicalCase | None:
        value = context.get("case")
        return value if isinstance(value, MedicalCase) else None

    @staticmethod
    def _admissible(context: Mapping[str, Any]) -> tuple[str, ...]:
        raw = context.get("admissible", ())
        if isinstance(raw, str) or not isinstance(raw, Sequence):
            return ()
        return tuple(str(value) for value in raw)

    def _route(self, context: Mapping[str, Any]) -> str:
        admissible = self._admissible(context)
        return admissible[0] if admissible else "clinical_extraction"

    def _react_tool_choice(self, context: Mapping[str, Any]) -> str:
        """Deterministic tool choice for the single-agent architecture.

        The fixture is a stand-in for a model, not a model, so it picks the
        first admissible tool.  That is what makes the canonical ReAct plan
        reproducible offline; a live provider is what exercises the agent
        actually preferring a different admissible tool.
        """

        admissible = self._admissible(context)
        tool = admissible[0] if admissible else "extract_clinical_evidence"
        return json.dumps(
            {"thought": f"synthetic fixture selects {tool}", "tool": tool},
            separators=(",", ":"),
        )

    def _clinical_extraction(self, context: Mapping[str, Any]) -> str:
        case = self._case(context)
        level = _level(case)
        hop = int(context.get("hop", 1))
        target = int(context.get("target_hops", fixture_target_hops(level)))
        sufficient = hop >= target
        if level == 0:
            notes = "The synthetic referral supports a standard coverage assessment. "
        else:
            notes = (
                "The synthetic referral has interacting clinical factors requiring "
                "cross-checking of prior therapies tried, severity, and diagnostics before "
                "coverage can be assessed. "
            ) * _EXTRACTION_EMPHASIS[level]
        return json.dumps(
            {"sufficient": sufficient, "notes": f"pass {hop}: {notes.strip()}"},
            separators=(",", ":"),
        )

    def _coverage_assessment(self, context: Mapping[str, Any]) -> str:
        case = self._case(context)
        raw_documents = context.get("documents", ())
        documents = [value for value in raw_documents if isinstance(value, MedicalDocument)]
        prior = str(context.get("prior_assessment", "")).strip()
        mode = str(context.get("assessment_mode", "initial"))
        if prior and mode in {"criteria_repair", "necessity_revision"}:
            suffix = (
                " Each criterion was re-linked to the supplied synthetic source identifiers."
                if mode == "criteria_repair"
                else " The revision explicitly records documentation gaps and uncertainty."
            )
            return prior + suffix
        if case is None:
            return "Synthetic fixture coverage assessment unavailable."
        level = _level(case)
        count = min(_CITED_SOURCES[level], len(documents))
        statements: list[str] = []
        for document in documents[:count]:
            first = document.text.strip().split(".", 1)[0].strip()
            if len(first) > 180:
                first = first[:177].rstrip() + "..."
            statements.append(f"{first}. [{document.document_id}]")
        detail = " ".join(statements) or "No synthetic source statement was available."
        qualifier = _ASSESSMENT_QUALIFIER[level]
        # "prior-auth" and "alert-triage" are both 10 and 12 characters; the
        # tag is chosen by domain so a fixture determination does not name the
        # wrong task family, and the pair is length-matched so fixture egress
        # remains comparable between families.
        tag = _DOMAIN_TAG.get(getattr(case, "domain", None), "prior-auth")
        return f"{case.specialty.title()} — {case.topic} {tag}. {detail} {qualifier}"

    def _criteria_check(self, context: Mapping[str, Any]) -> str:
        case = self._case(context)
        check = int(context.get("criteria_checks", 0))
        requested = bool(case and case.fixture.get("criteria_repair", False)) and check == 0
        return json.dumps(
            {
                "needs_repair": requested,
                "feedback": "re-link criteria to source ids" if requested
                else "criteria source-supported",
            },
            separators=(",", ":"),
        )

    def _necessity_review(self, context: Mapping[str, Any]) -> str:
        case = self._case(context)
        checks = int(context.get("necessity_checks", 0))
        requested = bool(case and case.fixture.get("necessity_revision", False)) and checks == 0
        return json.dumps(
            {
                "needs_revision": requested,
                "feedback": "record gaps and uncertainty" if requested else "review complete",
            },
            separators=(",", ":"),
        )

    @staticmethod
    def _determination(context: Mapping[str, Any]) -> str:
        assessment = str(context.get("assessment", "")).strip()
        if not assessment:
            return (
                "The run failed closed before a synthetic determination could be produced."
            )
        # Lenient rubric rendering: the upper rungs of the ladder leave residual
        # documentation gaps and pend; the lower rungs' criteria are met and approve.
        pended = "INSUFFICIENT" in assessment or "NOT_MET" in assessment
        case = context.get("case")
        domain = getattr(case, "domain", None)
        pend_label, pass_label, signoff, advice = _DOMAIN_DISPOSITION.get(
            domain, _DOMAIN_DISPOSITION[None]
        )
        decision = pend_label if pended else pass_label
        return (
            f"DETERMINATION: {decision}.\n{assessment}\n\n"
            f"AI-assisted draft; requires {signoff}. "
            f"For research demonstration only; {advice}."
        )


@dataclass(frozen=True, slots=True)
class DeploymentCapabilities:
    """What one Azure deployment actually accepts on the wire.

    Azure deployments of different model generations disagree about their
    request schema: newer generations reject ``max_tokens`` in favour of
    ``max_completion_tokens``, and reasoning-tier deployments additionally
    reject ``seed``, ``temperature`` and the penalty parameters.  Rather than
    hardcode a table of model names that will be wrong within a release, the
    adapter discovers this once per deployment and records the answer.

    ``seed_honoured`` is a correctness property, not a nicety.  The experiment
    runner derives a per-cell seed and records it in the journal and in the
    resume key, so a deployment that silently ignores ``seed`` makes those
    recorded seeds decorative and breaks the implication that a replayed cell
    reproduces.  It is therefore surfaced rather than swallowed.
    """

    max_output_field: str = "max_completion_tokens"
    seed_honoured: bool = True
    accepts_temperature: bool = False
    supports_json_object: bool = True
    probe_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_output_field": self.max_output_field,
            "seed_honoured": self.seed_honoured,
            "accepts_temperature": self.accepts_temperature,
            "supports_json_object": self.supports_json_object,
            "probe_note": self.probe_note,
        }


# Probed capabilities are a property of (endpoint, deployment), not of a run, so
# they are cached per process.  Probing happens at construction time, never
# inside a measured step: a discovery retry inside ShieldRuntime.execute would
# corrupt the per-step timing that is the whole observable this project studies.
_CAPABILITY_CACHE: dict[tuple[str, str], DeploymentCapabilities] = {}

# The probe payload is this fixed, non-sensitive string.  That matters: it is why
# probe error text may be inspected to discover which argument was rejected,
# whereas a real call's error must stay payload-free (transports echo request
# bodies, which would carry the private query).
_PROBE_PROMPT = "Reply with the single character: 1"

_UNSUPPORTED_MARKERS = (
    "unsupported",
    "unrecognized",
    "not supported",
    "invalid_request_error",
    "does not support",
)


def _rejects(error_text: str, parameter: str) -> bool:
    lowered = error_text.lower()
    return parameter in lowered and any(marker in lowered for marker in _UNSUPPORTED_MARKERS)


class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI Chat Completions adapter.

    Chat Completions is used rather than the Responses API because responses
    default to server-side retention (``store=true``), which conflicts with the
    no-retention posture of this module.  Note the scope limit this cannot
    reach: Azure may retain prompts for abuse monitoring independently of this
    choice unless the resource holds Microsoft's Limited Access exemption.  The
    manuscript states that limit rather than implying the adapter removes it.

    ``model`` on a request is an Azure **deployment name**, not a base model
    name.  The two are independent, and the deployment name is what is recorded
    in run provenance.
    """

    provenance = "live_azure_openai"

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        api_key: str | None = None,
        endpoint: str | None = None,
        api_version: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        settings = settings or Settings.from_env()
        key = api_key or settings.azure_openai_api_key
        if not key:
            raise ValueError("AZURE_OPENAI_API_KEY is required for the azure provider")
        base = (endpoint or settings.azure_openai_endpoint or "").rstrip("/")
        if not base:
            raise ValueError("AZURE_OPENAI_ENDPOINT is required for the azure provider")
        # An AI Foundry *project* URL is not an inference endpoint; the resource
        # host is. Rejecting it here turns a confusing 404 at first call into a
        # clear configuration error.
        if "/api/projects/" in base:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT must be the resource host "
                "(https://<resource>.services.ai.azure.com), not an "
                "/api/projects/<project> Foundry project URL"
            )
        self._endpoint = base
        self._api_version = api_version or settings.azure_openai_api_version
        try:
            from openai import AzureOpenAI
        except ImportError as exc:  # pragma: no cover - dependency is declared by the package
            raise RuntimeError("install the declared 'openai' dependency for live runs") from exc
        # The key is held only by the official client and is never exposed by
        # this adapter.  max_retries=0 is load-bearing for the measurement, not
        # only for privacy: the SDK retries twice by default, and a retry inside
        # a measured step would silently inflate that step's observed duration.
        self._client = AzureOpenAI(
            api_key=key,
            azure_endpoint=base,
            api_version=self._api_version,
            timeout=timeout_s or settings.provider_timeout_s,
            max_retries=0,
        )
        self._deployments = tuple(
            name
            for name in dict.fromkeys(
                (settings.model_fast, settings.model_deep, settings.model_review)
            )
            if name
        )
    # -- capability discovery ------------------------------------------------ #

    def warm_up(self) -> dict[str, DeploymentCapabilities]:
        """Probe every configured deployment before any measured step runs.

        Construction stays cheap and offline so the adapter is unit-testable and
        so ``create_provider`` cannot make a network call as a side effect. The
        runner calls this explicitly, which is what guarantees discovery happens
        *outside* the timing window: a probe retry inside
        ``ShieldRuntime.execute`` would inflate that step's observed duration,
        and per-step duration is the observable this project measures.
        """

        return {name: self.capabilities(name) for name in self._deployments}

    # -- capability discovery ------------------------------------------------ #

    def capabilities(self, deployment: str) -> DeploymentCapabilities:
        """Return (and cache) what this deployment accepts, probing if needed."""

        cache_key = (self._endpoint, deployment)
        cached = _CAPABILITY_CACHE.get(cache_key)
        if cached is not None:
            return cached
        resolved = self._probe(deployment)
        _CAPABILITY_CACHE[cache_key] = resolved
        return resolved

    def _probe(self, deployment: str) -> DeploymentCapabilities:
        """Discover the request schema for one deployment.

        Starts from the newest-generation parameter set and removes whatever the
        service rejects, so a deployment is never assumed to be reasoning-tier
        or not.  At most a handful of one-token calls per deployment, made once
        per process and outside any measured step.
        """

        max_output_field = "max_completion_tokens"
        seed_honoured = True
        notes: list[str] = []

        for _ in range(4):
            kwargs: dict[str, Any] = {
                "model": deployment,
                max_output_field: 16,
                "messages": [{"role": "user", "content": _PROBE_PROMPT}],
            }
            if seed_honoured:
                kwargs["seed"] = 7
            try:
                self._client.chat.completions.create(**kwargs)
                break
            except Exception as exc:  # noqa: BLE001 - probe payload is a constant
                # Safe to read: the probe prompt is the fixed string above, so
                # this error cannot contain private case content.
                text = str(exc)
                if max_output_field == "max_completion_tokens" and _rejects(
                    text, "max_completion_tokens"
                ):
                    max_output_field = "max_tokens"
                    notes.append("deployment requires max_tokens")
                    continue
                if seed_honoured and _rejects(text, "seed"):
                    seed_honoured = False
                    notes.append("deployment rejects seed; per-cell seeds are not honoured")
                    continue
                notes.append(f"probe failed: {type(exc).__name__}")
                break

        return DeploymentCapabilities(
            max_output_field=max_output_field,
            seed_honoured=seed_honoured,
            # Temperature is never sent, but a deployment that rejects it is a
            # reasoning-tier deployment, which is worth recording as provenance.
            accepts_temperature=False,
            supports_json_object=True,
            probe_note="; ".join(notes) or None,
        )

    def capability_provenance(self) -> dict[str, Any]:
        """Per-deployment capability record for a run manifest."""

        return {
            "endpoint": self._endpoint,
            "api_version": self._api_version,
            "deployments": {
                name: self.capabilities(name).to_dict() for name in self._deployments
            },
        }

    # -- the measured call --------------------------------------------------- #

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        capability = self.capabilities(request.model)
        # Temperature and top_p are omitted for every deployment; reasoning-tier
        # deployments reject non-default sampling args and the rest do not need
        # them under this deterministic-seeded protocol.
        kwargs: dict[str, Any] = {
            "model": request.model,
            capability.max_output_field: request.max_tokens,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "timeout": request.timeout_s,
        }
        if capability.seed_honoured:
            kwargs["seed"] = request.seed

        # Egress padding.  The host sees the size of the request body we put on
        # the wire, so bounding that observable means bounding *this* dict.  The
        # filler goes in the `user` field -- an opaque end-user identifier the
        # API forwards for abuse tracking and never conditions generation on --
        # so a padded request and its unpadded counterpart yield the same
        # completion.  Padding the prompt instead would buy a constant egress at
        # the cost of changing the answer, which is not a trade we can make
        # silently.
        request_bytes = _request_body_bytes(kwargs)
        target = request.pad_request_to_bytes
        if target is not None:
            overhead = _request_body_bytes({**kwargs, "user": ""}) - request_bytes
            slack = target - request_bytes - overhead
            if slack < 0:
                raise EgressCeilingExceeded(
                    "request exceeds the public egress ceiling; raise "
                    "TRACEGUARD_STEP_EGRESS_BYTES or shorten the prompt"
                )
            kwargs["user"] = "0" * slack
            request_bytes = _request_body_bytes(kwargs)

        try:
            completion = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            # Do not propagate an SDK exception string: some transports echo request
            # bodies, which could contain the private query or source text.
            raise ProviderCallError("Azure OpenAI Chat Completions API call failed") from exc

        choices = getattr(completion, "choices", None) or []
        first = choices[0] if choices else None
        message = getattr(first, "message", None)
        text = str(getattr(message, "content", "") or "")
        usage = getattr(completion, "usage", None)
        return ProviderResponse(
            text=text,
            # Ingress: what the provider sent back.  Chosen by the provider, so no
            # in-guest mechanism bounds it; recorded truthfully in every condition.
            response_bytes=len(text.encode("utf-8")),
            # Egress: what we actually transmitted, after any padding.
            request_bytes=request_bytes,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            stop_reason=str(getattr(first, "finish_reason", "") or "") or None,
        )


# Provider strings that appear in archived journals but that no current code
# path can emit. The OpenAI and Anthropic adapters were removed when the project
# moved to a single Azure OpenAI resource; the runs they produced remain in
# artifacts/ as historical evidence, and analysis code still reads them. Keeping
# the names here answers "what produced these rows?" from the tree itself.
HISTORICAL_PROVIDERS = ("openai", "anthropic")


def create_provider(settings: Settings) -> LLMProvider:
    if settings.provider == "fixture":
        return FixtureProvider()
    if settings.provider == "azure":
        return AzureOpenAIProvider(settings)
    if settings.provider in HISTORICAL_PROVIDERS:
        raise ValueError(
            f"provider {settings.provider!r} was removed when this project moved to "
            "Azure OpenAI; archived journals naming it are still readable, but new "
            "runs must use 'azure' or 'fixture'"
        )
    raise ValueError(f"unsupported provider mode {settings.provider!r}")

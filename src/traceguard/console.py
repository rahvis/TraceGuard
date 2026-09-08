"""Console-only API extensions for the TraceGuard research UI.

Everything in this module stays inside the artifact's privacy posture:

- Playground and taxonomy demonstrations execute the *real* crew, shield
  runtime, receipt signer, and verifier — always with the offline fixture
  provider, so they are deterministic, free, and never paper evidence.
- Responses expose only host-observable metadata (step type, timing, egress
  bytes), receipts, and verifier verdicts.  Fixture answers are synthetic by
  construction and are labeled as such when returned.
"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from . import verifier as verifier_module
from .attack import (
    evaluate_attack,
    extract_trace_features,
    permutation_control,
    randomized_response_frontier,
    run_ablation_suite,
    shape_fixed_control,
)
from .config import Settings
from .corpus import CorpusCatalog, default_case_count
from .experiment import catalog_cases
from .graph import ARCHITECTURES, DEFAULT_ARCHITECTURE, architecture_spec
from .ledger import HashChainLedger
from .plans import registry_manifest
from .providers import FixtureProvider
from .receipt import ReceiptSigner
from .sdk import TraceGuardSDK
from .storage import metadata_event, redact_secrets, utc_now
from .types import MedicalCase, ProviderRequest, ProviderResponse

# --------------------------------------------------------------------------
# Crew topology
# --------------------------------------------------------------------------
#
# Derived from graph.ARCHITECTURE_SPECS rather than restated. This dict used to
# be a hand-written mirror of build_crew_graph, which meant the console could
# describe a plan the crew did not run -- and with three selectable topologies
# that drift would be three times as likely and no more visible.

OBSERVABLE_CONTRACT: dict[str, Any] = {
    "released_per_step": ["step_type", "wall_time_s", "duration_s", "egress_bytes"],
    "never_released": [
        "query",
        "retrieved documents",
        "prompts",
        "model output",
        "labels",
        "credentials",
    ],
}


def crew_topology(architecture: str = DEFAULT_ARCHITECTURE) -> dict[str, Any]:
    """Topology of one crew architecture, plus the shared observable contract."""

    return {
        **architecture_spec(architecture).to_dict(),
        "default_architecture": DEFAULT_ARCHITECTURE,
        "architectures": {
            name: architecture_spec(name).to_dict() for name in ARCHITECTURES
        },
        # The frozen plan set the verifier checks a receipt's plan digest
        # against. Published here so a reviewer can see the legitimate plans
        # without reading the source.
        "plans": registry_manifest(),
        "observable_contract": OBSERVABLE_CONTRACT,
    }


#: The released pipeline_crew topology, kept under its historical name.
CREW_TOPOLOGY: dict[str, Any] = crew_topology()


# --------------------------------------------------------------------------
# Failure taxonomy
# --------------------------------------------------------------------------

TAXONOMY: list[dict[str, Any]] = [
    {
        "id": "leakage",
        "title": "Trace-shape leakage channels",
        "summary": (
            "The attack surface of the paper: three observable coordinates of an "
            "agent trace that remain visible to the host even when payloads are "
            "encrypted end to end."
        ),
        "failures": [
            {
                "id": "L1_structure",
                "code": "L1",
                "title": "Structure leakage — adaptive control flow",
                "category": "leakage",
                "severity": "high",
                "paper_anchor": "Adaptive attribute AUC 0.995 (paper, reported unverified)",
                "description": (
                    "The number of clinical-extraction passes and repair loops depends on "
                    "the private case. Sensitive fixtures take 7 passes, routine ones 4, "
                    "and repair loops re-enter coverage assessment — the step sequence alone "
                    "separates the classes."
                ),
                "demo": "Run one sensitive and one routine case adaptively, compare shapes.",
            },
            {
                "id": "L2_timing",
                "code": "L2",
                "title": "Timing leakage — data-dependent step durations",
                "category": "leakage",
                "severity": "medium",
                "paper_anchor": "Total-timing single-feature attribute AUC 0.94 (paper)",
                "description": (
                    "Per-step wall time correlates with the amount of private context "
                    "processed. Even with a canonical structure, durations remain a "
                    "usable attack feature until deadline padding is enforced."
                ),
                "demo": "Compare per-step durations across conditions for one case.",
            },
            {
                "id": "L3_egress_size",
                "code": "L3",
                "title": "Size leakage survives structure canonicalization",
                "category": "leakage",
                "severity": "high",
                "paper_anchor": "Structure-only attribute AUC 0.986 (paper)",
                "description": (
                    "Canonicalizing the step sequence does NOT close the size channel: "
                    "sensitive drafts emit more bytes. This is the taxonomy-based "
                    "failure the paper highlights — defending one coordinate while "
                    "leaving another open still loses."
                ),
                "demo": (
                    "Run sensitive vs routine under structure_only: identical step "
                    "sequences, clearly different egress bytes."
                ),
            },
        ],
    },
    {
        "id": "runtime",
        "title": "Runtime enforcement failures (fail-closed)",
        "summary": (
            "The full-pad shield promises a data-independent observable. When a step "
            "cannot meet the promise the runtime must fail closed: release the "
            "constant, drop the payload, and mark the run uncertified."
        ),
        "failures": [
            {
                "id": "F1_deadline_overrun",
                "code": "F1",
                "title": "Deadline overrun — provider slower than the padded deadline",
                "category": "runtime",
                "severity": "high",
                "paper_anchor": "~4% of steps would exceed the modeled deadline (paper limitation)",
                "description": (
                    "Under full padding every model step must finish inside the public "
                    "deadline. A slow provider call times out, the step releases the "
                    "constant deadline/size anyway, the payload is discarded, and the "
                    "receipt certifies NOTHING (guarantee kind 'none', budget breach)."
                ),
                "demo": (
                    "Inject 60ms provider latency under a 15ms deadline and watch every "
                    "step fail closed."
                ),
            },
            {
                "id": "F2_egress_overrun",
                "code": "F2",
                "title": "Egress overrun — response exceeds the fixed envelope",
                "category": "runtime",
                "severity": "high",
                "paper_anchor": "Fixed egress envelope, Section 5 (paper)",
                "description": (
                    "If a model response is larger than the public egress ceiling the "
                    "step cannot be released at the fixed size without truncation, so "
                    "the runtime fails closed instead of leaking the true size."
                ),
                "demo": "Inflate the draft to 4KB with a 512-byte ceiling.",
            },
            {
                "id": "F3_token_budget",
                "code": "F3",
                "title": "Token-budget exhaustion aborts the matter",
                "category": "runtime",
                "severity": "medium",
                "paper_anchor": "Per-matter cost guardrail (artifact)",
                "description": (
                    "The per-matter token guardrail aborts the crew mid-flight. The "
                    "run fails without a receipt; the console reports a payload-free "
                    "error. This is an availability failure, not a privacy failure."
                ),
                "demo": "Set a 30-token matter budget and watch the run abort at step 3.",
            },
        ],
    },
    {
        "id": "integrity",
        "title": "Receipt & ledger integrity failures",
        "summary": (
            "Receipts are Ed25519-signed fixed-size envelopes chained in a ledger. "
            "Any bit flip must be detected by the offline verifier."
        ),
        "failures": [
            {
                "id": "I1_receipt_tamper",
                "code": "I1",
                "title": "Tampered receipt body — signature check fails",
                "category": "integrity",
                "severity": "high",
                "paper_anchor": "Trace Receipts, Section 6 (paper)",
                "description": (
                    "Flipping the condition field of a signed receipt (e.g. claiming "
                    "full_pad instead of adaptive) invalidates the Ed25519 signature."
                ),
                "demo": "Sign a real receipt, flip body.condition, verify.",
            },
            {
                "id": "I2_envelope_break",
                "code": "I2",
                "title": "Non-canonical envelope — fixed-size encoding violated",
                "category": "integrity",
                "severity": "medium",
                "paper_anchor": "Fixed-size receipt envelope (artifact)",
                "description": (
                    "Receipts must re-encode to exactly the fixed envelope size. "
                    "Stripping padding or re-serializing with different key order is "
                    "rejected before any signature math runs."
                ),
                "demo": "Truncate the padding of a real receipt, verify.",
            },
            {
                "id": "I3_ledger_chain_break",
                "code": "I3",
                "title": "Ledger chain break — hash chain detects reordering",
                "category": "integrity",
                "severity": "high",
                "paper_anchor": "Hash-chained ledger (artifact)",
                "description": (
                    "Each receipt commits to the previous receipt's hash. Tampering "
                    "with receipt N breaks verification of receipt N+1."
                ),
                "demo": "Chain two receipts, tamper the first, verify the chain.",
            },
        ],
    },
    {
        "id": "evaluation",
        "title": "Evaluation pitfalls (honest measurement)",
        "summary": (
            "Ways an attack evaluation silently lies, and the controls this artifact "
            "runs to refuse them."
        ),
        "failures": [
            {
                "id": "V1_single_class",
                "code": "V1",
                "title": "Degenerate labels — AUC refuses single-class data",
                "category": "evaluation",
                "severity": "medium",
                "paper_anchor": "Mann-Whitney AUC requires both classes",
                "description": (
                    "Feeding only sensitive cases to the attacker cannot produce an "
                    "AUC. The pipeline raises instead of fabricating 0.5."
                ),
                "demo": "Evaluate the attack on sensitive-only traces.",
            },
            {
                "id": "V2_permutation",
                "code": "V2",
                "title": "Permutation control — shuffled labels must score ~0.5",
                "category": "evaluation",
                "severity": "low",
                "paper_anchor": "Permutation AUC 0.527 (paper)",
                "description": (
                    "Random labels on real traces must yield chance AUC. If they do "
                    "not, the pipeline is leaking labels into features."
                ),
                "demo": "Run the permutation control on leaky adaptive traces.",
            },
            {
                "id": "V3_shape_fixed",
                "code": "V3",
                "title": "Shape-fixed control — constant features must score 0.5",
                "category": "evaluation",
                "severity": "low",
                "paper_anchor": "Shape-fixed AUC 0.5 (paper)",
                "description": (
                    "Replacing every observable feature with its median forces the "
                    "attacker to chance. This is what a perfect full-pad defense looks "
                    "like to the attack."
                ),
                "demo": "Run the shape-fixed control on the same traces.",
            },
        ],
    },
]


# --------------------------------------------------------------------------
# Playground provider — a FixtureProvider with injectable corner cases
# --------------------------------------------------------------------------

class PlaygroundProvider(FixtureProvider):
    """Deterministic fixture provider with user-controlled corner cases.

    All knobs affect only the synthetic execution shape; provenance stays
    ``synthetic_fixture`` so results can never masquerade as evidence.
    """

    def __init__(
        self,
        *,
        delay_ms: float = 0.0,
        research_target_hops: int | None = None,
        force_citation_repair: bool = False,
        force_review_revision: bool = False,
        inflate_response_bytes: int = 0,
        fail_at_role: str | None = None,
    ) -> None:
        self.delay_ms = max(0.0, float(delay_ms))
        self.research_target_hops = research_target_hops
        self.force_citation_repair = force_citation_repair
        self.force_review_revision = force_review_revision
        self.inflate_response_bytes = max(0, int(inflate_response_bytes))
        self.fail_at_role = fail_at_role

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        if self.fail_at_role and request.role == self.fail_at_role:
            raise RuntimeError("playground-injected provider failure")
        if self.delay_ms:
            time.sleep(self.delay_ms / 1000.0)
        context = dict(request.fixture_context)
        if request.role == "clinical_extraction" and self.research_target_hops is not None:
            context["target_hops"] = self.research_target_hops
        if request.role == "criteria_check" and self.force_citation_repair:
            checks = int(context.get("criteria_checks", 0))
            text = json.dumps(
                {
                    "needs_repair": checks == 0,
                    "feedback": "re-link criteria to source ids"
                    if checks == 0
                    else "criteria source-supported",
                },
                separators=(",", ":"),
            )
            return self._respond(text)
        if request.role == "necessity_review" and self.force_review_revision:
            checks = int(context.get("necessity_checks", 0))
            text = json.dumps(
                {
                    "needs_revision": checks == 0,
                    "feedback": "record gaps and uncertainty"
                    if checks == 0
                    else "review complete",
                },
                separators=(",", ":"),
            )
            return self._respond(text)
        patched = ProviderRequest(
            role=request.role,
            model=request.model,
            max_tokens=request.max_tokens,
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            seed=request.seed,
            timeout_s=request.timeout_s,
            fixture_context=context,
        )
        response = super().complete(patched)
        if self.inflate_response_bytes and request.role in {
            "coverage_assessment",
            "determination",
        }:
            filler = " synthetic-filler" * (self.inflate_response_bytes // 17 + 1)
            text = response.text + filler
            text = text[: max(len(response.text), self.inflate_response_bytes)]
            return self._respond(text)
        return response

    @staticmethod
    def _respond(text: str) -> ProviderResponse:
        encoded = text.encode("utf-8")
        return ProviderResponse(
            text=text,
            response_bytes=len(encoded),
            request_bytes=len(encoded),
            input_tokens=0,
            output_tokens=max(1, (len(text) + 3) // 4) if text else 0,
            stop_reason="end_turn",
        )


# --------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------

class PlaygroundKnobs(BaseModel):
    research_target_hops: int | None = Field(default=None, ge=1, le=10)
    min_research_hops: int | None = Field(default=None, ge=1, le=10)
    max_research_hops: int | None = Field(default=None, ge=7, le=12)
    step_deadline_ms: float | None = Field(default=None, ge=1.0, le=500.0)
    egress_ceiling_bytes: int | None = Field(default=None, ge=16, le=65536)
    max_tokens_per_matter: int | None = Field(default=None, ge=10, le=400_000)
    provider_delay_ms: float = Field(default=0.0, ge=0.0, le=250.0)
    inflate_response_bytes: int = Field(default=0, ge=0, le=16384)
    force_citation_repair: bool = False
    force_review_revision: bool = False
    fail_at_role: (
        Literal[
            "intake",
            "clinical_extraction",
            "coverage_assessment",
            "criteria_check",
            "necessity_review",
            "determination",
        ]
        | None
    ) = None


class PlaygroundRunRequest(BaseModel):
    case_id: str
    condition: Literal["adaptive", "structure", "full", "structure_only", "full_pad"] = "adaptive"
    seed: int = 0
    label: str | None = Field(default=None, max_length=80)
    knobs: PlaygroundKnobs = Field(default_factory=PlaygroundKnobs)


class TaxonomyDemoRequest(BaseModel):
    demo_id: str
    seed: int = 0


class AttackAnalysisRequest(BaseModel):
    source: Literal["managed_run", "fresh_fixture"] = "fresh_fixture"
    run_id: str | None = None
    condition: Literal["adaptive", "structure", "full"] = "adaptive"
    target: Literal["attribute", "membership", "both"] = "both"
    case_limit: int = Field(default=12, ge=12, le=default_case_count())
    repetitions: int = Field(default=1, ge=1, le=3)
    seed: int = 0
    n_bootstrap: int = Field(default=200, ge=0, le=1000)
    include_ablations: bool = True
    include_frontier: bool = True
    include_controls: bool = True


CONDITION_ALIASES = {
    "adaptive": "adaptive",
    "structure": "structure_only",
    "structure_only": "structure_only",
    "full": "full_pad",
    "full_pad": "full_pad",
}


# --------------------------------------------------------------------------
# Console service
# --------------------------------------------------------------------------

class ConsoleService:
    """Fixture-only execution helpers shared by playground/taxonomy/attack routes."""

    def __init__(self, settings: Settings, catalog: CorpusCatalog) -> None:
        self.base_settings = settings
        self.catalog = catalog
        self._lock = threading.Lock()
        self._playground_runs: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._records_cache: dict[str, list[dict[str, Any]]] = {}
        self._counter = 0

    # -- shared execution -------------------------------------------------

    def _fixture_settings(self, **overrides: Any) -> Settings:
        base = self.base_settings
        values: dict[str, Any] = {
            "provider": "fixture",
            "azure_openai_api_key": None,
            "model_fast": base.model_fast,
            "model_deep": base.model_deep,
            "model_review": base.model_review,
            "model_vision": base.model_vision,
            "max_tokens_per_matter": base.max_tokens_per_matter,
            "max_research_hops": base.max_research_hops,
            "min_research_hops": base.min_research_hops,
            "canonical_research_hops": 7,
            "step_deadline_s": base.step_deadline_s,
            "egress_ceiling_bytes": base.egress_ceiling_bytes,
            "receipt_envelope_bytes": base.receipt_envelope_bytes,
            "provider_timeout_s": base.provider_timeout_s,
            "max_tokens_fast": base.max_tokens_fast,
            "max_tokens_deep": base.max_tokens_deep,
            "dataset_path": base.dataset_path,
            "policy_id": base.policy_id,
        }
        values.update(overrides)
        return Settings(**values)

    def execute_fixture_run(
        self,
        *,
        case_id: str,
        condition: str,
        seed: int,
        settings: Settings,
        provider: PlaygroundProvider | FixtureProvider | None = None,
        include_answer: bool = True,
    ) -> dict[str, Any]:
        """Run the real SDK offline and return a metadata-first result bundle."""

        events: list[dict[str, Any]] = []

        def on_event(value: Any) -> None:
            events.append(metadata_event(value))

        sdk = TraceGuardSDK(
            settings=settings,
            provider=provider or FixtureProvider(),
            catalog=self.catalog,
        )
        started = time.perf_counter()
        error: dict[str, Any] | None = None
        result: Any = None
        try:
            result = sdk.run(case_id, condition, event_callback=on_event, seed=seed)
        except Exception as exc:  # payload-free by SDK contract
            error = {
                "type": type(exc).__name__,
                "message": str(exc)[:200],
            }
        elapsed = time.perf_counter() - started

        bundle: dict[str, Any] = {
            "case_id": case_id,
            "condition": condition,
            "seed": seed,
            "provider": "fixture",
            "status": "failed" if error else "completed",
            "wall_clock_s": round(elapsed, 4),
            "events": events,
            "error": error,
            "settings": settings.to_public_dict(),
            "evidence": {
                "tier": "fixture_diagnostic",
                "scientific_evidence": False,
                "paper_reproduction_claim": False,
            },
        }
        if result is not None:
            trace = result.trace.to_dict()
            receipt = result.receipt.to_dict()
            verification = verifier_module.verify_receipt(result.receipt)
            bundle.update(
                {
                    "trace": trace,
                    "receipt": receipt,
                    "receipt_verification": verification.to_dict(),
                    "metrics": result.metrics.to_dict(),
                    "runtime": redact_secrets(dict(result.runtime)),
                    "features": extract_trace_features(trace),
                }
            )
            if include_answer:
                bundle["answer"] = {
                    "text": result.answer,
                    "note": "Synthetic fixture output; not clinical advice, not paper evidence.",
                }
        return bundle

    def case_meta(self, case_id: str) -> dict[str, Any]:
        case = self.catalog.get_case(case_id)
        return {
            "case_id": case.case_id,
            "specialty": case.specialty,
            "topic": case.topic,
            "sensitivity_level": case.sensitivity_level,
            "framing": case.framing,
            "sensitive": case.sensitive,
            "canary_member": case.canary_member,
        }

    def pick_cases(
        self,
        *,
        sensitive: bool | None = None,
        level: int | None = None,
        limit: int = 1,
    ) -> list[MedicalCase]:
        """Select demonstration cases by rung, or by the coarse boolean.

        ``sensitive=True`` deliberately returns the *top* rung first rather than
        the first case whose level happens to exceed zero. These demos exist to
        show the routine/sensitive contrast; with a graded corpus, taking the
        first truthy match would have silently demonstrated a middle rung and
        made the contrast look weaker than the corpus supports.
        """

        cases = self.catalog.list_cases()
        if level is not None:
            cases = [case for case in cases if case.sensitivity_level == level]
        elif sensitive is True:
            cases = sorted(
                (case for case in cases if case.sensitive),
                key=lambda case: (-case.sensitivity_level, case.case_id),
            )
        elif sensitive is False:
            cases = [case for case in cases if not case.sensitive]
        return cases[:limit]

    # -- playground -------------------------------------------------------

    def playground_run(self, request: PlaygroundRunRequest) -> dict[str, Any]:
        condition = CONDITION_ALIASES[request.condition]
        knobs = request.knobs
        overrides: dict[str, Any] = {}
        if knobs.step_deadline_ms is not None:
            overrides["step_deadline_s"] = knobs.step_deadline_ms / 1000.0
        if knobs.egress_ceiling_bytes is not None:
            overrides["egress_ceiling_bytes"] = knobs.egress_ceiling_bytes
        if knobs.max_tokens_per_matter is not None:
            overrides["max_tokens_per_matter"] = knobs.max_tokens_per_matter
        max_hops = knobs.max_research_hops or self.base_settings.max_research_hops
        if knobs.research_target_hops is not None:
            max_hops = max(max_hops, 7, knobs.research_target_hops)
        overrides["max_research_hops"] = max_hops
        if knobs.min_research_hops is not None:
            overrides["min_research_hops"] = min(knobs.min_research_hops, max_hops)
        settings = self._fixture_settings(**overrides)
        provider = PlaygroundProvider(
            delay_ms=knobs.provider_delay_ms,
            research_target_hops=knobs.research_target_hops,
            force_citation_repair=knobs.force_citation_repair,
            force_review_revision=knobs.force_review_revision,
            inflate_response_bytes=knobs.inflate_response_bytes,
            fail_at_role=knobs.fail_at_role,
        )
        bundle = self.execute_fixture_run(
            case_id=request.case_id,
            condition=condition,
            seed=request.seed,
            settings=settings,
            provider=provider,
        )
        bundle["case"] = self.case_meta(request.case_id)
        bundle["knobs"] = knobs.model_dump()
        bundle["label"] = request.label
        with self._lock:
            self._counter += 1
            playground_id = f"pg-{self._counter:04d}"
            bundle["playground_id"] = playground_id
            bundle["created_at"] = utc_now()
            self._playground_runs[playground_id] = bundle
            while len(self._playground_runs) > 50:
                self._playground_runs.popitem(last=False)
        return bundle

    def playground_list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "playground_id": item["playground_id"],
                    "label": item.get("label"),
                    "case": item.get("case"),
                    "condition": item["condition"],
                    "status": item["status"],
                    "created_at": item["created_at"],
                    "metrics": item.get("metrics"),
                    "knobs": item.get("knobs"),
                }
                for item in reversed(self._playground_runs.values())
            ]

    def playground_get(self, playground_id: str) -> dict[str, Any]:
        with self._lock:
            try:
                return self._playground_runs[playground_id]
            except KeyError as error:
                raise KeyError(playground_id) from error

    # -- attack records ----------------------------------------------------

    def fixture_records(
        self, *, condition: str, case_limit: int, repetitions: int, seed: int
    ) -> list[dict[str, Any]]:
        """Generate (and cache) labeled fixture trace records for the attack."""

        key = f"{condition}|{case_limit}|{repetitions}|{seed}"
        with self._lock:
            if key in self._records_cache:
                return self._records_cache[key]
        sdk_condition = CONDITION_ALIASES[condition]
        settings = self._fixture_settings(step_deadline_s=0.01)
        cases = catalog_cases(self.catalog)[:case_limit]
        records: list[dict[str, Any]] = []
        for repetition in range(repetitions):
            for case in cases:
                sdk = TraceGuardSDK(
                    settings=settings, provider=FixtureProvider(), catalog=self.catalog
                )
                result = sdk.run(
                    case["case_id"],
                    sdk_condition,
                    seed=seed + repetition,
                )
                trace = result.trace.to_dict()
                records.append(
                    {
                        "trace": trace,
                        "run_id": result.run_id,
                        "case_id": case["case_id"],
                        "condition": condition,
                        "service": case["specialty"],
                        "topic": case["topic"],
                        "attribute_label": case["attribute_label"],
                        "membership_label": case["membership_label"],
                        "provenance": {"provider": "fixture"},
                    }
                )
        with self._lock:
            self._records_cache[key] = records
            while len(self._records_cache) > 12:
                self._records_cache.pop(next(iter(self._records_cache)))
        return records

    # -- taxonomy demos ----------------------------------------------------

    def run_demo(self, demo_id: str, seed: int) -> dict[str, Any]:
        handlers = {
            "L1_structure": self._demo_structure,
            "L2_timing": self._demo_timing,
            "L3_egress_size": self._demo_size,
            "F1_deadline_overrun": self._demo_deadline,
            "F2_egress_overrun": self._demo_egress_overrun,
            "F3_token_budget": self._demo_token_budget,
            "I1_receipt_tamper": self._demo_receipt_tamper,
            "I2_envelope_break": self._demo_envelope_break,
            "I3_ledger_chain_break": self._demo_ledger_break,
            "V1_single_class": self._demo_single_class,
            "V2_permutation": self._demo_permutation,
            "V3_shape_fixed": self._demo_shape_fixed,
        }
        handler = handlers.get(demo_id)
        if handler is None:
            raise KeyError(demo_id)
        started = time.perf_counter()
        payload = handler(seed)
        payload.update(
            {
                "demo_id": demo_id,
                "elapsed_s": round(time.perf_counter() - started, 3),
                "created_at": utc_now(),
                "evidence_tier": "fixture_diagnostic",
                "scientific_evidence": False,
            }
        )
        return redact_secrets(payload)

    # leakage -------------------------------------------------------------

    def _demo_structure(self, seed: int) -> dict[str, Any]:
        sensitive = self.pick_cases(sensitive=True, limit=1)[0]
        routine = self.pick_cases(sensitive=False, limit=1)[0]
        settings = self._fixture_settings(step_deadline_s=0.01)
        provider = PlaygroundProvider(force_citation_repair=True)
        runs = {
            "sensitive_adaptive": self.execute_fixture_run(
                case_id=sensitive.case_id,
                condition="adaptive",
                seed=seed,
                settings=settings,
                provider=provider,
                include_answer=False,
            ),
            "routine_adaptive": self.execute_fixture_run(
                case_id=routine.case_id,
                condition="adaptive",
                seed=seed,
                settings=settings,
                include_answer=False,
            ),
        }
        shapes = {
            name: [step["step_type"] for step in run.get("trace", {}).get("steps", [])]
            for name, run in runs.items()
        }
        return {
            "outcome": "leak_demonstrated",
            "headline": (
                f"Sensitive case emitted {len(shapes['sensitive_adaptive'])} steps with a repair "
                f"loop; routine case emitted {len(shapes['routine_adaptive'])}. The step sequence "
                "alone separates the two private classes."
            ),
            "cases": {
                "sensitive_adaptive": self.case_meta(sensitive.case_id),
                "routine_adaptive": self.case_meta(routine.case_id),
            },
            "step_sequences": shapes,
            "runs": runs,
        }

    def _demo_timing(self, seed: int) -> dict[str, Any]:
        case = self.pick_cases(sensitive=True, limit=1)[0]
        settings = self._fixture_settings(step_deadline_s=0.05)
        provider = PlaygroundProvider(delay_ms=8.0)
        runs = {
            condition: self.execute_fixture_run(
                case_id=case.case_id,
                condition=condition,
                seed=seed,
                settings=settings,
                provider=provider,
                include_answer=False,
            )
            for condition in ("adaptive", "structure_only", "full_pad")
        }
        durations = {
            name: [round(step["duration_s"] * 1000, 3) for step in run["trace"]["steps"]]
            for name, run in runs.items()
            if run.get("trace")
        }
        return {
            "outcome": "leak_demonstrated",
            "headline": (
                "Adaptive and structure-only durations follow the real provider latency; "
                "full_pad releases the constant deadline for every step."
            ),
            "case": self.case_meta(case.case_id),
            "per_step_duration_ms": durations,
            "runs": runs,
        }

    def _demo_size(self, seed: int) -> dict[str, Any]:
        sensitive = self.pick_cases(sensitive=True, limit=1)[0]
        routine = self.pick_cases(sensitive=False, limit=1)[0]
        settings = self._fixture_settings(step_deadline_s=0.01)
        runs = {
            "sensitive_structure_only": self.execute_fixture_run(
                case_id=sensitive.case_id,
                condition="structure_only",
                seed=seed,
                settings=settings,
                include_answer=False,
            ),
            "routine_structure_only": self.execute_fixture_run(
                case_id=routine.case_id,
                condition="structure_only",
                seed=seed,
                settings=settings,
                include_answer=False,
            ),
        }
        sizes = {
            name: [step["egress_bytes"] for step in run["trace"]["steps"]]
            for name, run in runs.items()
            if run.get("trace")
        }
        totals = {name: sum(values) for name, values in sizes.items()}
        return {
            "outcome": "leak_demonstrated",
            "headline": (
                "Both runs have the identical canonical 11-step structure, yet total egress "
                f"differs ({totals.get('sensitive_structure_only', 0)} vs "
                f"{totals.get('routine_structure_only', 0)} bytes): canonicalizing structure "
                "alone leaves the size channel wide open — the paper's structure-only "
                "attribute AUC stays at 0.986."
            ),
            "cases": {
                "sensitive_structure_only": self.case_meta(sensitive.case_id),
                "routine_structure_only": self.case_meta(routine.case_id),
            },
            "per_step_egress_bytes": sizes,
            "total_egress_bytes": totals,
            "runs": runs,
        }

    # runtime ---------------------------------------------------------------

    def _demo_deadline(self, seed: int) -> dict[str, Any]:
        case = self.pick_cases(sensitive=True, limit=1)[0]
        settings = self._fixture_settings(step_deadline_s=0.015)
        provider = PlaygroundProvider(delay_ms=60.0)
        run = self.execute_fixture_run(
            case_id=case.case_id,
            condition="full_pad",
            seed=seed,
            settings=settings,
            provider=provider,
            include_answer=True,
        )
        violations = run.get("runtime", {}).get("violations", [])
        return {
            "outcome": "fail_closed_demonstrated",
            "headline": (
                f"All {len(violations)} model steps overran the 15ms public deadline. Every step "
                "still released exactly deadline+ceiling, the payload was dropped, and the "
                "signed receipt refuses certification (guarantee kind 'none', budget breach)."
            ),
            "case": self.case_meta(case.case_id),
            "violations": violations,
            "receipt_guarantee": run.get("receipt", {}).get("body", {}).get("guarantee"),
            "receipt_budget_status": run.get("receipt", {}).get("body", {}).get("budget_status"),
            "runs": {"full_pad_overrun": run},
        }

    def _demo_egress_overrun(self, seed: int) -> dict[str, Any]:
        case = self.pick_cases(sensitive=True, limit=1)[0]
        settings = self._fixture_settings(step_deadline_s=0.05, egress_ceiling_bytes=512)
        provider = PlaygroundProvider(inflate_response_bytes=4096)
        run = self.execute_fixture_run(
            case_id=case.case_id,
            condition="full_pad",
            seed=seed,
            settings=settings,
            provider=provider,
            include_answer=True,
        )
        violations = run.get("runtime", {}).get("violations", [])
        return {
            "outcome": "fail_closed_demonstrated",
            "headline": (
                "The inflated draft/editor responses exceeded the 512-byte egress envelope; "
                f"{len(violations)} steps failed closed rather than leak their true size."
            ),
            "case": self.case_meta(case.case_id),
            "violations": violations,
            "receipt_guarantee": run.get("receipt", {}).get("body", {}).get("guarantee"),
            "runs": {"full_pad_egress_squeeze": run},
        }

    def _demo_token_budget(self, seed: int) -> dict[str, Any]:
        case = self.pick_cases(sensitive=True, limit=1)[0]
        settings = self._fixture_settings(max_tokens_per_matter=30, step_deadline_s=0.05)
        run = self.execute_fixture_run(
            case_id=case.case_id,
            condition="adaptive",
            seed=seed,
            settings=settings,
            include_answer=True,
        )
        return {
            "outcome": "run_aborted_demonstrated",
            "headline": (
                "The 30-token matter budget was exhausted after two model steps; the crew "
                "aborted with a payload-free error and no receipt was issued."
            ),
            "case": self.case_meta(case.case_id),
            "error": run.get("error"),
            "receipt_issued": bool(run.get("receipt")),
            "runs": {"budget_exhaustion": run},
        }

    # integrity --------------------------------------------------------------

    def _signed_receipt(self, seed: int) -> tuple[dict[str, Any], Any]:
        case = self.pick_cases(sensitive=False, limit=1)[0]
        settings = self._fixture_settings(step_deadline_s=0.01)
        run = self.execute_fixture_run(
            case_id=case.case_id,
            condition="adaptive",
            seed=seed,
            settings=settings,
            include_answer=False,
        )
        return run["receipt"], run

    def _demo_receipt_tamper(self, seed: int) -> dict[str, Any]:
        receipt, run = self._signed_receipt(seed)
        honest = verifier_module.verify_receipt(receipt).to_dict()
        tampered = json.loads(json.dumps(receipt))
        # 'full_pad' and 'adaptive' have equal length, so the fixed-size envelope
        # still parses and the failure is attributable to the signature + honesty
        # checks rather than a trivial size mismatch.
        tampered["body"]["condition"] = "full_pad"
        tampered_result = verifier_module.verify_receipt(tampered).to_dict()
        return {
            "outcome": "tamper_detected",
            "headline": (
                "Flipping body.condition from 'adaptive' to 'full_pad' — claiming a privacy "
                "guarantee that was never enforced — is caught twice: the Ed25519 signature "
                "fails AND the honesty invariants reject the unsupported (0,0) claim."
            ),
            "original_verification": honest,
            "tampered_field": "body.condition: adaptive -> full_pad",
            "tampered_verification": tampered_result,
            "receipt": receipt,
            "runs": {"source_run": run},
        }

    def _demo_envelope_break(self, seed: int) -> dict[str, Any]:
        receipt, run = self._signed_receipt(seed)
        honest = verifier_module.verify_receipt(receipt).to_dict()
        truncated = json.loads(json.dumps(receipt))
        truncated["padding"] = truncated["padding"][:-16]
        result = verifier_module.verify_receipt(truncated).to_dict()
        return {
            "outcome": "tamper_detected",
            "headline": (
                "Stripping 16 padding bytes breaks the fixed-size canonical encoding; the "
                "verifier rejects the envelope before any signature math runs."
            ),
            "original_verification": honest,
            "tampered_field": "padding truncated by 16 bytes",
            "tampered_verification": result,
            "runs": {"source_run": run},
        }

    def _demo_ledger_break(self, seed: int) -> dict[str, Any]:
        case_a, case_b = self.pick_cases(limit=2)
        settings = self._fixture_settings(step_deadline_s=0.01)
        signer = ReceiptSigner.generate(envelope_bytes=settings.receipt_envelope_bytes)
        ledger = HashChainLedger(None)
        sdk = TraceGuardSDK(settings=settings, provider=FixtureProvider(), catalog=self.catalog)
        sdk.signer = signer
        sdk.ledger = ledger
        result_a = sdk.run(case_a.case_id, "adaptive", seed=seed)
        result_b = sdk.run(case_b.case_id, "adaptive", seed=seed)
        honest = ledger.verify(signer.public_key)
        honest_dict = honest.to_dict() if hasattr(honest, "to_dict") else dict(honest)
        tampered_first = json.loads(json.dumps(result_a.receipt.to_dict()))
        # 'breach' and 'within' have equal length: the envelope still parses, so the
        # chain demo surfaces signature, honesty, AND hash-chain errors together.
        tampered_first["body"]["budget_status"] = "breach"
        chain = [tampered_first, result_b.receipt.to_dict()]
        broken = verifier_module.verify_ledger(
            chain, pinned_public_key=signer.public_key
        ).to_dict()
        return {
            "outcome": "tamper_detected",
            "headline": (
                "Receipt #1 commits to receipt #0's hash. Tampering receipt #0's "
                "budget_status invalidates its signature, trips the fail-closed honesty "
                "invariant, and breaks the chain commitment checked for receipt #1."
            ),
            "honest_chain_verification": honest_dict,
            "tampered_field": "receipt[0].body.budget_status: within -> breach",
            "broken_chain_verification": broken,
            "ledger_receipts": [result_a.receipt.to_dict(), result_b.receipt.to_dict()],
        }

    # evaluation ---------------------------------------------------------------

    def _demo_single_class(self, seed: int) -> dict[str, Any]:
        records = self.fixture_records(
            condition="adaptive", case_limit=12, repetitions=1, seed=seed
        )
        sensitive_only = [row for row in records if row["attribute_label"] == 1]
        try:
            evaluate_attack(sensitive_only, target="attribute", n_bootstrap=0, seed=seed)
            outcome = {"raised": False}
        except ValueError as error:
            outcome = {"raised": True, "error_type": "ValueError", "message": str(error)[:300]}
        return {
            "outcome": "honest_refusal_demonstrated",
            "headline": (
                f"Fed {len(sensitive_only)} sensitive-only traces, the evaluator raised "
                "instead of fabricating an AUC — degenerate labels cannot be scored."
            ),
            "input_records": len(sensitive_only),
            "labels_present": sorted({row["attribute_label"] for row in sensitive_only}),
            "evaluator_response": outcome,
        }

    def _demo_permutation(self, seed: int) -> dict[str, Any]:
        records = self.fixture_records(
            condition="adaptive", case_limit=16, repetitions=1, seed=seed
        )
        real = evaluate_attack(records, target="attribute", n_bootstrap=0, seed=seed)
        control = permutation_control(records, target="attribute", seed=seed + 101)
        return {
            "outcome": "control_holds",
            "headline": (
                f"Real labels: attribute AUC {real['auc']:.3f} (fixture traces are deliberately "
                f"leaky). Shuffled labels: AUC {control['auc']:.3f} — chance, as an honest "
                "pipeline requires."
            ),
            "real_auc": real["auc"],
            "permutation_auc": control["auc"],
            "record_count": len(records),
            "real_evaluation": {
                k: real[k] for k in ("auc", "pooled_auc", "selected_attacker", "models")
            },
            "control_evaluation": {k: control[k] for k in ("auc", "pooled_auc", "control")},
        }

    def _demo_shape_fixed(self, seed: int) -> dict[str, Any]:
        records = self.fixture_records(
            condition="adaptive", case_limit=16, repetitions=1, seed=seed
        )
        real = evaluate_attack(records, target="attribute", n_bootstrap=0, seed=seed)
        control = shape_fixed_control(records, target="attribute", seed=seed + 202)
        return {
            "outcome": "control_holds",
            "headline": (
                f"With every observable feature frozen to its median the attacker drops from "
                f"AUC {real['auc']:.3f} to {control['auc']:.3f} — this is exactly what the "
                "full-pad defense looks like to the attack."
            ),
            "real_auc": real["auc"],
            "shape_fixed_auc": control["auc"],
            "record_count": len(records),
        }

    # -- attack analysis ----------------------------------------------------

    def analyze(
        self, request: AttackAnalysisRequest, managed_traces: list[dict[str, Any]] | None
    ) -> dict[str, Any]:
        if request.source == "managed_run":
            records = [
                {
                    "trace": trace,
                    "service": trace.get("service", trace.get("specialty")),
                    "topic": trace.get("topic"),
                    "attribute_label": trace.get("attribute_label"),
                    "membership_label": trace.get("membership_label"),
                    "provenance": {"provider": "fixture"},
                }
                for trace in (managed_traces or [])
                if trace.get("condition") in (None, request.condition)
            ]
            if not records:
                raise ValueError(
                    "the selected run has no traces for that condition; run a fixture "
                    "experiment first"
                )
        else:
            records = self.fixture_records(
                condition=request.condition,
                case_limit=request.case_limit,
                repetitions=request.repetitions,
                seed=request.seed,
            )
        targets = (
            ["attribute", "membership"] if request.target == "both" else [request.target]
        )
        def safe(name: str, thunk: Any) -> Any:
            # Optional controls/ablations can hit a degenerate small-sample fold;
            # surface that as a per-component note instead of failing the request.
            try:
                return thunk()
            except ValueError as error:
                return {"unavailable": True, "reason": str(error)[:200], "component": name}

        analysis: dict[str, Any] = {}
        for target in targets:
            # The core attack still raises loudly: a degenerate main result is a
            # real error the caller must see, not a silently-skipped control.
            entry: dict[str, Any] = {
                "attack": evaluate_attack(
                    records,
                    target=target,
                    seed=request.seed,
                    n_bootstrap=request.n_bootstrap,
                )
            }
            if request.include_controls:
                entry["permutation_control"] = safe(
                    "permutation_control",
                    lambda t=target: permutation_control(
                        records, target=t, seed=request.seed + 101
                    ),
                )
                entry["shape_fixed_control"] = safe(
                    "shape_fixed_control",
                    lambda t=target: shape_fixed_control(
                        records, target=t, seed=request.seed + 202
                    ),
                )
            if request.include_ablations:
                entry["ablations"] = safe(
                    "ablations",
                    lambda t=target: run_ablation_suite(records, target=t, seed=request.seed + 303),
                )
            if request.include_frontier:
                entry["frontier"] = safe(
                    "frontier",
                    lambda t=target: randomized_response_frontier(
                        records, target=t, seed=request.seed + 404, repetitions=5
                    ),
                )
            analysis[target] = entry
        return {
            "schema_version": "traceguard.console-analysis.v1",
            "created_at": utc_now(),
            "condition": request.condition,
            "source": request.source,
            "record_count": len(records),
            "per_record_features": [
                {
                    "case_id": row.get("case_id", row.get("trace", {}).get("case_id")),
                    "attribute_label": row.get("attribute_label"),
                    "membership_label": row.get("membership_label"),
                    "service": row.get("service"),
                    "topic": row.get("topic"),
                    "step_count": len(row.get("trace", {}).get("steps", [])),
                    "total_egress_bytes": sum(
                        step.get("egress_bytes", 0)
                        for step in row.get("trace", {}).get("steps", [])
                    ),
                }
                for row in records
            ],
            "paper_reproduction_claim": False,
            "evidence": {
                "tier": "fixture_diagnostic",
                "scientific_evidence": False,
                "note": (
                    "Fixture traces are deliberately shaped to demonstrate the attack "
                    "mechanics; AUC values here validate the pipeline, not the paper."
                ),
            },
            "analysis": redact_secrets(analysis),
        }


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------

def build_console_router() -> APIRouter:
    router = APIRouter(prefix="/api")

    def service(request: Request) -> ConsoleService:
        state = request.app.state
        existing = getattr(state, "console_service", None)
        if existing is None:
            existing = ConsoleService(state.settings, state.catalog)
            state.console_service = existing
        return existing

    @router.get("/graph/topology")
    def graph_topology(
        request: Request,
        architecture: str = DEFAULT_ARCHITECTURE,
    ) -> dict[str, Any]:
        settings: Settings = request.app.state.settings
        if architecture not in ARCHITECTURES:
            raise HTTPException(status_code=400, detail="unknown crew architecture")
        return {
            **crew_topology(architecture),
            "models": {
                "fast": settings.model_fast,
                "deep": settings.model_deep,
                "review": settings.model_review,
            },
            "conditions": {
                "adaptive": {
                    "label": "Adaptive (leaky baseline)",
                    "structure_enforced": False,
                    "timing_size_enforced": False,
                },
                "structure_only": {
                    "label": "Structure-only canonicalization",
                    "structure_enforced": True,
                    "timing_size_enforced": False,
                },
                "full_pad": {
                    "label": "Full pad (structure + timing + size)",
                    "structure_enforced": True,
                    "timing_size_enforced": True,
                },
            },
        }

    @router.get("/store/runs")
    def store_runs(request: Request) -> dict[str, Any]:
        store = request.app.state.store
        return {"runs": store.list_runs()}

    @router.get("/store/runs/{run_id}")
    def store_run_detail(request: Request, run_id: str) -> dict[str, Any]:
        store = request.app.state.store
        if not store.exists(run_id):
            raise HTTPException(status_code=404, detail="stored run not found")
        detail: dict[str, Any] = {
            "run_id": run_id,
            "manifest": store.manifest(run_id),
            "traces": store.read_jsonl(run_id, "traces.jsonl"),
            "receipts": store.read_jsonl(run_id, "receipts.jsonl"),
            "events": store.read_jsonl(run_id, "events.jsonl"),
            "verify": store.verify(run_id),
        }
        try:
            detail["result"] = store.read_json(run_id, "result.json")
        except FileNotFoundError:
            detail["result"] = None
        return redact_secrets(detail)

    @router.get("/taxonomy")
    def taxonomy() -> dict[str, Any]:
        return {
            "coordinate_contract": ["structure", "timing", "egress_size"],
            "categories": TAXONOMY,
        }

    @router.post("/taxonomy/demo")
    def taxonomy_demo(request: Request, body: TaxonomyDemoRequest) -> dict[str, Any]:
        try:
            return service(request).run_demo(body.demo_id, body.seed)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=f"unknown demo {body.demo_id!r}") from error

    @router.post("/playground/run")
    def playground_run(request: Request, body: PlaygroundRunRequest) -> dict[str, Any]:
        svc = service(request)
        try:
            svc.catalog.get_case(body.case_id)
        except KeyError as error:
            raise HTTPException(status_code=422, detail=f"unknown case {body.case_id!r}") from error
        try:
            return svc.playground_run(body)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @router.get("/playground/runs")
    def playground_runs(request: Request) -> dict[str, Any]:
        return {"runs": service(request).playground_list()}

    @router.get("/playground/runs/{playground_id}")
    def playground_run_detail(request: Request, playground_id: str) -> dict[str, Any]:
        try:
            return service(request).playground_get(playground_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="playground run not found") from error

    @router.post("/analysis/attack")
    def analysis_attack(request: Request, body: AttackAnalysisRequest) -> dict[str, Any]:
        svc = service(request)
        managed_traces: list[dict[str, Any]] | None = None
        if body.source == "managed_run":
            if not body.run_id:
                raise HTTPException(status_code=422, detail="run_id required for managed_run")
            manager = request.app.state.run_manager
            try:
                managed = manager.get(body.run_id)
            except KeyError as error:
                raise HTTPException(status_code=404, detail="run not found") from error
            managed_traces = managed.traces
        try:
            return svc.analyze(body, managed_traces)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @router.get("/ledger")
    def ledger(request: Request) -> dict[str, Any]:
        state = request.app.state
        runtime_dir = state.store.root.parent / "runtime"
        path = runtime_dir / "ledger.jsonl"
        if not path.is_file():
            return {"path": str(path), "receipts": [], "verification": None}
        chain = HashChainLedger(path)
        receipts = [receipt.to_dict() for receipt in chain.receipts]
        try:
            outcome = chain.verify()
            verification = outcome.to_dict() if hasattr(outcome, "to_dict") else outcome
        except Exception as error:
            verification = {"valid": False, "error": type(error).__name__}
        return redact_secrets(
            {
                "path": str(path),
                "count": len(receipts),
                "receipts": receipts[-25:],
                "verification": verification,
            }
        )

    return router

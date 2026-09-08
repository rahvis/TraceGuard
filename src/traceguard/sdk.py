"""High-level public SDK for one end-to-end TraceGuard execution."""

from __future__ import annotations

import re
import threading
import uuid
from pathlib import Path

from .agents import TraceWriterCrew
from .config import PROVIDERS, Settings
from .corpus import CorpusCatalog
from .graph import DEFAULT_ARCHITECTURE, architecture_spec, build_crew_graph
from .instrumentation import TraceRecorder
from .ledger import HashChainLedger
from .providers import (
    AzureOpenAIProvider,
    FixtureProvider,
    LLMProvider,
    create_provider,
)
from .receipt import ReceiptSigner, build_receipt_body
from .shield_runtime import ShieldRuntime
from .types import Autonomy, Condition, EventCallback, RunMetrics, RunResult

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_UNSET = object()


class TraceGuardSDK:
    """Run the paper-aligned crew, defense, receipt, and ledger pipeline.

    If ``artifact_dir`` is supplied, the SDK writes only a global
    ``ledger.jsonl`` containing fixed-size metadata receipts.  It does not create
    per-run directories and never persists prompts, queries, documents, or API
    keys.  Callers remain free to own their run-artifact lifecycle.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        provider: LLMProvider | str | None = None,
        artifact_dir: str | Path | None = None,
        *,
        catalog: CorpusCatalog | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.corpus = catalog or (
            CorpusCatalog.load(self.settings.dataset_path)
            if self.settings.dataset_path is not None
            else CorpusCatalog.load_default()
        )
        if provider is None:
            self.provider = create_provider(self.settings)
        elif isinstance(provider, str):
            normalized = provider.strip().lower()
            if normalized == "fixture":
                self.provider = FixtureProvider()
            elif normalized == "azure":
                self.provider = AzureOpenAIProvider(self.settings)
            else:
                raise ValueError(
                    f"provider override must be one of {', '.join(sorted(PROVIDERS))} "
                    "or an LLMProvider"
                )
        elif hasattr(provider, "complete") and hasattr(provider, "provenance"):
            self.provider = provider
        else:
            raise TypeError("provider must implement complete() and provenance")
        # Discover each deployment's request schema now, before any step is
        # timed. Doing it lazily on first use would put a discovery retry inside
        # a measured step and corrupt that step's duration.
        warm_up = getattr(self.provider, "warm_up", None)
        if callable(warm_up):
            self.provider_capabilities = warm_up()
        else:
            self.provider_capabilities = {}
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        ledger_path = self.artifact_dir / "ledger.jsonl" if self.artifact_dir else None
        self.ledger = HashChainLedger(ledger_path)
        self.signer = ReceiptSigner.from_settings(self.settings)
        self._receipt_lock = threading.RLock()
        # Attestation must be minted under its own lock. Without it, N worker
        # threads all see the _UNSET sentinel and fire N concurrent attest()
        # calls -- N tpm2_nvread + tpm2_quote subprocesses against a single
        # vTPM, N MAA POSTs, N-1 discarded. On a plain host that is only waste;
        # inside a CVM, vTPM contention can fail a quote and degrade that
        # thread's binding to None, producing a row that reads *unattested*
        # for a reason unrelated to the TEE.
        self._attestation_lock = threading.RLock()
        self._attestation_binding: dict | None = _UNSET  # type: ignore[assignment]
        self._attestation_result: object | None = _UNSET  # type: ignore[assignment]

    def attestation_binding(self) -> dict | None:
        """Cached remote-attestation binding for receipts (computed once).

        Returns the compact, signable verdict from :mod:`traceguard.attestation`,
        or ``None`` if attestation could not be obtained.  On a plain host this is
        an honest ``simulated`` binding (``hardware_backed`` False); on an Azure
        Confidential VM it is a hardware-backed, MAA-verified binding.
        """

        with self._attestation_lock:
            if self._attestation_binding is _UNSET:  # type: ignore[comparison-overlap]
                try:
                    import base64 as _base64
                    import secrets as _secrets

                    from .attestation import attest

                    # Bind the receipt signing key into the attestation, with a
                    # fresh per-process nonce. Without this the receipt would
                    # prove only that an MAA-verified quote exists for some VM
                    # at some time, plus a signature under a key of unproven
                    # residency -- a captured token would satisfy it. The nonce
                    # is what makes the binding fresh; the quote is what makes
                    # it this enclave.
                    signing_public = _base64.b64decode(self.signer.public_key)
                    result = attest(
                        nonce=_secrets.token_hex(16),
                        bind_public_key=signing_public,
                    )
                    self._attestation_result = result
                    self._attestation_binding = result.receipt_binding()
                except Exception:  # noqa: BLE001 - never break a run
                    self._attestation_result = None
                    self._attestation_binding = None
            return self._attestation_binding

    def attestation_evidence(self) -> dict | None:
        """Full attestation evidence, including the MAA token.

        ``attestation_binding`` deliberately carries only digests, because it
        is signed into every receipt. The offline verifier needs the token
        itself to check MAA's signature, the launch measurement and the
        report-data join, so it is exposed separately and written to a sidecar
        rather than into the receipt.
        """

        with self._attestation_lock:
            if self._attestation_binding is _UNSET:  # type: ignore[comparison-overlap]
                self.attestation_binding()
            result = self._attestation_result
            if result is None or result is _UNSET:  # type: ignore[comparison-overlap]
                return None
            evidence = result.to_dict(include_token=True)  # type: ignore[attr-defined]
            # to_dict is the *receipt-independent* view and omits the key
            # binding, which lives on receipt_binding(). The offline verifier
            # needs both: the token to check MAA's signature, and the binding
            # to walk quote -> AK -> SNP report data -> signing key. Without it
            # the sidecar cannot support step 5 of the verification recipe.
            key_binding = getattr(result, "key_binding", None)
            if key_binding is not None:
                evidence["key_binding"] = key_binding
            return evidence

    @property
    def dataset_hash(self) -> str:
        return self.corpus.dataset_hash

    def run(
        self,
        case_id: str,
        condition: str | Condition,
        run_id: str | None = None,
        event_callback: EventCallback | None = None,
        seed: int = 0,
        autonomy: str | Autonomy | None = None,
        architecture: str | None = None,
    ) -> RunResult:
        selected = Condition.parse(condition)
        # Crew topology is a third orthogonal axis: `condition` shapes the
        # release, `autonomy` bounds how much the data may steer control flow,
        # and `architecture` decides which plan the crew executes at all. It
        # defaults to the released pipeline, so an archived configuration that
        # names neither reproduces exactly.
        spec = architecture_spec(architecture or DEFAULT_ARCHITECTURE)
        # Autonomy is orthogonal to the shaping condition. When unset it falls
        # back to the level the condition used to imply, so every archived run
        # keeps its original meaning; new experiments pass it explicitly.
        selected_autonomy = (
            Autonomy.parse(autonomy)
            if autonomy is not None
            else Autonomy.default_for(selected)
        )
        if selected is not Condition.ADAPTIVE and selected_autonomy is not Autonomy.SCRIPTED:
            # A canonicalized release requires a data-independent plan; letting
            # the agent branch under it would emit a trace that is not the
            # canonical one, and the invariant below would fail with a much less
            # informative error.
            raise ValueError(
                f"condition {selected.value!r} canonicalizes the plan and therefore "
                f"requires autonomy 'scripted', not {selected_autonomy.value!r}"
            )
        identifier = run_id or uuid.uuid4().hex
        if not _RUN_ID.fullmatch(identifier):
            raise ValueError(
                "run_id must be 1-128 characters using letters, digits, '.', '_' or '-'"
            )
        case = self.corpus.get_case(case_id)
        shield = ShieldRuntime(self.settings, selected)
        recorder = TraceRecorder(run_id=identifier, shield=shield, event_callback=event_callback)
        crew = TraceWriterCrew(
            settings=self.settings,
            provider=self.provider,
            corpus=self.corpus,
            recorder=recorder,
        )
        graph = build_crew_graph(crew, spec.name)
        recorder.emit_run_event("run_started", status="running")
        try:
            final_state = graph.invoke(
                {
                    "case": case,
                    "condition": selected,
                    "autonomy": selected_autonomy,
                    "seed": int(seed),
                },
                config={"recursion_limit": 64},
            )
        except Exception:
            recorder.emit_run_event("run_failed", status="failed")
            # LangGraph/provider exceptions may embed state in their string form.
            # Expose a payload-free error at the public SDK boundary.
            raise RuntimeError("TraceGuard crew execution failed") from None

        trace = recorder.trace
        # Each architecture has its own canonical plan; the invariant is checked
        # against that architecture's plan rather than against one global tuple.
        if shield.fixed_structure and trace.step_types != spec.canonical_step_types:
            raise RuntimeError(
                f"canonical execution invariant failed for architecture {spec.name!r}"
            )
        if selected is Condition.FULL_PAD:
            expected_deadline = self.settings.step_deadline_s
            expected_bytes = self.settings.egress_ceiling_bytes
            if any(
                step.duration_s != expected_deadline or step.egress_bytes != expected_bytes
                for step in trace.steps
            ):
                raise RuntimeError("full-pad observable invariant failed")

        answer = str(final_state.get("answer", ""))
        violations = [item.to_dict() for item in recorder.violations]
        attestation_binding = self.attestation_binding()
        with self._receipt_lock:
            sequence = self.ledger.next_sequence
            previous = self.ledger.previous_hash
            body = build_receipt_body(
                run_id=identifier,
                case_id=case.case_id,
                condition=selected,
                trace=trace,
                dataset_hash=self.corpus.dataset_hash,
                provider_provenance=self.provider.provenance,
                policy_id=self.settings.policy_id,
                canonical_research_hops=self.settings.canonical_research_hops,
                step_deadline_s=self.settings.step_deadline_s,
                egress_ceiling_bytes=self.settings.egress_ceiling_bytes,
                fail_closed=recorder.fail_closed,
                violations=violations,
                ledger_sequence=sequence,
                previous_hash=previous,
                attestation=attestation_binding,
                # A canonicalized release commits to the plan it ran; an
                # adaptive one commits to no step sequence, and
                # build_receipt_body drops the digest for it.
                plan_digest=spec.plan.digest,
            )
            receipt = self.signer.sign(body)
            self.ledger.append(receipt)

        research_hops = int(final_state.get("research_hops", 0))
        metrics = RunMetrics(
            step_count=len(trace.steps),
            research_hops=research_hops,
            observable_duration_s=trace.total_duration_s,
            total_egress_bytes=trace.total_egress_bytes,
            fail_closed=recorder.fail_closed,
            overrun_count=len(violations),
        )
        runtime = {
            "graph_engine": "langgraph",
            "architecture": spec.name,
            "crew_agents": [node.id for node in spec.nodes],
            "canonical_plan": spec.plan.plan_id,
            "canonical_plan_digest": spec.plan.digest,
            "canonical_step_types": list(spec.canonical_step_types),
            "canonical_research_hops": self.settings.canonical_research_hops,
            "structure_enforced": shield.fixed_structure,
            "timing_and_size_enforced": shield.full_padding,
            "fail_closed": recorder.fail_closed,
            "violations": violations,
            "paper_evidence": False,
            "evidence_scope": (
                "synthetic_fixture_not_paper_evidence"
                if self.provider.provenance == "synthetic_fixture"
                else "fresh_run_not_original_paper_record"
            ),
            "implemented": {
                "application_trace_runtime": True,
                "ed25519_receipts": True,
                "hash_chained_ledger": True,
            },
            "not_implemented": [
                "trusted_execution_environment",
                "oram_or_oblivious_retrieval",
                "output_differential_privacy",
                "provider_transport_padding",
                *(
                    []
                    if attestation_binding
                    and attestation_binding.get("hardware_backed")
                    and attestation_binding.get("maa_verified")
                    else ["hardware_attestation"]
                ),
            ],
            "attestation": attestation_binding,
        }
        result = RunResult(
            run_id=identifier,
            case_id=case.case_id,
            condition=selected,
            answer=answer,
            trace=trace,
            receipt=receipt,
            metrics=metrics,
            provider_provenance=self.provider.provenance,
            dataset_hash=self.corpus.dataset_hash,
            runtime=runtime,
        )
        recorder.emit_run_event(
            "run_completed",
            status="fail_closed" if recorder.fail_closed else "completed",
        )
        return result

"""FastAPI console for metadata-only TraceGuard research runs."""

from __future__ import annotations

import inspect
import json
import os
import platform
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from importlib import metadata, resources
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import Settings
from .console import build_console_router
from .corpus import CorpusCatalog, default_case_count
from .experiment import (
    SDK_CONDITION,
    ExperimentConfig,
    ExperimentRunner,
    catalog_cases,
)
from .storage import ArtifactStore, metadata_event, metadata_trace, redact_secrets, utc_now

TERMINAL_STATUSES = {"completed", "completed_with_failures", "failed", "cancelled"}
PUBLIC_CONDITIONS = {"adaptive", "structure", "full"}
CONDITION_ALIASES = {
    "adaptive": "adaptive",
    "structure": "structure",
    "structure_only": "structure",
    "full": "full",
    "full_pad": "full",
}


def _version() -> str:
    try:
        return metadata.version("traceguard-research")
    except metadata.PackageNotFoundError:
        return "0.1.0"


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        result = value.to_dict()
        return dict(result) if isinstance(result, Mapping) else {}
    if hasattr(value, "model_dump"):
        result = value.model_dump(mode="json")
        return dict(result) if isinstance(result, Mapping) else {}
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _safe_result(value: Any) -> dict[str, Any]:
    """Public result view: research metadata, no answer or provider payload."""

    raw = _mapping(value)
    allowed = {
        "run_id",
        "case_id",
        "condition",
        # Arm coordinates: public experimental factors, never payload.
        "autonomy",
        "architecture",
        "observability",
        "trace",
        "receipt",
        "receipts",
        "metrics",
        "provider_provenance",
        "dataset_hash",
        "runtime",
        "status",
        "evidence",
        "planned_runs",
        "executed_runs",
        "resumed_runs",
        "failed_runs",
        "design",
        "experiment_id",
    }
    return redact_secrets({key: value for key, value in raw.items() if key in allowed})


def _result_trace(value: Any) -> dict[str, Any]:
    raw = _mapping(value)
    return metadata_trace(raw.get("trace", {"steps": []}))


def _result_receipts(value: Any) -> list[dict[str, Any]]:
    raw = _mapping(value)
    receipts = raw.get("receipts")
    if receipts is None and raw.get("receipt") is not None:
        receipts = [raw["receipt"]]
    if not isinstance(receipts, list):
        receipts = list(receipts) if isinstance(receipts, tuple) else []
    return [redact_secrets(_mapping(receipt)) for receipt in receipts]


class RunRequest(BaseModel):
    case_id: str | None = None
    provider: Literal["fixture", "azure"] | None = None
    condition: Literal["adaptive", "structure", "full", "structure_only", "full_pad"] = "adaptive"
    seed: int = 0
    # Crew topology. Defaults to the released pipeline so an existing client
    # that never sends the field keeps its exact behaviour.
    architecture: Literal[
        "pipeline_crew", "hierarchical_supervisor", "react_single_agent"
    ] = "pipeline_crew"
    kind: Literal["single", "fixture_experiment"] = "single"
    repetitions: int = Field(default=1, ge=1, le=3)
    # Ceiling read from the packaged corpus, not written down: the corpus
    # grew from 24 to 48 cases when sensitivity became a four-rung ladder.
    case_limit: int = Field(default=4, ge=1, le=default_case_count())
    conditions: list[Literal["adaptive", "structure", "full"]] | None = None


class VerifyReceiptRequest(BaseModel):
    receipt: dict[str, Any]
    public_key: str | None = None


class AttestationVerifyRequest(BaseModel):
    include_token: bool = True


class _RunCancelled(RuntimeError):
    pass


@dataclass
class _ManagedRun:
    run_id: str
    kind: str
    case_id: str | None
    provider: str
    condition: str
    seed: int
    architecture: str = "pipeline_crew"
    status: str = "queued"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    events: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    traces: list[dict[str, Any]] = field(default_factory=list)
    receipts: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    cancel_requested: threading.Event = field(default_factory=threading.Event)
    changed: threading.Condition = field(default_factory=threading.Condition)
    future: Future[Any] | None = None

    def public(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "kind": self.kind,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "provenance": redact_secrets(self.provenance),
            "result": self.result,
            "error": self.error,
        }


class RunManager:
    """Small in-process background manager used by the local research console."""

    def __init__(
        self,
        *,
        settings: Settings,
        catalog: CorpusCatalog,
        store: ArtifactStore,
        sdk_factory: Callable[..., Any] | None = None,
        max_workers: int = 2,
    ) -> None:
        self.settings = settings
        self.catalog = catalog
        self.store = store
        self._uses_default_factory = sdk_factory is None
        self.sdk_factory = sdk_factory or self._default_sdk_factory
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="traceguard"
        )
        self._runs: dict[str, _ManagedRun] = {}
        self._sdk_cache: dict[str, Any] = {}
        self._lock = threading.RLock()

    def _default_sdk_factory(self, settings: Settings, catalog: CorpusCatalog) -> Any:
        from .sdk import TraceGuardSDK

        return TraceGuardSDK(
            settings=settings,
            catalog=catalog,
            artifact_dir=self.store.root.parent / "runtime",
        )

    def _sdk(self, settings: Settings) -> Any:
        cache_key = json.dumps(settings.to_public_dict(), sort_keys=True, separators=(",", ":"))
        if self._uses_default_factory:
            with self._lock:
                if cache_key not in self._sdk_cache:
                    self._sdk_cache[cache_key] = self.sdk_factory(settings, self.catalog)
                return self._sdk_cache[cache_key]
        signature = inspect.signature(self.sdk_factory)
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and parameter.default is inspect.Parameter.empty
        ]
        if len(positional) >= 2:
            return self.sdk_factory(settings, self.catalog)
        if len(positional) == 1:
            return self.sdk_factory(settings)
        return self.sdk_factory()

    def submit(self, request: RunRequest) -> _ManagedRun:
        provider = request.provider or self.settings.provider
        condition = CONDITION_ALIASES[request.condition]
        if provider == "azure" and not (
            self.settings.azure_openai_api_key and self.settings.azure_openai_endpoint
        ):
            raise ValueError("Azure OpenAI provider is not configured")
        if request.kind == "single" and not request.case_id:
            raise ValueError("case_id is required for a single run")
        if request.kind == "single":
            self.catalog.get_case(str(request.case_id))
        if request.kind == "fixture_experiment" and provider != "fixture":
            raise ValueError("fixture_experiment requires provider='fixture'")
        run_id = ArtifactStore.new_run_id("run" if request.kind == "single" else "fixture-exp")
        managed = _ManagedRun(
            run_id=run_id,
            kind=request.kind,
            case_id=request.case_id,
            provider=provider,
            condition=condition,
            seed=request.seed,
            architecture=request.architecture,
            provenance={
                "provider": provider,
                "condition": condition,
                "architecture": request.architecture,
                "case_id": request.case_id,
                "dataset_hash": self.catalog.dataset_hash,
                "synthetic_dataset": True,
                "paper_reproduction_claim": False,
                "models": {
                    "fast": self.settings.model_fast,
                    "deep": self.settings.model_deep,
                    "review": self.settings.model_review,
                },
            },
        )
        with self._lock:
            self._runs[run_id] = managed
        self._publish(managed, "queued", data={"step_type": "queued"})
        if request.kind == "single":
            managed.future = self._executor.submit(self._execute_single, managed)
        else:
            managed.future = self._executor.submit(
                self._execute_fixture_experiment, managed, request
            )
        return managed

    @staticmethod
    def _sse_event(managed: _ManagedRun, event_type: str, value: Any = None) -> dict[str, Any]:
        raw = metadata_event(value or {})
        duration_ms = raw.get("duration_ms")
        if duration_ms is None and isinstance(raw.get("duration_s"), (int, float)):
            duration_ms = round(float(raw["duration_s"]) * 1000.0, 3)
        data = {
            "step_type": raw.get("step_type", raw.get("tool", event_type)),
            "duration_ms": duration_ms,
            "egress_bytes": raw.get("egress_bytes"),
            "hop": raw.get("hop", raw.get("index")),
        }
        return {
            "type": "run",
            "event_type": event_type,
            "run_id": managed.run_id,
            "timestamp": raw.get("timestamp", utc_now()),
            "data": {key: item for key, item in data.items() if item is not None},
        }

    def _publish(
        self, managed: _ManagedRun, event_type: str, *, value: Any = None, data: Any = None
    ) -> dict[str, Any]:
        event = self._sse_event(managed, event_type, value if value is not None else data)
        with managed.changed:
            managed.events.append(event)
            managed.updated_at = utc_now()
            managed.changed.notify_all()
        return event

    def _settings_for(self, provider: str) -> Settings:
        if provider == self.settings.provider:
            return self.settings
        return replace(self.settings, provider=provider)

    def _execute_single(self, managed: _ManagedRun) -> None:
        writer = None
        try:
            if managed.cancel_requested.is_set():
                raise _RunCancelled()
            managed.status = "running"
            self._publish(managed, "started", data={"step_type": "started"})
            settings = self._settings_for(managed.provider)
            sdk = self._sdk(settings)
            writer = self.store.create_run(
                run_id=managed.run_id,
                case_id=managed.case_id,
                condition=managed.condition,
                seed=managed.seed,
                provenance=managed.provenance,
                config=settings.to_public_dict(),
            )

            def on_event(value: Any) -> None:
                if managed.cancel_requested.is_set():
                    raise _RunCancelled()
                clean = writer.append_event(value)
                self._publish(managed, clean.get("event_type", "step"), value=clean)

            result = sdk.run(
                str(managed.case_id),
                SDK_CONDITION[managed.condition],
                run_id=managed.run_id,
                event_callback=on_event,
                seed=managed.seed,
                architecture=managed.architecture,
            )
            if managed.cancel_requested.is_set():
                raise _RunCancelled()
            trace = _result_trace(result)
            case = next(
                case for case in catalog_cases(self.catalog) if case["case_id"] == managed.case_id
            )
            trace.update(
                {
                    "run_id": managed.run_id,
                    "case_id": managed.case_id,
                    "condition": managed.condition,
                    "architecture": managed.architecture,
                    "service": case["specialty"],
                    "topic": case["topic"],
                    "attribute_label": case["attribute_label"],
                    "membership_label": case["membership_label"],
                }
            )
            writer.append_trace(trace)
            receipts = _result_receipts(result)
            for receipt in receipts:
                writer.append_receipt(receipt)
            safe = _safe_result(result)
            writer.write_result(safe)
            stored = writer.finalize(
                status="completed",
                extra={"trace_count": 1, "receipt_count": len(receipts)},
            )
            managed.traces = [trace]
            managed.receipts = receipts
            managed.metrics = dict(safe.get("metrics", {}))
            managed.result = safe
            managed.provenance["manifest_sha256"] = (
                (stored.path / "manifest.sha256").read_text(encoding="utf-8").strip()
            )
            managed.status = "completed"
            self._publish(managed, "complete", data={"step_type": "complete"})
        except _RunCancelled:
            managed.status = "cancelled"
            managed.error = None
            if writer is not None:
                try:
                    writer.write_result({"status": "cancelled"})
                    writer.finalize(status="cancelled")
                except Exception as storage_error:
                    managed.provenance["storage_finalize_error"] = type(storage_error).__name__
            self._publish(managed, "cancelled", data={"step_type": "cancelled"})
        except Exception as error:
            managed.status = "failed"
            managed.error = {
                "type": type(error).__name__,
                "message": "Run failed without exposing provider or case payloads.",
            }
            if writer is not None:
                try:
                    writer.abort(f"{type(error).__name__}: metadata-only failure")
                except Exception as storage_error:
                    managed.provenance["storage_finalize_error"] = type(storage_error).__name__
            self._publish(managed, "error", data={"step_type": type(error).__name__})
        finally:
            managed.updated_at = utc_now()

    def _execute_fixture_experiment(self, managed: _ManagedRun, request: RunRequest) -> None:
        try:
            managed.status = "running"
            self._publish(managed, "started", data={"step_type": "experiment_started"})
            # Keep the browser's deliberately small fixture experiment fast.  Its
            # outputs are labeled diagnostic-only and never compared as evidence.
            settings = replace(self._settings_for("fixture"), step_deadline_s=0.01)
            sdk = self._sdk(settings)
            runner = ExperimentRunner(sdk, self.catalog, self.store)

            def callback(event: dict[str, Any]) -> None:
                if managed.cancel_requested.is_set():
                    return
                event_type = str(event.get("event_type", "experiment"))
                self._publish(managed, event_type, value=event.get("data", event))

            summary = runner.run(
                ExperimentConfig(
                    repetitions=request.repetitions,
                    conditions=tuple(request.conditions or ("adaptive", "structure", "full")),
                    seed=request.seed,
                    experiment_id=managed.run_id,
                    journal_path=self.store.root.parent / "runtime" / f"{managed.run_id}.jsonl",
                    require_balanced=False,
                    provider="fixture",
                    case_limit=request.case_limit,
                    provenance=managed.provenance,
                ),
                event_callback=callback,
                cancel_check=managed.cancel_requested.is_set,
            )
            managed.status = str(summary["status"])
            managed.result = _safe_result(summary)
            for row in summary.get("records", []):
                if row.get("trace"):
                    managed.traces.append(row["trace"])
                run_id = row.get("run_id")
                if run_id and self.store.exists(run_id):
                    managed.receipts.extend(self.store.read_jsonl(run_id, "receipts.jsonl"))
            self._publish(
                managed,
                "complete" if managed.status != "cancelled" else "cancelled",
                data={"step_type": managed.status},
            )
        except Exception as error:
            managed.status = "failed"
            managed.error = {
                "type": type(error).__name__,
                "message": "Fixture experiment failed; no payload was retained.",
            }
            self._publish(managed, "error", data={"step_type": type(error).__name__})
        finally:
            managed.updated_at = utc_now()

    def get(self, run_id: str) -> _ManagedRun:
        with self._lock:
            try:
                return self._runs[run_id]
            except KeyError as error:
                raise KeyError(f"unknown run_id {run_id!r}") from error

    def list(self) -> list[_ManagedRun]:
        with self._lock:
            return sorted(self._runs.values(), key=lambda run: run.created_at, reverse=True)

    def cancel(self, run_id: str) -> _ManagedRun:
        managed = self.get(run_id)
        if managed.status not in TERMINAL_STATUSES:
            managed.cancel_requested.set()
            if managed.future and managed.future.cancel():
                managed.status = "cancelled"
                self._publish(managed, "cancelled", data={"step_type": "cancelled"})
            else:
                self._publish(managed, "cancel_requested", data={"step_type": "cancel_requested"})
        return managed

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def _read_paper_results() -> dict[str, Any]:
    resource = resources.files("traceguard").joinpath("data/paper-reported-unverified.json")
    data = json.loads(resource.read_text(encoding="utf-8"))
    return {
        **data,
        "status": "reported_unverified",
        "reproduced": False,
        "scientific_evidence": False,
        "display_label": "Paper-reported, unverified reference values",
    }


def _verify_receipt(receipt: dict[str, Any], public_key: str | None = None) -> dict[str, Any]:
    """Bridge supported verifier call shapes without weakening invalid results."""

    try:
        from . import verifier
    except ImportError as error:
        return {
            "valid": False,
            "checks": {},
            "error": f"verifier unavailable: {type(error).__name__}",
        }
    function = getattr(verifier, "verify_receipt", None) or getattr(verifier, "verify", None)
    if callable(function):
        try:
            signature = inspect.signature(function)
            if public_key and "pinned_public_key" in signature.parameters:
                value = function(receipt, pinned_public_key=public_key)
            elif public_key and "public_key" in signature.parameters:
                value = function(receipt, public_key=public_key)
            else:
                value = function(receipt)
            result = _mapping(value)
            if not result and isinstance(value, bool):
                result = {"valid": value}
            result.setdefault("valid", bool(result.get("ok", False)))
            return redact_secrets(result)
        except Exception as error:
            return {
                "valid": False,
                "checks": {},
                "error": f"receipt rejected: {type(error).__name__}",
            }
    verifier_class = getattr(verifier, "ReceiptVerifier", None)
    if verifier_class:
        try:
            instance = verifier_class(public_key) if public_key else verifier_class()
            return redact_secrets(_mapping(instance.verify(receipt)))
        except Exception as error:
            return {
                "valid": False,
                "checks": {},
                "error": f"receipt rejected: {type(error).__name__}",
            }
    return {"valid": False, "checks": {}, "error": "verifier entry point unavailable"}


# Honest mirror of receipt.build_receipt_body's implementation_scope (receipt.py).
# The values MUST stay identical; tests/test_api.py::test_runtime_report_matches_receipt_scope
# asserts this against a freshly built receipt so the mirror can never silently drift.
RUNTIME_IMPLEMENTATION_SCOPE: dict[str, bool] = {
    "trace_runtime": True,
    "ed25519_receipt": True,
    "hash_chained_ledger": True,
    "tee": False,
    "oram_or_oblivious_retrieval": False,
    "output_differential_privacy": False,
    "provider_transport_padding": False,
    "hardware_attestation": False,
}

# Read-only probes for hardware-TEE guest interfaces. Presence alone never flips
# quote_available; only a genuine MAA-verified attestation does (see attestation.py).
# On Azure CVMs the paravisor hides /dev/sev-guest, so the vTPM devices are the
# real signal; the SEV/TDX/SGX nodes are kept for non-Azure transparency.
_TEE_MARKERS = (
    "/dev/tpmrm0",
    "/dev/tpm0",
    "/dev/sev-guest",
    "/dev/tdx_guest",
    "/dev/sgx_enclave",
)

# Cache the attestation verdict briefly so /api/runtime polling does not re-mint a
# token (or re-hit MAA on Azure) on every request. POST /api/attestation/verify
# forces a fresh evidence collection.
_ATTESTATION_CACHE_TTL_S = 30.0
_attestation_cache: dict[str, Any] = {"result": None, "at": 0.0}
_attestation_lock = threading.Lock()


def _current_attestation(*, force: bool = False) -> Any:
    """Return a cached-or-fresh AttestationResult (never raises)."""

    import time

    from . import attestation as attestation_module

    with _attestation_lock:
        now = time.monotonic()
        cached = _attestation_cache["result"]
        if not force and cached is not None and (now - _attestation_cache["at"]) < (
            _ATTESTATION_CACHE_TTL_S
        ):
            return cached
        try:
            result = attestation_module.attest()
        except Exception as error:  # noqa: BLE001 - attestation must never 500 the API
            result = attestation_module.AttestationResult(
                mode="unavailable",
                tee_type="none",
                hardware_backed=False,
                maa_verified=False,
                verdict="unavailable",
                summary=f"Attestation could not run: {type(error).__name__}.",
                checked_at=utc_now(),
                errors=[type(error).__name__],
            )
        _attestation_cache["result"] = result
        _attestation_cache["at"] = now
        return result


def _guest_kernel() -> dict[str, str]:
    """Report the running kernel (the load-bearing micro-VM evidence).

    Deliberately excludes the hostname/nodename so no host/infra identity leaks.
    """

    report = {"release": platform.release(), "source": "platform"}
    try:
        with open("/proc/version", encoding="utf-8") as handle:
            report["version_line"] = handle.readline().strip()
            report["source"] = "proc_version"
    except OSError:
        report["version_line"] = ""
    return report


def _runtime_report() -> dict[str, Any]:
    """Honest, read-only self-report of the deployment isolation posture.

    Never fabricates a hardware-TEE quote. The isolation *kind* is asserted by the
    deploy environment (TRACEGUARD_ISOLATION); VM-level claims are only made when a
    recognized Kata kind is declared, so the endpoint cannot overclaim.
    """

    kind = (os.getenv("TRACEGUARD_ISOLATION") or "").strip()
    declared = bool(kind)
    is_kata = "kata" in kind.lower()
    markers = [path for path in _TEE_MARKERS if os.path.exists(path)]

    attestation = _current_attestation()
    hardware_verified = bool(attestation.hardware_backed and attestation.maa_verified)
    is_cvm = hardware_verified or "confidential" in kind.lower() or "cvm" in kind.lower()

    scope = dict(RUNTIME_IMPLEMENTATION_SCOPE)
    # The one implementation claim the artifact can genuinely earn at runtime.
    scope["hardware_attestation"] = hardware_verified

    report = {
        "schema_version": "traceguard.runtime.v1",
        "isolation": {
            "kind": kind or "unknown",
            "declared_by": "deploy_env" if declared else "unset",
            "class": (
                "confidential_vm"
                if is_cvm
                else "micro_vm"
                if is_kata
                else ("container" if declared else "unknown")
            ),
            "vm_level_isolation": is_kata or hardware_verified,
            "separate_guest_kernel": is_kata or hardware_verified,
        },
        "kernel": _guest_kernel(),
        "hardware_tee": {
            # quote_available flips True ONLY for a genuine MAA-verified quote.
            "quote_available": hardware_verified,
            "attestation_type": attestation.tee_type if hardware_verified else "none",
            "memory_encryption": hardware_verified,
            "vendor_technology": attestation.tee_type if hardware_verified else "none",
            "detected_markers": markers,
        },
        "attestation": {
            # kind stays "none" unless a hardware quote verified; mode/verdict add
            # transparency (e.g. "simulated") without ever overclaiming.
            "kind": attestation.tee_type if hardware_verified else "none",
            "mode": attestation.mode,
            "verdict": attestation.verdict,
            "maa_verified": attestation.maa_verified,
            "provider": attestation.provider,
            "issuer": attestation.issuer,
            "checked_at": attestation.checked_at,
            "summary": attestation.summary,
            "note": (
                "Hardware-rooted remote attestation verified by Microsoft Azure Attestation."
                if hardware_verified
                else "No hardware-rooted remote attestation on this host; verdict is "
                f"'{attestation.verdict}'. Deploy on an Azure Confidential VM for a "
                "hardware-backed result (see deploy/azure)."
            ),
        },
        "receipt_signer": {
            "trust_root": (
                "azure_maa_attested_tcb" if hardware_verified else "demo_key_not_hardware_attested"
            )
        },
        "implementation_scope": scope,
        "disclaimer": (
            "Read-only honest self-report. The isolation kind is asserted by the deploy "
            "environment; TEE and attestation fields reflect a live attestation check and "
            "are never fabricated — quote_available and hardware_attestation flip True only "
            "for a genuine Microsoft Azure Attestation-verified quote."
        ),
    }
    return redact_secrets(report)


def create_app(
    *,
    settings: Settings | None = None,
    catalog: CorpusCatalog | None = None,
    store: ArtifactStore | None = None,
    sdk_factory: Callable[..., Any] | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    catalog = catalog or (
        CorpusCatalog.load(settings.dataset_path)
        if settings.dataset_path
        else CorpusCatalog.load_default()
    )
    store = store or ArtifactStore(os.getenv("TRACEGUARD_ARTIFACT_DIR", "artifacts/runs"))
    manager = RunManager(
        settings=settings,
        catalog=catalog,
        store=store,
        sdk_factory=sdk_factory,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        manager.shutdown()

    app = FastAPI(
        title="TraceGuard Research Console",
        version=_version(),
        description="Public, synthetic, metadata-only reproducibility console",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.catalog = catalog
    app.state.store = store
    app.state.run_manager = manager

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": _version(),
            "provider": settings.provider,
            "azure_openai_configured": bool(
                settings.azure_openai_api_key and settings.azure_openai_endpoint
            ),
            "llm_configured": bool(
                settings.azure_openai_api_key and settings.azure_openai_endpoint
            ),
        }

    @app.get("/api/runtime")
    def runtime() -> dict[str, Any]:
        return _runtime_report()

    @app.get("/api/attestation")
    def attestation() -> dict[str, Any]:
        """Current remote-attestation verdict (cached briefly).

        The payload is built entirely by the attestation module from public
        attestation data (an MAA token is a signed artifact meant to be shared
        with a relying party), so it is returned without the payload redactor,
        which would otherwise strip the ``token`` field.
        """

        return _current_attestation().to_dict(include_token=True)

    @app.post("/api/attestation/verify")
    def attestation_verify(request: AttestationVerifyRequest) -> dict[str, Any]:
        """Trigger a fresh attestation and independently re-verify its token.

        This is the live "Verify" action a reviewer can trigger: it collects fresh
        evidence, submits it to Microsoft Azure Attestation (on a real CVM), and
        then re-runs the RS256 + issuer verification against the provider JWKS so
        the verdict is reproduced, not merely reported.
        """

        from . import attestation as attestation_module

        result = _current_attestation(force=True)
        payload = result.to_dict(include_token=request.include_token)
        # Independent re-verification of the returned token against a JWKS.
        reverify: dict[str, Any] = {"attempted": False}
        if result.token:
            try:
                if result.mode == "simulated":
                    jwks = attestation_module._simulated_jwks(
                        attestation_module._get_simulated_key()
                    )
                    expected = attestation_module._SIMULATED_ISSUER
                else:
                    provider = (result.provider or attestation_module.DEFAULT_MAA_ENDPOINT).rstrip(
                        "/"
                    )
                    jwks = attestation_module.fetch_jwks(provider)
                    expected = provider if provider.startswith("http") else None
                verdict = attestation_module.verify_maa_token(
                    result.token, jwks, expected_issuer=expected
                )
                reverify = {
                    "attempted": True,
                    "signature_and_issuer_valid": verdict.get("valid"),
                    "hardware_rooted": verdict.get("hardware_rooted"),
                    "reasons": verdict.get("reasons", []),
                }
            except Exception as error:  # noqa: BLE001 - report, never 500
                reverify = {"attempted": True, "error": type(error).__name__}
        payload["reverification"] = reverify
        return payload

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        return {
            **settings.to_public_dict(),
            "supported_conditions": sorted(PUBLIC_CONDITIONS),
            "dataset_hash": catalog.dataset_hash,
            "synthetic_dataset": True,
            "paper_results_status": "reported_unverified",
        }

    @app.get("/api/cases")
    def cases() -> dict[str, Any]:
        return {
            "dataset": {
                "version": catalog.schema_version,
                "hash": catalog.dataset_hash,
                "synthetic": True,
            },
            "cases": catalog_cases(catalog),
        }

    @app.get("/api/paper/results")
    def paper_results() -> dict[str, Any]:
        return _read_paper_results()

    @app.get("/api/paper/pdf")
    def paper_pdf() -> FileResponse:
        candidates = [
            Path(os.getenv("TRACEGUARD_PAPER_PDF", "")),
            Path(__file__).resolve().parents[2] / "usenixsecurity2026.pdf",
            Path("/app/usenixsecurity2026.pdf"),
        ]
        for candidate in candidates:
            if candidate and candidate.is_file():
                return FileResponse(
                    candidate,
                    media_type="application/pdf",
                    filename="usenixsecurity2026.pdf",
                )
        raise HTTPException(status_code=404, detail="paper PDF not bundled in this deployment")

    @app.get("/api/runs")
    def list_runs() -> dict[str, Any]:
        return {"runs": [managed.public() for managed in manager.list()]}

    @app.post("/api/runs", status_code=202)
    def start_run(request: RunRequest) -> dict[str, Any]:
        try:
            managed = manager.submit(request)
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {"run_id": managed.run_id, "status": managed.status}

    def get_managed(run_id: str) -> _ManagedRun:
        try:
            return manager.get(run_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="run not found") from error

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str) -> dict[str, Any]:
        return get_managed(run_id).public()

    @app.get("/api/runs/{run_id}/events")
    def run_events(run_id: str) -> StreamingResponse:
        managed = get_managed(run_id)

        def stream() -> Any:
            index = 0
            while True:
                with managed.changed:
                    if index >= len(managed.events) and managed.status not in TERMINAL_STATUSES:
                        managed.changed.wait(timeout=15.0)
                    pending = managed.events[index:]
                    index = len(managed.events)
                    terminal = managed.status in TERMINAL_STATUSES
                for event in pending:
                    yield f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
                if terminal and index >= len(managed.events):
                    break
                if not pending:
                    yield ": keep-alive\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/runs/{run_id}/traces")
    def run_traces(run_id: str) -> dict[str, Any]:
        managed = get_managed(run_id)
        traces = managed.traces
        if not traces and store.exists(run_id):
            traces = store.read_jsonl(run_id, "traces.jsonl")
        return {"run_id": run_id, "traces": traces}

    @app.get("/api/runs/{run_id}/metrics")
    def run_metrics(run_id: str) -> dict[str, Any]:
        managed = get_managed(run_id)
        return {"run_id": run_id, "metrics": managed.metrics}

    @app.get("/api/runs/{run_id}/receipts")
    def run_receipts(run_id: str) -> dict[str, Any]:
        managed = get_managed(run_id)
        receipts = managed.receipts
        if not receipts and store.exists(run_id):
            receipts = store.read_jsonl(run_id, "receipts.jsonl")
        return {"run_id": run_id, "receipts": receipts}

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str) -> dict[str, Any]:
        try:
            managed = manager.cancel(run_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="run not found") from error
        return {"run_id": run_id, "status": managed.status, "cancel_requested": True}

    @app.post("/api/receipts/verify")
    def verify_receipt(request: VerifyReceiptRequest) -> dict[str, Any]:
        return _verify_receipt(request.receipt, request.public_key)

    app.include_router(build_console_router())

    web_override = os.getenv("TRACEGUARD_WEB_DIR")
    web = Path(web_override) if web_override else Path(__file__).with_name("web")
    if web.is_dir():
        app.mount("/", _SPAStaticFiles(directory=web, html=True), name="console")

    return app


class _SPAStaticFiles(StaticFiles):
    """Serve the built single-page console, falling back to index.html.

    Client-side routes such as /playground or /runs/abc have no file on disk;
    the SPA router resolves them after index.html loads.  API paths keep their
    real 404 so the JSON error contract is unchanged.
    """

    async def get_response(self, path: str, scope: Any) -> Any:
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as error:
            if error.status_code == 404 and not path.startswith("api"):
                return await super().get_response("index.html", scope)
            raise


app: FastAPI | None
try:
    # ASGI convenience.  Import remains usable in an unconfigured live-provider
    # environment because fixture is the Settings default.
    app = create_app()
except (FileNotFoundError, ImportError, RuntimeError, ValueError):
    # ``traceguard serve`` constructs an app again and surfaces an actionable error.
    app = None

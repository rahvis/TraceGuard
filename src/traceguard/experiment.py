"""Balanced TraceGuard experiment orchestration.

The runner expands the task design -- (specialty, topic) group x sensitivity level
x membership, read from the catalog rather than hardcoded -- executes each defense
condition separately, journals every completed cell for resume, and commits every run
through :mod:`traceguard.storage`.  It never labels fixture output as scientific
evidence or silently equates a new-model run with the paper's historical results.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .attack import analyze_records
from .storage import (
    ArtifactStore,
    canonical_json,
    metadata_event,
    metadata_trace,
    redact_secrets,
    utc_now,
)
from .types import SENSITIVITY_FRAMINGS

CONDITIONS = ("adaptive", "structure", "full")
SDK_CONDITION = {"adaptive": "adaptive", "structure": "structure_only", "full": "full_pad"}

# Autonomy levels, mirroring types.Autonomy. Kept as strings here because a
# journal must stay readable without importing the package.
AUTONOMY_LEVELS = ("scripted", "bounded", "free")

# The autonomy each condition implied before the two axes were separated. Using
# it as the default is what makes an archived run reproduce byte-for-byte.
DEFAULT_AUTONOMY_FOR_CONDITION = {
    "adaptive": "bounded",
    "structure": "scripted",
    "full": "scripted",
}

# Crew topologies. Every architecture carries its own canonical plan.
#
# Kept as literal strings here for the same reason the condition and autonomy
# vocabularies are: a journal has to stay readable without importing the
# package. The single source of truth is traceguard.graph.ARCHITECTURES, and
# _assert_architectures_agree below makes a drift between the two loud rather
# than silently producing arms the SDK cannot run.
ARCHITECTURES = ("pipeline_crew", "hierarchical_supervisor", "react_single_agent")


def _assert_architectures_agree() -> None:
    """Fail loudly if the journal vocabulary and the crew registry diverge."""

    try:
        from .graph import ARCHITECTURES as GRAPH_ARCHITECTURES
    except Exception:  # noqa: BLE001 - langgraph absent: analysis-only use is fine
        return
    if set(ARCHITECTURES) != set(GRAPH_ARCHITECTURES):  # pragma: no cover - import guard
        raise RuntimeError(
            "experiment.ARCHITECTURES and graph.ARCHITECTURES disagree; an arm "
            "could name a topology the SDK cannot build"
        )


_assert_architectures_agree()

# Two vocabularies exist for the same three conditions: the experiment's short
# names and the SDK/Condition enum's long ones. Normalizing in one place stops
# them drifting into a KeyError at the boundary between them.
_CONDITION_ALIASES = {sdk: short for short, sdk in SDK_CONDITION.items()}


def normalize_condition(value: str) -> str:
    """Accept either the experiment ('structure') or SDK ('structure_only') name."""

    text = str(value).strip().lower()
    if text in CONDITIONS:
        return text
    if text in _CONDITION_ALIASES:
        return _CONDITION_ALIASES[text]
    raise ValueError(
        f"unknown condition {value!r}; expected one of "
        f"{', '.join((*CONDITIONS, *_CONDITION_ALIASES))}"
    )
_ID_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


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


def normalize_case(value: Any) -> dict[str, Any]:
    raw = _mapping(value)
    case_id = raw.get("case_id", raw.get("id"))
    if case_id is None:
        raise ValueError("catalog case is missing case_id")
    specialty = raw.get("specialty", raw.get("service"))
    if specialty is None or raw.get("topic") is None:
        raise ValueError(f"case {case_id!r} is missing specialty/service or topic")
    sensitive = raw.get("sensitive", raw.get("attribute_label", False))
    canary = raw.get("canary_member", raw.get("membership_label", raw.get("member", False)))
    # The ordinal grade has to survive normalization. This function is the only
    # path from the catalog into the runner, and it previously dropped every
    # field it did not name -- so a graded corpus would arrive at the experiment
    # as a binary one, and the per-level design would silently collapse into the
    # old 2-level one with duplicate cells.
    level = raw.get("sensitivity_level")
    if level is None:
        # v1 vocabulary: the boolean named the fully-gapped framing, i.e. the
        # top rung, not the second one.
        level = (len(SENSITIVITY_FRAMINGS) - 1) if bool(sensitive) else 0
    level = int(level)
    return {
        "id": str(case_id),
        "case_id": str(case_id),
        "specialty": str(specialty),
        "service": str(specialty),
        "topic": str(raw["topic"]),
        "sensitivity_level": level,
        "framing": str(raw.get("framing") or SENSITIVITY_FRAMINGS[level]),
        # The binary attribute label is retained: the attack evaluator scores a
        # binary AUC, and every archived journal is keyed on it.
        "sensitive": level > 0,
        "attribute_label": int(level > 0),
        "canary_member": bool(canary),
        "membership_label": int(bool(canary)),
        # The task family. This function drops every field it does not name, so
        # without this line the family label never reaches the journal and two
        # task families would be indistinguishable in the analysis -- the same
        # class of silent drop that once collapsed the graded ladder to binary.
        "domain": str(raw.get("domain") or "prior-authorization"),
    }


def catalog_cases(catalog: Any) -> list[dict[str, Any]]:
    """Discover public case metadata across simple and SDK catalog interfaces."""

    values: Any = None
    for method_name in ("list_cases", "cases", "all_cases"):
        member = getattr(catalog, method_name, None)
        if callable(member):
            values = member()
            break
        if member is not None:
            values = member
            break
    if values is None and isinstance(catalog, Mapping):
        values = catalog.get("cases", catalog)
    if values is None and isinstance(catalog, Iterable) and not isinstance(catalog, (str, bytes)):
        values = catalog
    if isinstance(values, Mapping):
        values = values["cases"] if "cases" in values else list(values.values())
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
        raise TypeError("CorpusCatalog must expose list_cases(), cases, or be iterable")
    normalized = [normalize_case(value) for value in values]
    return sorted(
        normalized,
        key=lambda row: (
            row["specialty"],
            row["topic"],
            # Ordinal, so a truncated case_limit walks the ladder in order
            # instead of walking framing names alphabetically.
            row["sensitivity_level"],
            row["canary_member"],
            row["case_id"],
        ),
    )


def validate_balanced_design(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Check the task design is a balanced full factorial, from the data alone.

    The design is (specialty, topic) group x sensitivity level x membership. Its
    size is derived from the cases in hand rather than asserted, for two
    reasons. The obvious one is that the corpus grew a four-rung sensitivity
    ladder and the hardcoded 24 became wrong. The less obvious one is that the
    old expected-count formula was wrong even for the corpus it shipped with:
    it read ``len(services) * 2 * 2 * 2``, using the count of *specialties*
    where the design's first factor is the count of (specialty, topic) groups.
    It only ever agreed with reality because 3 specialties x 8 happened to equal
    6 topics x 4; a corpus with an uneven number of topics per specialty would
    have made it silently accept an unbalanced catalog.
    """

    services = sorted({str(case["specialty"]) for case in cases})
    topics = {(str(case["specialty"]), str(case["topic"])) for case in cases}
    levels = sorted({int(case["sensitivity_level"]) for case in cases})
    memberships = sorted({int(bool(case["canary_member"])) for case in cases})
    cells = {
        (
            str(case["specialty"]),
            str(case["topic"]),
            int(case["sensitivity_level"]),
            int(bool(case["canary_member"])),
        )
        for case in cases
    }
    expected_cells = len(topics) * len(levels) * len(memberships)
    balanced = len(cells) == expected_cells and len(cases) == len(cells)
    return {
        "balanced": balanced,
        "services": services,
        "service_count": len(services),
        "service_topic_groups": len(topics),
        "sensitivity_levels": levels,
        "membership_arms": memberships,
        "task_cells": len(cells),
        "expected_task_cells": expected_cells,
        "duplicates": len(cases) - len(cells),
        "design": (
            f"{len(topics)} x {len(levels)} x {len(memberships)}"
        ),
    }


@dataclass(frozen=True, slots=True)
class Arm:
    """One experimental configuration, orthogonal to the case being run.

    The experiment used to be indexed by (condition, case, repetition) alone.
    Adding factors without extending that index would be a silent-corruption
    bug rather than a missing feature: two different arms would hash to the same
    cell key, and ``resume`` would treat an already-completed cell of arm A as
    satisfying arm B, quietly merging distinct experiments into one journal.
    Every axis therefore lives here, and every axis enters the cell key and the
    per-cell seed.
    """

    condition: str = "adaptive"
    autonomy: str = "bounded"
    architecture: str = "pipeline_crew"
    # Per-role deployment names. A mapping rather than a single id because the
    # crew assigns different roles to different models.
    models: tuple[tuple[str, str], ...] = ()
    # What the adversary is allowed to observe; see attack.OBSERVABILITY.
    observability: str = "full"

    def __post_init__(self) -> None:
        object.__setattr__(self, "condition", normalize_condition(self.condition))
        if self.autonomy not in AUTONOMY_LEVELS:
            raise ValueError(f"unknown autonomy level {self.autonomy!r}")
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"unknown architecture {self.architecture!r}")
        # A canonicalized release requires a data-independent plan. Catching the
        # contradiction here names the arm; catching it in the SDK would surface
        # mid-run as a failed cell.
        if self.condition != "adaptive" and self.autonomy != "scripted":
            raise ValueError(
                f"condition {self.condition!r} canonicalizes the plan and requires "
                f"autonomy 'scripted', not {self.autonomy!r}"
            )

    @property
    def slug(self) -> str:
        """Stable, human-readable arm identifier used in ids and cell keys."""

        model_part = "-".join(f"{role}:{name}" for role, name in self.models) or "default"
        return "|".join(
            (self.condition, self.autonomy, self.architecture, self.observability, model_part)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "autonomy": self.autonomy,
            "architecture": self.architecture,
            "observability": self.observability,
            "models": dict(self.models),
        }


@dataclass(frozen=True)
class ExperimentConfig:
    repetitions: int = 1
    conditions: tuple[str, ...] = CONDITIONS
    # Explicit arms. When empty the runner derives one arm per condition at that
    # condition's historical autonomy, which reproduces every archived run
    # exactly; new sweeps pass arms directly.
    arms: tuple[Arm, ...] = ()
    seed: int = 0
    experiment_id: str | None = None
    journal_path: Path | None = None
    resume: bool = True
    require_balanced: bool = True
    provider: str = "fixture"
    case_limit: int | None = None
    workers: int = 1
    provenance: dict[str, Any] = field(default_factory=dict)
    # When True, run() refuses to execute a single cell unless the process
    # holds a hardware-bound attestation. This turns "spend 90 minutes and
    # real money, then discover the vTPM was not visible" into a two-second
    # refusal, and makes attestation a precondition of the arm rather than an
    # observation about it.
    require_attestation: bool = False

    def resolved_arms(self) -> tuple[Arm, ...]:
        """The arms to run: explicit if given, else one per condition."""

        if self.arms:
            return self.arms
        return tuple(
            Arm(condition=condition, autonomy=DEFAULT_AUTONOMY_FOR_CONDITION[condition])
            for condition in self.conditions
        )

    def __post_init__(self) -> None:
        if self.repetitions < 1:
            raise ValueError("repetitions must be positive")
        invalid = set(self.conditions) - set(CONDITIONS)
        if invalid:
            raise ValueError(f"unknown conditions: {', '.join(sorted(invalid))}")
        slugs = [arm.slug for arm in self.arms]
        if len(slugs) != len(set(slugs)):
            raise ValueError("duplicate arms would collide in the journal cell key")
        if self.case_limit is not None and self.case_limit < 1:
            raise ValueError("case_limit must be positive")
        if self.workers < 1:
            raise ValueError("workers must be positive")


class ExperimentRunner:
    def __init__(self, sdk: Any, catalog: Any, store: ArtifactStore) -> None:
        self.sdk = sdk
        self.catalog = catalog
        self.store = store
        self._journal_lock = threading.Lock()

    @staticmethod
    def _experiment_id(config: ExperimentConfig) -> str:
        if config.experiment_id:
            return _ID_SAFE.sub("-", config.experiment_id)[:64]
        return ArtifactStore.new_run_id("experiment")

    @staticmethod
    def _cell_key(arm: Arm, case_id: str, repetition: int, seed: int) -> str:
        # The arm slug carries every experimental axis. Omitting any of them
        # would let two distinct arms share a cell key, and resume would then
        # accept a completed cell of one arm as satisfying the other.
        return f"{arm.slug}|{case_id}|{repetition}|{seed}"

    @staticmethod
    def _cell_seed(base: int, arm: Arm, case_id: str, repetition: int) -> int:
        value = f"{base}|{arm.slug}|{case_id}|{repetition}".encode()
        return int.from_bytes(hashlib.sha256(value).digest()[:4], "big")

    @staticmethod
    def _load_completed(journal: Path) -> dict[str, dict[str, Any]]:
        completed: dict[str, dict[str, Any]] = {}
        if not journal.is_file():
            return completed
        with journal.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    # Only a final partial line (e.g. interrupted fsync) is recoverable.
                    if line_number and not line.endswith("\n"):
                        break
                    raise ValueError(f"invalid experiment journal line {line_number}") from error
                if row.get("status") == "completed" and row.get("cell_key"):
                    completed[str(row["cell_key"])] = row
        return completed

    def _append_journal(self, journal: Path, row: Mapping[str, Any]) -> None:
        journal.parent.mkdir(parents=True, exist_ok=True)
        encoded = (
            json.dumps(redact_secrets(dict(row)), sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        with self._journal_lock, journal.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _extract_trace(result: Any, events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        raw = _mapping(result)
        trace = raw.get("trace")
        if trace is None and isinstance(raw.get("traces"), Sequence) and raw["traces"]:
            trace = raw["traces"][0]
        if trace is None:
            trace = {"steps": list(events)}
        return metadata_trace(trace)

    @staticmethod
    def _extract_receipts(result: Any) -> list[dict[str, Any]]:
        raw = _mapping(result)
        receipts = raw.get("receipts")
        if receipts is None and raw.get("receipt") is not None:
            receipts = [raw["receipt"]]
        if not isinstance(receipts, Sequence) or isinstance(receipts, (str, bytes)):
            return []
        return [redact_secrets(_mapping(receipt)) for receipt in receipts]

    def run(
        self,
        config: ExperimentConfig,
        *,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        experiment_id = self._experiment_id(config)
        journal = config.journal_path or self.store.root.parent / f"{experiment_id}.jsonl"
        if config.require_attestation:
            # Evaluate before the first cell, not after the last.
            gate = attestation_provenance(self.sdk)
            if not gate.get("hardware_bound_receipt"):
                raise RuntimeError(
                    "--require-attestation was set but this process holds no "
                    "hardware-bound attestation: "
                    f"verdict={gate.get('verdict')!r} "
                    f"hardware_backed={gate.get('hardware_backed')} "
                    f"maa_verified={gate.get('maa_verified')} "
                    f"key_binding={'present' if gate.get('key_binding') else 'absent'}. "
                    "Refusing to spend a run that could not support the claim."
                )
            self._write_attestation_sidecar(journal, experiment_id)
        cases = catalog_cases(self.catalog)
        design = validate_balanced_design(cases)
        if config.require_balanced and not design["balanced"]:
            raise ValueError(
                "catalog is not a balanced full factorial over (specialty, topic) x "
                f"sensitivity level x membership: {design}"
            )
        if config.case_limit:
            cases = cases[: config.case_limit]
        completed = self._load_completed(journal) if config.resume else {}
        arms = config.resolved_arms()
        planned = len(cases) * len(arms) * config.repetitions
        state = {"executed": 0, "resumed": 0, "failed": 0}
        state_lock = threading.Lock()
        emit_lock = threading.Lock()
        cancelled = threading.Event()
        records: list[dict[str, Any]] = []

        def emit(event_type: str, **data: Any) -> None:
            if event_callback:
                with emit_lock:
                    event_callback(
                        {
                            "type": "experiment",
                            "event_type": event_type,
                            "timestamp": utc_now(),
                            "data": redact_secrets(data),
                        }
                    )

        emit("experiment_started", experiment_id=experiment_id, planned_runs=planned)
        pending: list[tuple[Arm, dict[str, Any], int, int, str]] = []
        for arm in arms:
            for case in cases:
                for repetition in range(config.repetitions):
                    cell_seed = self._cell_seed(config.seed, arm, case["case_id"], repetition)
                    key = self._cell_key(arm, case["case_id"], repetition, cell_seed)
                    if key in completed:
                        state["resumed"] += 1
                        records.append(completed[key])
                        emit("run_resumed", cell_key=key, run_id=completed[key].get("run_id"))
                        continue
                    pending.append((arm, case, repetition, cell_seed, key))

        def execute_cell(
            arm: Arm,
            case: dict[str, Any],
            repetition: int,
            cell_seed: int,
            key: str,
        ) -> None:
            condition = arm.condition
            if cancelled.is_set():
                return
            if cancel_check and cancel_check():
                cancelled.set()
                return
            arm_tag = _ID_SAFE.sub("-", arm.slug)
            # The run id must stay unique per cell, and truncating a readable
            # id cannot be relied on to preserve that: the model arms differ
            # only in a suffix, so `[:120]` cut off the case id and repetition
            # and collapsed all 96 cells of an arm onto one id -- 180 of 288
            # cells in the model spoke died on the resulting bundle collision.
            # A digest of the full identity is appended after truncation, so
            # readability is best-effort and uniqueness is not.
            identity = f"{experiment_id}|{arm.slug}|{case['case_id']}|{repetition}"
            digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
            readable = _ID_SAFE.sub(
                "-", f"{experiment_id}-{arm_tag}-{case['case_id']}-r{repetition}"
            )[:104].rstrip("-")
            run_id_base = f"{readable}-{digest}"
            events: list[dict[str, Any]] = []
            # A crashed or failed earlier attempt leaves its immutable bundle dir
            # behind; retried cells must not collide with it, so suffix reattempts.
            writer = None
            run_id = run_id_base
            for attempt in range(4):
                run_id = run_id_base if attempt == 0 else f"{run_id_base}-a{attempt}"
                try:
                    writer = self.store.create_run(
                        run_id=run_id,
                        case_id=case["case_id"],
                        condition=condition,
                        seed=cell_seed,
                        provenance={
                            **config.provenance,
                            "experiment_id": experiment_id,
                            "provider": config.provider,
                            "synthetic_dataset": True,
                            "new_replication_not_exact_paper_reproduction": True,
                        },
                        config={"repetition": repetition},
                    )
                    break
                except FileExistsError:
                    continue
            if writer is None:
                with state_lock:
                    state["failed"] += 1
                self._append_journal(
                    journal,
                    {
                        "cell_key": key,
                        "status": "failed",
                        "experiment_id": experiment_id,
                        "run_id": run_id,
                        "case": case,
                        "condition": condition,
                        "repetition": repetition,
                        "seed": cell_seed,
                        "error": {
                            "type": "FileExistsError",
                            "message": "Run bundle collision persisted across retries.",
                        },
                        "artifact_finalize_error": None,
                    },
                )
                emit("run_failed", run_id=run_id, error_type="FileExistsError")
                return

            def on_sdk_event(
                value: Any,
                event_rows: list[dict[str, Any]] = events,
                run_writer: Any = writer,
                current_run_id: str = run_id,
            ) -> None:
                clean = metadata_event(value)
                event_rows.append(clean)
                run_writer.append_event(clean)
                emit("run_event", run_id=current_run_id, **clean)

            # An arm that names crew models must actually be running them. The
            # SDK holds one Settings for the whole runner, so a caller that
            # sweeps models has to build one runner per arm; nothing enforced
            # that, and the crew-model spoke silently ran the ambient crew for
            # all three arms -- 288 paid cells that measured the same
            # configuration three times while their slugs claimed otherwise.
            if arm.models:
                actual = {
                    "fast": self.sdk.settings.model_fast,
                    "deep": self.sdk.settings.model_deep,
                    "review": self.sdk.settings.model_review,
                }
                expected = dict(arm.models)
                mismatch = {
                    role: (expected[role], actual.get(role))
                    for role in expected
                    if expected[role] != actual.get(role)
                }
                if mismatch:
                    raise RuntimeError(
                        "arm names crew models this runner is not configured for "
                        f"({mismatch}); build one runner per model arm with "
                        "dataclasses.replace(settings, model_fast=..., ...)"
                    )

            emit(
                "run_started",
                run_id=run_id,
                case_id=case["case_id"],
                condition=condition,
            )
            try:
                result = self.sdk.run(
                    case["case_id"],
                    SDK_CONDITION[condition],
                    run_id=run_id,
                    event_callback=on_sdk_event,
                    seed=cell_seed,
                    autonomy=arm.autonomy,
                    architecture=arm.architecture,
                )
                result_dict = redact_secrets(_mapping(result))
                trace = self._extract_trace(result, events)
                trace.update(
                    {
                        "run_id": run_id,
                        "case_id": case["case_id"],
                        "condition": condition,
                        "autonomy": arm.autonomy,
                        "architecture": arm.architecture,
                        "observability": arm.observability,
                        "service": case["specialty"],
                        "topic": case["topic"],
                        "attribute_label": case["attribute_label"],
                        # The ordinal label. attribute_label stays as the
                        # thresholded binary so existing analysis keeps working.
                        "sensitivity_level": case.get("sensitivity_level", case["attribute_label"]),
                        "membership_label": case["membership_label"],
                    }
                )
                writer.append_trace(trace)
                receipts = self._extract_receipts(result)
                for receipt in receipts:
                    writer.append_receipt(receipt)
                writer.write_result(result_dict)
                stored = writer.finalize(
                    status="completed",
                    extra={"event_count": len(events), "receipt_count": len(receipts)},
                )
                # Whether the release fell back to the fail-closed value is the
                # single event the padded arm's whole utility cost is mediated
                # by, and until now it reached only the per-run receipt. Reading
                # `row.get("violations")` from a journal therefore returned a
                # missing key rather than a rate, so an analysis over the journal
                # alone silently measured 0% -- which is exactly the mistake this
                # field exists to prevent. The receipt remains authoritative; this
                # is the same fact carried where the analysis reads it.
                fail_closed = bool(getattr(result, "fail_closed", False))
                violations = [
                    dict(v) if isinstance(v, Mapping) else _mapping(v)
                    for v in (getattr(result, "violations", None) or [])
                ]
                row = {
                    "cell_key": key,
                    "status": "completed",
                    "fail_closed": fail_closed,
                    "violations": violations,
                    "experiment_id": experiment_id,
                    "run_id": run_id,
                    "case": case,
                    "condition": condition,
                    "repetition": repetition,
                    "seed": cell_seed,
                    "trace": trace,
                    "manifest_sha256": (stored.path / "manifest.sha256")
                    .read_text(encoding="utf-8")
                    .strip(),
                    "provenance": {
                        "provider": config.provider,
                        "synthetic": True,
                        "paper_reproduction_claim": False,
                        "models": {
                            "fast": self.sdk.settings.model_fast,
                            "deep": self.sdk.settings.model_deep,
                            "review": self.sdk.settings.model_review,
                        },
                        # Names the commit whose image produced this row.
                        "source_revision": os.getenv(
                            "TRACEGUARD_SOURCE_REVISION", "unknown"
                        ),
                        # Where this row executed, carried with the row so the
                        # claim is checkable from the journal alone.
                        "attestation": attestation_provenance(self.sdk),
                    },
                }
                self._append_journal(journal, row)
                with state_lock:
                    records.append(row)
                    state["executed"] += 1
                    done = state["executed"] + state["resumed"]
                emit("run_completed", run_id=run_id, completed=done)
            except Exception as error:
                with state_lock:
                    state["failed"] += 1
                try:
                    writer.abort(f"{type(error).__name__}: metadata-only failure")
                except Exception as storage_error:
                    failure_storage_error = type(storage_error).__name__
                else:
                    failure_storage_error = None
                failure = {
                    "cell_key": key,
                    "status": "failed",
                    "experiment_id": experiment_id,
                    "run_id": run_id,
                    "case": case,
                    "condition": condition,
                    "repetition": repetition,
                    "seed": cell_seed,
                    "error": {
                        "type": type(error).__name__,
                        "message": (
                            "Run failed without retaining provider or case payloads."
                        ),
                    },
                    "artifact_finalize_error": failure_storage_error,
                }
                self._append_journal(journal, failure)
                emit("run_failed", run_id=run_id, error_type=type(error).__name__)

        # Runs are I/O-bound on the provider API, so a thread pool multiplies
        # throughput without perturbing per-run step timings: each run's steps
        # stay strictly sequential inside its own worker.
        if config.workers == 1:
            for cell in pending:
                if cancelled.is_set():
                    break
                execute_cell(*cell)
        else:
            with ThreadPoolExecutor(max_workers=config.workers) as pool:
                futures = [pool.submit(execute_cell, *cell) for cell in pending]
                for future in futures:
                    future.result()

        executed = state["executed"]
        resumed = state["resumed"]
        failed = state["failed"]
        if cancelled.is_set():
            emit(
                "experiment_cancelled",
                completed=executed + resumed,
                planned_runs=planned,
            )
            return {
                "experiment_id": experiment_id,
                "status": "cancelled",
                "planned_runs": planned,
                "executed_runs": executed,
                "resumed_runs": resumed,
                "failed_runs": failed,
                "journal": str(journal),
                "design": design,
                "evidence": self._evidence(config.provider),
            }
        status = "completed" if failed == 0 else "completed_with_failures"
        emit("experiment_completed", status=status, completed=executed + resumed, failed=failed)
        return {
            "experiment_id": experiment_id,
            "status": status,
            "planned_runs": planned,
            "executed_runs": executed,
            "resumed_runs": resumed,
            "failed_runs": failed,
            "journal": str(journal),
            "design": design,
            "records": records,
            "evidence": self._evidence(config.provider),
        }

    def _write_attestation_sidecar(self, journal: Path, experiment_id: str) -> None:
        """Write the full attestation evidence beside the journal.

        The MAA token cannot travel in the journal: ``redact_secrets`` drops any
        key named ``token``, so routing it through the journal writer would
        silently strip exactly the evidence an offline verifier needs. It is
        written here directly instead. Each row references its epoch by
        ``evidence_sha256``, so a reader can tell which evidence covers which
        row rather than trusting a file-level association.

        The JWKS snapshot is archived alongside because MAA rotates signing
        keys: without it, a token outlives the ability to verify it.
        """

        evidence = self.sdk.attestation_evidence()
        if evidence is None:
            return
        jwks = None
        try:
            from .attestation import fetch_jwks

            issuer = evidence.get("issuer")
            if issuer:
                jwks = fetch_jwks(issuer)
        except Exception:  # noqa: BLE001 - a missing snapshot is not fatal
            jwks = None
        path = journal.with_name(journal.stem + ".attestation.json")
        path.parent.mkdir(parents=True, exist_ok=True)

        # Accumulate rather than overwrite. One journal is routinely built by
        # several invocations -- one per condition, one per repetition ladder --
        # and each process mints its own nonce and therefore its own epoch. An
        # overwriting writer keeps only the last, so every row written by an
        # earlier invocation references an epoch that is no longer archived and
        # a reviewer cannot resolve its evidence digest. Merge on the epoch's
        # own canonical digest, which is exactly the key the rows reference.
        existing: dict[str, Any] = {}
        if path.is_file():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                existing = {}
        epochs: list[Any] = list(existing.get("epochs") or [])
        seen = {hashlib.sha256(canonical_json(e)).hexdigest() for e in epochs}
        if hashlib.sha256(canonical_json(evidence)).hexdigest() not in seen:
            epochs.append(evidence)

        # Keep every JWKS snapshot ever fetched, keyed by kid, so a token whose
        # signing key has since rotated is still verifiable.
        merged_jwks = existing.get("maa_jwks_snapshot") or {}
        if jwks:
            by_kid = {
                k.get("kid"): k for k in (merged_jwks.get("keys") or []) if k.get("kid")
            }
            for key in jwks.get("keys") or []:
                if key.get("kid"):
                    by_kid[key["kid"]] = key
            merged_jwks = {"keys": list(by_kid.values())}

        payload = {
            "schema_version": "traceguard.experiment.attestation.v1",
            "experiment_id": experiment_id,
            "journal": journal.name,
            "signing_public_key_b64": self.sdk.signer.public_key,
            "maa_jwks_snapshot": merged_jwks or None,
            "epochs": epochs,
        }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _evidence(provider: str) -> dict[str, Any]:
        fixture = provider in {"fixture", "synthetic-fixture"}
        return {
            "tier": "fixture_diagnostic" if fixture else "fresh_replication",
            "scientific_evidence": not fixture,
            "paper_reproduction_claim": False,
            "note": (
                "Fixture runs validate software behavior and are not scientific evidence."
                if fixture
                else (
                    "A fresh run may be compared with the paper but is not the "
                    "missing original run."
                )
            ),
        }


def attestation_provenance(sdk: Any) -> dict[str, Any]:
    """Digest-only attestation provenance for one journal row.

    A journal row is the unit the analysis and the paper generator consume, so
    a claim about *where* a row executed has to travel with the row. This
    projects the SDK's cached binding to digests and identifiers only -- never
    the MAA token, which goes to the sidecar -- and reuses
    ``attestation_is_hardware_bound``, the same predicate that gates the signed
    receipt flag, so the journal and the receipt can never disagree.

    Off Azure the identical code path emits the honest counterpart
    (``verdict: simulated``, ``hardware_backed: false``, ``key_binding: null``).
    That two journals produced by one image differ in exactly this field is the
    point: it is the honesty invariant exercised rather than asserted.
    """

    from .receipt import attestation_is_hardware_bound

    try:
        binding = sdk.attestation_binding()
    except Exception:  # noqa: BLE001 - provenance must never break a run
        binding = None
    if not binding:
        return {"present": False, "hardware_bound_receipt": False}

    key_binding = binding.get("key_binding")
    kb: dict[str, Any] | None = None
    if isinstance(key_binding, Mapping):
        kb = {
            "scheme": key_binding.get("scheme"),
            "epoch": key_binding.get("epoch", 0),
            "nonce": key_binding.get("nonce"),
            "signing_key_sha256": key_binding.get("signing_key_sha256"),
            "qualifying_digest": key_binding.get("qualifying_digest"),
            "report_data": key_binding.get("report_data"),
            "sha256": hashlib.sha256(canonical_json(dict(key_binding))).hexdigest(),
        }

    evidence_sha256 = None
    try:
        evidence = sdk.attestation_evidence()
        if evidence is not None:
            evidence_sha256 = hashlib.sha256(canonical_json(evidence)).hexdigest()
    except Exception:  # noqa: BLE001
        evidence_sha256 = None

    return {
        "present": True,
        "verdict": binding.get("verdict"),
        "mode": binding.get("mode"),
        "hardware_backed": bool(binding.get("hardware_backed")),
        "maa_verified": bool(binding.get("maa_verified")),
        # The same predicate that gates the signed receipt's
        # implementation_scope.hardware_attestation flag.
        "hardware_bound_receipt": bool(attestation_is_hardware_bound(binding)),
        "tee_type": binding.get("tee_type"),
        "issuer": binding.get("issuer"),
        "provider": binding.get("provider"),
        "token_sha256": binding.get("token_sha256"),
        "measurement_sha256": binding.get("measurement_sha256"),
        "launch_measurement": binding.get("launch_measurement"),
        "key_binding": kb,
        "evidence_sha256": evidence_sha256,
    }


def load_experiment_records(
    path: str | Path, *, completed_only: bool = True
) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at line {line_number}") from error
            if not completed_only or row.get("status") == "completed":
                rows.append(row)
    return rows


def analyze_experiment(
    path: str | Path, *, condition: str = "adaptive", seed: int = 0, n_bootstrap: int = 1000
) -> dict[str, Any]:
    rows = [row for row in load_experiment_records(path) if row.get("condition") == condition]
    records = []
    for row in rows:
        case = row.get("case", {})
        trace = row.get("trace", {})
        records.append(
            {
                "trace": trace,
                "service": case.get("specialty", case.get("service")),
                "topic": case.get("topic"),
                "attribute_label": case.get("attribute_label", int(bool(case.get("sensitive")))),
                # Carried through so a downstream ordinal analysis can slice by
                # rung. A journal written before the ladder existed has no such
                # field, and its binary label is the top rung by definition.
                "sensitivity_level": case.get(
                    "sensitivity_level",
                    (len(SENSITIVITY_FRAMINGS) - 1) if case.get("sensitive") else 0,
                ),
                "membership_label": case.get(
                    "membership_label", int(bool(case.get("canary_member")))
                ),
                "provenance": row.get("provenance", {}),
            }
        )
    if not records:
        raise ValueError(f"no completed records for condition {condition!r}")
    analysis = analyze_records(records, seed=seed, n_bootstrap=n_bootstrap)
    return {
        "schema_version": "traceguard.analysis.v1",
        "created_at": utc_now(),
        "source": str(Path(path)),
        "condition": condition,
        "record_count": len(records),
        "paper_reproduction_claim": False,
        "analysis": analysis,
    }


def run_experiment(
    sdk: Any,
    catalog: Any,
    store: ArtifactStore,
    *,
    repetitions: int = 1,
    conditions: tuple[str, ...] = CONDITIONS,
    seed: int = 0,
    provider: str = "fixture",
    journal_path: str | Path | None = None,
) -> dict[str, Any]:
    """Convenience wrapper used by notebooks and the CLI."""

    config = ExperimentConfig(
        repetitions=repetitions,
        conditions=conditions,
        seed=seed,
        provider=provider,
        journal_path=Path(journal_path) if journal_path else None,
    )
    return ExperimentRunner(sdk, catalog, store).run(config)

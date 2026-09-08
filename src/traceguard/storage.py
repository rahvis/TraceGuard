"""Immutable, content-addressed storage for TraceGuard runs.

The store is deliberately boring: a run is a directory containing newline-delimited
metadata, a result, a manifest, and a checksum list.  Payload-bearing fields are
discarded at the boundary.  A ``FINALIZED`` marker makes the directory immutable to
this API; callers should treat the marker as the commit record for a run.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import secrets
import sys
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "traceguard.run.v1"
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_FORBIDDEN_KEYS = {
    "answer",
    "api_key",
    "authorization",
    "completion",
    "content",
    "document",
    "documents",
    "input",
    "messages",
    "output",
    "password",
    "payload",
    "prompt",
    "query",
    "response",
    "secret",
    "text",
    "token",
}
_EVENT_FIELDS = {
    "agent",
    "duration_ms",
    "duration_s",
    "egress_bytes",
    "event_type",
    # The response direction. Note the name: `_PAYLOAD_KEYS` strips anything
    # containing "response", so calling this `response_bytes` would have made it
    # vanish here silently -- which is exactly what happened to the first
    # attempt, producing a journal of nulls after an hour of live runs.
    "ingress_bytes",
    "hop",
    "index",
    "kind",
    "name",
    "run_id",
    "sequence",
    "status",
    "step_type",
    "timestamp",
    "tool",
    "type",
    "wall_time_s",
}
_TRACE_FIELDS = {
    "agent",
    "duration_ms",
    "duration_s",
    "egress_bytes",
    "event_type",
    # The response direction. Note the name: `_PAYLOAD_KEYS` strips anything
    # containing "response", so calling this `response_bytes` would have made it
    # vanish here silently -- which is exactly what happened to the first
    # attempt, producing a journal of nulls after an hour of live runs.
    "ingress_bytes",
    "hop",
    "index",
    "kind",
    "name",
    "run_id",
    "sequence",
    "status",
    "step_type",
    "timestamp",
    "tool",
    "type",
    "wall_time_s",
}


def utc_now() -> str:
    """Return a stable UTC timestamp suitable for manifests and events."""

    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def canonical_json(value: Any) -> bytes:
    """Serialize JSON deterministically for hashing."""

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def redact_secrets(value: Any) -> Any:
    """Recursively remove obvious secret and payload fields from public metadata."""

    value = _jsonable(value)
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            normalized = key.lower().replace("-", "_")
            if normalized in _FORBIDDEN_KEYS or normalized.endswith("_api_key"):
                continue
            clean[key] = redact_secrets(item)
        return clean
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value


def metadata_event(event: Any) -> dict[str, Any]:
    """Reduce an event to the paper's host-observable metadata surface."""

    raw = _jsonable(event)
    if not isinstance(raw, dict):
        raw = {"event_type": str(raw)}
    if isinstance(raw.get("data"), dict):
        raw = {**raw, **raw["data"]}
    clean = {key: redact_secrets(value) for key, value in raw.items() if key in _EVENT_FIELDS}
    clean.setdefault("event_type", clean.get("type", clean.get("kind", "step")))
    clean.setdefault("timestamp", utc_now())
    return clean


def metadata_trace(trace: Any) -> dict[str, Any]:
    """Normalize a trace record without retaining any content-bearing fields."""

    raw = _jsonable(trace)
    if not isinstance(raw, dict):
        raise TypeError("a trace must be a mapping or expose to_dict()")
    result: dict[str, Any] = {}
    for key in (
        "run_id",
        "case_id",
        "condition",
        # The experimental axes. Each is a public arm coordinate, not payload:
        # omitting any of them here silently drops it between the runner and
        # the journal, and the analysis then cannot slice by that factor at all.
        "autonomy",
        "architecture",
        "observability",
        "seed",
        "service",
        "specialty",
        "topic",
    ):
        if key in raw:
            result[key] = redact_secrets(raw[key])
    for key in (
        "sensitive",
        "attribute_label",
        # Ordinal sensitivity grade. A label, not a payload field, so it belongs
        # in this passthrough rather than in _TRACE_FIELDS.
        "sensitivity_level",
        "canary_member",
        "membership_label",
    ):
        if key in raw:
            result[key] = raw[key]
    steps = raw.get("steps", raw.get("events", raw.get("trace", [])))
    if isinstance(steps, dict):
        steps = steps.get("steps", steps.get("events", []))
    if not isinstance(steps, list):
        steps = []
    result["steps"] = []
    for index, step in enumerate(steps):
        item = _jsonable(step)
        if not isinstance(item, dict):
            item = {"step_type": str(item)}
        if isinstance(item.get("data"), dict):
            item = {**item, **item["data"]}
        clean = {key: redact_secrets(value) for key, value in item.items() if key in _TRACE_FIELDS}
        clean.setdefault("sequence", index)
        clean.setdefault(
            "step_type", clean.get("tool", clean.get("name", clean.get("type", "step")))
        )
        result["steps"].append(clean)
    return result


def default_provenance() -> dict[str, Any]:
    """Return environment provenance that contains no credentials or user payloads."""

    return {
        "artifact_schema": SCHEMA_VERSION,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": Path(sys.executable).name,
        "source_revision": os.getenv("TRACEGUARD_SOURCE_REVISION", "unknown"),
    }


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@dataclass(frozen=True)
class StoredRun:
    run_id: str
    path: Path
    manifest: dict[str, Any]


class RunWriter:
    """Transaction-like writer for one run directory."""

    def __init__(self, store: ArtifactStore, run_id: str, manifest: Mapping[str, Any]) -> None:
        self.store = store
        self.run_id = run_id
        self.path = store.run_path(run_id)
        self._lock = threading.RLock()
        self._closed = False
        self._manifest_base = redact_secrets(dict(manifest))
        self.path.mkdir(parents=True, exist_ok=False)
        # Create every stream up front so completed runs always have the full layout.
        for name in ("events.jsonl", "traces.jsonl", "receipts.jsonl"):
            _atomic_write(self.path / name, b"")

    def _assert_open(self) -> None:
        if self._closed or (self.path / "FINALIZED").exists():
            raise RuntimeError(f"run {self.run_id!r} is immutable")

    def _append(self, filename: str, value: Any) -> None:
        with self._lock:
            self._assert_open()
            path = self.path / filename
            with path.open("ab") as handle:
                handle.write(canonical_json(value))
                handle.flush()
                os.fsync(handle.fileno())

    def append_event(self, event: Any) -> dict[str, Any]:
        clean = metadata_event(event)
        self._append("events.jsonl", clean)
        return clean

    def append_trace(self, trace: Any) -> dict[str, Any]:
        clean = metadata_trace(trace)
        self._append("traces.jsonl", clean)
        return clean

    def append_receipt(self, receipt: Any) -> dict[str, Any]:
        clean = redact_secrets(receipt)
        if not isinstance(clean, dict):
            raise TypeError("a receipt must be a mapping or expose to_dict()")
        self._append("receipts.jsonl", clean)
        return clean

    def write_result(self, result: Any) -> dict[str, Any]:
        with self._lock:
            self._assert_open()
            clean = redact_secrets(result)
            if not isinstance(clean, dict):
                clean = {"value": clean}
            _atomic_write(self.path / "result.json", canonical_json(clean))
            return clean

    def finalize(
        self, *, status: str = "completed", extra: Mapping[str, Any] | None = None
    ) -> StoredRun:
        """Commit a run and return its final manifest.

        Hashes cover every data artifact.  ``manifest.sha256`` separately commits the
        manifest, avoiding an impossible self-referential checksum.
        """

        with self._lock:
            self._assert_open()
            result_path = self.path / "result.json"
            if not result_path.exists():
                self.write_result({"status": status})
            artifact_names = ("events.jsonl", "traces.jsonl", "result.json", "receipts.jsonl")
            artifacts = {
                name: {
                    "sha256": sha256_file(self.path / name),
                    "bytes": (self.path / name).stat().st_size,
                }
                for name in artifact_names
            }
            manifest = {
                **self._manifest_base,
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "status": status,
                "finalized_at": utc_now(),
                "artifacts": artifacts,
            }
            if extra:
                manifest["summary"] = redact_secrets(dict(extra))
            manifest_bytes = canonical_json(manifest)
            _atomic_write(self.path / "manifest.json", manifest_bytes)
            manifest_hash = sha256_bytes(manifest_bytes)
            checksum_lines = [
                f"{item['sha256']}  {name}" for name, item in sorted(artifacts.items())
            ]
            checksum_lines.append(f"{manifest_hash}  manifest.json")
            _atomic_write(
                self.path / "checksums.sha256", ("\n".join(checksum_lines) + "\n").encode()
            )
            _atomic_write(self.path / "manifest.sha256", (manifest_hash + "\n").encode())
            _atomic_write(
                self.path / "FINALIZED", canonical_json({"manifest_sha256": manifest_hash})
            )
            self._closed = True
            return StoredRun(self.run_id, self.path, manifest)

    def abort(self, error: str) -> StoredRun:
        self.append_event({"event_type": "error", "status": "failed"})
        self.write_result({"status": "failed", "error": str(error)[:1000]})
        return self.finalize(status="failed")


class ArtifactStore:
    """Filesystem store with one immutable directory per run."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def new_run_id(prefix: str = "run") -> str:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"{prefix}-{stamp}-{secrets.token_hex(4)}"

    def run_path(self, run_id: str) -> Path:
        if not _RUN_ID_RE.fullmatch(run_id):
            raise ValueError("run_id must contain only letters, numbers, '.', '_' or '-'")
        path = (self.root / run_id).resolve()
        if path.parent != self.root:
            raise ValueError("run_id escapes artifact root")
        return path

    def create_run(
        self,
        *,
        run_id: str | None = None,
        case_id: str | None = None,
        condition: str | None = None,
        seed: int | None = None,
        provenance: Mapping[str, Any] | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> RunWriter:
        run_id = run_id or self.new_run_id()
        manifest = {
            "created_at": utc_now(),
            "case_id": case_id,
            "condition": condition,
            "seed": seed,
            "provenance": {**default_provenance(), **redact_secrets(dict(provenance or {}))},
            "config": redact_secrets(dict(config or {})),
        }
        return RunWriter(self, run_id, manifest)

    def exists(self, run_id: str) -> bool:
        return self.run_path(run_id).is_dir()

    def is_finalized(self, run_id: str) -> bool:
        return (self.run_path(run_id) / "FINALIZED").is_file()

    def manifest(self, run_id: str) -> dict[str, Any]:
        with (self.run_path(run_id) / "manifest.json").open(encoding="utf-8") as handle:
            return json.load(handle)

    def read_json(self, run_id: str, filename: str) -> dict[str, Any]:
        if filename not in {"manifest.json", "result.json", "FINALIZED"}:
            raise ValueError("unsupported JSON artifact")
        with (self.run_path(run_id) / filename).open(encoding="utf-8") as handle:
            return json.load(handle)

    def read_jsonl(self, run_id: str, filename: str) -> list[dict[str, Any]]:
        if filename not in {"events.jsonl", "traces.jsonl", "receipts.jsonl"}:
            raise ValueError("unsupported JSONL artifact")
        path = self.run_path(run_id) / filename
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def list_runs(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in sorted(self.root.iterdir(), reverse=True):
            manifest_path = path / "manifest.json"
            if path.is_dir() and manifest_path.is_file():
                try:
                    rows.append(json.loads(manifest_path.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError):
                    continue
        return rows

    def verify(self, run_id: str) -> dict[str, Any]:
        path = self.run_path(run_id)
        manifest = self.manifest(run_id)
        mismatches: list[str] = []
        for name, expected in manifest.get("artifacts", {}).items():
            artifact = path / name
            if not artifact.is_file() or sha256_file(artifact) != expected.get("sha256"):
                mismatches.append(name)
        manifest_expected = (path / "manifest.sha256").read_text(encoding="utf-8").strip()
        if sha256_file(path / "manifest.json") != manifest_expected:
            mismatches.append("manifest.json")
        return {
            "run_id": run_id,
            "valid": not mismatches and (path / "FINALIZED").is_file(),
            "mismatches": mismatches,
            "finalized": (path / "FINALIZED").is_file(),
        }

    def iter_traces(self, run_ids: Iterable[str] | None = None) -> Iterable[dict[str, Any]]:
        ids = list(run_ids) if run_ids is not None else [row["run_id"] for row in self.list_runs()]
        for run_id in ids:
            yield from self.read_jsonl(run_id, "traces.jsonl")

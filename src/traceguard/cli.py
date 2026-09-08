"""Command-line entry point for the TraceGuard reproducibility artifact."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from .api import RunManager, RunRequest, _verify_receipt, create_app
from .config import PROVIDERS, Settings, assert_judge_reserved
from .corpus import CORPORA, CorpusCatalog
from .experiment import (
    ARCHITECTURES,
    ExperimentConfig,
    ExperimentRunner,
    analyze_experiment,
    catalog_cases,
    validate_balanced_design,
)
from .storage import ArtifactStore, redact_secrets


def _json(value: Any) -> str:
    return json.dumps(redact_secrets(value), indent=2, sort_keys=True, ensure_ascii=False)


def _settings(provider: str | None = None) -> Settings:
    settings = Settings.from_env()
    return (
        replace(settings, provider=provider)
        if provider and provider != settings.provider
        else settings
    )


def _catalog(settings: Settings, domain: str | None = None) -> CorpusCatalog:
    if settings.dataset_path:
        return CorpusCatalog.load(settings.dataset_path)
    if domain:
        return CorpusCatalog.load_named(domain)
    return CorpusCatalog.load_default()


def _store(path: str | Path | None) -> ArtifactStore:
    return ArtifactStore(path or os.getenv("TRACEGUARD_ARTIFACT_DIR", "artifacts/runs"))


def _sdk(
    settings: Settings, catalog: CorpusCatalog, *, artifact_dir: str | Path | None = None
) -> Any:
    from .sdk import TraceGuardSDK

    return TraceGuardSDK(settings=settings, catalog=catalog, artifact_dir=artifact_dir)


def command_doctor(args: argparse.Namespace) -> int:
    checks: list[dict[str, Any]] = []
    try:
        settings = _settings(args.provider)
        checks.append({"name": "settings", "ok": True, "provider": settings.provider})
    except Exception as error:
        settings = None
        checks.append({"name": "settings", "ok": False, "error": type(error).__name__})
    if settings:
        try:
            catalog = _catalog(settings)
            # Checked against the design the catalog itself implies rather than
            # against a written-down case count: the corpus went from 24 to 48
            # cases when sensitivity became a four-rung ladder, and a literal
            # here would have reported a correct corpus as broken.
            design = validate_balanced_design(catalog_cases(catalog))
            checks.append(
                {
                    "name": "synthetic_dataset",
                    "ok": design["balanced"],
                    "case_count": len(catalog.list_cases()),
                    "design": design["design"],
                    "sensitivity_levels": design["sensitivity_levels"],
                    "dataset_hash": catalog.dataset_hash,
                }
            )
        except Exception as error:
            checks.append({"name": "synthetic_dataset", "ok": False, "error": type(error).__name__})
        if settings.provider == "azure":
            checks.append(
                {
                    "name": "azure_openai_configuration",
                    "ok": bool(
                        settings.azure_openai_api_key and settings.azure_openai_endpoint
                    ),
                    "credential_displayed": False,
                }
            )
        else:
            checks.append(
                {
                    "name": "fixture_provider",
                    "ok": True,
                    "note": "offline diagnostic only; not scientific evidence",
                }
            )
    try:
        store = _store(args.artifact_dir)
        probe = store.root / ".doctor-write-probe"
        probe.touch(exist_ok=False)
        probe.unlink()
        checks.append({"name": "artifact_store", "ok": True, "path": str(store.root)})
    except Exception as error:
        checks.append({"name": "artifact_store", "ok": False, "error": type(error).__name__})
    ok = all(check["ok"] for check in checks)
    print(_json({"status": "ok" if ok else "failed", "checks": checks}))
    return 0 if ok else 1


def command_serve(args: argparse.Namespace) -> int:
    import uvicorn

    settings = _settings(args.provider)
    catalog = _catalog(settings)
    app = create_app(settings=settings, catalog=catalog, store=_store(args.artifact_dir))
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def command_cases(args: argparse.Namespace) -> int:
    settings = _settings(args.provider)
    catalog = _catalog(settings)
    print(
        _json(
            {
                "dataset": {
                    "hash": catalog.dataset_hash,
                    "version": catalog.schema_version,
                    "synthetic": True,
                },
                "cases": catalog_cases(catalog),
            }
        )
    )
    return 0


def command_run(args: argparse.Namespace) -> int:
    settings = _settings(args.provider)
    catalog = _catalog(settings)
    store = _store(args.artifact_dir)
    manager = RunManager(settings=settings, catalog=catalog, store=store)
    try:
        managed = manager.submit(
            RunRequest(
                case_id=args.case_id,
                provider=settings.provider,
                condition=args.condition,
                seed=args.seed,
                architecture=args.architecture,
                kind="single",
            )
        )
        if managed.future:
            managed.future.result()
        print(_json(managed.public()))
        return 0 if managed.status == "completed" else 1
    finally:
        manager.shutdown()


def command_experiment(args: argparse.Namespace) -> int:
    settings = _settings(args.provider)
    assert_judge_reserved(settings)
    catalog = _catalog(settings, getattr(args, "domain", None))
    store = _store(args.artifact_dir)
    sdk = _sdk(settings, catalog, artifact_dir=store.root.parent / "runtime")
    conditions = tuple(args.conditions)
    summary = ExperimentRunner(sdk, catalog, store).run(
        ExperimentConfig(
            repetitions=args.repetitions,
            conditions=conditions,
            seed=args.seed,
            experiment_id=args.experiment_id,
            journal_path=Path(args.journal) if args.journal else None,
            resume=not args.no_resume,
            require_balanced=not args.allow_partial,
            provider=settings.provider,
            case_limit=args.case_limit,
            workers=args.workers,
            require_attestation=args.require_attestation,
            provenance={
                "models": {
                    "fast": settings.model_fast,
                    "deep": settings.model_deep,
                    "review": settings.model_review,
                }
            },
        )
    )
    # Avoid dumping every trace to the terminal; the journal is the machine-readable source.
    summary = {key: value for key, value in summary.items() if key != "records"}
    print(_json(summary))
    return 0 if summary["status"] == "completed" else 1


def command_analyze(args: argparse.Namespace) -> int:
    result = analyze_experiment(
        args.journal,
        condition=args.condition,
        seed=args.seed,
        n_bootstrap=args.bootstrap,
    )
    encoded = _json(result) + "\n"
    if args.output:
        output = Path(args.output)
        if output.exists() and not args.force:
            raise FileExistsError(f"output already exists: {output}; pass --force to replace it")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
        print(
            _json(
                {"status": "written", "path": str(output), "record_count": result["record_count"]}
            )
        )
    else:
        print(encoded, end="")
    return 0


def command_verify(args: argparse.Namespace) -> int:
    if args.journal:
        return _verify_journal_attestation(Path(args.journal))
    if args.run_id:
        result = _store(args.artifact_dir).verify(args.run_id)
    else:
        if not args.receipt:
            raise ValueError("provide a receipt JSON path or --run-id")
        with Path(args.receipt).open(encoding="utf-8") as handle:
            value = json.load(handle)
        receipt = value.get("receipt", value) if isinstance(value, dict) else value
        if not isinstance(receipt, dict):
            raise ValueError("receipt JSON must be an object")
        result = _verify_receipt(receipt, args.public_key)
    print(_json(result))
    return 0 if result.get("valid") else 1


def _verify_journal_attestation(journal: Path) -> int:
    """Join each journal row to its archived attestation epoch.

    Step (2) of the paper's verification recipe. It exists as a command because
    doing it by hand requires knowing which of the package's two canonical-JSON
    serializers the digest was taken with -- they produce different bytes, and
    hashing with the wrong one makes every row look unresolvable. We hit that
    ourselves while auditing the artifact and briefly concluded the chain was
    broken when it was not, so the join is now a command rather than an
    instruction.
    """

    import hashlib

    from .storage import canonical_json as journal_canonical_json

    sidecar = journal.with_name(journal.stem + ".attestation.json")
    if not sidecar.is_file():
        print(f"traceguard: no attestation sidecar beside {journal.name}", file=sys.stderr)
        return 1
    epochs = json.loads(sidecar.read_text(encoding="utf-8")).get("epochs") or []
    archived = {
        hashlib.sha256(journal_canonical_json(epoch)).hexdigest(): epoch for epoch in epochs
    }

    total = resolved = hardware = 0
    unresolved: list[str] = []
    for line in journal.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") != "completed":
            continue
        total += 1
        att = (row.get("provenance") or {}).get("attestation") or {}
        digest = att.get("evidence_sha256")
        if digest in archived:
            resolved += 1
            epoch = archived[digest]
            if epoch.get("hardware_backed") and epoch.get("maa_verified"):
                hardware += 1
        elif digest:
            unresolved.append(str(digest)[:16])

    print(
        _json(
            {
                "journal": journal.name,
                "sidecar": sidecar.name,
                "epochs_archived": len(archived),
                "completed_rows": total,
                "rows_resolved_to_an_epoch": resolved,
                "rows_hardware_backed_and_maa_verified": hardware,
                "unresolved_digests": sorted(set(unresolved)),
                "all_rows_resolve": resolved == total and total > 0,
            }
        )
    )
    return 0 if resolved == total and total > 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="traceguard",
        description="TraceGuard synthetic reproducibility SDK and research console",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="immutable run store (default: TRACEGUARD_ARTIFACT_DIR or artifacts/runs)",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    doctor = subcommands.add_parser("doctor", help="validate local offline/live configuration")
    doctor.add_argument("--provider", choices=PROVIDERS, default=None)
    doctor.set_defaults(handler=command_doctor)

    serve = subcommands.add_parser("serve", help="serve the local research console")
    serve.add_argument("--provider", choices=PROVIDERS, default=None)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--log-level", choices=("critical", "error", "warning", "info"), default="info"
    )
    serve.set_defaults(handler=command_serve)

    cases = subcommands.add_parser("cases", help="list public synthetic case metadata")
    cases.add_argument("--provider", choices=PROVIDERS, default=None)
    cases.set_defaults(handler=command_cases)

    run = subcommands.add_parser(
        "run", help="run one synthetic case and persist immutable artifacts"
    )
    run.add_argument("case_id")
    run.add_argument("--provider", choices=PROVIDERS, default=None)
    run.add_argument(
        "--condition",
        choices=("adaptive", "structure", "full", "structure_only", "full_pad"),
        default="adaptive",
    )
    run.add_argument("--seed", type=int, default=0)
    run.add_argument(
        "--architecture",
        choices=ARCHITECTURES,
        default="pipeline_crew",
        help="crew topology to execute (default: the released pipeline_crew)",
    )
    run.set_defaults(handler=command_run)

    experiment = subcommands.add_parser(
        "experiment", help="run the balanced full-factorial suite with resumable JSONL"
    )
    experiment.add_argument("--provider", choices=PROVIDERS, default=None)
    experiment.add_argument(
        "--repetitions",
        "--n-per-cell",
        dest="repetitions",
        type=int,
        default=1,
        help="runs per design cell (README/scripts use the --n-per-cell spelling)",
    )
    experiment.add_argument(
        "--conditions",
        nargs="+",
        choices=("adaptive", "structure", "full"),
        default=["adaptive", "structure", "full"],
    )
    experiment.add_argument("--seed", type=int, default=0)
    experiment.add_argument("--experiment-id")
    experiment.add_argument("--journal")
    experiment.add_argument("--no-resume", action="store_true")
    experiment.add_argument("--case-limit", type=int)
    experiment.add_argument("--allow-partial", action="store_true")
    experiment.add_argument(
        "--workers",
        type=int,
        default=1,
        help="concurrent runs; per-run step timings stay sequential within a worker",
    )
    experiment.add_argument(
        "--domain",
        choices=sorted(CORPORA),
        help=(
            "task family to run; each has its own packaged corpus and its own "
            "dataset digest, and the two are never pooled"
        ),
    )
    experiment.add_argument(
        "--require-attestation",
        action="store_true",
        help=(
            "refuse to execute any cell unless this process holds a "
            "hardware-bound attestation (use when the point of the run is "
            "that it executed inside the TEE)"
        ),
    )
    experiment.set_defaults(handler=command_experiment)

    analyze = subcommands.add_parser("analyze", help="analyze completed experiment JSONL")
    analyze.add_argument("journal")
    analyze.add_argument(
        "--condition", choices=("adaptive", "structure", "full"), default="adaptive"
    )
    analyze.add_argument("--bootstrap", type=int, default=1000)
    analyze.add_argument("--seed", type=int, default=0)
    analyze.add_argument("--output")
    analyze.add_argument("--force", action="store_true")
    analyze.set_defaults(handler=command_analyze)

    verify = subcommands.add_parser("verify", help="verify a receipt or immutable run directory")
    verify.add_argument("receipt", nargs="?")
    verify.add_argument("--public-key")
    verify.add_argument("--run-id")
    verify.add_argument(
        "--journal",
        help=(
            "join every row of an experiment journal to its archived attestation "
            "epoch and report which rows resolve"
        ),
    )
    verify.set_defaults(handler=command_verify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("traceguard: interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        # The CLI never prints provider exception strings because transports may echo payloads.
        print(f"traceguard: {type(error).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

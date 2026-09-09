from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from traceguard.api import create_app
from traceguard.config import Settings
from traceguard.storage import ArtifactStore


@dataclass
class _Case:
    case_id: str = "cardiology-demo-routine-control"
    specialty: str = "cardiology"
    topic: str = "demo"
    sensitive: bool = False
    canary_member: bool = False

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "specialty": self.specialty,
            "topic": self.topic,
            "sensitive": self.sensitive,
            "canary_member": self.canary_member,
            "document_count": 6,
        }


class _Catalog:
    dataset_hash = "f" * 64
    schema_version = "synthetic-test-v1"

    def __init__(self) -> None:
        self.case = _Case()

    def list_cases(self) -> list[_Case]:
        return [self.case]

    def get_case(self, case_id: str) -> _Case:
        if case_id != self.case.case_id:
            raise KeyError(case_id)
        return self.case


class _FixtureSDK:
    """Network-free provider-shaped SDK used to test the HTTP contract."""

    def run(
        self,
        case_id,
        condition,
        run_id=None,
        event_callback=None,
        seed=0,
        autonomy=None,
        architecture="pipeline_crew",
    ):
        assert condition == "adaptive"
        # Crew topology reaches the SDK as an explicit argument rather than as
        # settings, so this double has to accept it or the HTTP contract test
        # would pass while the real call site was broken.
        assert architecture in {
            "pipeline_crew",
            "hierarchical_supervisor",
            "react_single_agent",
        }
        self.architecture = architecture
        if event_callback:
            event_callback(
                {
                    "event_type": "step",
                    "step_type": "clinical_extraction",
                    "duration_s": 0.01,
                    "egress_bytes": 123,
                    "hop": 1,
                    "payload": "must not cross the API boundary",
                }
            )
        return {
            "run_id": run_id,
            "case_id": case_id,
            "condition": condition,
            "answer": "synthetic tenant answer is intentionally not returned by this API",
            "trace": {
                "steps": [
                    {
                        "index": 0,
                        "step_type": "clinical_extraction",
                        "wall_time_s": 0.01,
                        "duration_s": 0.01,
                        "egress_bytes": 123,
                        "content": "must not be stored",
                    }
                ]
            },
            "receipt": {
                "schema_version": "fixture-receipt-v1",
                "request_id": run_id,
                "signature": "fixture-signature",
                "scientific_evidence": False,
            },
            "metrics": {
                "step_count": 1,
                "research_hops": 1,
                "observable_duration_s": 0.01,
                "total_egress_bytes": 123,
            },
            "provider_provenance": "synthetic_fixture",
            "dataset_hash": "f" * 64,
            "runtime": {"fixture": True, "paper_reproduction_claim": False},
        }


@pytest.fixture
def api(tmp_path: Path):
    settings = Settings(
        provider="fixture",
        azure_openai_api_key="dummy-unit-test-credential",
        azure_openai_endpoint="https://example-resource.services.ai.azure.com",
    )
    store = ArtifactStore(tmp_path / "runs")
    app = create_app(
        settings=settings,
        catalog=_Catalog(),
        store=store,
        sdk_factory=lambda settings, catalog: _FixtureSDK(),
    )
    with TestClient(app) as client:
        yield client, store
    app.state.run_manager.shutdown()


def _await_terminal(client: TestClient, run_id: str) -> dict:
    for _ in range(200):
        detail = client.get(f"/api/runs/{run_id}").json()
        if detail["status"] in {"completed", "failed", "cancelled"}:
            return detail
        time.sleep(0.005)
    raise AssertionError("background fixture run did not finish")


def test_health_config_cases_and_paper_labels(api) -> None:
    client, _ = api
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["provider"] == "fixture"
    assert health.json()["azure_openai_configured"] is True
    assert health.json()["llm_configured"] is True

    config = client.get("/api/config").json()
    serialized = json.dumps(config)
    assert "dummy-unit-test-credential" not in serialized
    assert "api_key" not in serialized
    assert config["synthetic_dataset"] is True
    assert set(config["supported_conditions"]) == {"adaptive", "structure", "full"}

    cases = client.get("/api/cases").json()
    assert cases["dataset"]["synthetic"] is True
    assert cases["cases"][0]["case_id"] == "cardiology-demo-routine-control"
    assert "query" not in cases["cases"][0]

    paper = client.get("/api/paper/results").json()
    assert paper["status"] == "reported_unverified"
    assert paper["reproduced"] is False
    assert paper["scientific_evidence"] is False
    assert "unverified" in paper["display_label"].lower()


def test_background_run_metadata_only_contract_and_artifacts(api) -> None:
    client, store = api
    response = client.post(
        "/api/runs",
        json={
            "kind": "single",
            "case_id": "cardiology-demo-routine-control",
            "provider": "fixture",
            "condition": "adaptive",
            "seed": 9,
        },
    )
    assert response.status_code == 202
    run_id = response.json()["run_id"]
    detail = _await_terminal(client, run_id)
    assert detail["status"] == "completed"
    assert detail["provenance"]["paper_reproduction_claim"] is False
    assert "answer" not in json.dumps(detail)

    traces = client.get(f"/api/runs/{run_id}/traces").json()
    step = traces["traces"][0]["steps"][0]
    assert step == {
        "index": 0,
        "step_type": "clinical_extraction",
        "wall_time_s": 0.01,
        "duration_s": 0.01,
        "egress_bytes": 123,
        "sequence": 0,
    }
    assert "content" not in json.dumps(traces)

    metrics = client.get(f"/api/runs/{run_id}/metrics").json()
    assert metrics["metrics"]["step_count"] == 1
    receipts = client.get(f"/api/runs/{run_id}/receipts").json()
    assert receipts["receipts"][0]["scientific_evidence"] is False
    assert store.verify(run_id)["valid"] is True

    with client.stream("GET", f"/api/runs/{run_id}/events") as stream:
        text = "".join(stream.iter_text())
    assert "must not cross" not in text
    assert '"duration_ms":10.0' in text
    assert '"egress_bytes":123' in text
    assert '"event_type":"complete"' in text


def test_runtime_report_is_honest_and_scope_matches_receipt(api, monkeypatch) -> None:
    """The /api/runtime self-report must never overclaim, and its implementation_scope
    mirror must stay byte-identical to a freshly built receipt body."""

    from traceguard.api import RUNTIME_IMPLEMENTATION_SCOPE
    from traceguard.receipt import build_receipt_body
    from traceguard.types import Condition, ObservableTrace

    body = build_receipt_body(
        run_id="runtime-scope-check",
        case_id="cardiology-demo-routine-control",
        condition=Condition.ADAPTIVE,
        trace=ObservableTrace(()),
        dataset_hash="f" * 64,
        provider_provenance="synthetic_fixture",
        policy_id="traceguard-public-canon-v1",
        canonical_research_hops=7,
        step_deadline_s=3.0,
        egress_ceiling_bytes=4096,
        fail_closed=False,
        violations=[],
        ledger_sequence=0,
        previous_hash="0" * 64,
    )
    assert RUNTIME_IMPLEMENTATION_SCOPE == body["implementation_scope"]

    client, _ = api
    monkeypatch.setenv("TRACEGUARD_ISOLATION", "kata-qemu")
    report = client.get("/api/runtime").json()
    assert report["schema_version"] == "traceguard.runtime.v1"
    assert report["isolation"]["kind"] == "kata-qemu"
    assert report["isolation"]["vm_level_isolation"] is True
    # Honesty posture: never claims a TEE or a hardware attestation quote.
    assert report["hardware_tee"]["quote_available"] is False
    assert report["hardware_tee"]["memory_encryption"] is False
    assert report["attestation"]["kind"] == "none"
    assert report["implementation_scope"]["tee"] is False
    assert report["implementation_scope"]["hardware_attestation"] is False
    assert report["receipt_signer"]["trust_root"] == "demo_key_not_hardware_attested"


def test_run_submit_rejects_unconfigured_azure_provider(tmp_path: Path) -> None:
    """Requesting a live provider with no credential must be refused up front.

    This needs its own app: the shared `api` fixture is deliberately configured,
    so it is the wrong instrument for asserting the unconfigured path.
    """

    store = ArtifactStore(tmp_path / "runs")
    app = create_app(
        settings=Settings(provider="fixture"),  # no Azure credential at all
        catalog=_Catalog(),
        store=store,
        sdk_factory=lambda settings, catalog: _FixtureSDK(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/runs",
            json={
                "kind": "single",
                "case_id": "cardiology-demo-routine-control",
                "provider": "azure",
                "condition": "adaptive",
            },
        )
        assert response.status_code == 422
        assert response.json()["detail"] == "Azure OpenAI provider is not configured"
    app.state.run_manager.shutdown()


def test_run_validation_and_receipt_verifier_contract(api) -> None:
    client, _ = api
    missing = client.post("/api/runs", json={"kind": "single", "condition": "adaptive"})
    assert missing.status_code == 422
    unknown = client.get("/api/runs/not-a-run")
    assert unknown.status_code == 404
    verified = client.post(
        "/api/receipts/verify",
        json={"receipt": {"schema_version": "invalid", "signature": "invalid"}},
    )
    assert verified.status_code == 200
    assert isinstance(verified.json()["valid"], bool)


def test_immutable_store_hashes_redaction_and_tamper_detection(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "runs")
    writer = store.create_run(
        run_id="fixture-run",
        case_id="case-1",
        condition="adaptive",
        seed=1,
        provenance={"provider": "fixture", "api_key": "discard-me"},
    )
    writer.append_event(
        {
            "step_type": "coverage_assessment",
            "duration_s": 0.2,
            "egress_bytes": 80,
            "payload": "discard",
        }
    )
    writer.append_trace(
        {
            "query": "discard",
            "steps": [
                {
                    "index": 0,
                    "step_type": "coverage_assessment",
                    "duration_s": 0.2,
                    "wall_time_s": 0.2,
                    "egress_bytes": 80,
                    "content": "discard",
                }
            ],
        }
    )
    writer.append_receipt({"signature": "sig", "secret": "discard"})
    writer.write_result({"answer": "discard", "metrics": {"step_count": 1}})
    stored = writer.finalize()

    assert store.verify("fixture-run") == {
        "run_id": "fixture-run",
        "valid": True,
        "mismatches": [],
        "finalized": True,
    }
    all_text = "".join(
        path.read_text(encoding="utf-8") for path in stored.path.iterdir() if path.is_file()
    )
    assert "discard" not in all_text
    assert "api_key" not in all_text
    manifest = store.manifest("fixture-run")
    assert set(manifest["artifacts"]) == {
        "events.jsonl",
        "traces.jsonl",
        "result.json",
        "receipts.jsonl",
    }
    with pytest.raises(RuntimeError, match="immutable"):
        writer.append_event({"step_type": "late"})

    with (stored.path / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    verification = store.verify("fixture-run")
    assert verification["valid"] is False
    assert verification["mismatches"] == ["events.jsonl"]

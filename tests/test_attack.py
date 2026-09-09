from __future__ import annotations

import math

import pytest

from traceguard.attack import (
    bootstrap_auc,
    evaluate_attack,
    extract_trace_features,
    inversion_safe_auc,
    mann_whitney_auc,
    randomized_response_frontier,
    randomized_response_keep_probability,
    run_ablation_suite,
    shape_fixed_control,
)


def _records(repetitions: int = 3) -> list[dict]:
    records = []
    topics = {
        "cardiology": ("anticoagulation", "discharge"),
        "oncology": ("neutropenia", "symptoms"),
        "psychiatry": ("reconciliation", "crisis"),
    }
    for service, service_topics in topics.items():
        for topic in service_topics:
            for sensitive in (False, True):
                for member in (False, True):
                    for repetition in range(repetitions):
                        hop_count = 7 if sensitive else 3
                        steps = [
                            {
                                "step_type": "intake",
                                "duration_s": 0.01,
                                "egress_bytes": 80,
                                "payload": "must-never-be-a-feature",
                            }
                        ]
                        steps.extend(
                            {
                                "step_type": "clinical_extraction",
                                "duration_s": (0.080 if sensitive else 0.015) + repetition / 10000,
                                "egress_bytes": (520 if sensitive else 120) + (25 if member else 0),
                            }
                            for _ in range(hop_count)
                        )
                        steps.append(
                            {
                                "step_type": "coverage_assessment",
                                "duration_s": 0.2 if sensitive else 0.03,
                                "egress_bytes": (1800 if sensitive else 320)
                                + (80 if member else 0),
                            }
                        )
                        records.append(
                            {
                                "service": service,
                                "topic": topic,
                                "attribute_label": int(sensitive),
                                "membership_label": int(member),
                                "trace": {"steps": steps},
                                "provenance": {"provider": "fixture"},
                            }
                        )
    return records


def test_feature_extraction_matches_paper_aggregates() -> None:
    trace = {
        "steps": [
            {"step_type": "clinical_extraction", "duration_s": 0.1, "egress_bytes": 100},
            {"step_type": "clinical_extraction", "duration_s": 0.2, "egress_bytes": 200},
            {"step_type": "coverage_assessment", "duration_s": 0.3, "egress_bytes": 300},
        ],
        "query": "not observable",
    }
    features = extract_trace_features(trace)
    assert features["step_count"] == 3
    assert features["distinct_tools"] == 2
    assert features["hop_count"] == 2
    # No `_sum`, deliberately: a feature set carrying both sum and mean hands the
    # attacker sum/mean == n, so a projection that hides structure would still
    # leak the exact step count. mean and max are count-invariant.
    assert "timing_ms_sum" not in features
    assert "egress_bytes_sum" not in features
    assert features["timing_ms_mean"] == pytest.approx(200)
    assert features["timing_ms_mean"] == pytest.approx(200)
    assert features["timing_ms_max"] == pytest.approx(300)
    assert features["egress_bytes_mean"] == pytest.approx(200)
    assert features["tool_count:clinical_extraction"] == 2
    assert features["bigram:clinical_extraction->clinical_extraction"] == 1
    assert features["bigram:clinical_extraction->coverage_assessment"] == 1
    assert all("query" not in name and "payload" not in name for name in features)


def test_mann_whitney_and_inversion_safe_auc() -> None:
    labels = [0, 0, 1, 1]
    assert mann_whitney_auc(labels, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert mann_whitney_auc(labels, [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert inversion_safe_auc(labels, [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert mann_whitney_auc(labels, [1, 1, 1, 1]) == 0.5


def test_grouped_attack_cluster_bootstrap_and_fixture_label() -> None:
    report = evaluate_attack(_records(), target="attribute", n_bootstrap=80, seed=7)
    assert report["protocol"] == "leave-one-(service,topic)-out"
    assert report["n_groups"] == 6
    assert len(report["folds"]) == 6
    assert report["auc"] > 0.98
    assert report["inversion_safe"] is True
    assert report["confidence_interval"]["method"] == "cluster_percentile"
    assert report["confidence_interval"]["valid_replicates"] == 80
    assert report["evidence"]["tier"] == "fixture_diagnostic"
    assert report["evidence"]["scientific_evidence"] is False
    assert report["evidence"]["paper_reproduction_claim"] is False


def test_bootstrap_falls_back_to_stratified_rows() -> None:
    interval = bootstrap_auc(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        clusters=["only"] * 4,
        n_bootstrap=20,
        seed=2,
    )
    assert interval["method"] == "stratified_percentile"
    assert interval["valid_replicates"] == 20
    assert interval["low"] == interval["high"] == 1.0


def test_shape_fixed_control_and_full_ablation_are_chance() -> None:
    records = _records(repetitions=2)
    control = shape_fixed_control(records, target="attribute", seed=3)
    assert control["auc"] == pytest.approx(0.5)
    assert control["control"] == "shape_fixed"
    ablations = run_ablation_suite(records, target="attribute", seed=4)
    by_name = {row["ablation"]: row["auc"] for row in ablations["rows"]}
    assert by_name["none"] > 0.98
    # Closing all four coordinates is the in-boundary counterfactual and is
    # chance by construction.  The row that matters for the shipped system is
    # `enforceable_remote_model`, which closes the three coordinates the guest
    # controls and leaves provider-chosen response size live -- so it is NOT
    # expected to be chance, and asserting that it were would re-introduce the
    # very overclaim this split exists to remove.
    assert by_name["all_four_coordinates"] == pytest.approx(0.5)
    assert "enforceable_remote_model" in by_name


def test_k3_randomized_response_formula_and_frontier_schema() -> None:
    assert randomized_response_keep_probability(0.0, k=3) == pytest.approx(1 / 3)
    expected = math.exp(1) / (math.exp(1) + 2)
    assert randomized_response_keep_probability(1.0, k=3) == pytest.approx(expected)
    frontier = randomized_response_frontier(
        _records(repetitions=2),
        target="attribute",
        epsilons=(1.0, 0.0),
        repetitions=2,
        seed=11,
    )
    assert frontier["k"] == 3
    assert frontier["paper_reproduction_claim"] is False
    assert [row["epsilon"] for row in frontier["rows"]] == [1.0, 0.0]
    assert frontier["rows"][1]["keep_probability"] == pytest.approx(1 / 3)
    # K is no longer pinned to 3: the mechanism is now sweepable alongside the
    # other factors, so any k >= 2 is allowed provided k-1 boundaries are given.
    wider = randomized_response_frontier(
        _records(1), k=4, repetitions=1, epsilons=(1.0,), depth_boundaries=[3.0, 4.0, 5.0]
    )
    assert wider["k"] == 4
    with pytest.raises(ValueError, match="at least two templates"):
        randomized_response_frontier(_records(1), k=1, repetitions=1)

"""Statistics for a graded secret, and the guards that keep them interpretable."""

from __future__ import annotations

import numpy as np
import pytest

from traceguard.attack import (
    evaluate_ordinal_leakage,
    holm_bonferroni,
    inversion_safe_somers_d,
    jonckheere_terpstra,
    mann_whitney_auc,
    randomized_response_frontier,
    somers_d,
)


def test_somers_d_reduces_to_the_binary_auc_identity() -> None:
    """D == 2*AUC - 1 on a binary label.

    This is what keeps the ordinal results comparable with every binary number
    already in the manuscript; without it the two families of results could not
    be read against each other.
    """

    rng = np.random.default_rng(1)
    for _ in range(5):
        labels = rng.integers(0, 2, 80)
        scores = labels * 0.6 + rng.normal(0, 1, 80)
        auc = mann_whitney_auc(labels.tolist(), scores.tolist())
        assert somers_d(labels.tolist(), scores.tolist()) == pytest.approx(2 * auc - 1, abs=1e-9)


def test_somers_d_bounds_and_inversion_safety() -> None:
    labels = [0, 0, 1, 1, 2, 2, 3, 3]
    assert somers_d(labels, [1, 2, 3, 4, 5, 6, 7, 8]) == pytest.approx(1.0)
    assert somers_d(labels, [8, 7, 6, 5, 4, 3, 2, 1]) == pytest.approx(-1.0)
    # An attacker free to flip its score sign is not weakened by anti-correlation,
    # so magnitude is the honest measure -- matching the max(AUC, 1-AUC) convention.
    assert inversion_safe_somers_d(labels, [8, 7, 6, 5, 4, 3, 2, 1]) == pytest.approx(1.0)
    # A constant score orders nothing.
    assert somers_d(labels, [1] * 8) == pytest.approx(0.0)


def test_somers_d_requires_a_graded_label() -> None:
    with pytest.raises(ValueError, match="two distinct label values"):
        somers_d([1, 1, 1], [1.0, 2.0, 3.0])


def test_jonckheere_terpstra_separates_trend_from_flat() -> None:
    """A trend test, not an omnibus difference test.

    "Leakage rises with sensitivity" is the claim worth making on a ladder, and
    it is a strictly stronger statement than "the groups differ somewhere".
    """

    rng = np.random.default_rng(7)
    rising = [rng.normal(level * 0.9, 1, 30).tolist() for level in range(4)]
    flat = [rng.normal(0, 1, 30).tolist() for _ in range(4)]
    assert jonckheere_terpstra(rising)["z"] > 4.0
    assert abs(jonckheere_terpstra(flat)["z"]) < 2.0
    with pytest.raises(ValueError, match="at least three ordered groups"):
        jonckheere_terpstra([[1.0], [2.0]])


def test_holm_bonferroni_is_step_down_and_beats_bonferroni() -> None:
    family = {"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.9}
    report = holm_bonferroni(family, alpha=0.05)
    adjusted = [entry["p_adjusted"] for entry in report["results"].values()]
    # Step-down: adjusted p-values are monotone in sorted raw order, so a
    # retained hypothesis forces every larger p-value to be retained too.
    assert adjusted == sorted(adjusted)
    assert report["results"]["a"]["rejected"] is True
    assert report["results"]["d"]["rejected"] is False
    # Strictly more powerful than plain Bonferroni at the same familywise rate.
    assert report["results"]["a"]["p_adjusted"] < 0.001 * len(family) + 1e-12
    assert holm_bonferroni({})["family_size"] == 0


def test_frontier_rejects_a_collapsed_template_partition() -> None:
    """The shipped table reported K=3 for a mechanism that released 2 templates.

    Boundaries were empirical tertiles of a depth distribution whose mass sits
    at the extremes, so both boundaries landed on the same value (3.0) and one
    template was never occupied. Reporting K keep probabilities for a mechanism
    that cannot emit K templates describes something that did not run.
    """

    records = _synthetic_records()
    with pytest.raises(ValueError, match="collapse to fewer than k"):
        randomized_response_frontier(records, target="attribute", depth_boundaries=[3.0, 3.0])


def test_frontier_partition_is_public_and_data_independent() -> None:
    """The default partition must not depend on the sample.

    A template set the adversary cannot know in advance is not a public
    template set, so the default is derived from the published plan depth
    range. Two samples with different observed depths must therefore yield the
    same boundaries.
    """

    a = randomized_response_frontier(_synthetic_records(), target="attribute", epsilons=(1.0,))
    b = randomized_response_frontier(
        _synthetic_records(scale=2), target="attribute", epsilons=(1.0,)
    )
    assert a["boundaries"] == b["boundaries"]
    # Data independence is the property under test. Whether a given sample
    # happens to span every band is a separate, legitimate fact and is reported
    # separately rather than treated as a partition defect.
    assert "sample_occupies_all_templates" in a


def test_frontier_supports_k_other_than_three() -> None:
    records = _synthetic_records()
    report = randomized_response_frontier(
        records, target="attribute", k=4, epsilons=(1.0,), depth_boundaries=[3.0, 4.0, 5.0]
    )
    assert report["k"] == 4
    assert len(report["boundaries"]) == 3
    with pytest.raises(ValueError, match="exactly 3 depth boundaries"):
        randomized_response_frontier(records, target="attribute", k=4, depth_boundaries=[3.0])
    with pytest.raises(ValueError, match="at least two templates"):
        randomized_response_frontier(records, target="attribute", k=1)


def _synthetic_records(scale: int = 1) -> list[dict[str, object]]:
    """Two groups, both classes present in each, with graded depths."""

    records = []
    for group in ("cardiology::a", "oncology::b"):
        for label in (0, 1):
            for index in range(6):
                depth = (2 + label * 4) * scale
                records.append(
                    {
                        "service": group.split("::")[0],
                        "topic": group.split("::")[1],
                        "attribute_label": label,
                        "membership_label": index % 2,
                        "trace": {
                            "steps": [
                                {
                                    "step_type": "clinical_extraction",
                                    "duration_s": 1.0 + 0.1 * index,
                                    "egress_bytes": 100 + 10 * index,
                                }
                            ]
                            * depth
                        },
                    }
                )
    return records


def test_adversary_spectrum_projects_to_each_threat_model() -> None:
    """Each restricted adversary sees strictly what its threat model allows."""

    from traceguard.attack import OBSERVABILITY, observability_transform

    rows = [
        {
            "step_count": 12.0,
            "hop_count": 7.0,
            "distinct_tools": 4.0,
            "timing_ms_sum": 900.0,
            "timing_ms_mean": 75.0,
            "egress_bytes_sum": 2000.0,
            "egress_bytes_max": 500.0,
            "tool_count:intake": 1.0,
            "bigram:intake->extract": 1.0,
        }
    ]
    projected = {name: observability_transform(name)(rows)[0] for name in OBSERVABILITY}

    # The full adversary keeps everything; the count-only one keeps one scalar.
    assert projected["full"] == rows[0]
    assert set(projected["count_only"]) == {"step_count"}

    # A timing-only observer must see no size feature, and vice versa. Getting
    # this backwards would silently model a different adversary than claimed.
    assert not any(k.startswith("egress_bytes") for k in projected["timing_only"])
    assert not any(k.startswith("timing_ms") for k in projected["size_only"])
    # A structure-only host sees the shape, including tool order, but no
    # timing and no sizes.
    assert "distinct_tools" in projected["structure_only"]
    assert "bigram:intake->extract" in projected["structure_only"]
    assert not any(
        k.startswith(("timing_ms", "egress_bytes")) for k in projected["structure_only"]
    )

    with pytest.raises(ValueError, match="unknown observability"):
        observability_transform("omniscient")


def test_adversary_spectrum_never_claims_unrecorded_capabilities() -> None:
    """Out-of-scope adversaries must stay out of scope.

    Single-stepping, page-fault tracking and KV-cache timing read intra-VM
    state that no application-level trace contains, so no projection of these
    traces can speak to them. The result must say so rather than leaving a
    reader to assume the spectrum is exhaustive.
    """

    from traceguard.attack import OBSERVABILITY, evaluate_adversary_spectrum

    report = evaluate_adversary_spectrum(_synthetic_records(), target="attribute")
    assert set(report["spectrum"]) == set(OBSERVABILITY)
    for term in ("single-stepping", "page-fault", "KV-cache"):
        assert term in report["out_of_scope"]


# --------------------------------------------------------------------------- #
# evaluate_ordinal_leakage: the attacker is binary-trained, ordinally scored.
# --------------------------------------------------------------------------- #


def _graded_records(*, graded: bool) -> list[dict[str, object]]:
    """Six groups x four rungs x three repeats.

    ``graded=True`` makes the observable rise with the rung, so the ladder is
    recoverable. ``graded=False`` makes it depend only on the binary label, so
    rungs 1..3 are observationally identical -- the case where a monotone
    Somers' D would be an artefact and must not appear.
    """

    records = []
    for service in ("cardiology", "oncology", "psychiatry"):
        for topic in ("a", "b"):
            for level in range(4):
                for repeat in range(3):
                    steps = 8 + (level * 2 if graded else (4 if level else 0))
                    records.append(
                        {
                            "trace": {
                                "steps": [
                                    {
                                        "index": i,
                                        "step_type": "clinical_extraction",
                                        "duration_s": 1.0 + 0.01 * repeat,
                                        "egress_bytes": 100 * steps,
                                    }
                                    for i in range(steps)
                                ]
                            },
                            "service": service,
                            "topic": topic,
                            "attribute_label": int(level > 0),
                            "membership_label": 0,
                            "sensitivity_level": level,
                        }
                    )
    return records


def test_ordinal_leakage_finds_a_graded_channel() -> None:
    report = evaluate_ordinal_leakage(_graded_records(graded=True), seed=7, n_perm=200)
    assert report["rungs"] == [0, 1, 2, 3]
    assert report["attacker"] == "binary-trained, ordinally scored"
    # Monotone in the rung means, and clearing its own permutation null.
    means = report["rung_mean_score"]
    assert means == sorted(means), means
    assert report["somers_d"] > report["null"]["null_q95"]
    assert report["null"]["p_value"] < 0.05
    assert report["trend"]["z"] > 2.0
    # A genuinely graded channel also clears the restricted test, which the
    # binary-only channel does not. That contrast is the whole point.
    within = report["within_sensitive"]
    assert within["rungs"] == [1, 2, 3]
    assert within["null"]["p_value"] < 0.05, within


def test_ordinal_leakage_reports_no_trend_when_the_ladder_is_not_observable() -> None:
    """A binary-only channel must not produce a significant ordinal claim.

    This is the guard that matters: the estimator is inversion-safe and so
    biased upward, and rung means computed on 72 records will wobble. Without
    the null, a flat channel reads as a graded one.
    """

    report = evaluate_ordinal_leakage(_graded_records(graded=False), seed=7, n_perm=400)
    # The full-ladder D is legitimately large here and MUST NOT be read as a
    # graded finding: the attacker separates rung 0 from the rest, and a step
    # function is monotone. This is the trap the restricted statistic exists to
    # close, so assert the trap is still there rather than pretending it isn't.
    assert report["null"]["p_value"] < 0.05, report
    # The incremental claim is the honest one, and it must find nothing.
    within = report["within_sensitive"]
    assert within["null"]["p_value"] > 0.05, within
    assert within["somers_d"] <= within["null"]["null_q95"], within


def test_ordinal_leakage_refuses_a_binary_record_set() -> None:
    """The archived two-level journals must fail loudly, not silently degrade."""

    binary = [
        {**record, "sensitivity_level": record["attribute_label"]}
        for record in _graded_records(graded=True)
    ]
    with pytest.raises(ValueError, match="at least three occupied rungs"):
        evaluate_ordinal_leakage(binary, seed=7)


def test_ordinal_leakage_requires_a_level_on_every_record() -> None:
    records = _graded_records(graded=True)
    del records[0]["sensitivity_level"]
    with pytest.raises(ValueError, match="integer sensitivity_level"):
        evaluate_ordinal_leakage(records, seed=7)


def test_evaluate_attack_does_not_leak_the_valid_mask_into_its_report() -> None:
    """The mask is a numpy array; a JSON report carrying it would fail to dump."""

    import json

    from traceguard.attack import evaluate_attack

    report = evaluate_attack(_graded_records(graded=True), n_bootstrap=0, seed=7)
    assert "valid_mask" not in report
    json.dumps(report)


def test_paired_group_difference_field_names_are_a_contract() -> None:
    """scripts/make_matrix_artifacts.py reads these keys by name.

    It once read 'difference' and 'confidence_interval', which do not exist, and
    the resulting KeyError was swallowed by an over-broad except that degraded to
    "contrast unevaluable" -- so the manuscript would simply have been missing
    its model-contrast macros after a multi-hour paid run, with no error. Pinning
    the names here makes a rename fail in CI in under a second instead.
    """

    from traceguard.attack import paired_group_difference

    high = _graded_records(graded=True)
    low = [
        {**record, "sensitivity_level": record["attribute_label"] * 2}
        for record in _graded_records(graded=False)
    ]
    report = paired_group_difference(high, low, target="attribute", n_bootstrap=200, seed=3)

    assert {"mean_difference", "ci95", "n_groups", "per_group_difference"} <= set(report)
    assert isinstance(report["mean_difference"], float)
    low_ci, high_ci = report["ci95"]
    assert low_ci <= report["mean_difference"] <= high_ci
    assert report["n_groups"] == 6

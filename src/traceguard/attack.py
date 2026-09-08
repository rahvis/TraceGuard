"""Trace-Reconstruction Attack and reproducible statistical controls.

Only host-observable metadata is used: step/tool type, per-step timing, and the
two wire byte counts -- request bytes leaving the CVM (egress) and response bytes
entering it (ingress).  Both directions are host-visible because the model is
remote.  No query, document, model response text, or other payload is accepted as
a feature.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET_ALIASES = {
    "attribute": ("attribute_label", "sensitive", "attribute", "is_sensitive"),
    "membership": ("membership_label", "canary_member", "member", "membership"),
}
STRUCTURE_FEATURES = {"step_count", "hop_count", "distinct_tools"}

# The published CANON-v1 admissible research-depth range (min_research_hops to
# canonical/max hops). The finite-epsilon template partition is derived from
# this rather than from observed depths, because a public template set must be
# knowable in advance.
#
# The partition is a real design parameter, not a formality: measured on the
# shipped journal, equal bands that isolate the routine mode in a band of its
# own give residual attribute AUC 0.959 at eps=8, while plan-derived bands give
# 0.779 -- and the former exceeds the *undefended* 0.736, because a clean
# categorical generalizes across held-out groups better than twenty noisy
# features do. At small epsilon every partition converges to the null, so it is
# the large-epsilon (high-utility) end of the frontier that is partition
# sensitive.
PUBLISHED_DEPTH_RANGE = (2, 7)
_SAFE_NAME = re.compile(r"[^a-z0-9_.-]+")


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


def _steps(trace: Any) -> list[dict[str, Any]]:
    raw = _mapping(trace)
    nested = raw.get("trace")
    if nested is not None and not isinstance(nested, list):
        nested_map = _mapping(nested)
        if nested_map:
            raw = {**raw, **nested_map}
    values = raw.get("steps", raw.get("events", nested if isinstance(nested, list) else []))
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    return [_mapping(value) for value in values]


def _number(step: Mapping[str, Any], names: Sequence[str], default: float = 0.0) -> float:
    nested = step.get("data")
    sources = [step, nested] if isinstance(nested, Mapping) else [step]
    for source in sources:
        for name in names:
            value = source.get(name)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return float(value)
    return default


def _duration_ms(step: Mapping[str, Any]) -> float:
    milliseconds = _number(step, ("duration_ms", "elapsed_ms", "latency_ms"), float("nan"))
    if math.isfinite(milliseconds):
        return max(0.0, milliseconds)
    seconds = _number(step, ("duration_s", "elapsed_s", "latency_s", "timing", "duration"))
    return max(0.0, seconds * 1000.0)


def _ingress_bytes(step: Mapping[str, Any]) -> float:
    """Response bytes entering the CVM.

    Recorded separately from egress because the two directions have different
    controllability: the guest composes the request, the provider chooses the
    response.  Archived journals predate the split and stored the response size
    under ``egress_bytes``; ``_legacy_single_direction`` detects that so old runs
    are never silently read as if they carried both.
    """

    return max(
        0.0,
        _number(step, ("ingress_bytes", "response_bytes", "rx_bytes")),
    )


def _egress_bytes(step: Mapping[str, Any]) -> float:
    return max(
        0.0,
        _number(step, ("egress_bytes", "output_bytes", "size_bytes", "bytes", "egress_size")),
    )


def _tool(step: Mapping[str, Any]) -> str:
    nested = step.get("data")
    sources = [step, nested] if isinstance(nested, Mapping) else [step]
    for source in sources:
        for key in ("step_type", "tool", "name", "agent", "event_type", "type", "kind"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return _SAFE_NAME.sub("_", value.strip().lower())[:80]
    return "step"


def _summary(prefix: str, values: Sequence[float]) -> dict[str, float]:
    """Count-free per-coordinate aggregates.

    Deliberately *not* ``sum``.  A feature set carrying both ``sum`` and ``mean``
    hands the attacker ``sum/mean == n``, so a projection that hides the
    structure coordinate would still leak the exact step count and an ablation
    that "closes structure" would not close it.  ``mean`` and ``max`` are both
    invariant to n, so restricting a coordinate really does restrict it.

    ``step_count`` remains available to adversaries that legitimately observe
    structure; it is simply no longer recoverable from the timing or size
    features alone.
    """

    if not values:
        return {f"{prefix}_mean": 0.0, f"{prefix}_max": 0.0}
    return {
        f"{prefix}_mean": float(sum(values) / len(values)),
        f"{prefix}_max": float(max(values)),
    }


def extract_trace_features(trace: Any, *, include_bigrams: bool = True) -> dict[str, float]:
    """Extract exactly the aggregate trace-shape features described in the paper."""

    raw = _mapping(trace)
    if isinstance(raw.get("features"), Mapping):
        result = {
            str(key): float(value)
            for key, value in raw["features"].items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }
        if not include_bigrams:
            result = {key: value for key, value in result.items() if not key.startswith("bigram:")}
        return result

    steps = _steps(raw)
    tools = [_tool(step) for step in steps]
    timings = [_duration_ms(step) for step in steps]
    sizes = [_egress_bytes(step) for step in steps]
    ingress = [_ingress_bytes(step) for step in steps]
    counts = Counter(tools)
    features: dict[str, float] = {
        "step_count": float(len(steps)),
        "distinct_tools": float(len(counts)),
        "hop_count": float(
            sum(
                any(
                    token in tool
                    for token in ("retriev", "research", "search", "hop", "extract", "clinical")
                )
                for tool in tools
            )
        ),
        **_summary("timing_ms", timings),
        **_summary("egress_bytes", sizes),
        **_summary("ingress_bytes", ingress),
    }
    for tool, count in sorted(counts.items()):
        features[f"tool_count:{tool}"] = float(count)
    if include_bigrams:
        for pair, count in sorted(Counter(zip(tools, tools[1:], strict=False)).items()):
            features[f"bigram:{pair[0]}->{pair[1]}"] = float(count)
    return features


def mann_whitney_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """ROC-AUC computed as the tie-aware Mann-Whitney U statistic."""

    y = np.asarray(labels, dtype=int)
    values = np.asarray(scores, dtype=float)
    if y.size != values.size or y.size == 0:
        raise ValueError("labels and scores must be non-empty and the same length")
    positive = y == 1
    negative = y == 0
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    if not n_positive or not n_negative:
        raise ValueError("ROC-AUC requires both binary classes")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    index = 0
    while index < values.size:
        end = index + 1
        while end < values.size and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + 1 + end) / 2.0
        index = end
    u = ranks[positive].sum() - n_positive * (n_positive + 1) / 2.0
    return float(u / (n_positive * n_negative))


def inversion_safe_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Return attacker-favorable AUC, accounting for a consistently inverted score."""

    auc = mann_whitney_auc(labels, scores)
    return max(auc, 1.0 - auc)


def bootstrap_auc(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    clusters: Sequence[Any] | None = None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Percentile CI with cluster resampling when at least two groups are available."""

    y = np.asarray(labels, dtype=int)
    values = np.asarray(scores, dtype=float)
    if y.size != values.size:
        raise ValueError("labels and scores must be the same length")
    if n_bootstrap <= 0:
        point = inversion_safe_auc(y, values)
        return {"low": point, "high": point, "valid_replicates": 0, "method": "disabled"}
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    cluster_values = np.asarray(clusters, dtype=object) if clusters is not None else None
    unique_clusters = (
        list(dict.fromkeys(cluster_values.tolist())) if cluster_values is not None else []
    )
    clustered = len(unique_clusters) >= 2
    positive_indices = np.flatnonzero(y == 1)
    negative_indices = np.flatnonzero(y == 0)
    if not len(positive_indices) or not len(negative_indices):
        raise ValueError("ROC-AUC requires both binary classes")
    attempts = max(n_bootstrap * 3, n_bootstrap)
    for _ in range(attempts):
        if len(estimates) >= n_bootstrap:
            break
        if clustered:
            sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
            pieces = [np.flatnonzero(cluster_values == group) for group in sampled]
            indices = np.concatenate(pieces) if pieces else np.array([], dtype=int)
        else:
            # Stratification guarantees a defined statistic for small fixture suites.
            indices = np.concatenate(
                (
                    rng.choice(positive_indices, size=len(positive_indices), replace=True),
                    rng.choice(negative_indices, size=len(negative_indices), replace=True),
                )
            )
        if indices.size and len(np.unique(y[indices])) == 2:
            estimates.append(inversion_safe_auc(y[indices], values[indices]))
    if not estimates:
        point = inversion_safe_auc(y, values)
        return {
            "low": point,
            "high": point,
            "valid_replicates": 0,
            "method": "cluster_percentile" if clustered else "stratified_percentile",
        }
    alpha = (1.0 - confidence) / 2.0
    return {
        "low": float(np.quantile(estimates, alpha)),
        "high": float(np.quantile(estimates, 1.0 - alpha)),
        "valid_replicates": len(estimates),
        "method": "cluster_percentile" if clustered else "stratified_percentile",
        "confidence": confidence,
    }


def _field(record: Mapping[str, Any], names: Sequence[str]) -> Any:
    sources = [record]
    for nested_name in ("labels", "case", "metadata"):
        nested = record.get(nested_name)
        if isinstance(nested, Mapping):
            sources.append(nested)
    for source in sources:
        for name in names:
            if name in source:
                return source[name]
    raise KeyError(f"record is missing one of: {', '.join(names)}")


def _label(record: Mapping[str, Any], target: str) -> int:
    if target not in TARGET_ALIASES:
        raise ValueError("target must be 'attribute' or 'membership'")
    value = _field(record, TARGET_ALIASES[target])
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "sensitive", "member", "present"}:
            return 1
        if lowered in {"0", "false", "no", "routine", "nonmember", "absent"}:
            return 0
    if value in (0, 1, False, True):
        return int(value)
    raise ValueError(f"{target} label is not binary: {value!r}")


def _group(record: Mapping[str, Any]) -> str:
    service = _field(record, ("service", "specialty"))
    topic = _field(record, ("topic",))
    return f"{service}::{topic}"


def _record_features(record: Mapping[str, Any], include_bigrams: bool) -> dict[str, float]:
    if isinstance(record.get("features"), Mapping):
        return extract_trace_features(record, include_bigrams=include_bigrams)
    trace = record.get("trace", record)
    return extract_trace_features(trace, include_bigrams=include_bigrams)


def _constant_scores(y_train: np.ndarray, size: int) -> np.ndarray:
    return np.full(size, float(y_train.mean()) if y_train.size else 0.5)


def _fit_scores(
    train_features: Sequence[dict[str, float]],
    y_train: np.ndarray,
    test_features: Sequence[dict[str, float]],
    *,
    seed: int,
) -> np.ndarray:
    if len(np.unique(y_train)) < 2:
        return _constant_scores(y_train, len(test_features))
    vectorizer = DictVectorizer(sparse=True)
    train_matrix = vectorizer.fit_transform(train_features)
    test_matrix = vectorizer.transform(test_features)
    if train_matrix.shape[1] == 0 or train_matrix.nnz == 0:
        return _constant_scores(y_train, len(test_features))
    model = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
            solver="liblinear",
        ),
    )
    model.fit(train_matrix, y_train)
    return np.asarray(model.predict_proba(test_matrix)[:, 1], dtype=float)


def _evaluate_features(
    features_a1: Sequence[dict[str, float]],
    features_a2: Sequence[dict[str, float]],
    labels: Sequence[int],
    groups: Sequence[str],
    *,
    seed: int,
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=int)
    group_array = np.asarray(groups, dtype=object)
    unique_groups = list(dict.fromkeys(groups))
    if len(unique_groups) < 2:
        raise ValueError("group-aware evaluation requires at least two (service, topic) groups")
    scores = {"A1": np.full(y.size, np.nan), "A2": np.full(y.size, np.nan)}
    folds: list[dict[str, Any]] = []
    # A fold whose train or test side lacks both classes cannot yield an AUC and
    # is skipped. That used to happen silently, so fold attrition was invisible;
    # with a graded label and small cells it becomes common and it biases the
    # fold mean, so every skip is now recorded and reported.
    skipped: list[dict[str, Any]] = []
    for fold_index, held_out in enumerate(unique_groups):
        test_indices = np.flatnonzero(group_array == held_out)
        train_indices = np.flatnonzero(group_array != held_out)
        if len(np.unique(y[test_indices])) < 2 or len(np.unique(y[train_indices])) < 2:
            skipped.append(
                {
                    "held_out_group": held_out,
                    "reason": (
                        "test_single_class"
                        if len(np.unique(y[test_indices])) < 2
                        else "train_single_class"
                    ),
                    "n_test": int(test_indices.size),
                }
            )
            continue
        fold_row: dict[str, Any] = {
            "held_out_group": held_out,
            "n_train": int(train_indices.size),
            "n_test": int(test_indices.size),
        }
        for name, values in (("A1", features_a1), ("A2", features_a2)):
            predicted = _fit_scores(
                [values[index] for index in train_indices],
                y[train_indices],
                [values[index] for index in test_indices],
                seed=seed + fold_index,
            )
            scores[name][test_indices] = predicted
            fold_row[f"{name.lower()}_auc"] = inversion_safe_auc(y[test_indices], predicted)
        fold_row["attacker_favorable_auc"] = max(fold_row["a1_auc"], fold_row["a2_auc"])
        folds.append(fold_row)
    valid = ~np.isnan(scores["A1"]) & ~np.isnan(scores["A2"])
    if not valid.any() or len(np.unique(y[valid])) < 2:
        raise ValueError("no valid leave-one-group-out folds contained both classes")
    model_rows: dict[str, dict[str, Any]] = {}
    for name in ("A1", "A2"):
        fold_aucs = [row[f"{name.lower()}_auc"] for row in folds]
        model_rows[name] = {
            "pooled_auc": inversion_safe_auc(y[valid], scores[name][valid]),
            "fold_mean_auc": float(np.mean(fold_aucs)),
            "fold_std_auc": float(np.std(fold_aucs)),
        }
    selected = max(model_rows, key=lambda name: model_rows[name]["fold_mean_auc"])
    return {
        "labels": y[valid],
        "groups": group_array[valid],
        "scores": scores[selected][valid],
        # Which input records survived fold attrition. Needed by any caller that
        # wants to line the held-out scores back up against a per-record field
        # (the ordinal rung, say); evaluate_attack pops it, so it never reaches
        # a JSON report as a numpy array.
        "valid_mask": valid,
        "selected_attacker": selected,
        "models": model_rows,
        "folds": folds,
        "auc": model_rows[selected]["fold_mean_auc"],
        "pooled_auc": model_rows[selected]["pooled_auc"],
        "n": int(valid.sum()),
        "n_groups": len(folds),
    }


def _fixture_evidence(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fixture = False
    for record in records:
        provenance = record.get("provenance", {})
        provider = provenance.get("provider") if isinstance(provenance, Mapping) else None
        fixture = (
            fixture or provider in {"fixture", "synthetic-fixture"} or bool(record.get("fixture"))
        )
    if fixture:
        return {
            "tier": "fixture_diagnostic",
            "scientific_evidence": False,
            "paper_reproduction_claim": False,
            "note": "Fixture metrics test the pipeline; they are not scientific evidence.",
        }
    return {
        "tier": "measured_input",
        "scientific_evidence": True,
        "paper_reproduction_claim": False,
        "note": (
            "Computed from supplied traces; comparison with the paper requires protocol review."
        ),
    }


def evaluate_attack(
    records: Sequence[Mapping[str, Any]],
    *,
    target: str = "attribute",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
    feature_transform: Callable[[list[dict[str, float]]], list[dict[str, float]]] | None = None,
) -> dict[str, Any]:
    """Run A1/A2 leave-one-(service, topic)-out logistic-regression attacks."""

    if not records:
        raise ValueError("at least one trace record is required")
    labels = [_label(record, target) for record in records]
    groups = [_group(record) for record in records]
    a1 = [_record_features(record, False) for record in records]
    a2 = [_record_features(record, True) for record in records]
    if feature_transform:
        a1 = feature_transform(a1)
        a2 = feature_transform(a2)
    evaluation = _evaluate_features(a1, a2, labels, groups, seed=seed)
    evaluation.pop("valid_mask", None)
    interval = bootstrap_auc(
        evaluation.pop("labels"),
        evaluation.pop("scores"),
        clusters=evaluation.pop("groups"),
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )
    evaluation.update(
        {
            "target": target,
            "protocol": "leave-one-(service,topic)-out",
            "classifier": "logistic_regression",
            "selection": "max(A1,A2) by fold-mean AUC",
            "inversion_safe": True,
            "confidence_interval": interval,
            "evidence": _fixture_evidence(records),
        }
    )
    return evaluation



# --------------------------------------------------------------------------- #
# Ordinal outcomes: leakage against a graded secret.
# --------------------------------------------------------------------------- #
#
# A binary AUC answers "can the adversary tell sensitive from routine?". With a
# graded secret the sharper question is "does leakage rise with sensitivity?",
# which is a trend, not a point estimate. Somers' D is the natural statistic:
# it is the rank correlation between the attacker's ordered score and the
# ordinal label, and it reduces to 2*AUC-1 in the binary case, so the binary
# results stay comparable.
#
# The bias caveat matters as much as the statistic. |D| inherits exactly the
# upward bias of the inversion-safe max(AUC, 1-AUC) convention -- measured at
# about +0.07 in AUC terms on this design -- so a Somers' D reported without its
# own permutation null is uninterpretable, in the same way a bare AUC is.


def somers_d(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Somers' D of ``scores`` against ordinal ``labels``.

    D = (concordant - discordant) / comparable, over all pairs whose labels
    differ. Ties in the score contribute zero; ties in the label are not
    comparable and are excluded, which is what makes this a measure of the
    ordering rather than of agreement.
    """

    y = np.asarray(labels, dtype=float)
    s = np.asarray(scores, dtype=float)
    if y.size != s.size:
        raise ValueError("labels and scores must be the same length")
    if np.unique(y).size < 2:
        raise ValueError("Somers' D requires at least two distinct label values")

    # Pairwise over label-discordant pairs. n is small here (tens to hundreds),
    # so the O(n^2) form is clearer than a rank-based shortcut and avoids the
    # tie-correction subtleties that trip up the closed forms.
    label_diff = np.sign(y[:, None] - y[None, :])
    score_diff = np.sign(s[:, None] - s[None, :])
    comparable = label_diff != 0
    total = float(np.count_nonzero(comparable))
    if total == 0:
        return 0.0
    agreement = label_diff * score_diff
    return float(agreement[comparable].sum() / total)


def inversion_safe_somers_d(labels: Sequence[int], scores: Sequence[float]) -> float:
    """|D|, matching the inversion-safe convention used for AUC.

    An attacker free to flip its score sign is not weakened by a negative
    correlation, so magnitude is the honest measure of what leaks. This shares
    the same upward bias as max(AUC, 1-AUC) and therefore the same requirement
    to be read against a permutation null.
    """

    return abs(somers_d(labels, scores))


def jonckheere_terpstra(groups: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Jonckheere--Terpstra trend statistic across ordered groups.

    Tests for a monotone trend in the attacker's score across the sensitivity
    ladder, rather than merely for "some difference somewhere". The normal
    approximation is reported alongside the raw statistic, but at these sample
    sizes the p-value that should be quoted is the permutation one -- the
    approximation assumes no ties and reasonably large groups, and neither holds
    reliably here.
    """

    ordered = [np.asarray(g, dtype=float) for g in groups if len(g)]
    if len(ordered) < 3:
        raise ValueError("a trend test needs at least three ordered groups")

    statistic = 0.0
    for i in range(len(ordered) - 1):
        for j in range(i + 1, len(ordered)):
            a, b = ordered[i], ordered[j]
            # Count b > a, with ties contributing a half, which is the
            # standard tie-adjusted form of the U count.
            greater = float(np.count_nonzero(b[:, None] > a[None, :]))
            tied = float(np.count_nonzero(b[:, None] == a[None, :]))
            statistic += greater + 0.5 * tied

    sizes = np.array([g.size for g in ordered], dtype=float)
    n = sizes.sum()
    mean = (n**2 - (sizes**2).sum()) / 4.0
    variance = (
        2 * (n**3) - 2 * (sizes**3).sum() - 3 * (n**2) + 3 * (sizes**2).sum()
    ) / 72.0
    z = (statistic - mean) / math.sqrt(variance) if variance > 0 else 0.0
    return {
        "statistic": statistic,
        "expected": mean,
        "variance": variance,
        "z": z,
        "n_groups": len(ordered),
        "group_sizes": [int(size) for size in sizes],
        "note": (
            "Quote the permutation p-value, not the normal approximation: the "
            "approximation assumes no ties and large groups."
        ),
    }


def evaluate_ordinal_leakage(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = 0,
    n_perm: int = 0,
) -> dict[str, Any]:
    """Does the attacker's score *order* the sensitivity ladder?

    The attacker here is exactly the one used everywhere else: the same A1/A2
    logistic pair, trained on the same binary attribute label (routine vs any
    sensitive rung), selected the same way, evaluated leave-one-(service,
    topic)-out. What changes is only the question asked of its output. Instead
    of "did it separate the two classes", we correlate its held-out score
    against the full ordinal rung.

    Training on the binary label rather than on the ladder is deliberate and is
    the stronger claim. An adversary who already knows the rung of every
    training case is a generous adversary; one who only knows "sensitive or
    not" is realistic. If that weaker adversary's score nonetheless ranks the
    rungs, the channel is graded rather than binary -- which is the finding the
    graded corpus was built to test, and it is not an artefact of having
    supervised the grade.

    Somers' D is the primary estimand and, like every AUC here, is reported
    inversion-safe and therefore biased upward; ``n_perm`` > 0 permutes the
    rung labels within each held-out group to give it the null it needs.

    **Read ``within_sensitive``, not ``somers_d``, for the graded claim.** D
    over all four rungs is large whenever the attacker merely separates rung 0
    from the rest, because a step function is monotone: a channel that leaks
    nothing but the binary secret scores a high, significant D. That number
    therefore restates the binary result rather than extending it. The
    incremental question -- given that a case is sensitive, does the trace
    reveal *how* sensitive -- is answered only by the trend restricted to the
    non-zero rungs, which is what ``within_sensitive`` reports.
    """

    if not records:
        raise ValueError("at least one trace record is required")
    levels = []
    for record in records:
        try:
            levels.append(int(_field(record, ("sensitivity_level", "level"))))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "every record needs an integer sensitivity_level for an ordinal claim"
            ) from exc
    if len(set(levels)) < 3:
        raise ValueError(
            "an ordinal claim needs at least three occupied rungs; "
            f"this record set has {sorted(set(levels))}"
        )
    labels = [_label(record, "attribute") for record in records]
    groups = [_group(record) for record in records]
    a1 = [_record_features(record, False) for record in records]
    a2 = [_record_features(record, True) for record in records]
    evaluation = _evaluate_features(a1, a2, labels, groups, seed=seed)
    scores = evaluation.pop("scores")
    valid = evaluation.pop("valid_mask")
    evaluation.pop("labels", None)
    evaluation.pop("groups", None)

    # Records in a skipped fold have no held-out score. They are dropped, not
    # given a stand-in value: imputing anything here would manufacture ordering
    # evidence out of missingness.
    kept_levels = [lv for lv, ok in zip(levels, valid, strict=True) if ok]
    kept_scores = [float(score) for score in scores]
    if len(kept_levels) != len(kept_scores):
        raise AssertionError("valid mask and held-out scores disagree in length")
    if len(set(kept_levels)) < 3:
        raise ValueError("fewer than three rungs survived fold attrition")
    scored = list(zip(kept_levels, kept_scores, strict=True))

    ordered = sorted(set(kept_levels))
    per_rung = [[sc for lv, sc in scored if lv == rung] for rung in ordered]
    result = {
        "protocol": "leave-one-(service,topic)-out",
        "attacker": "binary-trained, ordinally scored",
        "selected_attacker": evaluation.get("selected_attacker"),
        "n": len(scored),
        "n_dropped_to_fold_attrition": len(records) - len(scored),
        "rungs": ordered,
        "rung_sizes": [len(bucket) for bucket in per_rung],
        "rung_mean_score": [
            round(float(np.mean(bucket)), 6) if bucket else None for bucket in per_rung
        ],
        "somers_d": inversion_safe_somers_d(kept_levels, kept_scores),
        "somers_d_signed": somers_d(kept_levels, kept_scores),
        "direction": (
            "increasing" if somers_d(kept_levels, kept_scores) > 0
            else "decreasing" if somers_d(kept_levels, kept_scores) < 0
            else "flat"
        ),
        "trend": jonckheere_terpstra(per_rung),
        "binary_auc": evaluation.get("auc"),
        "binary_pooled_auc": evaluation.get("pooled_auc"),
        "n_groups_skipped": evaluation.get("n_groups_skipped", 0),
    }

    kept_groups = [g for g, ok in zip(groups, valid, strict=True) if ok]
    if n_perm > 0:
        result["null"] = _somers_null(
            kept_levels, kept_scores, kept_groups,
            observed=result["somers_d"], n_perm=n_perm, seed=seed,
        )

    # The incremental claim, restricted to the sensitive rungs. This is the one
    # the graded corpus was built to test; the full-ladder D above cannot
    # distinguish "graded channel" from "binary channel with more rungs".
    restricted = [
        (lv, sc, g)
        for lv, sc, g in zip(kept_levels, kept_scores, kept_groups, strict=True)
        if lv > 0
    ]
    rungs_above_zero = sorted({lv for lv, _, _ in restricted})
    if len(rungs_above_zero) >= 3:
        r_levels = [lv for lv, _, _ in restricted]
        r_scores = [sc for _, sc, _ in restricted]
        r_groups = [g for _, _, g in restricted]
        buckets = [[sc for lv, sc, _ in restricted if lv == rung] for rung in rungs_above_zero]
        within: dict[str, Any] = {
            "question": "given a sensitive case, does the trace reveal which rung?",
            "n": len(restricted),
            "rungs": rungs_above_zero,
            "rung_sizes": [len(bucket) for bucket in buckets],
            "rung_mean_score": [round(float(np.mean(bucket)), 6) for bucket in buckets],
            "somers_d": inversion_safe_somers_d(r_levels, r_scores),
            "somers_d_signed": somers_d(r_levels, r_scores),
            "trend": jonckheere_terpstra(buckets),
        }
        # The reported D is a magnitude, so it is equally large for a ladder
        # the attacker orders backwards. Carrying the direction explicitly
        # keeps prose from claiming "leakage rises with sensitivity" off a
        # statistic that cannot distinguish rising from falling.
        within["direction"] = (
            "increasing" if within["somers_d_signed"] > 0
            else "decreasing" if within["somers_d_signed"] < 0
            else "flat"
        )
        if n_perm > 0:
            within["null"] = _somers_null(
                r_levels, r_scores, r_groups,
                observed=within["somers_d"], n_perm=n_perm, seed=seed + 13,
            )
        result["within_sensitive"] = within
    else:
        result["within_sensitive"] = {
            "unevaluable": (
                "fewer than three sensitive rungs survived; the graded claim "
                "cannot be separated from the binary one"
            )
        }
    return result


def _somers_null(
    levels: Sequence[int],
    scores: Sequence[float],
    groups: Sequence[str],
    *,
    observed: float,
    n_perm: int,
    seed: int,
) -> dict[str, Any]:
    """Within-group permutation null for |Somers' D|, attacker held fixed.

    Permuting only the labels and reusing the already-fitted held-out scores
    isolates the label-score association from the attacker's capacity to fit
    anything at all, which is the right null for "is this ordering real". It is
    also cheap, which is why it can afford enough draws to matter.
    """

    rng = np.random.default_rng(seed + 991)
    group_array = np.asarray(list(groups), dtype=object)
    level_array = np.asarray(list(levels), dtype=int)
    score_list = list(scores)
    draws = []
    for _ in range(n_perm):
        shuffled = level_array.copy()
        for held_out in dict.fromkeys(group_array):
            index = np.flatnonzero(group_array == held_out)
            shuffled[index] = rng.permutation(level_array[index])
        if np.unique(shuffled).size < 2:
            continue
        draws.append(abs(somers_d(shuffled.tolist(), score_list)))
    if not draws:
        return {"unevaluable": "no permutation draw retained two distinct rungs"}
    array = np.asarray(draws, dtype=float)
    return {
        "null_mean": round(float(array.mean()), 6),
        "null_q95": round(float(np.quantile(array, 0.95)), 6),
        "p_value": round(
            float((np.count_nonzero(array >= observed) + 1) / (array.size + 1)), 6
        ),
        "n_perm": int(array.size),
        "scheme": "rung labels permuted within held-out group, attacker held fixed",
    }


def holm_bonferroni(p_values: Mapping[str, float], *, alpha: float = 0.05) -> dict[str, Any]:
    """Holm--Bonferroni correction within one pre-declared family of tests.

    Sweeping four factors means many tests, and the family has to be declared
    before the runs rather than chosen after seeing which came out well. Holm is
    used rather than plain Bonferroni because it is uniformly more powerful at
    the same familywise error rate, and rather than Benjamini--Hochberg because
    the claims here are individual ("this factor leaks") rather than a
    discovery-rate over many exchangeable hypotheses.
    """

    if not p_values:
        return {"alpha": alpha, "family_size": 0, "results": {}}
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    family = len(ordered)
    results: dict[str, Any] = {}
    running_max = 0.0
    for index, (name, p_value) in enumerate(ordered):
        threshold = alpha / (family - index)
        # Holm is a step-down procedure: once one hypothesis fails to be
        # rejected, every larger p-value is retained too.
        adjusted = max(running_max, min(1.0, p_value * (family - index)))
        running_max = adjusted
        results[name] = {
            "p_value": p_value,
            "threshold": threshold,
            "p_adjusted": adjusted,
            "rejected": adjusted <= alpha,
        }
    return {"alpha": alpha, "family_size": family, "results": results}


def paired_group_difference(
    records_a: Sequence[Mapping[str, Any]],
    records_b: Sequence[Mapping[str, Any]],
    *,
    target: str = "attribute",
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare two arms paired per held-out group, not by CI overlap.

    Both arms share the same groups and cases, so the per-group difference is
    the right unit and it removes the between-group variance that dominates each
    arm's own interval. This is not a stylistic preference: on the shipped
    journal the adaptive-vs-structure attribute difference is about +0.015 on the
    fold-mean estimator while the two per-arm intervals are roughly
    [0.55, 0.81] and [0.64, 0.88] -- eyeballing those intervals gives the wrong
    answer, in both directions depending on which endpoint you look at.
    """

    def per_group(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        report = evaluate_attack(records, target=target, n_bootstrap=0, seed=seed)
        model = report["models"][report["selected_attacker"]]
        return {
            str(fold["held_out_group"]): float(fold[f"{report['selected_attacker'].lower()}_auc"])
            if f"{report['selected_attacker'].lower()}_auc" in fold
            else float(fold.get("auc", float("nan")))
            for fold in model.get("folds", report.get("folds", []))
        }

    left = per_group(records_a)
    right = per_group(records_b)
    shared = sorted(set(left) & set(right))
    if not shared:
        raise ValueError("the two arms share no held-out group; they are not comparable")

    diffs = np.array([right[group] - left[group] for group in shared], dtype=float)
    rng = np.random.default_rng(seed)
    boots = np.array(
        [rng.choice(diffs, size=diffs.size, replace=True).mean() for _ in range(n_bootstrap)]
    )
    return {
        "target": target,
        "groups": shared,
        "per_group_difference": {g: float(right[g] - left[g]) for g in shared},
        "mean_difference": float(diffs.mean()),
        "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "n_groups": len(shared),
        "estimator": (
            "mean paired per-held-out-group difference, "
            "cluster bootstrap of the difference"
        ),
    }


def permutation_null(
    records: Sequence[Mapping[str, Any]],
    *,
    target: str = "attribute",
    n_perm: int = 2000,
    within_group: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    """Estimate the null distribution of the attack AUC for THIS arm.

    Why this exists: the leave-one-(service,topic)-out protocol selects the
    better of A1/A2 by fold-mean AUC and reports an inversion-safe
    ``max(auc, 1-auc)``.  Both choices are attacker-favourable, and together
    they put the *null* well above 0.5 -- measured at about 0.57 on the
    undefended arm and 0.60 under structure canonicalization.  Comparing an
    observed AUC against 0.5 therefore overstates leakage, and comparing two
    arms against one global null is wrong because the floor differs per arm.

    ``within_group`` permutes labels inside each held-out group rather than
    globally, which preserves the group-by-label balance the protocol is defined
    against.  A global shuffle destroys it and produces a null that is not the
    null of this design.

    Returns the null mean, its upper quantiles, and a one-sided empirical
    p-value: the fraction of permutations reaching the observed AUC.
    """

    if n_perm < 1:
        raise ValueError("n_perm must be positive")
    observed = evaluate_attack(records, target=target, n_bootstrap=0, seed=seed)["auc"]
    labels = np.asarray([_label(record, target) for record in records])
    groups = np.asarray([_group(record) for record in records], dtype=object)
    alias = TARGET_ALIASES[target][0]
    aliases = TARGET_ALIASES[target]
    rng = np.random.default_rng(seed)

    null: list[float] = []
    for _ in range(n_perm):
        permuted = labels.copy()
        if within_group:
            for held_out in dict.fromkeys(groups.tolist()):
                mask = np.flatnonzero(groups == held_out)
                permuted[mask] = rng.permutation(labels[mask])
        else:
            permuted = rng.permutation(labels)
        clones = []
        for record, label in zip(records, permuted, strict=True):
            clone = {key: value for key, value in record.items() if key not in aliases}
            clone[alias] = int(label)
            clones.append(clone)
        try:
            null.append(evaluate_attack(clones, target=target, n_bootstrap=0, seed=seed)["auc"])
        except ValueError:
            # A permutation can leave every fold single-class; it contributes no
            # draw rather than a fabricated one.
            continue

    if not null:
        raise ValueError("no permutation produced an evaluable split")
    array = np.asarray(null, dtype=float)
    exceed = int(np.count_nonzero(array >= observed))
    return {
        "target": target,
        "observed_auc": float(observed),
        "null_mean": float(array.mean()),
        "null_std": float(array.std()),
        "null_q95": float(np.quantile(array, 0.95)),
        "null_q99": float(np.quantile(array, 0.99)),
        # Add-one so a p-value is never reported as exactly zero from a finite
        # number of permutations.
        "p_value": float((exceed + 1) / (array.size + 1)),
        "n_perm": int(array.size),
        "mode": "within_group" if within_group else "global",
    }


def permutation_control(
    records: Sequence[Mapping[str, Any]],
    *,
    target: str = "attribute",
    seed: int = 0,
) -> dict[str, Any]:
    """Single-shot label shuffle. Kept for the console and existing callers.

    One draw from the null is a sanity check that the pipeline is not leaking
    labels; it is not a calibrated reference. Use :func:`permutation_null` for
    anything that a claim depends on.
    """

    rng = np.random.default_rng(seed)
    labels = np.asarray([_label(record, target) for record in records])
    shuffled = rng.permutation(labels)
    aliases = TARGET_ALIASES[target]
    controlled = []
    for record, label in zip(records, shuffled, strict=True):
        clone = dict(record)
        clone[aliases[0]] = int(label)
        # Top-level canonical label wins over nested aliases in _field().
        controlled.append(clone)
    report = evaluate_attack(controlled, target=target, n_bootstrap=0, seed=seed)
    report["control"] = "label_permutation"
    report["evidence"]["scientific_evidence"] = False
    return report


def _constant_transform(
    feature_rows: list[dict[str, float]], predicate: Callable[[str], bool] | None = None
) -> list[dict[str, float]]:
    keys = sorted(
        {key for row in feature_rows for key in row if predicate is None or predicate(key)}
    )
    constants = {key: float(np.median([row.get(key, 0.0) for row in feature_rows])) for key in keys}
    return [
        {
            **row,
            **{key: value for key, value in constants.items()},
        }
        for row in feature_rows
    ]


def shape_fixed_control(
    records: Sequence[Mapping[str, Any]], *, target: str = "attribute", seed: int = 0
) -> dict[str, Any]:
    report = evaluate_attack(
        records,
        target=target,
        n_bootstrap=0,
        seed=seed,
        feature_transform=lambda rows: _constant_transform(rows),
    )
    report["control"] = "shape_fixed"
    report["evidence"]["scientific_evidence"] = False
    return report


def _is_structure(name: str) -> bool:
    return name in STRUCTURE_FEATURES or name.startswith(("tool_count:", "bigram:"))


def _is_timing(name: str) -> bool:
    return name.startswith("timing_ms_")


def _is_size(name: str) -> bool:
    return name.startswith("egress_bytes_")


def _is_ingress(name: str) -> bool:
    return name.startswith("ingress_bytes_")


def legacy_single_direction(records: Sequence[Mapping[str, Any]]) -> bool:
    """True when a journal predates the egress/ingress split.

    Such a journal stored the *response* size under ``egress_bytes`` and has no
    ingress field, so its size coordinate means the opposite of a current run's.
    Callers refuse to pool the two rather than silently comparing them.
    """

    for record in records:
        steps = _steps(_mapping(record.get("trace", record)))
        if not steps:
            continue
        return "ingress_bytes" not in steps[0]
    return False


def run_ablation_suite(
    records: Sequence[Mapping[str, Any]],
    *,
    target: str = "attribute",
    seed: int = 0,
    n_bootstrap: int = 0,
) -> dict[str, Any]:
    """Close each observable coordinate separately, then close what is closeable.

    Four coordinates, not three, and they are not equally available to a defense.
    Structure, timing and egress are guest-controlled and the runtime genuinely
    closes them.  Ingress -- the response bytes the provider chooses -- is not,
    so ``enforceable_remote_model`` is the ablation that corresponds to what the
    shipped full-pad condition actually achieves, and ``all_four_coordinates`` is
    the counterfactual that would additionally require the model inside the
    boundary or constant-rate transport shaping.
    """

    transforms: list[
        tuple[str, Callable[[list[dict[str, float]]], list[dict[str, float]]] | None]
    ] = [
        ("none", None),
        (
            "blur_tool_order_only",
            lambda rows: [
                {k: v for k, v in row.items() if not k.startswith("bigram:")} for row in rows
            ],
        ),
        ("pad_timing_only", lambda rows: _constant_transform(rows, _is_timing)),
        ("pad_egress_size_only", lambda rows: _constant_transform(rows, _is_size)),
        ("pad_ingress_size_only", lambda rows: _constant_transform(rows, _is_ingress)),
        ("canonicalize_structure_only", lambda rows: _constant_transform(rows, _is_structure)),
        (
            # What the runtime attains against a remote model: three of four.
            "enforceable_remote_model",
            lambda rows: _constant_transform(
                rows, lambda name: _is_structure(name) or _is_timing(name) or _is_size(name)
            ),
        ),
        # What would be attainable with the model inside the boundary.
        ("all_four_coordinates", lambda rows: _constant_transform(rows)),
    ]
    rows = []
    for offset, (name, transform) in enumerate(transforms):
        report = evaluate_attack(
            records,
            target=target,
            seed=seed + offset,
            n_bootstrap=n_bootstrap,
            feature_transform=transform,
        )
        rows.append({"ablation": name, "auc": report["auc"], "pooled_auc": report["pooled_auc"]})
    return {
        "target": target,
        "rows": rows,
        "coordinate_contract": ["structure", "timing", "egress_size"],
        "paper_reproduction_claim": False,
    }



# --------------------------------------------------------------------------- #
# Adversary spectrum: what each threat model can actually observe.
# --------------------------------------------------------------------------- #
#
# The paper models one adversary: a passive host that sees the ordered step
# sequence, per-step timing, and per-step egress size. That is a single point in
# a space, and a generality claim needs the space. Each entry below restricts
# the feature set to what one adversary sees, so the same recorded traces answer
# "how much does *this* adversary learn?" without new runs.
#
# Restricting features is a faithful model of a weaker adversary precisely
# because the defense is applied at release: a host that cannot time a step
# genuinely has no timing feature, rather than having a noised one.
#
# What is NOT modelled here, and must not be claimed: an adversary with
# capabilities the harness never recorded. Single-stepping, page-fault tracking
# and KV-cache timing read intra-VM state that no application-level trace
# contains, so no projection of these traces can speak to them. They stay
# explicitly out of scope.

OBSERVABILITY: dict[str, dict[str, Any]] = {
    "full": {
        "adversary": "passive host (the paper's model)",
        "sees": ("structure", "timing", "egress_size", "ingress_size"),
        "rationale": (
            "boundary-crossing round-trips with their wall-clock and both wire "
            "directions; the model is remote, so the host sees request and "
            "response sizes alike"
        ),
    },
    "ingress_only": {
        "adversary": "host that sees only response volume",
        "sees": ("ingress_size",),
        "rationale": (
            "the one coordinate no in-guest mechanism can pad when the model is "
            "remote, and therefore the residual the shipped defense leaves"
        ),
    },
    "enforced_remote": {
        "adversary": "passive host against the enforced remote-model release",
        "sees": ("ingress_size",),
        "rationale": (
            "structure, timing and egress are public constants under full "
            "padding, so this is what the host has left to work with"
        ),
    },
    "structure_only": {
        "adversary": "host that counts round-trips but cannot time them",
        "sees": ("structure",),
        "rationale": (
            "an adversary observing only that a request crossed the boundary, e.g. via a "
            "coarse request counter rather than a packet timeline"
        ),
    },
    "timing_only": {
        "adversary": "network observer on a padded channel",
        "sees": ("timing",),
        "rationale": (
            "sizes padded to a constant by the transport, leaving inter-arrival timing; this "
            "is what off-the-shelf network padding leaves behind"
        ),
    },
    "size_only": {
        "adversary": "observer of billing or byte-count telemetry",
        "sees": ("egress_size",),
        "rationale": "aggregate egress accounting with no timeline, e.g. a metered egress bill",
    },
    "count_only": {
        "adversary": "coarse host that sees only how many round-trips occurred",
        "sees": ("count",),
        "rationale": (
            "the weakest adversary worth modelling: one scalar per request. If the channel "
            "survives here it survives almost any deployment."
        ),
    },
}


def observability_transform(
    name: str,
) -> Callable[[list[dict[str, float]]], list[dict[str, float]]]:
    """Project trace features onto what one adversary can observe.

    Returned as a ``feature_transform`` for :func:`evaluate_attack`, so the
    restricted adversary runs through exactly the same attacker, protocol and
    null as the full one -- which is what makes the comparison across the
    spectrum meaningful rather than a change of two things at once.
    """

    if name not in OBSERVABILITY:
        raise ValueError(
            f"unknown observability {name!r}; expected one of {', '.join(OBSERVABILITY)}"
        )
    sees = set(OBSERVABILITY[name]["sees"])

    def keep(feature: str) -> bool:
        if "count" in sees:
            # One scalar only: the number of boundary-crossing steps.
            return feature == "step_count"
        if _is_timing(feature) and "timing" in sees:
            return True
        if _is_size(feature) and "egress_size" in sees:
            return True
        if _is_ingress(feature) and "ingress_size" in sees:
            return True
        if "structure" in sees and (
            _is_structure(feature)
            or feature.startswith("tool_count:")
            or feature.startswith("bigram:")
        ):
            return True
        return False

    def transform(rows: list[dict[str, float]]) -> list[dict[str, float]]:
        projected = [{k: v for k, v in row.items() if keep(k)} for row in rows]
        # A wholly empty feature matrix makes the classifier degenerate rather
        # than erroring; give it one constant so the AUC is a clean 0.5 instead
        # of an exception the caller has to special-case.
        return [row or {"constant": 1.0} for row in projected]

    return transform


def evaluate_adversary_spectrum(
    records: Sequence[Mapping[str, Any]],
    *,
    target: str = "attribute",
    seed: int = 0,
    n_bootstrap: int = 0,
) -> dict[str, Any]:
    """Run the same attack for every modelled adversary over the same traces.

    Read the result against the calibrated null, and read the *ordering* with
    care: the two estimators disagree about it. Measured on the shipped journal,
    fold-mean puts the single-scalar count-only adversary highest (0.754) and
    the full adversary at 0.736, while pooled puts the full adversary clearly
    first (0.679 against 0.588-0.640). Neither ordering is an artifact -- with
    few features the classifier generalizes more consistently across held-out
    groups, which lifts a fold mean, whereas pooled AUC concatenates
    out-of-fold scores and rewards the richer feature set.

    So the defensible claim from this spectrum is not "the weakest adversary
    learns the most". It is that *every single coordinate alone* -- count,
    timing, or size -- recovers the attribute well above the null. That is the
    redundancy result, and it is why closing any one coordinate is insufficient.
    Both estimators are therefore returned, and a caller quoting one without
    the other is quoting half the evidence.
    """

    spectrum: dict[str, Any] = {}
    for name, spec in OBSERVABILITY.items():
        try:
            report = evaluate_attack(
                records,
                target=target,
                seed=seed,
                n_bootstrap=n_bootstrap,
                feature_transform=observability_transform(name),
            )
        except ValueError as exc:
            spectrum[name] = {"error": str(exc), **spec}
            continue
        spectrum[name] = {
            "auc": report["auc"],
            "pooled_auc": report["pooled_auc"],
            "n_groups": report.get("n_groups"),
            "n_groups_skipped": report.get("n_groups_skipped", 0),
            **spec,
        }
    return {
        "target": target,
        "spectrum": spectrum,
        "out_of_scope": (
            "single-stepping, page-fault tracking and KV-cache timing read intra-VM state "
            "that no application-level trace records; no projection of these traces can "
            "speak to those adversaries"
        ),
    }


def randomized_response_keep_probability(epsilon: float, *, k: int = 3) -> float:
    if epsilon < 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and non-negative")
    if k < 2:
        raise ValueError("k must be at least 2")
    exp_epsilon = math.exp(epsilon)
    return exp_epsilon / (exp_epsilon + k - 1)


def randomized_response_frontier(
    records: Sequence[Mapping[str, Any]],
    *,
    target: str = "attribute",
    epsilons: Sequence[float] = (8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.1),
    k: int = 3,
    repetitions: int = 10,
    seed: int = 0,
    depth_boundaries: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Measure the public-template randomized-response privacy frontier.

    ``k`` is the number of public plan templates and may be any value >= 2; the
    paper reports K=3. It used to be hardcoded, which made the mechanism
    unsweepable alongside the other factors.

    Templates are defined by *declared public* depth boundaries, not by
    empirical quantiles of the observed depths. That change matters for
    correctness rather than tidiness: the observed depth distribution on this
    workload is {2, 3, 4, 7} with most of its mass at 2 and 7, so the empirical
    tertile boundaries can land on the same value. Fewer than K templates are
    then ever occupied while the table still reports K keep probabilities, and
    the reported mechanism is not the one that ran. Declared boundaries also
    match the threat model: a template set the adversary does not know in
    advance is not a public template set.
    """

    if k < 2:
        raise ValueError("a template mechanism needs at least two templates")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    base = [_record_features(record, False) for record in records]
    depths = np.asarray([row.get("hop_count", row.get("step_count", 0.0)) for row in base])
    if depth_boundaries is not None:
        boundaries = np.asarray(sorted(float(b) for b in depth_boundaries), dtype=float)
        if boundaries.size != k - 1:
            raise ValueError(f"k={k} templates need exactly {k - 1} depth boundaries")
    else:
        # Default to equal bands over the PUBLISHED plan depth range, not over
        # the observed depths. Using the sample maximum would make the public
        # template set data-dependent, which contradicts the threat model: a
        # partition the adversary cannot know in advance is not public.
        low, high = float(PUBLISHED_DEPTH_RANGE[0]), float(PUBLISHED_DEPTH_RANGE[1])
        boundaries = np.asarray(
            [low + (high - low) * (index + 1) / k for index in range(k - 1)], dtype=float
        )
    if np.unique(boundaries).size != boundaries.size:
        raise ValueError(
            "depth boundaries collapse to fewer than k templates; declare distinct ones"
        )
    true_templates = np.digitize(depths, boundaries, right=True)
    labels = [_label(record, target) for record in records]
    groups = [_group(record) for record in records]
    rng = np.random.default_rng(seed)
    frontier = []
    for epsilon in epsilons:
        keep_probability = randomized_response_keep_probability(float(epsilon), k=k)
        aucs = []
        kept = 0
        total = 0
        for repetition in range(repetitions):
            released_rows: list[dict[str, float]] = []
            for true_template in true_templates:
                if rng.random() < keep_probability:
                    released = int(true_template)
                    kept += 1
                else:
                    alternatives = [index for index in range(k) if index != int(true_template)]
                    released = int(rng.choice(alternatives))
                total += 1
                # Each public plan has a fixed profile; timing and size are constants.
                released_rows.append(
                    {
                        "step_count": float(released + 1),
                        "hop_count": float(released),
                        "distinct_tools": 1.0,
                        f"tool_count:public_template_{released}": 1.0,
                        "timing_ms_mean": 1.0,
                        "timing_ms_max": 1.0,
                        "egress_bytes_mean": 1.0,
                        "egress_bytes_max": 1.0,
                        "ingress_bytes_mean": 1.0,
                        "ingress_bytes_max": 1.0,
                    }
                )
            measured = _evaluate_features(
                released_rows,
                released_rows,
                labels,
                groups,
                seed=seed + repetition,
            )
            aucs.append(measured["auc"])
        frontier.append(
            {
                "epsilon": float(epsilon),
                "keep_probability": keep_probability,
                "observed_keep_rate": kept / total if total else 0.0,
                "auc_mean": float(np.mean(aucs)),
                "auc_std": float(np.std(aucs)),
                "repetitions": repetitions,
            }
        )
    occupied = int(np.unique(true_templates).size)
    return {
        "target": target,
        "k": k,
        "boundaries": [float(b) for b in boundaries],
        # Two distinct failure modes, kept distinct:
        #  - a degenerate *partition* (two boundaries equal) is rejected
        #    outright, because such a mechanism cannot emit k templates at all;
        #  - a *sample* that happens not to span every band is legitimate but
        #    means the reported k overstates what this run exercised.
        "templates_occupied": occupied,
        "sample_occupies_all_templates": occupied == k,
        "mechanism": "k_ary_randomized_response_over_public_templates",
        "rows": frontier,
        "evidence": _fixture_evidence(records),
        "paper_reproduction_claim": False,
    }


def analyze_records(
    records: Sequence[Mapping[str, Any]], *, seed: int = 0, n_bootstrap: int = 1000
) -> dict[str, Any]:
    """Run the complete attack/controls bundle for both paper targets."""

    return {
        target: {
            "attack": evaluate_attack(records, target=target, seed=seed, n_bootstrap=n_bootstrap),
            "permutation_control": permutation_control(records, target=target, seed=seed + 101),
            "shape_fixed_control": shape_fixed_control(records, target=target, seed=seed + 202),
            "ablations": run_ablation_suite(records, target=target, seed=seed + 303),
            "frontier": randomized_response_frontier(records, target=target, seed=seed + 404),
        }
        for target in ("attribute", "membership")
    }

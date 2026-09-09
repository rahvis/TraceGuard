#!/usr/bin/env python3
"""Regenerate the paper's tables and numeric macros from a real experiment journal.

This is the ``analysis.py`` the manuscript refers to. It closes the reproducibility
loop: a journal produced by ``traceguard experiment`` is turned into the exact
``tables/macros.tex`` and ``tables/tab_*.tex`` files the paper ``\\input``s, so the
manuscript reflects a run rather than transcribed numbers.

Usage:
    uv run python scripts/make_paper_artifacts.py \\
        --journal artifacts/<run>.jsonl --out-dir tables

Provenance is honest by construction. The generated files carry a header stating
whether they came from a scientific ``live_replication`` run or from the deterministic
``synthetic_fixture`` harness, whose numbers are diagnostic only and must never be read
as evidence that a real agent leaks. Utility and latency (Table utility) require a live
judged run; when the journal lacks them the utility table is left untouched and the
script reports that it was skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traceguard.attack import (  # noqa: E402
    PUBLISHED_DEPTH_RANGE,
    evaluate_adversary_spectrum,
    evaluate_attack,
    inversion_safe_auc,
    permutation_control,
    permutation_null,
    randomized_response_frontier,
    run_ablation_suite,
    shape_fixed_control,
)
from traceguard.attack import extract_trace_features as _features  # noqa: E402
from traceguard.config import Settings  # noqa: E402
from traceguard.experiment import load_experiment_records  # noqa: E402

CONDITIONS = ("adaptive", "structure", "full")
SPECIALTIES = ("cardiology", "oncology", "psychiatry")

# The single trace features the attribution table reports, in paper order, one
# block per observable coordinate. Both size directions appear because the model
# is remote: the request is guest-composed and closeable, the response is
# provider-chosen and not (Prop. 2). `_sum` is deliberately absent from the
# feature vector -- carrying both sum and mean would hand the attacker
# sum/mean == n and defeat the coordinate projections -- so the size and timing
# rows are the count-invariant mean and max.
ATTRIBUTION_FEATURES = [
    ("extraction passes", "hop_count"),
    ("step count", "step_count"),
    ("distinct tools", "distinct_tools"),
    ("mean step timing", "timing_ms_mean"),
    ("max step timing", "timing_ms_max"),
    ("mean egress (request)", "egress_bytes_mean"),
    ("max egress (request)", "egress_bytes_max"),
    ("mean ingress (response)", "ingress_bytes_mean"),
    ("max ingress (response)", "ingress_bytes_max"),
]


# How a journal's machine-readable provider string is typeset in the manuscript.
# "azure" is the live path; "openai" and "anthropic" appear only in archived
# journals produced before the migration and are labelled as historical so a
# regenerated table can never silently present them as the current platform.
_PROVIDER_PROSE = {
    "azure": "via the Azure OpenAI Service",
    "openai": "via the OpenAI API",
    "anthropic": "via the Anthropic API",
    "fixture": "via the deterministic offline fixture provider",
}


def _records_for(
    rows: list[dict[str, Any]],
    condition: str,
    *,
    autonomy: str | None = None,
    architecture: str | None = None,
) -> list[dict[str, Any]]:
    """Project journal rows for one arm down to what the attack consumes.

    A field absent from this projection never reaches ``evaluate_attack``, so
    every experimental factor has to be listed here as well as in the journal.
    ``autonomy`` and ``architecture`` filter the arm; passing None means "any",
    which is what reproduces the single-arm archived journals.
    """

    out = []
    for row in rows:
        if row.get("condition") != condition or row.get("status") not in (None, "completed"):
            continue
        trace = row.get("trace", {}) or {}
        if autonomy is not None and trace.get("autonomy", "bounded") != autonomy:
            continue
        if architecture is not None and trace.get("architecture", "pipeline_crew") != architecture:
            continue
        case = row.get("case", {})
        attribute = case.get("attribute_label", int(bool(case.get("sensitive"))))
        out.append(
            {
                "trace": trace,
                "service": case.get("specialty", case.get("service")),
                "topic": case.get("topic"),
                "attribute_label": attribute,
                # Ordinal grade; falls back to the binary label so an archived
                # journal with no ladder still analyses unchanged.
                "sensitivity_level": case.get(
                    "sensitivity_level", trace.get("sensitivity_level", attribute)
                ),
                "membership_label": case.get(
                    "membership_label", int(bool(case.get("canary_member")))
                ),
                "autonomy": trace.get("autonomy", "bounded"),
                "architecture": trace.get("architecture", "pipeline_crew"),
                # The task family. A field absent from this projection never
                # reaches evaluate_attack, so the two families would be
                # indistinguishable in the analysis without this line.
                "domain": case.get("domain", "prior-authorization"),
                "provenance": row.get("provenance", {}),
            }
        )
    return out


def _step_durations_s(trace: Mapping[str, Any]) -> list[float]:
    durations = []
    for step in trace.get("steps") or []:
        if not isinstance(step, Mapping):
            continue
        if isinstance(step.get("duration_s"), (int, float)):
            durations.append(float(step["duration_s"]))
        elif isinstance(step.get("duration_ms"), (int, float)):
            durations.append(float(step["duration_ms"]) / 1000.0)
    return durations


def _total_wall_s(trace: Mapping[str, Any]) -> float:
    steps = [step for step in trace.get("steps") or [] if isinstance(step, Mapping)]
    walls = [
        float(step["wall_time_s"])
        for step in steps
        if isinstance(step.get("wall_time_s"), (int, float))
    ]
    # wall_time_s is cumulative (ObservableTrace.total_duration_s); sum durations
    # only for traces that never carried wall clocks.
    return max(walls) if walls else sum(_step_durations_s(trace))


def _enforced_deadline_s(rows: list[dict[str, Any]]) -> tuple[float, str]:
    """The step deadline this journal was produced under.

    Recovered from the padded rows, whose released per-step duration *is* the
    deadline by construction. Falling back to Settings() is a last resort and
    says so, because that value describes the reader's environment rather than
    the run's.
    """

    padded = Counter()
    for row in rows:
        if row.get("condition") not in ("full", "full_pad"):
            continue
        for step in (row.get("trace") or {}).get("steps") or []:
            if isinstance(step, Mapping) and isinstance(step.get("duration_s"), (int, float)):
                padded[round(float(step["duration_s"]), 3)] += 1
    if padded:
        value, count = padded.most_common(1)[0]
        share = count / sum(padded.values())
        if share >= 0.9:
            return float(value), f"journal full-pad rows ({share:.0%} at {value:g}s)"
        # Mixed constants mean the journal spans a deadline change, and averaging
        # them would invent a deadline no run used.
        raise SystemExit(
            "journal contains full-pad rows with inconsistent step durations "
            f"({dict(padded.most_common(4))}); it spans more than one deadline "
            "and cannot be analysed as one run"
        )
    fallback = Settings().step_deadline_s
    print(
        f"WARNING: journal has no full-pad rows; the deadline-overrun figures use "
        f"the ambient TRACEGUARD_STEP_DEADLINE_MS ({fallback:g}s), which describes "
        f"this environment and not the run. Treat them as provisional."
    )
    return float(fallback), f"ambient Settings() fallback ({fallback:g}s)"


# Which completion cap each crew node is given. This mirrors the
# ``max_tokens=self.settings.max_tokens_{fast,deep}`` choices in agents.py; the
# journal records bytes and not the cap, so the mapping cannot be recovered from
# it. _ingress_cap_headroom asserts the map covers every step type present, so
# adding a node fails the build rather than silently dropping it from the claim.
_NODE_CAP_KIND = {
    "intake": "fast",
    "supervisor": "fast",
    "clinical_extraction": "fast",
    "coverage_assessment": "deep",
    "criteria_check": "fast",
    "necessity_review": "fast",
    "determination": "deep",
    "react_step": "fast",
}

# Completions are English prose, which tokenizes near four bytes per token. The
# conversion is stated in the paper because it has to be: the caps are in tokens
# and the released traces are in bytes, so a reader recomputing the claim needs
# the same constant. Four is the conservative direction for "the cap does not
# bind" -- a smaller ratio makes utilisation look higher -- so the figure is
# reported together with the assumption rather than as a bare percentage.
_BYTES_PER_TOKEN = 4.0


def _ingress_cap_headroom(
    records: list[dict[str, Any]], settings: Any
) -> dict[str, Any] | None:
    r"""How close the response size gets to its configured completion cap.

    This is the evidence for the ``max_tokens`` objection to Proposition 2, and
    it was hand-typed: the manuscript claimed the largest step type reached 53%
    of its cap, which is coverage_assessment's figure, while clinical_extraction
    -- a 320-token cap against a 945 B peak -- actually reaches about 74%. The
    worst case is the only one worth quoting, so it is computed here.
    """

    per: dict[str, list[int]] = {}
    for record in records:
        for step in (record.get("trace") or {}).get("steps") or []:
            value = step.get("ingress_bytes")
            if value is None:
                continue
            per.setdefault(str(step.get("step_type")), []).append(int(value))
    if not per:
        return None
    unmapped = sorted(set(per) - set(_NODE_CAP_KIND))
    if unmapped:
        raise SystemExit(
            f"step types with no completion-cap mapping: {unmapped}; add them to "
            "_NODE_CAP_KIND (mirroring agents.py) so the cap-headroom claim stays true"
        )
    caps = {"fast": settings.max_tokens_fast, "deep": settings.max_tokens_deep}
    worst_node, worst_pct = None, -1.0
    for node, values in per.items():
        cap_bytes = caps[_NODE_CAP_KIND[node]] * _BYTES_PER_TOKEN
        pct = 100.0 * max(values) / cap_bytes
        if pct > worst_pct:
            worst_node, worst_pct = node, pct
    flat = [v for values in per.values() for v in values]
    mean = sum(flat) / len(flat)
    variance = sum((v - mean) ** 2 for v in flat) / (len(flat) - 1)
    ordered = sorted(flat)
    mid = len(ordered) // 2
    median = (
        ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    )
    return {
        "tightest_node": worst_node,
        "tightest_pct": worst_pct,
        "bytes_per_token": _BYTES_PER_TOKEN,
        "median_bytes": median,
        "max_bytes": max(flat),
        "cv": (variance ** 0.5) / mean,
        "n_steps": len(flat),
    }


def _canonical_hop_cap(padded: list[dict[str, Any]]) -> int | None:
    """The canonical hop count this journal was produced under.

    Recovered from the padded rows for the same reason the step deadline is:
    full-pad forces the canonical plan, so its hop count *is* the cap. Reading
    Settings().canonical_research_hops instead would make a reported figure
    depend on the analyst's environment rather than on the run -- verified here
    against both journals, where all padded rows sit at exactly 7 hops.
    """

    if not padded:
        return None
    counts = Counter(
        int(_features(record["trace"], include_bigrams=False).get("hop_count", 0))
        for record in padded
    )
    value, count = counts.most_common(1)[0]
    return int(value) if count >= 0.9 * sum(counts.values()) else None


def _canary_effect(records: list[dict[str, Any]]) -> dict[str, Any]:
    """What the planted record does to the host-visible observable, in raw units.

    An AUC establishes that membership is detectable; these medians say through
    which coordinate and by how much, which is what makes the mechanism
    falsifiable. ``phi_attribute`` is reported alongside because a membership
    result is only meaningful if the two labels are actually independent -- if
    the canary were correlated with sensitivity, this would be the attribute
    channel wearing a different label.
    """

    import math
    import statistics

    buckets: dict[int, dict[str, list[float]]] = {}
    for record in records:
        label = int(record.get("membership_label", 0))
        raw = (record.get("trace") or {}).get("steps") or []
        steps = [s for s in raw if isinstance(s, Mapping)]
        bucket = buckets.setdefault(label, {"egress": [], "wall": [], "depth": []})
        bucket["egress"].append(sum(int(s.get("egress_bytes") or 0) for s in steps))
        walls = [float(s["wall_time_s"]) for s in steps
                 if isinstance(s.get("wall_time_s"), (int, float))]
        bucket["wall"].append(max(walls) if walls else sum(_step_durations_s(record["trace"])))
        bucket["depth"].append(
            sum(1 for s in steps if "extraction" in str(s.get("step_type", "")).lower())
        )
    if 0 not in buckets or 1 not in buckets:
        return {}

    def med(label: int, key: str) -> float:
        return statistics.median(buckets[label][key])

    a = sum(1 for r in records if r.get("attribute_label") and r.get("membership_label"))
    b = sum(1 for r in records if r.get("attribute_label") and not r.get("membership_label"))
    c = sum(1 for r in records if not r.get("attribute_label") and r.get("membership_label"))
    d = sum(1 for r in records if not r.get("attribute_label") and not r.get("membership_label"))
    den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    base = med(0, "egress")
    return {
        "n_absent": len(buckets[0]["egress"]),
        "n_present": len(buckets[1]["egress"]),
        "egress_absent": base,
        "egress_present": med(1, "egress"),
        "egress_delta": med(1, "egress") - base,
        "egress_pct": ((med(1, "egress") - base) / base * 100.0) if base else None,
        "wall_delta": med(1, "wall") - med(0, "wall"),
        "depth_delta": med(1, "depth") - med(0, "depth"),
        "phi_attribute": ((a * d - b * c) / den) if den else 0.0,
    }


def condition_latency_means(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """Mean end-to-end observable wall time (s) per condition from journal rows."""

    per: dict[str, list[float]] = {condition: [] for condition in CONDITIONS}
    for row in rows:
        if row.get("condition") in per and row.get("status") in (None, "completed"):
            per[row["condition"]].append(_total_wall_s(row.get("trace", {})))
    return {condition: (sum(v) / len(v) if v else None) for condition, v in per.items()}


def _auc3(value: float | None) -> str:
    return "--" if value is None else f"{float(value):.3f}"


def _auc2(value: float | None) -> str:
    return "--" if value is None else f"{float(value):.2f}"


def _single_feature_auc(records: list[dict[str, Any]], key: str, target: str) -> float | None:
    labels, scores = [], []
    label_field = f"{target}_label"
    for record in records:
        feats = _features(record["trace"], include_bigrams=False)
        if key not in feats:
            return None
        labels.append(int(record[label_field]))
        scores.append(feats[key])
    if len(set(labels)) < 2:
        return None
    return inversion_safe_auc(labels, scores)


def _hop_distributions(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-framing hop counts, as a distribution and not only a mean.

    The mean alone is misleading on this workload: the sensitive arm is bimodal
    (runs either terminate at the floor or go to the public cap, with nothing in
    between), so its mean names a depth no individual run ever takes. The
    distribution is both more honest and a stronger finding.
    """

    buckets: dict[str, list[float]] = {"routine": [], "sensitive": []}
    for record in records:
        feats = _features(record["trace"], include_bigrams=False)
        hop = feats.get("hop_count", feats.get("step_count"))
        buckets["sensitive" if record["attribute_label"] else "routine"].append(hop)

    def summarize(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"n": 0, "mean": None, "histogram": {}, "modal": None, "bimodal": False}
        counts = Counter(int(v) for v in values)
        ordered = sorted(counts.items())
        mean = sum(values) / len(values)
        # Bimodal here means the two most common depths together account for
        # essentially the whole arm while the mean falls between them.
        top = counts.most_common(2)
        bimodal = (
            len(top) == 2
            and sum(n for _, n in top) >= 0.95 * len(values)
            and min(top[0][0], top[1][0]) < mean < max(top[0][0], top[1][0])
        )
        return {
            "n": len(values),
            "mean": mean,
            "histogram": {str(k): v for k, v in ordered},
            "modal": counts.most_common(1)[0][0],
            "bimodal": bimodal,
        }

    return {name: summarize(values) for name, values in buckets.items()}


def _hop_means(records: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    dist = _hop_distributions(records)
    return dist["routine"]["mean"], dist["sensitive"]["mean"]


def _evidence_tier(records: list[dict[str, Any]]) -> str:
    for record in records:
        provider = str(record.get("provenance", {}).get("provider", "")).lower()
        if provider and "fixture" not in provider:
            return "live_replication"
    return "synthetic_fixture"


def _header(tier: str) -> str:
    if tier == "live_replication":
        note = "Generated by scripts/make_paper_artifacts.py from a live_replication run."
    else:
        note = (
            "Generated by scripts/make_paper_artifacts.py from a synthetic_fixture run. "
            "DIAGNOSTIC ONLY: these numbers test the pipeline and are NOT scientific "
            "evidence that a real agent leaks."
        )
    return f"% {note}\n"


def format_p_value(report: Mapping[str, Any]) -> str:
    """Format an empirical p-value without ever claiming exactly zero.

    A permutation test can only resolve down to 1/(n+1). Printing "0.000" for a
    p-value that is really "below the resolution of 2000 permutations" overstates
    the result, so the floor is reported as an inequality instead.

    The returned string *includes the relation operator* ("<0.0005" or "=0.521")
    so a caller writes ``$p\\pAttr{}$`` and gets valid output either way. Holding
    only the number would force the caller to hardcode "=", which renders as the
    malformed ``p=<0.0005`` in the floor case.
    """

    n_perm = int(report.get("n_perm", 0))
    p_value = float(report.get("p_value", 1.0))
    if n_perm < 1:
        return "=--"
    floor = 1.0 / (n_perm + 1)
    if p_value <= floor:
        # The floor is ATTAINED, not undercut: with zero exceedances the
        # add-one estimate is exactly 1/(n+1). Printing "<" would claim a value
        # strictly below the smallest the test can produce, so this reports
        # equality at the resolution limit and names the limit.
        return "\\le " + f"{floor:.4f}".rstrip("0").rstrip(".")
    return f"={p_value:.3f}"


def load_cached_nulls(
    path: Path | None, *, journal: Path | None = None, seed: int | None = None
) -> dict[str, Any]:
    """Read a previously computed calibrated-null cache, and verify it belongs here.

    Permuting labels 1,000+ times re-fits the whole leave-one-group-out attack
    on every draw, which takes minutes -- too slow to redo on every push. The
    nulls are therefore computed deliberately (``make paper-nulls``) and cached
    as committed evidence, while table regeneration stays fast and exactly
    reproducible from that cache.

    The cache records the journal digest and seed it came from, and this function
    now *checks* them. It previously only claimed to: ``--nulls`` defaults to
    ``tables/nulls.json``, so pointing the tool at a different journal silently
    paired that journal's AUCs with the cached journal's nulls, and every
    p-value in the table footnote was then wrong for the numbers above it. A
    calibrated null is the only thing that makes these AUCs interpretable, so a
    mismatch is refused rather than warned about.
    """

    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return {}

    cached_journal = payload.get("journal")
    cached_digest = payload.get("journal_sha256")
    cached_seed = payload.get("seed")
    if journal is not None and cached_digest:
        actual = hashlib.sha256(journal.read_bytes()).hexdigest()
        if actual != cached_digest:
            raise SystemExit(
                f"{path} holds calibrated nulls for {cached_journal or 'another journal'} "
                f"(sha256 {cached_digest[:12]}...), but the tables are being built from "
                f"{journal} (sha256 {actual[:12]}...). Every reported AUC is read against "
                "these nulls, so they cannot be carried across journals. Recompute them "
                f"with `make paper-nulls JOURNAL={journal}` (minutes), or pass "
                "--nulls /dev/null to build tables with no significance line at all."
            )
    if seed is not None and cached_seed is not None and int(cached_seed) != int(seed):
        raise SystemExit(
            f"{path} holds nulls computed at seed {cached_seed}, but --seed is {seed}. "
            "The permutation null depends on the seed; recompute with `make paper-nulls`."
        )
    return payload.get("nulls", {})


def build(
    journal: Path,
    seed: int,
    bootstrap: int,
    n_perm: int = 0,
    nulls_path: Path | None = None,
) -> dict[str, Any]:
    rows = load_experiment_records(journal)
    per_condition = {c: _records_for(rows, c) for c in CONDITIONS}
    adaptive = per_condition["adaptive"]
    if not adaptive:
        raise SystemExit("no completed adaptive records in journal")

    result: dict[str, Any] = {"tier": _evidence_tier(adaptive), "tables": {}, "macros": {}}
    M = result["macros"]

    # --- main attack table + controls ------------------------------------------
    def attack(records: list[dict[str, Any]], target: str) -> dict[str, Any] | None:
        """None for an arm this journal cannot support, rather than a traceback.

        A journal from a run still in progress legitimately has an arm with too
        few held-out groups to evaluate, and inspecting a partial journal is a
        normal thing to do. The arm reads as -- in the table; it is never
        silently filled with a number from somewhere else.
        """

        if not records:
            return None
        try:
            return evaluate_attack(records, target=target, seed=seed, n_bootstrap=bootstrap)
        except ValueError as exc:
            print(f"WARNING: {target} arm unevaluable on this journal: {exc}")
            return None

    a_attr, a_memb = attack(adaptive, "attribute"), attack(adaptive, "membership")
    s_attr = attack(per_condition["structure"], "attribute")
    s_memb = attack(per_condition["structure"], "membership")
    # The full-pad column used to be the literal string "0.500" on the grounds
    # that it is (0,0) by construction. It is also directly measurable from the
    # padded runs in this journal, and measuring an asserted number costs
    # nothing, so it is computed like every other cell.
    f_attr = attack(per_condition["full"], "attribute")
    f_memb = attack(per_condition["full"], "membership")
    perm = permutation_control(adaptive, target="attribute", seed=seed + 101)
    shp = shape_fixed_control(adaptive, target="attribute", seed=seed + 202)

    # --- calibrated per-arm nulls ---------------------------------------------
    # The protocol picks the better of A1/A2 and reports max(auc, 1-auc); both
    # are attacker-favourable, so the null sits well above 0.5 and differs per
    # arm. Every headline AUC is therefore reported against its own null.
    if n_perm > 0:
        nulls: dict[str, dict[str, Any]] = {}
        # The enforced arm is included because the padded release is now read
        # against its own null: under a remote model its observable is not
        # constant, so it has a real null rather than a degenerate one, and
        # borrowing the adaptive arm's floor would be the very error this
        # per-arm calibration exists to avoid.
        for arm, records in (
            ("adaptive", adaptive),
            ("structure", per_condition["structure"]),
            ("full", per_condition["full"]),
        ):
            for target in ("attribute", "membership"):
                if not records:
                    continue
                try:
                    nulls[f"{arm}_{target}"] = permutation_null(
                        records, target=target, n_perm=n_perm, seed=seed + 505
                    )
                except ValueError:
                    continue
        result["nulls_recomputed"] = True
    else:
        nulls = load_cached_nulls(nulls_path, journal=journal, seed=seed)
        result["nulls_recomputed"] = False
    result["nulls"] = nulls

    # Multi-split mean/std comes from the selected attacker's per-fold statistics.
    def ms(rep: dict[str, Any] | None) -> tuple[float | None, float | None]:
        if not rep:
            return None, None
        model = rep["models"][rep["selected_attacker"]]
        return model["fold_mean_auc"], model["fold_std_auc"]

    attr_ms, attr_std = ms(a_attr)
    memb_ms, memb_std = ms(a_memb)

    # ONE estimator for every headline number, and it has to be one estimator.
    #
    # These macros previously mixed them: \attrAUC took pooled_auc while
    # \memberAUC took the fold mean, and \residAttrAUC took pooled while
    # \residMemberAUC took the fold mean. The two disagree by up to 0.10 on
    # this design -- the paper devotes a passage to *why* -- so mixing them
    # across the two secrets makes the headline comparison between them
    # meaningless. On the Azure journal the mixed form printed attribute 0.689
    # against membership 0.719, reversing the true ordering: consistently
    # estimated it is 0.763 against 0.719 (fold) or 0.689 against 0.670
    # (pooled), attribute ahead either way.
    #
    # The fold mean is chosen because it is the standard cross-validation
    # estimator for this protocol and it is what the multi-split mean and
    # standard deviation already report. Pooled values are emitted alongside
    # under a `Pooled` suffix so the disagreement stays visible.
    _EST = "auc"
    for _macro, _report in (
        ("attrAUC", a_attr), ("memberAUC", a_memb),
        ("enfAttrAUC", f_attr), ("enfMemberAUC", f_memb),
        ("residAttrAUC", s_attr), ("residMemberAUC", s_memb),
    ):
        M[_macro] = _auc3(_report[_EST] if _report else None)
        M[f"{_macro}Pooled"] = _auc3(_report["pooled_auc"] if _report else None)
    # The structure-canonical membership residual. Previously unemitted because
    # membership was at its null under every condition and there was nothing to
    # report; it is now the number that shows structure canonicalization does
    # not close the membership channel, so it has to be a macro rather than a
    # sentence. Uses the fold-mean, matching \memberAUC above it, so the
    # before/after comparison in the prose is between like estimators.
    M["permAUC"] = _auc2(perm.get("auc"))
    M["shapeFixedAUC"] = _auc2(shp.get("auc"))
    M["attrMS"] = _auc2(attr_ms)
    M["attrMSstd"] = _auc2(attr_std)
    M["memberMS"] = _auc2(memb_ms)
    M["memberMSstd"] = _auc2(memb_std)

    _NULL_MACRO = {
        "adaptive_attribute": ("nullAttr", "pAttr"),
        "adaptive_membership": ("nullMember", "pMember"),
        "structure_attribute": ("nullResidAttr", "pResidAttr"),
        "structure_membership": ("nullResidMember", "pResidMember"),
        # The enforced arm needs its own null too. Reading the padded release
        # against the adaptive arm's null would be the same error the paper warns
        # about elsewhere: each arm's protocol induces its own floor, and the
        # padded arm's feature distribution is not the adaptive arm's.
        "full_attribute": ("nullEnfAttr", "pEnfAttr"),
        "full_membership": ("nullEnfMember", "pEnfMember"),
    }
    for key, (null_macro, p_macro) in _NULL_MACRO.items():
        report = nulls.get(key)
        if not report:
            continue
        M[null_macro] = _auc2(report["null_mean"])
        M[p_macro] = format_p_value(report)
    if nulls:
        # Report the count the cached nulls were actually computed at, not the
        # flag this invocation happened to be passed.
        M["nPerm"] = str(max(int(r.get("n_perm", 0)) for r in nulls.values()))

    result["tables"]["tab_main"] = _tab_main(
        a_memb, a_attr, s_memb, s_attr, f_memb, f_attr, perm, shp, nulls
    )

    # --- attribution (single feature) ------------------------------------------
    attribution = []
    for label, key in ATTRIBUTION_FEATURES:
        memb = _single_feature_auc(adaptive, key, "membership")
        attr = _single_feature_auc(adaptive, key, "attribute")
        attribution.append((label, memb, attr))
    M["attFeatEgress"] = _auc2(_single_feature_auc(adaptive, "egress_bytes_mean", "attribute"))
    # Uplink/downlink volume ratio. The request direction was unmeasured before
    # and is the larger of the two on this workload, which is why omitting it
    # understated the observable surface rather than being conservative.
    _tx = [
        sum(int(st.get("egress_bytes") or 0) for st in r["trace"]["steps"])
        for r in adaptive if (r.get("trace") or {}).get("steps")
    ]
    _rx = [
        sum(int(st.get("ingress_bytes") or 0) for st in r["trace"]["steps"])
        for r in adaptive if (r.get("trace") or {}).get("steps")
    ]
    M["txMeanBytes"] = f"{sum(_tx) / max(1, len(_tx)):.0f}"
    M["rxMeanBytes"] = f"{sum(_rx) / max(1, len(_rx)):.0f}"
    M["txRxRatio"] = f"{(sum(_tx) / max(1, len(_tx))) / max(1.0, sum(_rx) / max(1, len(_rx))):.1f}"

    # The single-scalar ingress channel per arm. This is the evidence for the
    # corrected Prop. 2: the defense cannot pad this coordinate, but it changes
    # the requests, and the response volume follows -- closing the coordinate for
    # the control-flow-carried secret and not for the content-carried one. A
    # single scalar with no classifier and no folds, so no estimator confound.
    def _ingress_scalar(rows, target):
        vals, labs = [], []
        for row in rows:
            steps = (row.get("trace") or {}).get("steps") or []
            if not steps:
                continue
            vals.append(sum(int(st.get("ingress_bytes") or 0) for st in steps))
            labs.append(int((row.get("trace") or {}).get(f"{target}_label", 0)))
        pos = [v for v, lb in zip(vals, labs, strict=True) if lb]
        neg = [v for v, lb in zip(vals, labs, strict=True) if not lb]
        if not pos or not neg:
            return None, None
        u = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
        a = u / (len(pos) * len(neg))
        return max(a, 1 - a), (sum(pos) / len(pos)) - (sum(neg) / len(neg))

    # Power and clustering, generated rather than asserted. Two facts qualify the
    # headline and a reader is entitled to both: the paired undefended-vs-padded
    # attribute delta is not distinguishable from zero, and the design's minimum
    # detectable effect is nearly the whole signal -- so "the attribute is not
    # reduced" is a failure to detect, not a demonstrated absence. And the six
    # (specialty, topic) groups are not independent: clustering at specialty, the
    # real unit, the interval is uninformative and one specialty sits at its null.
    import math as _math
    import statistics as _stats

    def _folds(rows, target):
        rep = attack(rows, target)
        return {
            d["held_out_group"]: d["attacker_favorable_auc"]
            for d in (rep.get("fold_aucs") or rep.get("folds") or [])
            if d.get("attacker_favorable_auc") is not None
        }

    _fa, _ff = _folds(adaptive, "attribute"), _folds(per_condition["full"], "attribute")
    _keys = sorted(set(_fa) & set(_ff))
    if len(_keys) >= 3:
        _d = [_fa[k] - _ff[k] for k in _keys]
        _n = len(_d)
        _sd = _stats.stdev(_d)
        _m = _stats.mean(_d)
        _t975, _t80 = 2.571, 0.941  # 5 df
        _half = _t975 * _sd / _math.sqrt(_n)
        M["pairedAttrDelta"] = f"{_m:+.3f}"
        M["pairedAttrCI"] = f"{_m - _half:+.3f}, {_m + _half:+.3f}"
        M["attrMDE"] = f"{(_t975 + _t80) * _sd / _math.sqrt(_n):.3f}"
        M["attrSignalAboveNull"] = f"{attack(adaptive, 'attribute')['auc'] - 0.599:.3f}"

    _sp = {}
    for row in adaptive:
        _sp.setdefault((row.get("trace") or {}).get("service", "?"), []).append(row)
    _svals = {k: attack(v, "attribute")["pooled_auc"] for k, v in _sp.items() if len(v) > 20}
    if len(_svals) >= 3:
        _v = list(_svals.values())
        _sm, _ssd = _stats.mean(_v), _stats.stdev(_v)
        _sh = 2.920 * _ssd / _math.sqrt(len(_v))  # 2 df
        M["specClusterCI"] = f"{_sm - _sh:.3f}, {_sm + _sh:.3f}"
        M["specWeakest"] = min(_svals, key=_svals.get)
        M["specWeakestAUC"] = _auc2(min(_svals.values()))
        M["specStrongest"] = max(_svals, key=_svals.get)
        M["specStrongestAUC"] = _auc2(max(_svals.values()))

    # NB: this loop must not bind the name `rows`. It used to, and because
    # `_records_for` projects the journal down to what the attack consumes --
    # deliberately dropping `condition` -- the outer `rows` was left holding the
    # last arm's projected records. Everything downstream that reads the whole
    # journal then saw one arm with no condition field: `_enforced_deadline_s`
    # found no full-pad rows and silently fell back to the ambient
    # TRACEGUARD_STEP_DEADLINE_MS, which is the exact failure its docstring
    # exists to prevent, and the latency means and per-node overrun counts were
    # computed over one arm.
    for arm, key in (("adaptive", "Adaptive"), ("structure", "Struct"), ("full", "Full")):
        arm_rows = per_condition.get(arm) or []
        for target, tkey in (("attribute", "Attr"), ("membership", "Memb")):
            auc, gap = _ingress_scalar(arm_rows, target)
            if auc is not None:
                M[f"ingressOnly{tkey}{key}"] = _auc2(auc)
                if target == "membership":
                    M[f"ingressGap{key}"] = f"{gap:+.0f}"
    M["attFeatIngress"] = _auc2(_single_feature_auc(adaptive, "ingress_bytes_mean", "attribute"))
    # The max variants as well. The prose, Table 3 and Figure 2 must quote the
    # same statistic per coordinate, and for the size coordinates the max is both
    # the stronger carrier and the one the figure plots; quoting the mean in prose
    # beside a figure showing the max invites a reviewer to think they disagree.
    M["attFeatEgressMax"] = _auc2(_single_feature_auc(adaptive, "egress_bytes_max", "attribute"))
    M["attFeatIngressMax"] = _auc2(
        _single_feature_auc(adaptive, "ingress_bytes_max", "attribute")
    )
    M["attFeatTimingMax"] = _auc2(_single_feature_auc(adaptive, "timing_ms_max", "attribute"))
    M["memFeatTimingMax"] = _auc2(_single_feature_auc(adaptive, "timing_ms_max", "membership"))
    M["memFeatIngressMax"] = _auc2(
        _single_feature_auc(adaptive, "ingress_bytes_max", "membership")
    )
    M["attFeatTiming"] = _auc2(_single_feature_auc(adaptive, "timing_ms_mean", "attribute"))
    M["attFeatHops"] = _auc2(_single_feature_auc(adaptive, "hop_count", "attribute"))
    result["tables"]["tab_attribution"] = _tab_attribution(attribution)

    # --- per-specialty ----------------------------------------------------------
    specialty_rows = []
    macro_by_service = {
        "cardiology": "attAUCCard",
        "oncology": "attAUCOnco",
        "psychiatry": "attAUCPsyc",
    }
    for svc in SPECIALTIES:
        subset = [r for r in adaptive if str(r["service"]).lower() == svc]
        if not subset:
            continue
        s_a = attack(subset, "attribute")
        s_m = attack(subset, "membership")
        attr_auc = s_a["pooled_auc"] if s_a else None
        specialty_rows.append(
            (svc.capitalize(), len(subset), s_m["auc"] if s_m else None, attr_auc)
        )
        M[macro_by_service[svc]] = _auc2(attr_auc)
    result["tables"]["tab_specialty"] = _tab_specialty(specialty_rows)

    # --- ablation ---------------------------------------------------------------
    abl = {
        t: run_ablation_suite(adaptive, target=t, seed=seed + 303)
        for t in ("attribute", "membership")
    }
    result["tables"]["tab_ablation"] = _tab_ablation(abl)
    canon = next((row["auc"] for row in abl["attribute"]["rows"]
                  if row["ablation"] == "canonicalize_structure_only"), None)
    M["canonOnlyAttr"] = _auc2(canon)

    # --- adversary spectrum -----------------------------------------------------
    # Same traces, same attacker, same protocol; only the observable coordinate
    # set changes. Both estimators are emitted because they disagree about the
    # ordering and either alone would mislead.
    spectrum = evaluate_adversary_spectrum(adaptive, target="attribute", seed=seed)
    result["spectrum"] = spectrum

    # The same decomposition for membership. This is not symmetry for its own
    # sake: the two secrets ride on different coordinates, and which coordinate
    # carries a secret decides whether structure canonicalization alone can
    # close it. Reporting only the attribute decomposition hid that.
    member_spectrum = evaluate_adversary_spectrum(adaptive, target="membership", seed=seed)
    result["member_spectrum"] = member_spectrum
    for key, macro in (
        ("size_only", "memberFeatSize"),
        ("timing_only", "memberFeatTiming"),
        ("structure_only", "memberFeatStructure"),
        ("count_only", "memberFeatCount"),
    ):
        row = member_spectrum["spectrum"].get(key, {})
        if row.get("auc") is not None:
            M[macro] = _auc3(row["auc"])

    # What the canary actually does to the observable, in raw units. An AUC says
    # "detectable"; these say "why", and they are what makes the mechanism
    # falsifiable rather than a story told around a number.
    delta = _canary_effect(adaptive)
    result["canary_effect"] = delta
    if delta.get("egress_delta") is not None:
        M["memberEgressDelta"] = f"{delta['egress_delta']:.0f}"
        M["memberEgressPct"] = f"{delta['egress_pct']:+.1f}"
        M["memberWallDelta"] = f"{delta['wall_delta']:.1f}"
        M["memberDepthDelta"] = f"{delta['depth_delta']:+.1f}"
        M["memberAttrPhi"] = f"{delta['phi_attribute']:+.3f}"
    result["tables"]["tab_spectrum"] = _tab_spectrum(spectrum)
    # Which restricted adversaries clear the null on BOTH estimators. The
    # manuscript claimed "every coordinate alone clears the null" as the thing
    # both estimators support; that was true of the archived run and is false
    # here, where structure alone is 0.667 on the fold mean and 0.511 pooled
    # against a null of 0.597. The scope of the claim is therefore counted
    # rather than asserted.
    null_attr = (nulls.get("adaptive_attribute") or {}).get("null_mean")
    single = ("structure_only", "timing_only", "size_only", "count_only")
    if null_attr is not None:
        both, fold_only = [], []
        _pretty = {
            "structure_only": "structure",
            "timing_only": "timing",
            "size_only": "size",
            "count_only": "round-trip count",
        }
        for name in single:
            row = spectrum["spectrum"].get(name) or {}
            fold, pooled = row.get("auc"), row.get("pooled_auc")
            if fold is None or pooled is None:
                continue
            if fold > null_attr and pooled > null_attr:
                both.append(_pretty[name])
            elif fold > null_attr:
                fold_only.append(_pretty[name])
        M["specBothCount"] = str(len(both))
        M["specSingleCount"] = str(len(single))
        M["specBothList"] = ", ".join(both) if both else "none"
        M["specFoldOnlyList"] = ", ".join(fold_only) if fold_only else "none"
        M["specAllClearBoth"] = "yes" if len(both) == len(single) else "no"

    weakest = spectrum["spectrum"].get("count_only", {})
    if weakest.get("auc") is not None:
        M["specCountAttr"] = _auc2(weakest["auc"])
        M["specCountAttrPooled"] = _auc2(weakest["pooled_auc"])
    full_row = spectrum["spectrum"].get("full", {})
    if full_row.get("auc") is not None:
        M["specFullAttr"] = _auc2(full_row["auc"])
    for key, macro in (
        ("timing_only", "specTimingAttr"),
        ("size_only", "specSizeAttr"),
        # The residual the shipped full-pad release actually leaves.
        ("ingress_only", "specIngressAttr"),
    ):
        row = spectrum["spectrum"].get(key, {})
        if row.get("auc") is not None:
            M[macro] = _auc2(row["auc"])

    # --- frontier ---------------------------------------------------------------
    fr = {
        t: randomized_response_frontier(adaptive, target=t, seed=seed + 404)
        for t in ("attribute", "membership")
    }
    result["tables"]["tab_frontier"] = _tab_frontier(fr)

    # Whether the template mechanism has anything to release. It buckets the
    # depth, so on a crew whose depth collapses onto the public cap every run
    # lands in one band: the released template is then near-constant and the
    # table shows excellent "privacy" at every epsilon for a reason that has
    # nothing to do with privacy. Read as a privacy-utility tradeoff that would
    # be misleading, so the regime is measured and named.
    bands = PUBLISHED_DEPTH_RANGE
    width = (bands[1] - bands[0]) / 3.0
    edges = (bands[0] + width, bands[0] + 2 * width)
    depths = [
        int(_features(record["trace"], include_bigrams=False).get("hop_count", 0))
        for record in adaptive
    ]
    if depths:
        occupancy = Counter(
            0 if d < edges[0] else (1 if d < edges[1] else 2) for d in depths
        )
        top_share = 100.0 * max(occupancy.values()) / len(depths)
        M["frontierTopBandShare"] = f"{top_share:.0f}"
        M["frontierBandsOccupied"] = str(len(occupancy))
        # "vacuous" when one band holds essentially everything: the mechanism
        # is then noising a constant.
        M["frontierRegime"] = "vacuous" if top_share >= 85.0 else "informative"
    # Exposed so scripts/make_paper_figures.py plots the same rows the table
    # reports, rather than a second, independently-derived set.
    result["frontier"] = {target: report["rows"] for target, report in fr.items()}
    rows_attr = fr["attribute"]["rows"]
    if rows_attr:
        M["frontierHiAttr"] = _auc2(rows_attr[0]["auc_mean"])
        M["frontierLoAttr"] = _auc2(rows_attr[-1]["auc_mean"])
        M["frontierLoEps"] = f"{rows_attr[-1]['epsilon']:.2f}"
        # The partition is a design parameter, so report its sensitivity rather
        # than presenting one choice as canonical. This is the worst partition
        # we measured: equal bands over the observed range, which isolate the
        # routine mode in a band of its own.
        worst = randomized_response_frontier(
            adaptive,
            target="attribute",
            seed=seed + 404,
            depth_boundaries=[7 / 3, 14 / 3],
        )
        if worst["rows"]:
            M["frontierPartitionWorst"] = _auc2(worst["rows"][0]["auc_mean"])

    # --- hop means --------------------------------------------------------------
    hop_dist = _hop_distributions(adaptive)
    result["hop_distributions"] = hop_dist
    routine = hop_dist["routine"]["mean"]
    sensitive = hop_dist["sensitive"]["mean"]
    M["hopCommon"] = "--" if routine is None else f"{routine:.1f}"
    M["hopSensitive"] = "--" if sensitive is None else f"{sensitive:.1f}"
    # The sensitive arm is bimodal on this workload, so the mean names a depth
    # no run takes. Emit the modes so the figure and its caption can say so
    # instead of implying a shift of a unimodal distribution.
    for name in ("routine", "sensitive"):
        entry = hop_dist[name]
        if entry["modal"] is not None:
            M[f"hop{name.capitalize()}Modal"] = str(entry["modal"])
        if entry["histogram"]:
            span = sorted(int(k) for k in entry["histogram"])
            M[f"hop{name.capitalize()}Range"] = f"{span[0]}--{span[-1]}"
    M["hopSensitiveBimodal"] = "yes" if hop_dist["sensitive"]["bimodal"] else "no"

    # The manuscript described the depth distribution as bimodal in prose and in
    # a figure caption. That was true of the run it was written against and is
    # false of a stronger crew, where the distribution collapses onto the public
    # cap instead. The shape is therefore named by the data: a caption that can
    # contradict its own figure is worse than no caption.
    cap = _canonical_hop_cap(per_condition["full"])
    all_hops = [
        h
        for arm in ("routine", "sensitive")
        for depth, count in hop_dist[arm]["histogram"].items()
        for h in [int(depth)] * int(count)
    ]
    if all_hops and cap:
        at_cap = sum(1 for h in all_hops if h >= cap)
        M["hopSaturatedPct"] = f"{100.0 * at_cap / len(all_hops):.0f}"
        M["hopCap"] = str(cap)
        if at_cap >= 0.85 * len(all_hops):
            M["hopShape"] = "saturated at the public cap"
            M["hopShapeShort"] = "saturated"
        elif hop_dist["sensitive"]["bimodal"]:
            M["hopShape"] = "bimodal, with mass at the floor and at the public cap"
            M["hopShapeShort"] = "bimodal"
        else:
            M["hopShape"] = "unimodal"
            M["hopShapeShort"] = "unimodal"

    # --- counts -----------------------------------------------------------------
    M["nBaseline"] = str(len(adaptive))
    M["nShielded"] = str(len(per_condition["full"]))
    M["nReceipts"] = str(len(per_condition["full"]))
    # Receipts across every arm, not just the shielded one: the verifier claim is
    # about all of them, and counting only the padded arm understates what was
    # actually checked.
    M["nReceiptsAll"] = str(sum(len(v) for v in per_condition.values()))
    services = {str(r["service"]).lower() for r in adaptive}
    M["nSpecialties"] = str(len(services & set(SPECIALTIES)) or len(services))

    # The design paragraph used to state the case count and the number of
    # framings in prose. Both changed when the corpus gained a graded ladder
    # (24 cases in two framings became 48 in four), so both are generated: a
    # design description that can disagree with the corpus is a defect.
    M["nCases"] = str(len({(r["service"], r["topic"], r["sensitivity_level"],
                            r["membership_label"]) for r in adaptive}))
    M["nTopicGroups"] = str(len({(r["service"], r["topic"]) for r in adaptive}))
    rungs = sorted({int(r["sensitivity_level"]) for r in adaptive})
    M["nFramings"] = str(len(rungs))
    M["framingWord"] = {2: "two", 3: "three", 4: "four", 5: "five"}.get(
        len(rungs), str(len(rungs))
    )

    # --- latency ----------------------------------------------------------------
    # Wall-clock macros come from the journal's own observable traces.  The
    # full-pad overrun rate is measured on adaptive (unpadded) step timings
    # against the deadline: padded traces release the deadline constant, so the
    # measured timings are the paper's basis for its modeled overrun figure.
    lat = condition_latency_means(rows)
    if lat["adaptive"]:
        M["latBase"] = f"{lat['adaptive']:.1f}"
        if lat["structure"] is not None:
            pct = (lat["structure"] - lat["adaptive"]) / lat["adaptive"] * 100.0
            M["latStructPct"] = f"{pct:+.0f}"
        if lat["full"] is not None:
            M["latFull"] = f"{lat['full']:.1f}"
            M["padExtraS"] = f"{lat['full'] - lat['adaptive']:.1f}"
            # The multiplier and the absolute cost tell different stories and
            # the paper needs both: the multiplier is roughly stable across
            # crew models while the absolute cost tracks the deadline, which
            # tracks the model's tail latency. Reporting only the ratio hides
            # that a more capable model makes the same defense cost more
            # wall-clock.
            M["padRatio"] = f"{lat['full'] / lat['adaptive']:.1f}"

    # Median steps per run, per arm. Canonicalizing the plan can *remove* the
    # per-pass sufficiency decisions the adaptive arm makes, which is why the
    # cheap defense can also be the faster one -- but whether it does is a
    # property of the crew, not of the defense, so it is measured rather than
    # asserted. On one journal both arms run 12 steps; on another the adaptive
    # arm runs 16.
    import statistics as _stats

    for arm, macro in (("adaptive", "stepsAdaptive"), ("structure", "stepsStructure")):
        counts = [
            len((record.get("trace") or {}).get("steps") or [])
            for record in per_condition[arm]
        ]
        if counts:
            M[macro] = f"{_stats.median(counts):.0f}"
    # The enforced deadline must come from the journal, not from the ambient
    # environment. Reading Settings() made this figure depend on whoever ran the
    # script: the same journal reported 12.6% under a 3 s default and would
    # report a different number under the 10 s deadline the live deployment
    # actually uses, which is not a reproducible statistic. Padded traces
    # release the deadline as a constant step duration, so the value the
    # runtime enforced is recoverable from the run's own full-pad rows.
    deadline, deadline_source = _enforced_deadline_s(rows)
    durations = [d for record in adaptive for d in _step_durations_s(record["trace"])]
    result["deadline_s"] = deadline
    result["deadline_source"] = deadline_source
    result["deadline_overrun_pct"] = (
        100.0 * sum(1 for d in durations if d > deadline) / len(durations) if durations else None
    )

    # Which node overruns matters more than the aggregate: the padding cost is
    # not spread evenly, and reporting one prose constant hides that.
    per_node: dict[str, dict[str, Any]] = {}
    ceiling = Settings().egress_ceiling_bytes
    max_egress = 0
    over_half_ceiling = 0
    over_ceiling = 0
    total_steps = 0
    for record in adaptive:
        for step in (record["trace"].get("steps") or []):
            name = str(step.get("step_type", "?"))
            entry = per_node.setdefault(name, {"n": 0, "over": 0})
            entry["n"] += 1
            total_steps += 1
            duration = step.get("duration_s")
            if duration is None and step.get("duration_ms") is not None:
                duration = float(step["duration_ms"]) / 1000.0
            if duration is not None and float(duration) > deadline:
                entry["over"] += 1
            egress = int(step.get("egress_bytes") or 0)
            max_egress = max(max_egress, egress)
            if egress > ceiling:
                over_ceiling += 1
            # The paper's justification for the ceiling is that the tighter one
            # it started from would have refused real steps. That is a claim
            # about a counterfactual ceiling, so count it here rather than
            # leaving a hand-typed percentage in the prose -- it was typed from
            # a 48-case pilot (0.7%) and the headline journal makes it 17%.
            if egress > ceiling // 2:
                over_half_ceiling += 1
    for entry in per_node.values():
        entry["overrun_pct"] = 100.0 * entry["over"] / entry["n"] if entry["n"] else None
    result["deadline_overrun_by_step_type"] = dict(
        sorted(per_node.items(), key=lambda kv: -(kv[1]["overrun_pct"] or 0.0))
    )
    # The egress ceiling is part of the advertised mechanism, so whether it ever
    # actually binds on this workload is a fact the paper should state.
    result["egress"] = {
        "ceiling_bytes": ceiling,
        "max_observed_bytes": max_egress,
        "steps_over_ceiling": over_ceiling,
        "steps_over_half_ceiling": over_half_ceiling,
        "n_steps": total_steps,
        "ceiling_binds": over_ceiling > 0,
    }
    M["egressMaxObserved"] = str(max_egress)
    M["egressCeiling"] = str(ceiling)
    M["egressHalfCeiling"] = str(ceiling // 2)
    headroom = _ingress_cap_headroom(adaptive, Settings())
    if headroom:
        result["ingress_cap_headroom"] = headroom
        M["ingressTightestNode"] = str(headroom["tightest_node"]).replace("_", "\\_")
        M["ingressTightestPct"] = f"{headroom['tightest_pct']:.0f}"
        M["ingressBytesPerToken"] = f"{headroom['bytes_per_token']:g}"
        M["ingressMedian"] = f"{headroom['median_bytes']:.0f}"
        M["ingressMax"] = str(headroom["max_bytes"])
        M["ingressCV"] = f"{headroom['cv']:.2f}"
    M["egressStepsN"] = str(total_steps)
    if total_steps:
        M["egressOverHalfPct"] = f"{100.0 * over_half_ceiling / total_steps:.1f}"
    M["stepDeadline"] = f"{deadline:g}"
    if result["deadline_overrun_pct"] is not None:
        M["overrunPct"] = f"{result['deadline_overrun_pct']:.1f}"
    # Name the node the padding cost actually falls on. Reporting only the
    # aggregate hides that it is one step type, which is the actionable part.
    worst = next(iter(result["deadline_overrun_by_step_type"].items()), None)
    if worst and worst[1]["overrun_pct"]:
        M["overrunNode"] = worst[0].replace("_", "\\_")
        M["overrunNodePct"] = f"{worst[1]['overrun_pct']:.1f}"

    # --- crew models (checked against \crewModel in main, warn-only) -------------
    models: set[str] = set()
    providers: set[str] = set()
    for row in rows:
        provenance = row.get("provenance") or {}
        row_models = provenance.get("models")
        if isinstance(row_models, Mapping):
            models.update(str(value) for value in row_models.values() if value)
        if provenance.get("provider"):
            providers.add(str(provenance["provider"]))
    result["crew_models"] = sorted(models)
    result["crew_providers"] = sorted(providers)

    # \crewModel was hand-declared in macros.tex and only *warned* about when it
    # disagreed with the journal, on a code path that returned 0 regardless --
    # so a stale crew name was invisible to CI. Generate it, for the same reason
    # \crewApi is generated: a manuscript should not be able to name a model it
    # did not run. A heterogeneous crew is named in full rather than collapsed to
    # its first member, because "which model" is a claim the paper makes.
    if len(result["crew_models"]) == 1:
        M["crewModel"] = result["crew_models"][0]
    elif len(result["crew_models"]) == 2:
        M["crewModel"] = " and ".join(result["crew_models"])
    elif result["crew_models"]:
        M["crewModel"] = (
            ", ".join(result["crew_models"][:-1]) + ", and " + result["crew_models"][-1]
        )

    # The platform name used to be hardcoded prose sitting next to the
    # generated \crewModel macro, which is exactly how a manuscript ends up
    # naming a provider it did not run on. Derive it from the journal instead.
    if len(providers) == 1:
        M["crewApi"] = _PROVIDER_PROSE.get(
            next(iter(providers)), next(iter(providers))
        )
    elif providers:
        M["crewApi"] = "multiple provider backends (" + ", ".join(sorted(providers)) + ")"

    return result


# Which macros are worth carrying from a superseded journal, and what they are
# called when they arrive. The manuscript's central new claim is a comparison
# between a weaker and a stronger crew, so both sides of every such sentence
# have to be generated -- otherwise the historical half becomes a literal that
# no longer tracks the journal it came from, which is the defect the generated
# macros exist to prevent.
HISTORICAL_MACROS = {
    "memberAUC": "histMemberAUC",
    "nullMember": "histNullMember",
    "attrAUC": "histAttrAUC",
    "attrMS": "histAttrMS",
    "memberEgressDelta": "histMemberEgressDelta",
    "memberEgressPct": "histMemberEgressPct",
    "hopSaturatedPct": "histHopSaturatedPct",
    "hopShapeShort": "histHopShape",
    "frontierHiAttr": "histFrontierHiAttr",
    "frontierPartitionWorst": "histFrontierPartitionWorst",
    "frontierRegime": "histFrontierRegime",
    "crewModel": "histCrewModel",
    "crewApi": "histCrewApi",
    "latBase": "histLatBase",
    "latFull": "histLatFull",
    "padRatio": "histPadRatio",
    "stepDeadline": "histStepDeadline",
}


def historical_macros(
    journal: Path, seed: int, nulls_path: Path | None
) -> dict[str, str]:
    """Re-run the generator over a superseded journal, under a ``hist`` prefix.

    Only the handful of values the manuscript actually compares across crews is
    carried over, so this does not quietly reintroduce a second set of headline
    numbers.
    """

    prior = build(journal, seed, bootstrap=0, n_perm=0, nulls_path=nulls_path)
    macros = prior["macros"]
    return {
        new_name: macros[old_name]
        for old_name, new_name in HISTORICAL_MACROS.items()
        if old_name in macros
    }


def gen_macros(journal: Path, seed: int, bootstrap: int) -> dict[str, str]:
    """Cross-model generality AUC macros from a second (e.g. gpt-4o crew) journal."""

    adaptive = _records_for(load_experiment_records(journal), "adaptive")
    if not adaptive:
        raise SystemExit("no completed adaptive records in --gen-journal")
    attr = evaluate_attack(adaptive, target="attribute", seed=seed, n_bootstrap=bootstrap)
    memb = evaluate_attack(adaptive, target="membership", seed=seed, n_bootstrap=bootstrap)
    return {"genAttrAUC": _auc2(attr["pooled_auc"]), "genMemberAUC": _auc2(memb["auc"])}


# --- LaTeX table emitters (match the committed table style) --------------------
#
# LaTeX skeletons carry literal braces, so they are built with plain string
# concatenation (never f-strings) to avoid brace-escaping mistakes.

# One table vocabulary for the whole manuscript. Nine generated tables had
# drifted to three different column separations (3pt, 5pt, 6pt) and read as
# three different designs on facing pages; the widest table sets the floor, and
# everything else matches it. tabcolsep is now advisory: a caller may narrow it
# for a table that would otherwise overrun the column, and nothing else.
TABLE_COLSEP_PT = 5
TABLE_ARRAYSTRETCH = "1.15"


_OPEN = (
    "{\\setlength{\\tabcolsep}{%dpt}"
    "\\renewcommand{\\arraystretch}{" + TABLE_ARRAYSTRETCH + "}\n"
)
_CLOSE = "\\bottomrule\n\\end{tabular}}\n"


def _wrap(tabcolsep: int, colspec: str, header: str, body: str) -> str:
    """Wrap a generated table body in the shared house style.

    ``tabcolsep`` is clamped to the house value unless a caller genuinely needs
    it tighter (a seven-column table does), so a table cannot quietly adopt a
    different look than the one beside it.
    """

    tabcolsep = min(tabcolsep, TABLE_COLSEP_PT)
    return (
        (_OPEN % tabcolsep)
        + "\\begin{tabular}{@{}" + colspec + "@{}}\n\\toprule\n"
        + header
        + "\\midrule\n"
        + body
        + ("" if body.endswith("\n") else "\n")
        + _CLOSE
    )


# A \multicolumn over an `l` column cannot break a line, so a long footnote runs
# straight off the column: the utility table's "Full pad vs baseline" line, at
# 117 characters, overflowed by 128pt -- about 1.8in into the margin. Splitting
# at a character budget keeps every emitted row inside the column without
# needing a p{} width, which cannot be computed from inside the cell.
_MC_CHARS = 64


def _mc(text: str, limit: int = _MC_CHARS) -> str:
    r"""One or more full-width footnote rows, each short enough to fit.

    Splits on whitespace only, so a number or a macro is never broken, and it
    measures the *typeset* length by discounting control sequences and math
    delimiters -- counting "$p\approx0.003$" as 14 characters rather than 4
    would wrap far too early.
    """

    def visible(chunk: str) -> int:
        return len(re.sub(r"\\[a-zA-Z]+\s?|[{}$]", "", chunk))

    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and visible(candidate) > limit:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(
        "\\multicolumn{4}{@{}l@{}}{\\footnotesize " + line + "}\\\\"
        for line in lines
    )


def _tab_main(a_m, a_a, s_m, s_a, f_m, f_a, perm, shp, nulls) -> str:
    def cell(rep, key="auc"):
        """Fold mean by default, matching the headline macros.

        This defaulted to pooled_auc while two of the six cells passed "auc"
        explicitly, so the membership row read 0.566 (fold) against 0.504
        (pooled) and appeared to show a drop nearly twice the real one.
        """

        return _auc3(rep.get(key) if rep else None)

    # Every cell measured, including full pad.
    memb = (
        "Membership & " + cell(a_m) + " & " + cell(s_m)
        + " & " + cell(f_m) + " \\\\"
    )
    attr = "Attribute  & " + cell(a_a) + " & " + cell(s_a) + " & " + cell(f_a) + " \\\\"
    ms_a = a_a["models"][a_a["selected_attacker"]] if a_a else {}
    ms_m = a_m["models"][a_m["selected_attacker"]] if a_m else {}
    # Kept short on purpose: in a two-column layout the widest of these
    # \multicolumn lines sets the whole tabular's width, so a verbose footnote
    # silently stretches the data columns and leaves them looking sparse.
    ms = _mc(
        "Multi-split: memb.\\ $%s\\pm%s$, attr.\\ $%s\\pm%s$"  # noqa: UP031
        % (_auc3(ms_m.get("fold_mean_auc")), _auc3(ms_m.get("fold_std_auc")),
           _auc3(ms_a.get("fold_mean_auc")), _auc3(ms_a.get("fold_std_auc")))
    )
    ctrl = _mc("Controls: permutation %s, shape-fixed %s"  # noqa: UP031
               % (_auc3(perm.get("auc")), _auc3(shp.get("auc"))))
    # The calibrated null is the reference an AUC must be read against; the
    # protocol's floor is not 0.5, so printing it beside the estimates is the
    # difference between an interpretable table and a misleading one.
    null_bits = []
    for key, label in (("adaptive_attribute", "attr."), ("adaptive_membership", "memb.")):
        report = nulls.get(key)
        if report:
            null_bits.append(
                "%s $%s$ ($p%s$)"  # noqa: UP031
                % (label, _auc3(report["null_mean"]), format_p_value(report))
            )
    null_line = (
        _mc("Calibrated null: " + ", ".join(null_bits)) if null_bits else None
    )
    header = (
        "Attack target & Undefended & Struct.\\ canon. & Full pad \\\\\n"
        "              & (grouped)  & (residual)      &          \\\\\n"
    )
    body = memb + "\n" + attr + "\n\\midrule\n" + ms + "\n" + ctrl
    if null_line:
        body += "\n" + null_line
    return _wrap(5, "lccc", header, body)


_SPECTRUM_ORDER = (
    "full",
    "structure_only",
    "timing_only",
    "size_only",
    "ingress_only",
    "count_only",
)
_SPECTRUM_LABELS = {
    "full": "Full trace (all four coordinates)",
    "structure_only": "Structure only",
    "timing_only": "Timing only",
    "size_only": "Egress (request) size only",
    # Not a weaker hypothetical: this is exactly what the shipped full-pad
    # release leaves an adversary, since the other three are public constants.
    "ingress_only": "Ingress (response) size only",
    "count_only": "Round-trip count only",
}


def _tab_spectrum(spectrum: Mapping[str, Any]) -> str:
    rows = []
    for name in _SPECTRUM_ORDER:
        entry = spectrum.get("spectrum", {}).get(name)
        if not entry or entry.get("auc") is None:
            continue
        rows.append(
            "%s & %s & %s \\\\"  # noqa: UP031
            % (_SPECTRUM_LABELS[name], _auc3(entry["auc"]), _auc3(entry["pooled_auc"]))
        )
    header = (
        "Observable to the adversary & Fold-mean & Pooled \\\\\n"
        "                            & AUC       & AUC    \\\\\n"
    )
    return _wrap(6, "lcc", header, "\n".join(rows))


def _tab_attribution(rows) -> str:
    body = "\n".join(
        "%-17s & %s & %s \\\\" % (label, _auc3(memb), _auc3(attr))  # noqa: UP031
        for label, memb, attr in rows
    )
    header = "Single trace feature & Memb.\\ AUC & Attr.\\ AUC \\\\\n"
    return _wrap(6, "lcc", header, body)


def _tab_specialty(rows) -> str:
    body = "\n".join(
        "%-10s & %d & %s & %s \\\\" % (name, n, _auc3(memb), _auc3(attr))  # noqa: UP031
        for name, n, memb, attr in rows
    )
    header = "Service & $n$ & Memb.\\ AUC & Attr.\\ AUC \\\\\n"
    return _wrap(6, "lccc", header, body)


_ABLATION_LABELS = {
    "none": "none (undefended)",
    "blur_tool_order_only": "blur tool order only",
    "pad_timing_only": "pad timing only",
    "pad_egress_size_only": "pad egress (request) only",
    "pad_ingress_size_only": "pad ingress (response) only$^{\\dagger}$",
    "canonicalize_structure_only": "canonicalize structure only",
    # What the shipped runtime attains against a remote model: the three
    # guest-composed coordinates, leaving the provider-chosen one live.
    "enforceable_remote_model": "\\textbf{all three closeable} (shipped)",
    # The in-boundary counterfactual. Chance by construction, not by measurement.
    "all_four_coordinates": "all four$^{\\dagger}$ (model in-boundary)",
}


def _tab_ablation(abl) -> str:
    memb = {r["ablation"]: r["auc"] for r in abl["membership"]["rows"]}
    lines = []
    for row in abl["attribute"]["rows"]:
        name = _ABLATION_LABELS.get(row["ablation"], row["ablation"])
        lines.append(
            "%-33s & %s & %s \\\\"  # noqa: UP031
            % (name, _auc3(memb.get(row["ablation"])), _auc3(row["auc"]))
        )
    header = "Coordinate closed alone & Memb.\\ AUC & Attr.\\ AUC \\\\\n"
    lines.append("\\midrule")
    lines.append(
        "\\multicolumn{3}{@{}p{0.92\\linewidth}@{}}{\\footnotesize $^{\\dagger}$Not "
        "attainable by any in-guest mechanism against a remote model "
        "(Prop.~\\ref{prop:remote}); shown as the counterfactual a confidential "
        "GPU or constant-rate transport would buy.}\\\\"
    )
    return _wrap(5, "lcc", header, "\n".join(lines))


def _tab_frontier(fr) -> str:
    memb = {round(r["epsilon"], 4): r["auc_mean"] for r in fr["membership"]["rows"]}
    lines = []
    for row in fr["attribute"]["rows"]:
        eps = row["epsilon"]
        lines.append(
            "%.2f & %.2f & %s & %s \\\\"  # noqa: UP031
            % (eps, row["keep_probability"], _auc3(memb.get(round(eps, 4))), _auc3(row["auc_mean"]))
        )
    header = "$\\eps$ & keep prob. & Memb.\\ AUC & Attr.\\ AUC \\\\\n"
    return _wrap(6, "cccc", header, "\n".join(lines))


def _emit_macros(macros: dict[str, str]) -> str:
    lines = [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()]
    return "\n".join(lines) + "\n"


_MACRO_RE = re.compile(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}")


def read_macros(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {m.group(1): m.group(2) for m in _MACRO_RE.finditer(path.read_text(encoding="utf-8"))}


_VALID_MACRO_NAME = re.compile(r"^[A-Za-z]+$")


def assert_macro_names_are_latex_safe(names: Sequence[str]) -> None:
    r"""Refuse a macro name TeX cannot parse.

    A digit in a control-sequence name is not an error TeX reports usefully:
    \newcommand{\gradeEgressL0}{...} makes it read \gradeEgressL followed by a
    stray "0", which surfaces as "Missing \begin{document}" from inside
    macros.tex and buries the actual cause. Cheaper to refuse here.
    """

    bad = sorted(n for n in names if not _VALID_MACRO_NAME.match(n))
    if bad:
        raise SystemExit(
            "macro names must be letters only (TeX cannot parse a digit in a "
            f"control sequence): {', '.join(bad)}"
        )


def merge_macros(
    path: Path,
    updates: Mapping[str, str],
    header: str,
    *,
    drop: Sequence[str] = (),
) -> None:
    r"""Merge generated macros over the existing macros.tex.

    Hand-maintained entries (and macros produced by the other pipeline stage,
    e.g. the judged utility numbers) are preserved rather than clobbered.

    ``drop`` removes names outright, and exists because preservation is the
    wrong default for a macro whose input is missing. When no calibrated nulls
    are available, merging left the *previous* journal's \nullMember and
    \pMember in place, so the manuscript quietly kept a null belonging to
    different data -- exactly the failure the cache's digest check now refuses,
    reappearing through the escape hatch that bypasses it. A missing number
    must go missing.
    """

    merged = read_macros(path)
    merged.update(updates)
    for name in drop:
        merged.pop(name, None)
    assert_macro_names_are_latex_safe(list(merged))
    path.write_text(header + _emit_macros(merged), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--journal", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20260710)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument(
        "--permutations",
        type=int,
        default=0,
        help=(
            "recompute the calibrated per-arm nulls with this many label "
            "permutations and refresh the --nulls cache (slow: minutes). "
            "Default 0 reads the cache instead, which is what CI does."
        ),
    )
    ap.add_argument(
        "--nulls",
        type=Path,
        default=Path("tables/nulls.json"),
        help="calibrated-null cache to read (or rewrite with --permutations)",
    )
    ap.add_argument(
        "--macros-only",
        action="store_true",
        help="regenerate macros.tex but leave existing table files untouched",
    )
    ap.add_argument(
        "--historical-journal",
        type=Path,
        default=None,
        help=(
            "superseded journal to carry a few comparison macros from, under a "
            "'hist' prefix (e.g. the weaker-crew run the manuscript compares against)"
        ),
    )
    ap.add_argument(
        "--historical-nulls",
        type=Path,
        default=None,
        help="calibrated-null cache belonging to --historical-journal",
    )
    ap.add_argument(
        "--gen-journal",
        type=Path,
        default=None,
        help="cross-model generality journal; merges \\genAttrAUC and \\genMemberAUC",
    )
    args = ap.parse_args()

    result = build(
        args.journal, args.seed, args.bootstrap, args.permutations, args.nulls
    )
    if args.permutations > 0 and result.get("nulls"):
        args.nulls.parent.mkdir(parents=True, exist_ok=True)
        args.nulls.write_text(
            json.dumps(
                {
                    "_comment": (
                        "Calibrated per-arm permutation nulls. Regenerate with "
                        "`make paper-nulls`. Committed because recomputing takes "
                        "minutes; the digest below makes a stale cache detectable."
                    ),
                    "journal": str(args.journal),
                    "journal_sha256": hashlib.sha256(
                        args.journal.read_bytes()
                    ).hexdigest(),
                    "seed": args.seed,
                    "permutations": args.permutations,
                    "nulls": result["nulls"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote calibrated nulls -> {args.nulls}")
    if args.gen_journal:
        result["macros"].update(gen_macros(args.gen_journal, args.seed, args.bootstrap))
    if args.historical_journal:
        hist = historical_macros(
            args.historical_journal, args.seed, args.historical_nulls
        )
        result["macros"].update(hist)
        print(
            f"carried {len(hist)} historical macros from "
            f"{args.historical_journal} under the 'hist' prefix"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    header = _header(result["tier"])

    written = []
    if not args.macros_only:
        for name, body in result["tables"].items():
            path = args.out_dir / f"{name}.tex"
            path.write_text(header + body, encoding="utf-8")
            written.append(path.name)

    # Merge generated macros over the existing macros.tex so hand-maintained entries
    # (e.g. utility, which needs a judged run) are preserved.
    macros_path = args.out_dir / "macros.tex"
    if not result["crew_models"]:
        # No provenance at all means the manuscript would keep whatever
        # \crewModel happens to be sitting in macros.tex. Refuse instead.
        raise SystemExit(
            f"{args.journal} records no crew models in provenance; \\crewModel "
            "cannot be generated and a stale value must not be inherited"
        )
    # Null-derived macros are dropped when this run produced no nulls, rather
    # than inherited from whatever journal last wrote the file.
    null_macros = (
        "nullAttr", "pAttr", "nullMember", "pMember",
        "nullResidAttr", "pResidAttr", "nullResidMember", "pResidMember", "nPerm",
    )
    stale_nulls = () if result.get("nulls") else null_macros
    if stale_nulls:
        print(
            "note: no calibrated nulls for this journal, so "
            f"{len(stale_nulls)} null/p macros are being REMOVED rather than "
            "inherited. Every AUC in the manuscript is read against these, so "
            "run `make paper-nulls JOURNAL=<journal>` before building the PDF."
        )
    merge_macros(macros_path, result["macros"], header, drop=stale_nulls)
    written.append("macros.tex")

    print(f"provenance tier: {result['tier']}")
    print(f"wrote: {', '.join(written)} -> {args.out_dir}")
    if result["deadline_overrun_pct"] is not None:
        print(
            f"full-pad deadline overrun: {result['deadline_overrun_pct']:.1f}% of measured "
            f"steps exceeded the {result['deadline_s']:g}s deadline"
        )
    print("note: tab_utility (judge scores) comes from the judged run, not this journal; "
          "regenerate it offline with scripts/judge_utility.py --analyze-only.")
    if result["tier"] != "live_replication":
        print("WARNING: fixture output is diagnostic only and is NOT scientific evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Attacker-capacity ladder: how much of the trace channel is actually reachable?

Why this exists. The manuscript's headline attribute AUC and the strongest single
trace feature are the same number to three decimals. That is not a coincidence to
be buried: it says the deployed attacker extracts nothing a univariate threshold
does not already get, and it makes "a stronger adversary can only do better" an
argument rather than a measurement. This script turns it into a measurement by
scoring a ladder of attackers of increasing capacity against the SAME protocol.

The protocol is copied, not reinvented, because a ladder evaluated differently
from the headline would not be comparable to it:

  * leave-one-(specialty, topic)-group-out, the same folds, with the same
    skip-a-fold-that-lacks-both-classes rule and the same attrition accounting;
  * ``inversion_safe_auc`` per fold, so a systematically inverted classifier
    counts as a success;
  * the headline figure is the FOLD MEAN, which is what \\attrAUC{} reports;
  * a within-group permutation null per attacker, because a more flexible model
    has a higher null and comparing it against the linear model's floor would
    manufacture a gain.

That last point is the one that makes this honest. A gradient-boosted model can
fit group structure a linear model cannot, so its null sits higher. Reporting
each attacker against its own calibrated null is the only comparison that means
anything.

Nothing here contacts a provider or the confidential VM: it is reanalysis of a
journal already on disk. Outputs are written under iclr2027/ so the shared
tables/ that the other manuscripts read cannot change.

Usage:
  python3 scripts/make_attacker_ladder.py --journal artifacts/cvm-honest.jsonl \\
      --out-dir iclr2027/tables --permutations 2000 --seed 20260710
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import multiprocessing as _mp
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from traceguard.attack import (  # noqa: E402
    _egress_bytes,
    _group,
    _ingress_bytes,
    _label,
    _steps,
    extract_trace_features,
    inversion_safe_auc,
)
from traceguard.attack import _fit_scores as _attack_fit_scores  # noqa: E402
from make_paper_artifacts import _records_for, load_experiment_records  # noqa: E402

# Step types are a closed vocabulary fixed by the crew's plan, so a stable order
# gives every run the same column layout regardless of which types it realised.
STEP_TYPES = [
    "intake",
    "clinical_extraction",
    "coverage_assessment",
    "criteria_check",
    "necessity_review",
    "determination",
]


def _duration_s(step: Mapping[str, Any]) -> float:
    for key in ("duration_s", "wall_time_s", "duration"):
        value = step.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def positional_features(trace: Any, *, max_steps: int) -> dict[str, float]:
    """Per-step-index features: what the aggregate view throws away.

    The aggregate features carry means and maxima over the whole trace, so they
    are invariant to WHERE in the plan a value occurred. A secret that shows up
    as "the third extraction pass is long" is visible here and invisible there.
    Sequences are right-padded with zeros and the true length is carried
    explicitly, so a model can condition on it.
    """

    steps = _steps(trace)
    out: dict[str, float] = {"seq_len": float(len(steps))}
    for index in range(max_steps):
        step = steps[index] if index < len(steps) else None
        present = step is not None
        out[f"p{index}_present"] = 1.0 if present else 0.0
        out[f"p{index}_dur"] = _duration_s(step) if present else 0.0
        out[f"p{index}_egress"] = _egress_bytes(step) if present else 0.0
        out[f"p{index}_ingress"] = _ingress_bytes(step) if present else 0.0
        kind = str(step.get("step_type", "")) if present else ""
        for name in STEP_TYPES:
            out[f"p{index}_is_{name}"] = 1.0 if kind == name else 0.0
    return out


def max_egress(trace: Any) -> float:
    steps = _steps(trace)
    return max((_egress_bytes(step) for step in steps), default=0.0)


# ---------------------------------------------------------------- attackers


def _fit_predict(model, train_x, y_train, test_x, seed: int) -> np.ndarray:
    if len(np.unique(y_train)) < 2:
        return np.full(test_x.shape[0], float(y_train.mean()))
    model.fit(train_x, y_train)
    return np.asarray(model.predict_proba(test_x)[:, 1], dtype=float)


def _dense(feature_dicts, train_idx, test_idx):
    vec = DictVectorizer(sparse=False)
    train = vec.fit_transform([feature_dicts[i] for i in train_idx])
    test = vec.transform([feature_dicts[i] for i in test_idx])
    return train, test


def _lr(seed: int):
    return make_pipeline(
        StandardScaler(with_mean=True),
        LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=seed, solver="liblinear"
        ),
    )


def _hgb(seed: int):
    # Small and heavily regularised on purpose: n is 191 with six independent
    # groups, so an unconstrained booster memorises the group and the
    # leave-one-group-out score collapses. These settings were fixed before
    # looking at any held-out score.
    return HistGradientBoostingClassifier(
        max_depth=3,
        max_iter=100,
        learning_rate=0.05,
        min_samples_leaf=10,
        l2_regularization=1.0,
        random_state=seed,
    )


def _mlp(seed: int):
    return make_pipeline(
        StandardScaler(with_mean=True),
        MLPClassifier(
            hidden_layer_sizes=(16,),
            alpha=1e-2,
            max_iter=3000,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=seed,
        ),
    )


# Each entry: (key, label, representation, constructor or None for unfitted)
LADDER = [
    ("A0", "max request size alone (no fitting)", "univariate", None),
    ("A1", "logistic regression, aggregate features", "aggregate", _lr),
    ("A2", "logistic regression, aggregates + tool bigrams", "aggregate_bigram", _lr),
    ("A3", "gradient boosting, aggregate features", "aggregate_bigram", _hgb),
    ("A4", "gradient boosting, per-step sequence", "positional", _hgb),
    ("A5", "neural network, per-step sequence", "positional", _mlp),
]


def _representations(records: Sequence[Mapping[str, Any]], max_steps: int):
    traces = [record.get("trace", record) for record in records]
    return {
        "aggregate": [extract_trace_features(t, include_bigrams=False) for t in traces],
        "aggregate_bigram": [extract_trace_features(t, include_bigrams=True) for t in traces],
        "positional": [positional_features(t, max_steps=max_steps) for t in traces],
        "univariate": [{"max_egress": max_egress(t)} for t in traces],
    }


def evaluate_ladder(
    records: Sequence[Mapping[str, Any]],
    *,
    target: str,
    seed: int,
    group_key,
    reps=None,
    max_steps: int = 16,
) -> dict[str, dict[str, float]]:
    """Leave-one-group-out over `group_key`, mirroring the headline protocol."""

    y = np.asarray([_label(r, target) for r in records], dtype=int)
    groups = np.asarray([group_key(r) for r in records], dtype=object)
    unique = list(dict.fromkeys(groups.tolist()))
    if len(unique) < 2:
        raise ValueError("need at least two groups")
    reps = reps if reps is not None else _representations(records, max_steps)

    fold_aucs: dict[str, list[float]] = {key: [] for key, *_ in LADDER}
    skipped = 0
    for fold_index, held in enumerate(unique):
        test_idx = np.flatnonzero(groups == held)
        train_idx = np.flatnonzero(groups != held)
        if len(np.unique(y[test_idx])) < 2 or len(np.unique(y[train_idx])) < 2:
            skipped += 1
            continue
        for key, _label_text, rep_name, ctor in LADDER:
            feats = reps[rep_name]
            if ctor is None:
                # Unfitted univariate baseline: the feature IS the score.
                scores = np.asarray(
                    [list(feats[i].values())[0] for i in test_idx], dtype=float
                )
            elif ctor is _lr:
                # Delegate to attack.py's fitter verbatim so A1 and A2 reproduce
                # the manuscript's numbers exactly. Do not reimplement it.
                scores = _attack_fit_scores(
                    [feats[i] for i in train_idx],
                    y[train_idx],
                    [feats[i] for i in test_idx],
                    seed=seed + fold_index,
                )
            else:
                train_x, test_x = _dense(feats, train_idx, test_idx)
                scores = _fit_predict(ctor(seed + fold_index), train_x, y[train_idx], test_x,
                                      seed + fold_index)
            fold_aucs[key].append(inversion_safe_auc(y[test_idx], scores))
    if skipped == len(unique):
        raise ValueError("every fold was skipped")
    return {
        key: {
            "fold_mean_auc": float(np.mean(v)) if v else float("nan"),
            "fold_std_auc": float(np.std(v)) if v else float("nan"),
            "n_folds": len(v),
            "skipped_folds": skipped,
        }
        for key, v in fold_aucs.items()
    }


def permutation_null_ladder(
    records: Sequence[Mapping[str, Any]],
    *,
    target: str,
    seed: int,
    group_key,
    n_perm: int,
    max_steps: int = 16,
) -> dict[str, dict[str, float]]:
    """A separate calibrated null per attacker.

    A more flexible model has a higher null, so scoring every rung against the
    linear model's floor would report capacity that is really just flexibility.
    Labels are permuted WITHIN group, which preserves the group-by-label balance
    the protocol is defined against.
    """

    y = np.asarray([_label(r, target) for r in records], dtype=int)
    groups = np.asarray([_group(r) for r in records], dtype=object)
    reps = _representations(records, max_steps)
    draws: dict[str, list[float]] = {key: [] for key, *_ in LADDER}
    # Each permutation is independent, so fan them out. Seeds are derived from the
    # draw index rather than from a shared generator, so the null is reproducible
    # and does not depend on how many workers happened to be available.
    payload = (records, target, seed, max_steps, reps)
    workers = max(1, min(int(os.cpu_count() or 2) - 1, 12))
    ctx = _mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        for res in pool.map(
            _one_permutation,
            ((payload, i) for i in range(n_perm)),
            chunksize=8,
        ):
            if res is None:
                continue
            for key, value in res.items():
                draws[key].append(value)
    return {
        key: {
            "null_mean": float(np.nanmean(v)) if v else float("nan"),
            "null_q95": float(np.nanquantile(v, 0.95)) if v else float("nan"),
            "n_draws": len(v),
            "draws": [float(x) for x in v],
        }
        for key, v in draws.items()
    }


def _one_permutation(job):
    """One within-group label permutation, scored by the whole ladder."""
    (records, target, seed, max_steps, reps), index = job
    y = np.asarray([_label(r, target) for r in records], dtype=int)
    groups = np.asarray([_group(r) for r in records], dtype=object)
    rng = np.random.default_rng(seed + 1_000_003 * (index + 1))
    permuted = y.copy()
    for held in dict.fromkeys(groups.tolist()):
        mask = np.flatnonzero(groups == held)
        permuted[mask] = rng.permutation(y[mask])
    alias = "attribute_label" if target == "attribute" else "membership_label"
    clones = [dict(r) for r in records]
    for clone, lab in zip(clones, permuted, strict=True):
        clone[alias] = int(lab)
    try:
        res = evaluate_ladder(
            clones, target=target, seed=seed, group_key=_group, reps=reps, max_steps=max_steps
        )
    except ValueError:
        return None
    return {key: res[key]["fold_mean_auc"] for key, *_ in LADDER}


def _p_value(observed: float, draws: Sequence[float]) -> float:
    if not draws:
        return float("nan")
    arr = np.asarray(draws, dtype=float)
    return float((np.sum(arr >= observed) + 1) / (arr.size + 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--journal", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "iclr2027" / "tables")
    ap.add_argument("--seed", type=int, default=20260710)
    ap.add_argument("--permutations", type=int, default=2000)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    rows = load_experiment_records(args.journal)
    records = _records_for(rows, "adaptive")
    if not records:
        raise SystemExit(f"no adaptive records in {args.journal}")
    max_steps = max(len(_steps(r.get("trace", r))) for r in records)
    print(f"  {len(records)} adaptive runs; longest trace {max_steps} steps")

    by_group = _group
    def by_specialty(record):
        return str(record.get("service", "?"))

    report: dict[str, Any] = {
        "journal": str(args.journal),
        "seed": args.seed,
        "permutations": args.permutations,
        "n_records": len(records),
        "max_steps": max_steps,
        "ladder": [{"key": k, "label": lbl, "representation": rep} for k, lbl, rep, _ in LADDER],
    }

    for target in ("attribute", "membership"):
        obs = evaluate_ladder(records, target=target, seed=args.seed, group_key=by_group,
                              max_steps=max_steps)
        print(f"\n  == {target}, leave-one-(specialty,topic)-out ==")
        for key, lbl, _rep, _c in LADDER:
            r = obs[key]
            print(f"    {key} {lbl:46} {r['fold_mean_auc']:.3f} +/- {r['fold_std_auc']:.3f}"
                  f"  ({r['n_folds']} folds)")
        spec = evaluate_ladder(records, target=target, seed=args.seed, group_key=by_specialty,
                               max_steps=max_steps)
        print(f"  == {target}, leave-one-SPECIALTY-out ==")
        for key, lbl, _rep, _c in LADDER:
            print(f"    {key} {spec[key]['fold_mean_auc']:.3f} ({spec[key]['n_folds']} folds)")
        if args.permutations > 0:
            nulls = permutation_null_ladder(
                records, target=target, seed=args.seed, group_key=by_group,
                n_perm=args.permutations, max_steps=max_steps,
            )
            print(f"  == {target}, each attacker against ITS OWN calibrated null "
                  f"({args.permutations} within-group permutations) ==")
            for key, lbl, _rep, _c in LADDER:
                o = obs[key]["fold_mean_auc"]
                nl = nulls[key]
                pv = _p_value(o, nl["draws"])
                flag = "*" if pv <= 0.05 else " "
                print(f"    {key} obs {o:.3f}  null {nl['null_mean']:.3f}"
                      f"  q95 {nl['null_q95']:.3f}  p={pv:.4f}{flag}  {lbl}")
                nl.pop("draws", None)
                nl["observed"] = o
                nl["p_value"] = pv
            report[target] = {"group": obs, "specialty": spec, "null": nulls}
        else:
            report[target] = {"group": obs, "specialty": spec}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\n  wrote {args.report}")
    interim = args.out_dir / "ladder-report.json"
    interim.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"  wrote {interim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

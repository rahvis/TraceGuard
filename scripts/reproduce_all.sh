#!/usr/bin/env bash
# Regenerate every number, table, macro and figure the paper reports, from the
# confidential-VM journals committed under artifacts/.
#
# The reproduction claim this script makes is deliberately narrow and checkable:
# given the journals in artifacts/, the analysis is deterministic, so a correct
# run leaves the committed contents of tables/ and figures/ unchanged. Stage 6
# checks exactly that with 'git diff'. An empty diff is the reproduction; a
# non-empty diff is a finding, and the diff itself says which quantity moved.
#
# Nothing here contacts a model provider or an Azure endpoint. Collecting new
# journals is a separate, costly step documented in docs/REPRODUCIBILITY.md;
# this script only re-derives the paper from evidence already gathered.
#
# Two traps are worth naming, because both produced wrong numbers during the
# work and both are now guarded rather than remembered:
#
#   * make_paper_artifacts.py merges its macros over tables/macros.tex. Running
#     it with a low --permutations silently replaces the 2000-permutation
#     Holm family with an underpowered one, and nothing in nulls.json reveals
#     the substitution. PERMS is pinned below and asserted after stage 2.
#
#   * the fail-closed rate is recovered from the signed receipts when a journal
#     predates the runtime recording it, so --runs is not optional. Without it
#     the deadline stage correctly refuses to report a rate it cannot source,
#     and an earlier version of that stage reported a confident 0% instead.
#
# Usage:   bash scripts/reproduce_all.sh
# Options (environment):
#   FORCE_NULLS=1   recompute the calibrated permutation nulls from scratch
#                   instead of using the hash-bound cache in tables/nulls.json.
#                   Costs minutes of CPU and must produce identical output.
#   PYTHON=...      interpreter to use (default python3).

set -euo pipefail

PERMS=2000
SEED=20260710
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

ART="$ROOT/artifacts"
HEADLINE="$ART/cvm-honest.jsonl"
HIST="$ART/experiment-usenix26-main.jsonl"
RUNS="$ART/cvm-runs"

for required in "$HEADLINE" "$HIST" "$RUNS"; do
  [ -e "$required" ] || { echo "missing required input: $required" >&2; exit 1; }
done
mkdir -p tables figures

echo "===== 1/7 environment and dependency self-check"
# Pinned to the offline provider on purpose. No stage below contacts a model
# provider, so a missing or misconfigured Azure credential must not fail a
# reproduction; without the pin, doctor inherits TRACEGUARD_PROVIDER from the
# environment and exits non-zero on absent keys the run never needs.
"$PY" -m traceguard.cli doctor --provider fixture || {
  echo "  doctor reported a problem; the stages below will not be trustworthy" >&2
  exit 1
}

echo
echo "===== 2/7 attack tables, controls, ablations and macros from the headline journal"
# The calibrated nulls are the expensive part: 2000 within-group permutations.
# They are cached in tables/nulls.json and bound to the journal by sha256, so a
# cache built for a different journal is refused rather than silently reused.
# That makes reusing it safe, and FORCE_NULLS=1 the way to prove it.
# Note on the two switches, because they are easy to confuse and one of them
# quietly changes what gets written. Omitting --permutations is what selects the
# cached nulls; --macros-only is a different thing entirely, suppressing the six
# attack tables and emitting macros alone, so it must not appear here.
NULLS_VALID=0
if [ -z "${FORCE_NULLS:-}" ] && [ -f tables/nulls.json ]; then
  NULLS_VALID="$("$PY" - <<'PYEOF'
import hashlib, json, pathlib
try:
    cache = json.loads(pathlib.Path("tables/nulls.json").read_text())
    journal = pathlib.Path(cache["journal"])
    print(1 if hashlib.sha256(journal.read_bytes()).hexdigest() == cache["journal_sha256"] else 0)
except Exception:
    print(0)
PYEOF
)"
fi

NULL_ARGS=(--nulls tables/nulls.json --historical-nulls tables/nulls-historical.json)
if [ "$NULLS_VALID" = "1" ]; then
  echo "  the cached calibrated nulls match this journal by sha256; reusing them"
  echo "  (FORCE_NULLS=1 recomputes them at $PERMS permutations; takes minutes)"
else
  echo "  computing calibrated nulls at $PERMS permutations; this takes minutes"
  NULL_ARGS+=(--permutations "$PERMS")
fi
"$PY" scripts/make_paper_artifacts.py \
  --journal "$HEADLINE" --out-dir tables --seed "$SEED" \
  --historical-journal "$HIST" "${NULL_ARGS[@]}"

# Assert the pinned permutation count survived the merge over tables/macros.tex.
if grep -q "holmPerms}{$PERMS" tables/macros.tex; then
  echo "  ok: macros record $PERMS permutations"
else
  echo "  FAIL: tables/macros.tex does not record $PERMS permutations" >&2
  grep -o 'holmPerms}{[0-9]*' tables/macros.tex >&2 || true
  exit 1
fi

echo
echo "===== 3/7 generality sweep and second task family"
"$PY" scripts/make_matrix_artifacts.py \
  --journal-dir "$ART" --out-dir tables --seed "$SEED" \
  --permutations "$PERMS" --report tables/matrix-report.json
"$PY" scripts/make_family_artifacts.py \
  --journal-dir "$ART" --out-dir tables --seed "$SEED" \
  --permutations "$PERMS" --report tables/family-report.json

echo
echo "===== 4/7 deadline sweep and the cost of the padded defence"
DL_ARGS=()
for ms in 4000 6000 8000; do
  cand="$ART/cvm-dl$ms.jsonl"
  [ -f "$cand" ] && DL_ARGS+=(--journal "$ms=$cand")
done
# The headline arm is itself a 10 s full-pad run, at the largest n of any
# setting, so it is a row in its own right and not merely the source of the
# pad-cost macros.
DL_ARGS+=(--journal "10000=$HEADLINE")
"$PY" scripts/make_deadline_artifacts.py "${DL_ARGS[@]}" \
  --headline "10000=$HEADLINE" --runs "$RUNS" --tables tables

echo
echo "===== 5/7 judged utility, reanalysed offline from the committed judged run"
# --analyze-only runs no cells and needs no credentials. It is deliberately
# incapable of producing a score: the four grid cells that errored during the
# live judged run stay absent and are reported as a shortfall, rather than being
# filled by a stub. Without this the utility table could only be regenerated by
# paying for a fresh judged run, which is how it silently went stale once.
JUDGED="$ART/utility-cvm-enclave.jsonl"
if [ -f "$JUDGED" ]; then
  "$PY" scripts/judge_utility.py --analyze-only --n-per-cell 3 \
    --scores "$JUDGED" --journal "$HEADLINE" --tables-dir tables
else
  echo "  no committed judged run; skipping the utility stage"
fi

echo
echo "===== 6/7 wire-level measurement and figures"
WIRE="$ART/wire"
if [ -f "$WIRE/wire.jsonl" ] && [ -f "$WIRE/wire-packets.txt" ]; then
  # --peer is load-bearing. The capture also saw an unrelated peer carrying
  # 227 kB, and including it moved the step-level egress correlation from
  # +0.45 to -0.04, which inverts the paper's conclusion.
  PEER="$("$PY" -c "import json;print(json.load(open('$WIRE/wire-report.json'))['peer'])" 2>/dev/null || true)"
  PEER_ARGS=()
  [ -n "$PEER" ] && PEER_ARGS+=(--peer "$PEER") && echo "  restricting the capture to peer $PEER"
  "$PY" scripts/make_wire_artifacts.py \
    --journal "$WIRE/wire.jsonl" --runs "$WIRE/runs" \
    --packets "$WIRE/wire-packets.txt" --out-dir tables \
    --report tables/wire-report.json --permutations "$PERMS" --seed "$SEED" \
    "${PEER_ARGS[@]}"
  "$PY" scripts/make_wire_figure.py \
    --journal "$WIRE/wire.jsonl" --runs "$WIRE/runs" \
    --packets "$WIRE/wire-packets.txt" --report tables/wire-report.json \
    --out figures/fig_wire.pdf "${PEER_ARGS[@]}"
else
  echo "  no packet capture under artifacts/wire; skipping the wire stage"
fi
"$PY" scripts/make_paper_figures.py \
  --journal "$HEADLINE" --out-dir figures --seed "$SEED" --nulls tables/nulls.json
ASYM_ARGS=()
for ms in 4000 6000; do
  cand="$ART/cvm-dl$ms.jsonl"
  [ -f "$cand" ] && ASYM_ARGS+=(--deadline-journal "$ms=$cand")
done
"$PY" scripts/make_asymmetry_figure.py \
  --journal "$HEADLINE" "${ASYM_ARGS[@]}" --runs "$RUNS" \
  --out figures/fig_asymmetry.pdf

echo
echo "===== 7/7 verification"
"$PY" scripts/verify_citations.py --bib refs.bib
"$PY" -m pytest -q tests

echo
echo "===== reproduction check: did the regenerated output match what is committed?"
# The analysis is deterministic given the journals, so a correct run is a no-op
# on tracked files. This is the actual claim, and it is machine-checkable.
if git rev-parse --git-dir >/dev/null 2>&1; then
  # 'git status --porcelain' rather than 'git diff', because git diff is blind
  # to untracked files: a stage that wrote a table the release never committed
  # would otherwise pass this check silently.
  DRIFT="$(git status --porcelain -- tables figures)"
  if [ -z "$DRIFT" ]; then
    echo "  ok: tables/ and figures/ reproduce the committed values exactly"
  else
    echo "  DIFFERS from the committed values:"
    echo "$DRIFT" | sed 's/^/    /'
    echo "  Inspect with: git diff -- tables figures"
    exit 1
  fi
else
  echo "  not a git checkout; compare tables/ and figures/ against the release by hand"
fi

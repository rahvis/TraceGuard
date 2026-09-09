# Reproducibility contract

## Research questions implemented

The artifact implements the paper's four evaluation questions:

1. Can a passive host infer membership or a sensitive case framing from trace
   metadata alone?
2. Do permutation and shape-fixed controls locate the signal in trace shape?
3. Does structure canonicalization alone leave residual timing/size signal, and
   does a constant structure/timing/size release remove the application-level
   observable at a measurable cost?
4. Do signed receipts and the ledger verify, and do mutations fail verification?

The fixture provider validates only that these questions are computed correctly.
It deliberately generates labeled behavior and cannot establish that a real LLM
leaks. Only a fresh live provider run can supply empirical evidence.

## What "deterministic" does and does not mean here

The analysis is deterministic given the journals and a fixed analysis environment.
Two identical invocations of any stage produce byte-identical output, which is what
`scripts/reproduce_all.sh` asserts at the end by checking `git status` over `tables/`
and `figures/`: a correct run changes nothing that is committed.

It is not deterministic across scikit-learn versions, and this is worth stating plainly
because it bit this artifact. The second task family's attribute AUC was 0.763 when the
number first entered the manuscript and is 0.733 under the environment recorded in
`environment-tested.txt`, from the same journal bytes, the same seed and the same code.
Cross-validation fold construction changed underneath the analysis. The nulls moved with
it, so the permutation rank was unchanged and every qualitative finding survived: both
families still clear their own calibrated nulls on the attribute channel, membership
still fails to replicate on the second family and still sits below its own null, and
Holm still rejects three of four. The manuscript reports the values produced under the
recorded environment.

Two practical consequences. If you reproduce third decimals that differ slightly from
the paper, compare your versions against `environment-tested.txt` before concluding
that something is wrong. And if you are extending this work, pin the analysis
environment rather than the range in `pyproject.toml`, which is deliberately permissive
so the package installs broadly.

## Regenerating the paper's tables and macros

The paper's attack-derived tables and numeric macros are generated in place from a
real experiment journal, so the manuscript reflects a run rather than transcribed
numbers:

```bash
# 1. produce a journal (use the azure provider for evidential numbers)
set -a && . ./.env && set +a        # a valid, rotated AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT
uv run traceguard experiment --provider azure \
    --n-per-cell 12 --seed 20260710 --journal artifacts/run.jsonl

# 2. regenerate tables/macros in place (or OUT=<dir> to preview first)
make paper-tables JOURNAL=artifacts/run.jsonl
#   -> rewrites tables/tab_main.tex, tab_attribution.tex, tab_specialty.tex,
#      tab_ablation.tex, tab_frontier.tex, and the attack macros in tables/macros.tex

# 3. recompile the paper (byte-reproducibly -- see below)
make paper
```

### The generality sweep

The four per-factor results the reviewers asked for -- graded sensitivity,
autonomy, crew topology, crew model -- come from a separate star (one-factor-at-a-time)
design with its own journals and its own generator:

```bash
# 1. run the matrix (1,152 cells; ~2.5 h at 8 workers against Azure OpenAI)
uv run python -u scripts/run_experiment_matrix.py --workers 8 --reps 2 --baseline-reps 4
#   -> artifacts/matrix-{baseline,autonomy,architecture,model}.jsonl
#   Resume-safe: a cell already journalled under the same
#   (arm, case, repetition, seed) key is skipped, so an interrupted run continues.

# 2. regenerate the per-factor tables and macros
make paper-matrix                    # PERMUTATIONS=2000 by default; drop it to iterate
#   -> tables/tab_grades.tex, tab_autonomy.tex, tab_architecture.tex, tab_models.tex
#      plus the grade/autonomy/architecture/model macros and tables/matrix-report.json
```

A star design rather than a full factorial: crossing the four factors at their
measured levels would be 81 arms / 7,776 cells, and the star holds the baseline
fixed while moving one factor at a time, so cost scales additively. The trade is
that interactions are not identified, and the manuscript claims none.

Two properties of the generator are worth knowing before reading its output.
Every arm is reported against **its own** within-group permutation null, never
against 0.5, because the leave-one-group-out protocol selects the better of two
attackers and reports an inversion-safe `max(auc, 1-auc)`, which puts the floor
near 0.57--0.60. And the graded-sensitivity result carries two statistics: the
full-ladder Somers' D restates the binary result (a step function is monotone, so
separating rung 0 from the rest alone scores high), while the incremental claim
lives in the trend restricted to the non-zero rungs. Read `within_sensitive` in
`tables/matrix-report.json` for the latter.

### Byte-reproducible manuscript

`make paper` is byte-reproducible: pdfTeX otherwise stamps `/CreationDate`,
`/ModDate` and a random `/ID` into the trailer, so recompiling identical source
produced different bytes and repeatedly invalidated the SHA-256 anchor in
`src/traceguard/data/paper-reported-unverified.json`. The `\pdfinfoomitdate`,
`\pdftrailerid` and `\pdfsuppressptexinfo` directives in the preamble are
sufficient on their own -- a build with `SOURCE_DATE_EPOCH` unset and
`TZ=Asia/Kolkata` reproduces the same digest -- so the anchor means "this
content", not "this build on this machine".

### The step deadline is calibrated, not constant

`TRACEGUARD_STEP_DEADLINE_MS` must be measured against the deployment, not
inherited. On the gpt-5.6 Azure deployments the measured per-step latency is
p50 3.0 s / p95 8.1 s / p99 8.7 s, so the 3 s value the archived journal used
would fail closed on 48.3% of steps and the defense would be unusable; production
runs at 10 s. Because the overrun figures depend on it, the analysis recovers the
deadline from each journal's own full-pad rows -- whose released per-step duration
*is* the enforced deadline -- rather than from your environment, refuses to
average a journal spanning two deadlines, and warns that the figures are
provisional for a journal with no padded rows.

Provenance is honest by construction: every generated file carries a header stating
whether it came from a `live_replication` run or the diagnostic `synthetic_fixture`
harness, and the generator refuses to relabel fixture output as evidence. Utility and
latency (`tab_utility.tex`) come from a judged live run produced by
`scripts/judge_utility.py` (crew answers are scored in memory and never persisted;
the latency footnote derives from the main journal via `--journal`). The values
shipped in this checkout come from the archived `live_replication` journals under
`artifacts/` (`experiment-usenix26-main.jsonl`, `experiment-usenix26-gen.jsonl`,
`utility-usenix26.jsonl`), executed on an Azure Confidential VM (AMD SEV-SNP,
MAA-verified); running the steps above replaces them with the numbers your own
run measured.

## Evaluation cells

`synthetic-medical-v2.0.0` defines six `(specialty, topic)` groups:

- cardiology: advanced cardiac imaging; rhythm device;
- oncology: systemic therapy; molecular diagnostics;
- psychiatry: behavioral-health level of care; specialty psychotropic.

Each group expands over a four-rung sensitivity ladder (`routine`, `guarded`,
`elevated`, `sensitive`) crossed with canary-absent/present: `6 x 4 x 2 = 48`
cases. A canary replaces one distractor, so every case has exactly six documents.
The design size is derived from the catalog by `validate_balanced_design` rather
than asserted, so the ladder can be lengthened in data alone.

The archived journals under `artifacts/` were produced against
`synthetic-medical-v1.0.0`, whose binary routine/sensitive split gives
`6 x 2 x 2 = 24` cases. The paper's reported counts imply 12 baseline repetitions
and 6 shielded repetitions per cell against that corpus
(`6 groups x 4 cells x 12 = 288`; `6 x 4 x 6 = 144`). This inference is recorded
because the paper did not state the per-cell repetition count directly. Note that
`dataset_hash` covers the whole corpus file, so a fresh run on v2 records a
different digest than those journals do — expected, and the reason v1 is still
shipped byte-for-byte.

## Conditions

- `adaptive`: research depth and repair/review loops may depend on private input;
  timing and egress sizes are measured.
- `structure_only`: the public plan has a fixed research depth and fixed node
  sequence, while real timing and sizes remain observable. No formal bound is
  claimed.
- `full_pad`: fixed plan plus public timing and egress targets. The receipt states
  `(0,0)` only when the runtime reports every application-level invariant met.
  An overrun is fail-closed and recorded; it is not silently counted as success.

The local container emulates the confidential boundary. Operating-system,
network, TLS, model-provider, scheduler, and hardware side channels are outside
the proof supplied by this code.

## Attack protocol

New runs derive only metadata features:

- step and research-hop counts;
- distinct step types and per-step-type counts;
- sum, mean, and maximum duration;
- sum, mean, and maximum egress bytes;
- ordered step bigrams for the stronger attacker.

The classifier is logistic regression with scaling. Evaluation holds out an
entire `(specialty, topic)` group and produces out-of-fold scores for all six
groups. Membership and attribute are evaluated separately. Reported leakage is
`max(AUC, 1-AUC)` so a score inversion cannot make leakage look safe.

The analysis also produces:

- seeded label-permutation control;
- population-constant shape-fixed control;
- single-feature attribution;
- per-coordinate ablations;
- K=3 randomized-response template frontier with keep probability
  `exp(epsilon) / (exp(epsilon) + 2)`;
- clustered bootstrap intervals where the group/task structure permits them.

Every split, seed, preprocessing setting, package version, and warning is written
to the run result. Small fixture runs may be too small for meaningful intervals;
the analyzer reports that limitation instead of manufacturing precision.

## Run artifact contract

Each run is immutable and has a unique identifier. Its directory contains the
available subset of:

```text
manifest.json       provenance, start/end time, provider, exact models, seed,
                    dataset hash, code hash/version, environment versions
events.jsonl        metadata-only progress events
traces.jsonl        observable trace plus experiment labels outside the trace
receipts.jsonl      signed claims; never prompts, documents, or answers
ledger.jsonl        hash-chain entries
metrics.json        derived measurements and warnings
files.sha256        content hash for every artifact file
```

No secret, query, document, prompt, private model output, or API response body may
appear in those files. The answer returned to an SDK caller is an inside-boundary
value and is not part of an experiment trace export by default.

## Offline gate

```bash
./scripts/run_fixture_reproduction.sh
```

This must pass without a key or network call. Acceptance criteria:

- all 24 expanded cases have exactly six synthetic documents;
- traces contain only allowed metadata keys;
- full-pad neighbor traces are identical at the declared boundary;
- group splits have no `(specialty, topic)` overlap;
- controls and inversion-safe AUC behave as specified;
- receipt mutation and wrong pinned keys fail;
- ledger mutation/reordering fail;
- fixed-size receipt envelopes have the configured byte length;
- API and SSE events contain no payload fields;
- the container health endpoint is live.

## Live replication gate

```bash
read -s AZURE_OPENAI_API_KEY && export AZURE_OPENAI_API_KEY
export AZURE_OPENAI_ENDPOINT=https://<resource>.services.ai.azure.com   # resource host, not an /api/projects/<name> Foundry URL
N_PER_CELL=12 ./scripts/run_live_replication.sh
```

Before interpreting a live run, verify:

1. the exact model IDs in the manifest were available and no fallback occurred;
2. there were no silent retries, timeouts, cancellations, or cap overruns;
3. all cells completed with the intended fixed corpus size;
4. the attacker used only exported metadata;
5. all out-of-fold splits and intervals are present;
6. utility is evaluated under a predeclared rubric by a distinct model or human
   protocol, not inferred from attack metrics;
7. full-pad wall-clock and fail-closed utility are measured separately;
8. result labels remain `live_replication`.

## Original-result limitation

`src/traceguard/data/paper-reported-unverified.json` is an immutable transcription
anchored to the supplied PDF and TeX hashes. It is not an input to analysis of a
current run. Replacing a fresh metric with a paper reference value is a test and
release failure.

# TraceShield: trace privacy for LLM agents

This repository is the research artifact for *Trace Privacy for LLM Agents:
Differential Privacy over Adaptive Execution*. It contains the multi-agent
system the paper measures, the attack it mounts against that system, the
defense it proposes, the receipt and attestation machinery that certifies which
release actually ran, the deployment scripts that provision an attested
confidential virtual machine, and the experiment journals every reported number
is computed from. It is organized so that a reader can start from a clean
checkout, run something meaningful within a minute without a model key, and
then decide how far up the ladder of cost and fidelity they wish to go.

## The problem, in one paragraph

An autonomous language-model agent decides its own execution. How many
retrieval passes it issues, what it places in each request to the model,
whether it loops back to repair an unsupported claim, and when it has read
enough are all runtime decisions conditioned on the private query and on the
private documents the agent encounters. The execution trace is therefore not
incidental telemetry but a data-dependent encoding of the private input, and it
crosses the trust boundary in the clear even when the agent runs inside a
confidential virtual machine composed with oblivious retrieval and output
differential privacy. Content is sealed, access is oblivious, the answer is
private, and the trace still leaks. Existing privacy notions do not cover this,
because each of them privatizes a value computed by a *fixed* program, whereas
an agent's step count, step identity and stopping decision are themselves random
functions of the data. The paper defines an (epsilon, delta) notion over the
variable-length, adaptively generated trace itself; this artifact measures the
channel, closes the coordinates a defense can close, and emits a signed receipt
stating exactly which guarantee a given run carried.

## The one asymmetry everything here turns on

Because the model is a hosted API, which is the ordinary way these systems are
built, the host observes *four* scalar series rather than three: the step
sequence, the per-step timing, the *request* bytes the agent composes, and the
*response* bytes the provider returns. The first three are composed by the
agent. Reshaping the agent's policy makes them public constants, which is
(0, 0) under both neighbor relations simultaneously and for group changes of any
size. The fourth is chosen by a party the agent does not control, and its bytes
have already crossed the boundary before the agent is scheduled again, so no
agent-side release mechanism can put any guarantee on it at all. Which secret a
mechanism actually removes therefore depends on whether that secret is
control-flow-carried or request-content-carried, and the artifact contains the
measurement that separates the two: reshaping the control flow removes
membership even on the coordinate it never touches, while the attribute, carried
by request content, survives. That locus is not a fixed property of the task. It
moves with model capability and with the workload, which is why a
coordinate-specific mitigation validated on one crew can be wrong on the next.

## What is in here and what each part is for

The Python package under `src/traceguard` is the system itself. Within it,
`agents.py` and `graph.py` implement the six-agent prior-authorization crew on
LangGraph, `instrumentation.py` records the host-observable trace and nothing
else, `shield_runtime.py` implements the three release conditions, `plans.py`
holds the canonical plan a defended release executes, `providers.py` wraps the
model providers and performs the request padding at the point where bytes
actually leave, `attack.py` implements the trace-reconstruction attack together
with its statistical controls, `attestation.py` collects and verifies the
confidential-VM evidence, `receipt.py` and `verifier.py` implement the Ed25519
receipts and the independent verifier, `ledger.py` implements the hash chain,
`corpus.py` and `domains.py` load the synthetic corpora, `experiment.py` drives
the resumable experiment matrix, and `cli.py` provides the command line. The
directory `sdk` documents the programmatic surface and holds four runnable
examples; start there if you intend to embed the runtime in your own program
rather than reproduce a measurement.

The directory `artifacts` holds the evidence: the experiment journals, the
per-run receipt stores that let a third party check any individual run rather
than trusting an aggregate, and the header-only packet capture behind the
wire-level result. Its own `artifacts/README.md` says which journal backs which
claim. The directory `deploy/azure` provisions the attested confidential virtual
machine.

## What is deliberately not in here

The manuscript-facing analysis pipeline is not published in this repository.
The scripts that recompute the paper's tables, calibrated nulls, ablations,
figures and numeric macros from the journals, the generated outputs they write,
the bibliography checker, and the test suite are all held back, and the ignore
rules exclude them so that a stray `git add` cannot reintroduce them.

State the consequence plainly rather than discover it later. The journals under
`artifacts` are the recorded evidence and every run in them is verifiable from a
clean checkout, and the attack itself ships in `src/traceguard/attack.py` and is
reachable from the command line, so the headline attack numbers can be
recomputed here. What cannot be done from this tree alone is re-derive a
specific table or figure in the paper byte for byte, because the generator that
writes it is not part of this repository.

## A note about names

The system described in the paper is called TraceShield. The Python package and
the command line entry point are both called `traceguard`, and they are left
that way here deliberately. The paper cites `traceguard experiment`,
`traceguard verify --journal`, `deploy/azure/deploy.sh`, and the module
filenames `agents.py`, `graph.py`, `instrumentation.py`, `shield_runtime.py` and
`attestation.py`. Renaming any of those in the published artifact would put the
artifact and the paper out of agreement, which is precisely the kind of
discrepancy artifact evaluation is meant to catch, so the names stay as the
paper cites them.

## Requirements

You need Python 3.11, 3.12 or 3.13. The dependency set is installed from
`pyproject.toml` and includes cryptography for the Ed25519 receipts, LangGraph
for the crew, the OpenAI client for the model calls, and NumPy, SciPy and
scikit-learn for the analysis. Everything that recomputes a number from a
committed journal runs on a laptop with no network access and no credentials.
Only a fresh live replication needs a model deployment, and only the deployment
scripts need an Azure subscription. One caveat is worth flagging rather than
burying: the analysis is deterministic given the journals and a fixed
environment, but not across scikit-learn versions, so an AUC recomputed under a
different solver version can differ in the third decimal place.

## Installation

The project is built with `uv`, which is the fastest path and the one the paper
quotes.

```sh
uv sync --extra dev
uv run traceguard doctor
```

The `doctor` subcommand reports whether this host is configured for offline
operation or for live model calls, and whether a hardware trusted execution
environment is present. It never invents a capability: with no virtual trusted
platform module available it reports the attestation state as simulated and
refuses to set the hardware-backed flag.

If you would rather not use `uv`, a standard virtual environment works. The
package lives under `src`, so an editable install or an explicit `PYTHONPATH`
is required.

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
traceguard doctor
```

Everything in this README can also be run inside the container, which is the
form the paper describes as the offline functional gate.

```sh
docker compose up --build
```

## The three provenance tiers, which matter more than any single number

This artifact separates what it produces into three tiers, and the separation
is enforced in code rather than asserted in prose, because conflating them is
the easiest way to make an artifact look stronger than it is.

The first tier is the offline fixture. A deterministic harness stands in for the
model, so the pipeline runs end to end with no key and no network. Every result
it produces is labelled `synthetic_fixture`. This tier demonstrates that the
software works. It is not evidence that a real agent leaks anything, and it must
never be read as such.

```sh
make offline
make examples
```

The second tier is reanalysis. The journals committed under `artifacts` are the
recorded output of real runs against real model deployments, and the attack
recomputes from them with no credentials. This is the tier most readers will
want. What it does not do is re-derive the journals themselves.

The third tier is live replication. A fresh run against a configured model
deployment produces new journal rows labelled `live_replication`. This is a
genuine new measurement rather than a relabeling of a prior result, it costs
money, and the numbers will differ slightly from the paper's because the models
are stochastic and the fail-closed rate depends on the latency profile of the
machine you run on.

## Recomputing the attack from a committed journal

The attack is part of the package, so it runs directly against any journal in
this repository without the manuscript pipeline. It performs the
leave-one-group-out protocol the paper specifies, scores both attackers and
takes the inversion-safe maximum, and reports both targets.

```sh
uv run traceguard analyze artifacts/cvm-honest.jsonl --condition adaptive
uv run traceguard analyze artifacts/cvm-honest.jsonl --condition structure
uv run traceguard analyze artifacts/cvm-honest.jsonl --condition full
```

Read the result against the calibrated null rather than against one half. The
null of this design is not 0.5: holding out an entire group, selecting the
better of two attackers and taking the inversion-safe maximum pushes it well
above that, and the binary attribute label runs its classes at one to three. The
paper estimates the null directly with two thousand within-group label
permutations and reports every AUC against it, which is why the permutation
calibration dominates the runtime of the full pipeline. The `analyze`
subcommand reports the AUC and its bootstrap interval; it does not recompute the
permutation null, and a number read from it against 0.5 would overstate the
effect substantially.

## Where the paper's headline numbers come from

The headline attack is computed from `artifacts/cvm-honest.jsonl`, the journal
of the arm that ran inside a genuine AMD SEV-SNP confidential virtual machine
with attestation verified by Microsoft Azure Attestation. Every row carries a
hardware-rooted binding checkable from the journal alone with
`traceguard verify --journal`. From it, and from metadata alone with content
confidentiality, single-access obliviousness and output DP all assumed in force,
the attack recovers the sensitive attribute at a group-aware AUC of 0.784
against a calibrated null of 0.60 and canary membership at 0.721 against its own
null of 0.58, over 191 undefended runs on six held-out groups.

Under the fully padded release the agent-composed triple is byte-identical
across all 192 runs, so the (0, 0) claim on those coordinates holds by
measurement rather than by assertion. Membership falls to 0.577 and no longer
clears its null, while the attribute stands at 0.764. That asymmetry is the
paper's central practical result, and restricting the adversary to the single
coordinate no mechanism can close isolates its cause: ingress-only membership
falls from 0.68 to 0.51 as the mechanism tightens, because membership was
carried into the response by the control flow, whereas ingress-only attribute
does not follow, holding at 0.69 to 0.73, because it is carried by request
content that the provider's reply reflects whatever plan produced it.

Two results in that journal are worth stating because they cut against an
intuitive reading. It is not the adaptive-depth channel that carries the leak
here: depth alone scores 0.55, *below* its own null, because 97 percent of runs
sit at the public cap of seven, and what carries the secret is size, with the
request at 0.78 and the response at 0.78 against 0.72 for timing. And structure
canonicalization *without* padding raises the attribute to 0.799, above the
undefended 0.784, so the intermediate mode is a trap: a deployer who enables it
as a cheap partial measure gets the opposite of what they intended.

The extended-corpus replication comes from `artifacts/cvm-v3.jsonl`, 383 runs
over twelve groups across six specialties, where the attribute replicates and
strengthens at 0.879 against a null of 0.597 and membership at 0.697 against
0.584. That corpus is also what resolves the paired undefended-versus-padded
contrast the six-group design could not, taking the minimum detectable effect
from 0.174 to 0.106 and putting the attribute difference at -0.152 with a 95
percent interval of [-0.228, -0.076]; `artifacts/cvm-v4.jsonl` doubles the
groups again to twenty-four and replicates the effect without tightening it,
because heterogeneity grows faster than the square root of the group count.

The depth-cap sweep, which shows the locus of a secret moving under
manipulation rather than observation, comes from `artifacts/depth` with the pass
budget announced in the prompt and `artifacts/quiet` with it removed, and
`artifacts/w5/w5-quiet-adaptive.jsonl` is the undefended arm with the
announcement removed and nothing else changed. The utility and latency numbers
come from `artifacts/utility-cvm-enclave.jsonl`, scored by an independent and
stronger judge model held distinct from the crew's deployments. The deadline
provisioning sweep comes from `artifacts/cvm-dl4000.jsonl` and
`artifacts/cvm-dl6000.jsonl` together with the headline journal, and the per-run
receipts that establish the fail-closed rate live under `artifacts/cvm-runs`,
because that rate is recovered from receipts rather than from a journal field.
The generality sweep across a graded secret, three autonomy levels, three crew
topologies and three model deployments comes from the four
`artifacts/matrix-*.jsonl` journals, the second task family from
`artifacts/family-aml.jsonl`, and the historical comparison against a weaker
crew from `artifacts/experiment-usenix26-main.jsonl`.

The packet-level cross check lives under `artifacts/wire`, which holds the
capture, the journal recorded alongside it and the per-run receipts that place
each run in absolute time. One property of that capture is worth stating
explicitly because it is a deliberate design choice rather than an accident of
tooling: it was recorded with a snapshot length that keeps the internet and
transport headers and discards every payload byte, so the released capture holds
sizes and arrival times and contains no ciphertext and no plaintext at any
point. That is both what an honest-but-curious host actually sees and the only
form of such a capture we were willing to publish. At the wire, the
provider-chosen response direction still clears its calibrated null on the
attribute at 0.742 against 0.578: the coordinate proved unclosable is the one
packet-level measurement confirms.

## What the (0, 0) release costs

Canonicalizing the plan costs no detected utility, 0.64 against a 0.66 baseline,
and it runs 27 percent *faster* than the 55.6 second baseline, because
canonicalizing removes the passes an adaptive run adds. Full padding is
answer-invisible whenever every step returns inside the public deadline, but a
step that misses it is released empty and the answer collapses rather than
degrades, giving an overall 0.55 at 120.0 seconds. The cost is therefore not a
uniform quality tax but a near-total failure concentrated on the minority of runs
where a step ran long, 14.1 percent at the ten-second deadline, and since a
padded run costs plan length times deadline, that rate is a provisioning
parameter bought down in wall-clock alone.

## Running a fresh live replication

A live run needs a model deployment. Copy the template, fill in your own
endpoint and deployment names, and keep the result out of version control; the
ignore rules already exclude it.

```sh
cp .env.example .env
uv run traceguard doctor
make experiment
```

The experiment writes into `artifacts/local` rather than over the committed
journals, so a live run cannot quietly replace the paper's evidence with your
own.

## Deploying to an attested confidential virtual machine

The deployment provisions a confidential virtual machine with a virtual trusted
platform module, secure boot and guest attestation, installs the container, and
delivers the runtime environment over a channel that does not put credentials in
the image.

```sh
cd deploy/azure
cp deploy.env.example deploy.env
./deploy.sh
```

Read `deploy/azure/README.md` before you run this, because it documents the
region and size constraints, the deterministic hostname the template derives,
and the teardown path. The machine bills by the hour, so tear it down when you
are not using it.

Once the service is running there, the attestation module collects evidence from
the virtual trusted platform module, submits it to Microsoft Azure Attestation,
and verifies the returned token. On a machine with no such device the same code
reports the isolation as simulated and never asserts a hardware-backed result.

## Verifying rather than trusting

Two kinds of verification are available and they answer different questions.
Verifying a receipt answers whether a particular run's release carried the
guarantee it claims, and verifying the journal answers whether every row joins
to an archived attestation epoch.

```sh
uv run traceguard verify --receipt artifacts/runs/<run-id>/receipts.jsonl
uv run traceguard verify --journal artifacts/cvm-honest.jsonl
```

The interesting property of the verifier is not that it detects a mutated field,
which fails immediately on the fixed envelope size and therefore proves nothing
about the claims. It is that a prover who holds a valid signing key still cannot
assert more than the mechanism delivered. A receipt claiming an unconditional
result over the whole observable is refused with the reason named, because the
provider-chosen coordinate admits no such claim, and a partial claim that omits
its open coordinate or its measured residual is refused as well. The third SDK
example demonstrates this by re-signing a mutated body with a valid key, so that
the size and the signature are both correct and the claim is the only thing
wrong.

## What this artifact does not establish

The observables are recorded by the guest's own runtime, so the timing is
application level and driven by model round trips. A real host additionally
observes virtual machine exits and direct memory access activity, which this
artifact does not measure at all, so the confidential-VM arm establishes that
the channel survives a genuine boundary rather than measuring everything a host
could extract. The packet-level result narrows this gap for the byte counts
specifically and confirms that the response-direction channel survives at the
wire, but it is forty-eight runs at one repetition on the undefended condition
and is a cross check rather than a replacement for the headline protocol.

The adversary is passive and offline. A single-stepping or page-fault adversary
reads intra-virtual-machine branches that no application-level defense can hide,
and the artifact does not defend the content-level cache and token-length
channels a co-tenant or a network adversary can mount. Robustness against active
and adaptive adversaries is not claimed.

The headline attacker is a logistic regression over aggregate features, so every
positive result is a lower bound rather than a worst case. The paper measures
that slack rather than asserting it, and the slack is real: an unfitted threshold
on maximum request size beats the deployed attacker on the attribute, 0.816
against 0.784, so an adversary here needs no machine learning at all, and a
representation that preserves *where* in the plan each value occurred takes
membership to 0.794 against 0.721, which is exactly what a control-flow-carried
secret predicts.

The statistical resolution is bounded and the paper says so. Holding out whole
groups makes the group count the effective sample size, so the headline design
resolves the existence, direction and sign of an effect rather than its magnitude
in an arbitrary deployment, and more groups buy generality rather than precision.
Read a negative result as a failure to detect rather than as a demonstration of
absence. In the notion's own units the undefended release is certified only from
below, spending at least epsilon 0.67 under the query-neighbor relation at 95
percent confidence, and nothing here certifies membership or the padded residual.

Finally, the corpus is entirely synthetic and fictional, authored for this study.
It contains no protected health information, no human subjects were involved, and
no personal data was collected. Half the extended corpus was drafted with AI
assistance, which is a real hazard rather than a formality, so the corpus builder
enforces a per-role character band across the sensitivity rungs and refuses to
emit a corpus that violates it, and the paper audits the residual length confound
rather than asserting its absence. The workload imitates clinical prior
authorization but is not a decision-support system, is not a medical device, and
produces no medical advice.

## Repository size

The committed artifacts are roughly one hundred and thirty megabytes, of which
the per-run receipt store under `artifacts/runs` is the great majority. That
store is what allows a third party to check any individual run rather than
trusting an aggregate, which is why it is committed rather than summarized. If
you are cloning only to recompute the attack, a shallow clone is enough, and if
you are mirroring this repository somewhere with a file size policy, consider
serving the artifact directories as a release asset or through large file
storage instead.

## License

The code is released under the Apache License, version 2.0, and the full text is
in `LICENSE`, with attribution notes in `NOTICE`. Until the paper appears, cite
the paper this artifact accompanies.

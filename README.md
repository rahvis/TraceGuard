# TraceShield: Trace-Reconstruction Attacks and Trace-Private Defenses for Confidential LLM Agents

This repository is the research artifact for the paper of the same name. It
contains the multi-agent system the paper measures, the attack the paper mounts
against it, the defense the paper proposes, the receipt and attestation
machinery that certifies which release actually ran, the deployment scripts that
provision an attested confidential virtual machine, the experiment journals
every reported number is computed from, and the analysis scripts that recompute
those numbers from those journals. It is organized so that a reader can start
from a clean checkout, run something meaningful within a minute without a model
key, and then decide how far up the ladder of cost and fidelity they wish to go.

## The problem, in one paragraph

An autonomous language-model agent decides its own execution. How many
retrieval passes it issues, whether it loops back to repair an unsupported
claim, and whether it revises after review are all runtime decisions
conditioned on the private query and on the private documents the agent
encounters. The shape of the resulting execution is therefore a function of the
private inputs, and that shape remains visible to the operator of the
infrastructure even when the agent runs inside a confidential virtual machine
composed with oblivious retrieval and output differential privacy. Content is
sealed, access is oblivious, the answer is private, and the trace still leaks.
This artifact measures how much, shows which coordinates carry the leak, closes
the ones a defense can close, proves that one of them cannot be closed by any
mechanism inside the guest, and emits a signed receipt stating exactly which of
those two situations a given run was in.

## What is in here and what each part is for

The Python package under `src/traceguard` is the system itself. Within it,
`agents.py` and `graph.py` implement the six-agent prior-authorization crew on
LangGraph, `instrumentation.py` records the host-observable trace and nothing
else, `shield_runtime.py` implements the three release conditions,
`providers.py` wraps the model providers and performs the request padding at the
point where bytes actually leave, `attack.py` implements the
Trace-Reconstruction Attack together with its statistical controls,
`attestation.py` collects and verifies the confidential-VM evidence,
`receipt.py` and `verifier.py` implement the Ed25519 receipts and the
independent verifier, `ledger.py` implements the hash chain, `corpus.py` and
`domains.py` load the synthetic corpora, `experiment.py` drives the resumable
experiment matrix, and `cli.py` provides the command line. The directory
`sdk` documents the programmatic surface and holds four runnable examples;
start there if you intend to embed the runtime in your own program rather than
reproduce a table.

The directory `scripts` holds the analysis and reproduction entry points. The
single script `scripts/reproduce_all.sh` runs the whole offline pipeline in the
order the pipeline requires. The individual generators behind it are
`make_paper_artifacts.py` for the main attack tables, controls, ablations and
numeric macros, `make_matrix_artifacts.py` for the generality sweep,
`make_family_artifacts.py` for the second task family,
`make_deadline_artifacts.py` for the deadline provisioning sweep,
`make_wire_artifacts.py` for the packet-level cross check, and
`make_paper_figures.py`, `make_asymmetry_figure.py` and `make_wire_figure.py`
for the figures. Three further scripts exist for release hygiene rather than for
results: `verify_citations.py` resolves every arXiv identifier in the
bibliography against the arXiv record and fails on a title, first-author or
asserted-venue mismatch, `scan_secrets.py` scans tracked files for credential
material and self-tests its own patterns against a runtime-assembled canary, and
`verify_release.sh` runs the whole gate.

The directory `deploy/azure` provisions the confidential virtual machine. The
entry point is `deploy/azure/deploy.sh`, which reads a local `deploy.env` that
you create from the committed `deploy.env.example`, applies `main.bicep`, and
waits for the cloud-init host preparation to finish. The teardown script is
alongside it, and you should use it, because the machine bills by the hour.

The directory `artifacts` holds the evidence. The directory `tables` holds the
generated numbers, including the cached calibrated nulls, so that a fresh
reproduction can be compared against the values the paper reports rather than
merely producing numbers of its own. The directory `figures` holds the generated
figures. The directory `schemas` holds the JSON schemas for the trace and the
receipt. The directory `.github/workflows` holds the continuous integration
that gates this repository, and it is worth reading rather than assuming: beyond
linting and the test suite it regenerates the paper's tables from the archived
journals and fails if any of them differ from what is committed, recomputes the
calibrated permutation nulls and compares them against the cached ones, scans
every tracked file for provider credential material, and asserts that the
attestation module reports an honest simulated verdict off Azure rather than
fabricating a hardware claim. The directory `tests` holds the test suite, and the
directory `docs`
holds the longer-form design and reproducibility notes. If you are reviewing
this artifact rather than building on it, read `docs/CLAIMS.md` first. It maps
every quantitative claim in the paper to the journal it was computed from, the
script that computes it and the generated file the value lands in, and it also
records the claims the paper deliberately declines to make and the reasons why,
including the statistical power limitation that bounds the headline result.

## A note about names

The system described in the paper is called TraceShield. The Python package and
the command line entry point are both called `traceguard`, and they are left
that way here deliberately. The paper cites `traceguard experiment`,
`traceguard verify --journal`, the paths `scripts/make_paper_artifacts.py`,
`scripts/verify_citations.py` and `deploy/azure/deploy.sh`, and the module
filenames `agents.py`, `graph.py`, `instrumentation.py`, `shield_runtime.py` and
`attestation.py`. Renaming any of those in the published artifact would put the
artifact and the paper out of agreement, which is precisely the kind of
discrepancy artifact evaluation is meant to catch, so the names stay as the
paper cites them.

## Requirements

You need Python 3.11, 3.12 or 3.13. The exact versions the committed tables and
figures were produced under are recorded in `environment-tested.txt`, which is a
record rather than an install target; the analysis is deterministic given the
journals and a fixed environment, but not across scikit-learn versions, and
`docs/REPRODUCIBILITY.md` explains where that mattered and what it did not change. The dependency set is installed from
`pyproject.toml` and includes cryptography for the Ed25519 receipts, LangGraph
for the crew, the OpenAI client for the model calls, and NumPy, SciPy and
scikit-learn for the analysis. The development extra additionally installs
pytest, ruff and matplotlib, and you need the development extra to run the tests
or regenerate the figures. Everything that recomputes a number from a committed
journal runs on a laptop with no network access and no credentials. Only a fresh
live replication needs a model deployment, and only the deployment scripts need
an Azure subscription. Reproducing the packet-level appendix from the committed
capture needs nothing extra, although recording a new capture needs `tcpdump`
and root on the guest.

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

The second tier is reanalysis. The journals committed under `artifacts` are the
recorded output of real runs against real model deployments, and the analysis
scripts recompute every table, null, ablation, figure and macro from them. This
is the tier that regenerates the paper's numbers, it needs no credentials, and
it is what most readers will want. What it does not do is re-derive the journals
themselves; it recomputes the analysis over the journals as committed.

The third tier is live replication. A fresh run against a configured model
deployment produces new journal rows labelled `live_replication`. This is a
genuine new measurement rather than a relabeling of a prior result, it costs
money, and the numbers will differ slightly from the paper's because the models
are stochastic and the fail-closed rate depends on the latency profile of the
machine you run on.

## Reproducing the paper's numbers

The whole offline pipeline runs from one script. On a laptop the calibrated
nulls dominate the runtime, because they permute labels two thousand times
within each held-out group and rerun the full leave-one-group-out attack on
every draw.

```sh
scripts/reproduce_all.sh
```

If you want a faster pass while you are still finding your way around, set a
lower permutation count. Do so knowingly: the script will warn you, because a
lower count writes an underpowered null into `tables/macros.tex` and nothing in
`tables/nulls.json` records the substitution.

```sh
PERMS=200 scripts/reproduce_all.sh
```

When it finishes, `tables` holds every attack table, calibrated null, ablation,
deadline row and numeric macro regenerated from the committed journals, and
`figures` holds the figures. Compare those against the values quoted in the
paper. The individual stages are also available as Make targets, so that
`make analysis`, `make matrix`, `make family`, `make deadline`, `make wire`,
`make figures` and `make citations` each run one part, and `make help` lists
them with a one-line description of what each needs.

## Where the paper's headline numbers come from

The headline attack is computed from `artifacts/cvm-honest.jsonl`, which is the
journal of the arm that ran inside a genuine AMD SEV-SNP confidential virtual
machine with attestation verified by Microsoft Azure Attestation. Every row in
it carries a hardware-rooted binding that can be checked from the journal alone
with `traceguard verify --journal`. From that journal the attack recovers the
sensitive attribute at a group-aware area under the curve of zero point seven
eight four against a calibrated null of zero point six zero, and canary
membership at zero point seven two one against its own null of zero point five
eight. Under the fully padded release membership falls to zero point five seven
seven and no longer clears its null, while the attribute stands at zero point
seven six four, and that asymmetry is the paper's central practical result.

The utility and latency numbers come from `artifacts/utility-cvm-enclave.jsonl`,
the judged run scored by an independent and stronger judge model held distinct
from the crew's models. The deadline provisioning sweep comes from
`artifacts/cvm-dl4000.jsonl` and `artifacts/cvm-dl6000.jsonl` together with the
headline journal, and the per-run receipts that establish the fail-closed rate
live under `artifacts/cvm-runs`. The generality sweep across a graded secret,
three autonomy levels, three crew topologies and three model deployments comes
from the four `artifacts/matrix-*.jsonl` journals, and the second task family
comes from `artifacts/family-aml.jsonl`. The packet-level cross check that
measures the observable at the guest's virtual network interface rather than
with the guest's own byte counters comes from `artifacts/wire`, which holds the
header-only capture, the journal recorded alongside it, and the per-run
receipts that place each run in absolute time. The historical comparison the
paper draws against a weaker crew comes from
`artifacts/experiment-usenix26-main.jsonl`.

One property of the capture is worth stating explicitly because it is a
deliberate design choice rather than an accident of tooling. It was recorded
with a snapshot length that keeps the internet and transport headers and
discards every payload byte, so the released capture holds sizes and arrival
times and contains no ciphertext and no plaintext at any point. That is both
what an honest-but-curious host actually sees and the only form of such a
capture we were willing to publish.

## Running a fresh live replication

A live run needs a model deployment. Copy the template, fill in your own
endpoint and deployment names, and keep the result out of version control; the
ignore rules already exclude it, and `scripts/scan_secrets.py` will complain if
credential material reaches a tracked file.

```sh
cp .env.example .env
uv run traceguard doctor
make experiment
```

The experiment writes into `artifacts/local` rather than over the committed
journals, so a live run cannot quietly replace the paper's evidence with your
own. The judged utility run is separate, because it spends money on a second and
stronger model, and it is available as `make judge`.

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
reports the isolation as simulated and never asserts a hardware-backed result,
which is the behaviour a test in the suite specifically pins.

## Verifying rather than trusting

Two kinds of verification are available and they answer different questions.
Verifying a receipt answers whether a particular run's release carried the
guarantee it claims, and verifying the ledger answers whether the sequence of
receipts has been tampered with.

```sh
uv run traceguard verify --receipt artifacts/runs/<run-id>/receipts.jsonl
uv run traceguard verify --journal artifacts/cvm-honest.jsonl
```

The interesting property of the verifier is not that it detects a mutated field,
which fails immediately on the fixed envelope size and therefore proves nothing
about the claims. It is that a prover who holds a valid signing key still cannot
assert more than the mechanism delivered. A receipt claiming an unconditional
result over the whole observable is refused with the reason named, and a partial
claim that omits its open coordinate or its residual is refused as well. The
third SDK example demonstrates this by re-signing a mutated body with a valid
key, so that the size and the signature are both correct and the claim is the
only thing wrong.

The bibliography can be verified the same way, which matters because a citation
that does not exist or that names the wrong authors is treated as misconduct
rather than as a typo.

```sh
uv run python scripts/verify_citations.py --bib refs.bib
```

That script resolves every arXiv identifier against the arXiv record and fails
on a mismatch of title, first author, or an asserted venue the record does not
support. It also lists the entries carrying only a digital object identifier or
a venue address, which it cannot check, so that the residual gap is visible
rather than implied.

## Tests

```sh
uv run pytest
uv run ruff check .
```

The suite covers the trace and receipt types, the attack statistics, the ordinal
statistics, the attestation honesty properties, the topology and domain
plumbing, the packet attribution, and a set of publication properties. Three of
the publication tests assert properties of the manuscript itself and skip with a
clear reason in this repository, because the LaTeX sources of the paper are not
part of the artifact. Everything else runs, and a clean checkout should report
no failures.

## What this artifact does not establish

The observables are recorded by the guest's own runtime, so the timing is
application level and driven by model round trips. A real host additionally
observes virtual machine exits and direct memory access activity, which this
artifact does not measure at all, so the confidential-VM arm establishes that
the channel survives a genuine boundary rather than measuring everything a host
could extract. The packet-level appendix narrows this gap for the byte counts
specifically and confirms that the response-direction channel survives at the
wire, but it is forty eight runs at one repetition on the undefended condition
and is a cross check rather than a replacement for the headline protocol.

The adversary is passive and offline. A single-stepping or page-fault adversary
reads intra-virtual-machine branches that no application-level defense can hide,
and the artifact does not defend the content-level cache and token-length
channels a co-tenant or a network adversary can mount. Robustness against active
and adaptive adversaries is not claimed.

The statistical resolution is bounded and the paper says so. The headline design
has six held-out groups drawn from three clinical specialties, so the minimum
effect it can detect at eighty percent power is close to the size of the total
signal above the null. A defense that halved the attribute leak would have been
reported as no reduction. Read the negative results as failures to detect rather
than as demonstrations of absence.

Finally, the corpus is entirely synthetic and fictional, authored for this study.
It contains no protected health information, no human subjects were involved, and
no personal data was collected. The workload imitates clinical prior
authorization but is not a decision-support system and produces no medical
advice.

## Repository size

The committed artifacts are roughly one hundred and thirty megabytes, of which
the per-run receipt store under `artifacts/runs` is the great majority. That
store is what allows a third party to check any individual run rather than
trusting an aggregate, which is why it is committed rather than summarized. If
you are cloning only to reproduce the tables, a shallow clone is enough, and if
you are mirroring this repository somewhere with a file size policy, consider
serving the artifact directories as a release asset or through large file
storage instead.

## License, security and citation

The code is released under the Apache License, version 2.0, and the full text is
in `LICENSE` with attribution notes in `NOTICE` and third-party acknowledgements
in `THIRD_PARTY.md`. Please read `SECURITY.md` before reporting a vulnerability,
and `CONTRIBUTING.md` if you intend to send a change. Citation metadata is in
`CITATION.cff`, which will name the paper once it is published; until then, cite
the paper this artifact accompanies.

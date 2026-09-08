# The evidence directory

This directory holds the experiment journals every number in the paper is
computed from, the per-run receipt stores that let a third party check any
individual run rather than trusting an aggregate, and the packet capture behind
the wire-level appendix. It is roughly one hundred and thirty megabytes, of
which the receipt store under `runs` is the great majority. Everything here
derives from a fully synthetic and fictional corpus authored for this study.
No real patient, payer or provider data exists in any of these files, and no
prompt, document, query or answer is recorded in any journal: a journal row
carries the host-observable trace and its provenance, and nothing else.

## Which journal is the headline

The headline arm is `cvm-honest.jsonl`. It holds 575 completed runs across the
undefended, structure-canonicalized and fully padded conditions, executed inside
a genuine AMD SEV-SNP confidential virtual machine. All 575 rows carry a
hardware-backed attestation binding issued through Microsoft Azure Attestation,
which you can check from the journal alone without trusting this sentence.

```sh
uv run traceguard verify --journal artifacts/cvm-honest.jsonl
```

Its companion `cvm-honest.attestation.json` is the archived evidence sidecar,
accumulated one epoch per command invocation, that the journal rows join
against. The two deadline settings for the provisioning sweep are
`cvm-dl4000.jsonl` and `cvm-dl6000.jsonl`, at 24 completed padded runs each, and
both are likewise fully attested. The per-run receipts for all of these live
under `cvm-runs`, which holds 654 run directories and is what establishes the
fail-closed rate, because the rate is recovered from the receipts rather than
from a journal field.

## The generality sweep and the second task family

Four journals hold the star design that varies one factor at a time. The file
`matrix-baseline.jsonl` holds 576 completed runs of the graded sensitivity
ladder, `matrix-autonomy.jsonl` holds 96 runs across three autonomy levels,
`matrix-architecture.jsonl` holds 192 runs across three crew topologies, and
`matrix-model.jsonl` holds 288 runs across three model deployments. The file
`matrix-cvm.jsonl` holds 572 completed runs of the same design executed inside
the confidential virtual machine, all attested, and `matrix-cvm.attestation.json`
is its evidence sidecar. The summaries `matrix-summary.json` and
`matrix-report.json` are generated from those journals.

The second task family is `family-aml.jsonl`, holding 288 completed runs of an
anti-money-laundering alert triage workload whose sensitivity ladder runs in the
opposite direction to the clinical one, which is why it is a control on the
confound rather than merely a second domain. Its summary is
`family-report.json`.

An important scoping property of the sweep journals: they were recorded under
earlier single-direction instrumentation that measured the response size and not
the request, so they are statements about structure, timing and the response
coordinate and are not commensurable with the headline arm's four coordinates.
The analysis code refuses to pool them, and it should. Because the request
direction only ever adds an observable, each of those arms is a lower bound on
what a host seeing both directions would recover.

## Utility

Two judged runs exist and they measure the same thing on different machines.
The file `utility-cvm-enclave.jsonl` holds 428 scored cells from the in-enclave
run and is the one the paper reports. The file `utility-azure.jsonl` holds 432
scored cells from an earlier host run and is retained because the paper compares
against it: the two disagree on the fail-closed rate by a wide margin, which is
the evidence that the rate is a property of a deployment's latency profile
rather than of the mechanism. In both, utility is scored by an independent and
stronger judge model held distinct from the crew's models, and that independence
is asserted in code rather than promised.

## The wire-level capture

The directory `wire` holds the packet-level cross check. Inside it,
`wire.jsonl` is the journal of 48 sequential undefended runs,
`wire-packets.txt` is the capture in text form, `wire-report.json` is the
generated analysis, and `runs` holds the 48 per-run receipts that place each run
in absolute time, which is what makes packet attribution possible at all.

The capture was recorded at the guest's virtual network interface with a
snapshot length that keeps the internet and transport headers and discards every
payload byte. It therefore contains sizes and arrival times and no ciphertext
and no plaintext at any point. That is both what an honest-but-curious host
actually sees and the only form of such a capture we were willing to publish.
The runs are sequential on purpose, because concurrent runs interleave packets
and no segment can then be attributed to a step.

## The superseded comparison

The file `experiment-usenix26-main.jsonl` holds 576 completed runs on an earlier
and weaker crew, recorded on a host rather than in an enclave, and it carries no
attestation provenance. It is retained because the paper leans on the
comparison rather than discarding it: membership is undetectable on that crew
and a real leak on the stronger one, and the useful half of that result is that
a channel can sit below the detection threshold of one study and above it for
the next model. The paper's historical macros are generated from this journal
under a `hist` prefix so that both halves of every weak-against-strong sentence
are generated rather than typed.

## The receipt store

The directory `runs` holds 1869 run directories, each with its signed receipt.
Along with `cvm-runs` and `wire/runs` it is why this repository is large. It is
committed rather than summarized because the point of a receipt is that someone
other than us can check it, and an aggregate cannot be checked.

## A local run writes elsewhere

A fresh reproduction writes into `artifacts/local`, which the ignore rules
exclude, so running the experiment yourself cannot quietly replace the paper's
evidence with your own output. The same applies to the judged utility run. If
you want to compare, run the analysis over both and diff the generated tables.

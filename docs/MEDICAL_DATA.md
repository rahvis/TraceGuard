# Synthetic medical data statement

`synthetic-medical-v2.0.0` was manually authored for this artifact. It does not
derive from, paraphrase, or encode a patient chart, clinician note, hospital
protocol, contact record, or protected identifier.

The dataset uses fictional source IDs and qualitative documentation conflicts to
exercise an adaptive writing graph. Numerical clinical values, doses, treatment
instructions, real organizations, and real people are intentionally absent.

Each of six topics expands over a four-rung sensitivity ladder crossed with the
membership arm, giving eight cells per topic and 48 cases in total:

| Level | Framing | Step-therapy documentation | Diagnostics | Fixture depth |
|---|---|---|---|---|
| L0 | `routine` | complete | confirmatory | 4 hops |
| L1 | `guarded` | minor gap | suggestive | 5 hops |
| L2 | `elevated` | material gap | equivocal | 6 hops |
| L3 | `sensitive` | absent | pending | 7 hops |

The gradation is carried by the prose itself — the step-therapy and diagnostic
sentences, the service requested, and the code — not by a label. Character counts
are held constant across the four rungs so text volume cannot stand in for the
level; `scripts/gen_pa_corpus.py` refuses to write a corpus that violates that
band.

The canary replaces a distractor so every corpus contains exactly six documents.
It names no service: naming the top-of-ladder service in every canary (as
v1 did) would have entangled the membership and attribute labels in the text, so
it carries only a distinctive prior-therapy timeline and a named comorbidity, at
the same length as the distractor it replaces. The `sensitivity_level` and
`canary_member` labels exist only for privacy-attack research. They are not
clinical labels.

Fixture-provider behavior is deliberately parameterized by those labels to test
the pipeline. It is circular by design and must never be cited as evidence that a
real agent leaks. Live-provider behavior is measured separately.

This software is for research only. It is not medical advice, a medical device,
or a clinical decision-support system.


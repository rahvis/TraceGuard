# The TraceShield research SDK

Constructing `TraceGuardSDK` loads the synthetic corpus, selects a model
provider, generates or loads an Ed25519 receipt signing key, and prepares a hash
chained ledger. Calling `run` executes one case under one release condition and
returns a `RunResult` carrying the answer, the host observable trace, a signed
receipt, timing and byte metrics, the provider provenance label and the dataset
digest. The trace is metadata only. No prompt, document, query, answer or API key
is ever written into it, which is why the same object can be handed to the attack
code and to the verifier without a redaction step.

The provider is chosen by name. Passing `fixture` selects a deterministic
harness that needs no key and no network, and every result it produces is
labelled `synthetic_fixture`. Passing `azure` selects a live model deployment,
and results are labelled `live_replication`. The labels are enforced in code
rather than written by hand, because the difference between the two matters more
than any number either produces: fixture output demonstrates that the pipeline
runs, and only a live run measures anything about a real agent.

The release condition is the second axis. Passing `adaptive` runs the agent
undefended, so its control flow, timing and request sizes all vary with the
private data. Passing `structure` forces a canonical plan. Passing `full`
additionally holds each step to the public deadline and pads each request to the
public ceiling, which is the release the paper's zero epsilon and zero delta
claim applies to. Because a canonicalized release requires a data independent
plan, the defended conditions require `autonomy="scripted"` and will refuse any
other value rather than emit a trace that is not the canonical one.

## The examples

Run them from the repository root so that the relative artifact paths resolve.
Each one is independent and none needs a model key.

```sh
python sdk/examples/01_offline_single_run.py
python sdk/examples/02_defended_run_and_receipt.py
python sdk/examples/03_verify_receipt_and_ledger.py
python sdk/examples/04_attack_a_journal.py
```

The first runs one case undefended and prints the four host observable
coordinates for every step, which is the concrete form of the object the whole
paper is about. The second runs the same case under full padding and prints the
guarantee the receipt actually claims, so you can see that it names
`ingress_size` as an open coordinate rather than asserting an unconditional
result. The third verifies a receipt and the ledger, and then re signs a mutated
body with a valid key so that the size and the signature are both correct and
the claim is the only thing wrong, which is the refusal that matters: a signing
key alone must not be enough to assert a guarantee the runtime did not deliver.
The fourth scores the attack on a committed journal and prints the single
coordinate area under the curve for each observable against both secrets.

## A note on names

The system described in the paper is called TraceShield. The Python package and
the command line entry point are both called `traceguard`, and they stay that
way here on purpose. The paper cites `traceguard experiment`,
`traceguard verify --journal` and the module filenames `agents.py`, `graph.py`,
`instrumentation.py`, `shield_runtime.py` and `attestation.py` directly, so
renaming them in the published artifact would put the artifact and the paper out
of agreement, which is exactly what artifact evaluation checks for.

## Reading the numbers an example prints

Every area under the curve this artifact reports has to be read against its own
calibrated null rather than against one half. The attack protocol selects the
better of two attackers by fold mean and reports the inversion safe maximum of
the score and one minus the score. Both choices favour the attacker, and together
they place the null of the design well above chance, near zero point five eight
for membership and near zero point six for the attribute on the headline arm. The
nulls for every arm reported in the paper are cached in `tables/nulls.json`
alongside the digest of the journal they were computed from, and the generator
refuses to pair that cache with a journal it was not built from.

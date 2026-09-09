# Python SDK

## Install

From the repository:

```bash
uv sync
```

Or build a wheel with `uv build` and install it into Python 3.11/3.12.

## Run a single case

```python
from traceguard import TraceGuardSDK

sdk = TraceGuardSDK(provider="fixture")
case = sdk.corpus.list_cases()[0]

result = sdk.run(
    case_id=case.id,
    condition="adaptive",
    seed=7,
)

print(result.to_dict())
```

`fixture` makes no network call. It exists to verify the graph and artifact
pipeline. Its behavior is label-parameterized and is not scientific evidence.

## Run live on Azure OpenAI

```python
import os
from traceguard import Settings, TraceGuardSDK

assert os.environ.get("AZURE_OPENAI_API_KEY")
assert os.environ.get("AZURE_OPENAI_ENDPOINT")
settings = Settings.from_env()
sdk = TraceGuardSDK(settings=settings, provider="azure")
result = sdk.run(
    case_id=sdk.corpus.list_cases()[0].id,
    condition="adaptive",
    seed=7,
)
```

The SDK uses Azure OpenAI Chat Completions (not the Responses API, which defaults
to server-side retention). The adapter probes each deployment once to learn whether
it accepts `max_tokens` or `max_completion_tokens`, and whether it accepts `seed`
and `temperature`, then omits whatever that deployment rejects; a deployment that
silently ignores `seed` is surfaced rather than swallowed. `model` is an Azure
**deployment** name, not a base model name. The exact resolved deployment names and
package versions are written to the run manifest by the experiment layer. No model
fallback is allowed without an explicit new configuration.

## Conditions

```python
for condition in ("adaptive", "structure_only", "full_pad"):
    result = sdk.run(case_id=case.id, condition=condition, seed=7)
```

Only a successful full-pad run whose application-level invariants all hold may
carry a `(0,0)` receipt claim. Adaptive and structure-only receipts report
empirical status rather than a certified epsilon.

## Metadata events

```python
events = []
result = sdk.run(
    case_id=case.id,
    condition="adaptive",
    event_callback=events.append,
)
```

Callbacks receive safe lifecycle/step metadata. Applications must not add query,
document, prompt, result, or answer fields to this channel.

## Verification

```python
from traceguard.verifier import verify_receipt

verification = verify_receipt(
    result.receipt,
    trusted_public_key=result.receipt["signature"]["public_key"],
)
assert verification.valid
```

Pin the key from an independent trust channel in a real deployment. Trusting only
the key carried inside a receipt does not authenticate the signer.

## Result handling

`RunResult.trace` is safe for the modeled host observer. The generated summary,
if present, is an inside-boundary return value. Do not persist or stream it from
an experiment service unless a separate data policy permits that action.


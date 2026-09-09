# Architecture

```text
browser / Python SDK / CLI
          |
          v
 FastAPI run manager ---- immutable run artifact directory
          |
          v
 TraceGuardSDK -> LangGraph prior-authorization crew -> fixture or Azure OpenAI provider
          |                    |
          |                    +-- private synthetic query/documents/answer
          v
 metadata recorder -> TraceShield release -> attack analysis
          |                    |
          v                    v
  Ed25519 receipt ------> hash-chained ledger
```

## Trust boundaries

The program separates an inside-boundary value plane from an observable metadata
plane. The value plane contains the synthetic query, source text, prompts, model
responses, and final summary. The observable plane contains only step type,
elapsed time, and egress bytes. Labels needed for supervised evaluation live in
the experiment envelope, not in the trace object supplied to the attacker.

The container is not a TEE. This separation is an application-level emulation
used to test the paper's measurement and release logic.

## Runtime conditions

Adaptive execution uses LangGraph conditional edges. Structure-only and full-pad
execution select a public fixed graph path with seven research iterations and no
data-dependent repair/revision edges. Full-pad additionally applies public
timing/size targets and emits an honest overrun status.

## Realtime console

The API runs jobs in background workers and appends safe events to the run store.
The browser consumes them over server-sent events. The console never accepts an
API key and has no build-time result bundle. Paper reference values and current
run values come from different endpoints and remain visually labeled.

## Persistence

`artifacts/` is a bind-mounted `/data` directory in Docker. Signing keys live only
under `/data/runtime`; run evidence lives under `/data/runs`. Both are excluded
from the image and source archive.


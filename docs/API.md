# Local research API

The API is intentionally local and unauthenticated. Do not expose it to an
untrusted network. It never accepts an API key from the browser.

## Service metadata

```text
GET  /api/health
GET  /api/config
GET  /api/cases
GET  /api/paper/results
GET  /api/runtime
GET  /api/attestation
POST /api/attestation/verify        {"include_token": true}
```

`/api/config` returns only safe resolved settings — provider mode, Azure OpenAI
endpoint and API version, deployment names, and budgets; the API key is never
included. `/api/health` reports the active `provider` plus
`azure_openai_configured`. `/api/paper/results` always returns
`paper_reported_unverified` provenance.

`/api/runtime` is an honest, read-only self-report of the isolation posture; its
`hardware_tee.quote_available` and `implementation_scope.hardware_attestation`
flip `true` only for a genuine, MAA-verified Azure Confidential VM quote.
`/api/attestation` returns the current attestation verdict (`verified` on an
attested CVM, `simulated` otherwise) with the MAA token, measurement claims, and a
`token_sha256`. `POST /api/attestation/verify` re-collects evidence and
independently re-checks the RS256 signature + `*.attest.azure.net` issuer against
the provider JWKS, returning the result under `reverification`.

## Start and inspect a run

```http
POST /api/runs
Content-Type: application/json

{
  "kind": "single",
  "case_id": "cardiology-anticoagulation-review-routine-m0",
  "provider": "fixture",
  "condition": "adaptive",
  "seed": 20260710
}
```

The response contains a `run_id`; poll or stream:

```text
GET  /api/runs/{run_id}
GET  /api/runs/{run_id}/events       # text/event-stream
GET  /api/runs/{run_id}/traces
GET  /api/runs/{run_id}/metrics
GET  /api/runs/{run_id}/receipts
POST /api/runs/{run_id}/cancel
```

SSE step events expose only step type, timing, egress bytes, hop/progress, and
public release status. They never include query, source, prompt, answer, or API
response content.

## Verify a receipt

```http
POST /api/receipts/verify
Content-Type: application/json

{
  "receipt": {"...": "..."},
  "trusted_public_key": "optional independently pinned key"
}
```

Omitting the trusted key checks signature integrity against the embedded key but
does not authenticate signer identity.

## Errors

Errors are JSON and contain a public error category plus run identifier. Provider
response bodies, source text, prompts, environment values, and credentials are
never returned. Full-pad failures are marked as failures/overruns rather than
silently converted into successful privacy receipts.


# Implementation and verification plan

This plan is part of the public artifact so reviewers can distinguish intended
scope from completed evidence. A release may mark a gate complete only when its
command and generated evidence are available.

## Phase 1 - evidence audit

- [x] Read and visually inspect all 11 paper pages.
- [x] Inventory the target directory and surrounding workspace.
- [x] Identify the missing original traces/code/results chain.
- [x] Record paper values as `paper_reported_unverified`, anchored to PDF/TeX
  hashes.
- [x] Exclude the static product/investor demo from the artifact boundary.

## Phase 2 - public research package

- [x] Establish Apache-2.0 source boundary and CC0 synthetic-data statement.
- [x] Pin Python/provider/runtime dependencies in `uv.lock`.
- [x] Define trace and receipt schemas that exclude payload/cardinality.
- [x] Add fixed-six-document synthetic medical cells and provenance.
- [ ] Complete SDK, six-agent graph, providers, trace recorder, TraceShield,
  receipts, ledger, and verifier.
- [ ] Complete group-aware attack, controls, ablations, frontier, and manifests.

## Phase 3 - live console and container

- [ ] Serve a single minimal research landing page from the running API.
- [ ] Stream safe multi-agent events over SSE from actual jobs.
- [ ] Keep paper-reference values visibly separate from current-run values.
- [ ] Verify receipt integrity from the console.
- [x] Define a single-container Docker/Compose stack with persistent artifacts.

## Phase 4 - verification

- [ ] Unit tests for corpus, trace allow-list, graph paths, provider redaction,
  padding invariants, signatures, key pinning, ledger integrity, and statistics.
- [ ] Offline end-to-end fixture run with committed verification summary.
- [ ] API integration tests and browser-driven live-run test.
- [ ] Docker image build, health check, API run, SSE completion, and persisted
  artifact check.
- [ ] Secret scan, source archive inspection, and clean installation test.
- [ ] Final limitation and claim-language review.

## Phase 5 - publisher-owned release gates

- [ ] Revoke the credential exposed before reconstruction and use a new secret.
- [ ] Add real author, copyright, repository URL, and citation/DOI metadata.
- [ ] Confirm authority to publish the paper/code and any patent-disclosure timing.
- [ ] Run the chosen live Azure OpenAI sample size and retain all raw run artifacts.
- [ ] Conduct independent statistical/methods review.
- [ ] Generate SBOM, sign/tag release, and archive an immutable DOI snapshot.

The implementation can complete Phases 1-4 locally. Phase 5 requires identities,
authority, credentials, external review, and archival services that this code
cannot infer or fabricate.


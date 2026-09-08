# Contributing

Contributions should preserve the artifact's evidence boundaries.

1. Create a focused change with tests.
2. Run `./scripts/verify_release.sh`.
3. Do not commit generated run directories, credentials, real medical data, or
   model outputs containing private source text.
4. Never replace a measured value with a paper-reference value.
5. Add a release note for changes to data, prompts, trace features, split logic,
   defense behavior, receipt schema, or statistical methods.
6. Treat model/provider changes as a new replication protocol and version the
   manifest accordingly.

Scientific changes should include a rationale, predeclared expected effect, and
an explanation of whether old runs remain comparable. Security-sensitive changes
to trace filtering, receipts, signing, or ledger verification require two-person
review before release.

The supplied paper has no author or ownership metadata. The repository owner must
adopt a real contributor agreement and code of conduct before accepting public
contributions; placeholders are not invented here.


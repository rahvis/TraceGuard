# Security policy

## Supported version

Only the latest tagged release is supported. This is a research prototype, not a
production security control or a clinical system.

## Reporting

Before public release, replace this section with a monitored private security
contact owned by the publisher. Do not open a public issue containing a secret,
private trace, unpublished result, or vulnerability exploit.

## Secret handling

- Revoke any API key exposed in chat, source, a terminal transcript, an issue, or
  a build log.
- Inject `AZURE_OPENAI_API_KEY` only through the process environment or a container
  secret mechanism. Never put it in Compose YAML, source, examples, receipts,
  run manifests, browser storage, or screenshots.
- The application reports only whether a provider is configured; it never echoes
  the credential.
- Run `scripts/verify_release.sh` and an independent secret scanner before every
  source archive or image publication.

## Receipt trust

An Ed25519 signature proves that a receipt was unchanged after a holder of the
corresponding private key signed it. A verifier that accepts the public key
embedded in the same receipt has integrity but no external identity assurance.
Pin an expected public key or bind it to an independently verified attestation in
any real deployment.

A local hash chain detects mutation/reordering relative to a known head. It does
not prevent a signer from replacing, truncating, or forking the entire ledger.
Publish checkpoints to a trusted external system when non-omission matters.

## Threat-model boundary

The container does not deploy confidential hardware, ORAM/oblivious retrieval,
output DP, a KMS/HSM, or remote attestation. It emulates a passive application-
level observer. Do not advertise receipts as proof of any absent substrate.

## Medical safety

Only the supplied fictional corpus is supported. Do not load real clinical data
without a separate privacy, security, legal, and ethics review. Generated text is
not medical advice and must not be used for patient care.


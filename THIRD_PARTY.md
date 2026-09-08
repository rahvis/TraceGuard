# Third-party dependencies

The source tree does not vendor third-party packages. `uv.lock` is the machine-
readable dependency inventory and pins the transitive graph used for verification.

The main direct dependencies are:

- OpenAI Python SDK (`openai`, Apache-2.0) for its `AzureOpenAI` client, used for
  optional live Azure OpenAI runs. No OpenAI-hosted endpoint is contacted.
- LangGraph (`langgraph`, MIT) for the six-agent state graph.
- FastAPI and Uvicorn for the local research API and console.
- Cryptography for Ed25519 receipts.
- NumPy, SciPy, and scikit-learn for measured experiment analysis.

Run `uv tree` after `uv sync` to inspect the resolved graph. A publisher should
generate an SBOM from the final release image and review every resolved license
before distribution.


#!/usr/bin/env python3
"""Run one case offline and print the host-observable trace.

This example needs no API key and no network. It uses the deterministic fixture
provider, whose outputs are labelled ``synthetic_fixture`` and are a plumbing
harness rather than evidence about a real agent. Use it to confirm the install
works and to see the shape of what the runtime records.

    python sdk/examples/01_offline_single_run.py
"""

from __future__ import annotations

from traceguard import TraceGuardSDK

# Written to artifacts/local/, never to artifacts/runs. The latter holds the
# 1,868 signed receipt stores that back the paper's fail-closed rates, and a
# fixture run writing beside them would mix synthetic receipts into the
# evidence. artifacts/local/ is git-ignored for exactly this reason.
sdk = TraceGuardSDK(provider="fixture", artifact_dir="artifacts/local/runs")

case_id = sdk.corpus.list_cases()[0].case_id
result = sdk.run(case_id=case_id, condition="adaptive", seed=20260710)

trace = result.trace
print(f"case        {case_id}")
print("condition   adaptive (undefended)")
print(f"provenance  {result.provider_provenance}")
print(f"steps       {len(trace.steps)}")
print()
print(f"{'idx':>3}  {'step type':24}  {'wall s':>7}  {'egress B':>9}  {'ingress B':>9}")
for step in trace.steps:
    print(
        f"{step.index:>3}  {step.step_type:24}  {step.duration_s:>7.2f}  "
        f"{step.egress_bytes:>9}  {step.ingress_bytes:>9}"
    )

print()
print("The four coordinates above are what an honest-but-curious host can read:")
print("the step sequence and its length, the per-step wall clock, the request")
print("bytes the guest transmits, and the response bytes the provider returns.")
print("The first three are guest-composed. The fourth is chosen by the provider.")

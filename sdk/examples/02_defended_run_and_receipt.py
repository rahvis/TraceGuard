#!/usr/bin/env python3
"""Run the same case under full padding and inspect what the receipt claims.

Under the fully padded release the three guest-composed coordinates become
public constants: a canonical plan of fixed length, every step held to the
public deadline, every request padded to the public ceiling. The fourth
coordinate, the response size, is chosen by the model provider and crosses the
boundary before the guest is scheduled again, so no in-guest mechanism can pad
it. The receipt therefore claims a partial guarantee and names that coordinate
as open, and the verifier refuses any receipt that claims more.

    python sdk/examples/02_defended_run_and_receipt.py
"""

from __future__ import annotations

import json

from traceguard import TraceGuardSDK

# Written to artifacts/local/, never to artifacts/runs. The latter holds the
# 1,868 signed receipt stores that back the paper's fail-closed rates, and a
# fixture run writing beside them would mix synthetic receipts into the
# evidence. artifacts/local/ is git-ignored for exactly this reason.
sdk = TraceGuardSDK(provider="fixture", artifact_dir="artifacts/local/runs")
case_id = sdk.corpus.list_cases()[0].case_id

result = sdk.run(case_id=case_id, condition="full", autonomy="scripted", seed=20260710)
trace = result.trace

egress = {step.egress_bytes for step in trace.steps}
timing = {round(step.duration_s, 3) for step in trace.steps}

print(f"steps                {len(trace.steps)}")
print(f"distinct egress      {sorted(egress)}")
print(f"distinct step timing {sorted(timing)}")
print()
print("A single value in each of those sets is the point: on the closed")
print("coordinates the release is the same for every input, which is what makes")
print("the guarantee (0,0) there rather than a small number.")
print()

body = result.receipt.to_dict() if hasattr(result.receipt, "to_dict") else result.receipt
guarantee = (body.get("body") or body).get("guarantee", {})
print("guarantee as recorded in the receipt:")
print(json.dumps(guarantee, indent=2)[:900])

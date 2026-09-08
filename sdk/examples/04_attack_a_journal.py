#!/usr/bin/env python3
"""Score the Trace-Reconstruction Attack on a shipped journal.

Reads one of the journals under artifacts/ and reports the single-coordinate
ROC-AUC for each observable against both secrets. Every AUC has to be read
against its own calibrated null rather than against 0.5, because the attack
protocol reports the inversion-safe maximum over the better of two attackers,
which places the null of the design well above chance. The nulls for the
paper's arms are cached in tables/nulls.json.

    python sdk/examples/04_attack_a_journal.py [journal.jsonl]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from traceguard.attack import inversion_safe_auc

journal = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/cvm-honest.jsonl")
if not journal.is_file():
    raise SystemExit(f"no such journal: {journal}")

rows = []
for line in journal.read_text(errors="replace").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if row.get("status") == "completed" and row.get("condition") == "adaptive":
        rows.append(row)
if not rows:
    raise SystemExit("no completed undefended rows in that journal")

coordinates = {
    "step count (structure)": lambda r: float(len(r["trace"]["steps"])),
    "max step timing": lambda r: max(s["duration_s"] for s in r["trace"]["steps"]),
    "max request (egress)": lambda r: float(max(s["egress_bytes"] for s in r["trace"]["steps"])),
    "max response (ingress)": lambda r: float(
        max(s.get("ingress_bytes") or 0 for s in r["trace"]["steps"])
    ),
}
labels = {
    "attribute": [int(r["trace"]["attribute_label"]) for r in rows],
    "membership": [int(r["trace"]["membership_label"]) for r in rows],
}

print(f"{journal}  ({len(rows)} undefended runs)")
print(f"\n  {'coordinate':24} {'attribute':>10} {'membership':>11}")
for name, fn in coordinates.items():
    scores = [fn(r) for r in rows]
    a = inversion_safe_auc(labels["attribute"], scores)
    m = inversion_safe_auc(labels["membership"], scores)
    print(f"  {name:24} {a:>10.3f} {m:>11.3f}")
print("\nRead each against its calibrated null, not against 0.5. See")
print("tables/nulls.json for the nulls of the arms reported in the paper.")

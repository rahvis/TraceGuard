"""Tests for the packet-level re-analysis.

This script turns a header-only capture into the paper's headline
cross-check -- the attack recomputed on what a host's NIC sees rather than on
the guest's own counters -- so its two fragile parts are pinned here. Both have
a silent failure mode: a parser that drops packets reports a smaller channel,
and an attribution window that misaligns reports no channel at all, and neither
raises.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_wire_artifacts import attribute, read_packets  # noqa: E402


def test_packet_parser_keeps_payload_segments_and_assigns_direction(tmp_path: Path) -> None:
    cap = tmp_path / "pkts.txt"
    cap.write_text(
        # egress: destination port 443
        "1000.000000 IP 10.0.0.4.54321 > 20.1.2.3.443: tcp 1200\n"
        # ingress: source port 443
        "1000.100000 IP 20.1.2.3.443 > 10.0.0.4.54321: tcp 800\n"
        # pure ACK carries no volume and must not be counted
        "1000.150000 IP 10.0.0.4.54321 > 20.1.2.3.443: tcp 0\n"
        # unrelated port is not the provider flow
        "1000.200000 IP 10.0.0.4.5000 > 10.0.0.5.9999: tcp 500\n"
        # ip6 lines use the same shape and must parse
        "1000.300000 IP6 fe80::1.54322 > 2600::1.443: tcp 640\n"
        "garbage line that must be ignored\n"
    )
    pkts = read_packets(cap)
    assert [(d, n) for _, d, n in pkts] == [("out", 1200), ("in", 800), ("out", 640)]
    # sorted by timestamp, which the window scan relies on
    assert [t for t, _, _ in pkts] == sorted(t for t, _, _ in pkts)


def _run(run_id: str, steps: list[tuple[float, float]]) -> dict:
    return {
        "run_id": run_id,
        "status": "completed",
        "condition": "adaptive",
        "case": {"attribute_label": 1, "membership_label": 0},
        "trace": {
            "steps": [
                {
                    "index": i,
                    "step_type": f"s{i}",
                    "wall_time_s": w,
                    "duration_s": d,
                    "egress_bytes": 100 + i,
                    "ingress_bytes": 10 + i,
                }
                for i, (w, d) in enumerate(steps)
            ]
        },
    }


def test_attribution_uses_the_receipt_to_place_relative_step_times(tmp_path: Path) -> None:
    r"""The journal's wall_time_s is relative; the receipt's issued_at is not.

    run_start = issued_at - (last wall_time_s + duration_s), so a step's packets
    must land in [run_start + wall_time_s, + duration_s]. Getting this backwards
    silently yields zero matched packets and an AUC of 0.5, which reads as "no
    channel on the wire" rather than as a bug.
    """
    # Steps at t=+1 (1s long) and t=+5 (2s long); run therefore spans 7s and
    # ends -- receipt issued -- at epoch 1007.
    run = _run("r1", [(1.0, 1.0), (5.0, 2.0)])
    runs_dir = tmp_path / "runs" / "r1"
    runs_dir.mkdir(parents=True)
    (runs_dir / "receipts.jsonl").write_text(
        json.dumps({"body": {"run_id": "r1", "issued_at": "1970-01-01T00:16:47.000Z"}}) + "\n"
    )

    packets = [
        (1001.5, "out", 1000), (1001.6, "in", 300),   # step 0's interval
        (1003.0, "out", 7777),                        # still step 0's: it owns
                                                      # everything up to step 1
        (1005.5, "out", 2000), (1006.9, "in", 400),   # step 1's interval
    ]
    from make_wire_artifacts import _issued_at

    stamps = _issued_at(tmp_path / "runs")
    assert stamps == {"r1": 1007.0}

    out = attribute([run], packets, stamps)
    assert len(out) == 1
    s0, s1 = out[0]["steps"]
    # Partitioned, so step 0 owns [1001, 1005) including the 7777-byte segment
    # and step 1 owns [1005, 1007). Nothing is counted twice and nothing inside
    # the run is dropped.
    assert (s0["wire_egress"], s0["wire_ingress"]) == (1000 + 7777, 300)
    assert (s1["wire_egress"], s1["wire_ingress"]) == (2000, 400)
    assert out[0]["run_wire_egress"] == 1000 + 7777 + 2000
    assert out[0]["matched"] == 2


def test_a_run_with_no_receipt_is_dropped_not_zero_filled(tmp_path: Path) -> None:
    """A run we cannot place in absolute time has no wire measurement.

    Zero-filling it would put "0 bytes on the wire" into the feature matrix and
    bias the recomputed AUC downward, which is the same shape of error as
    reading an absent journal field as a zero rate.
    """
    run = _run("orphan", [(1.0, 1.0)])
    out = attribute([run], [(1001.5, "out", 999)], stamps={})
    assert out == []

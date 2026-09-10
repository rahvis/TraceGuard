#!/usr/bin/env python3
r"""ICLR-scoped macros for the masking ablation's own residuals.

Why. Table~\ref{tab:ablation} is a masking ablation over recorded undefended
traces and Table~\ref{tab:main} is the executed release. Read as if both
measured the same thing they contradict each other: 0.683/0.806 against
0.577/0.764 for what looked like the same configuration. They do not measure the
same thing, and the difference between them is a finding rather than an error --
masking holds the provider-chosen coordinate at its undefended values, whereas
reshaping the policy changes what the provider is asked and therefore what it
returns. Saying that in the caption requires naming the masked residuals, and
the manuscript types no number as a literal, so they are extracted here.

Extracted rather than recomputed on purpose: re-running the analysis to recover
two numbers already committed would risk overwriting the 2000-permutation Holm
family in tables/macros.tex with a cheaper run, which has happened before.

Run:  python3 scripts/make_iclr_ablation_macros.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_paper_artifacts import assert_macro_names_are_latex_safe  # noqa: E402

ROWS = {
    "enforceable_remote_model": ("ablMaskedMemb", "ablMaskedAttr"),
    "none": ("ablNoneMemb", "ablNoneAttr"),
}
LABELS = {
    "enforceable_remote_model": r"\textbf{all three closeable} (masked)",
    "none": "none (undefended)",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, default=ROOT / "tables" / "tab_ablation.tex")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "iclr2027" / "tables" / "macros_ablation.tex")
    args = ap.parse_args()

    text = args.table.read_text()
    macros: dict[str, str] = {}
    for key, (memb_name, attr_name) in ROWS.items():
        label = re.escape(LABELS[key])
        found = re.search(label + r"\s*&\s*([0-9.]+)\s*&\s*([0-9.]+)\s*\\\\", text)
        if not found:
            raise SystemExit(
                f"could not find the {key!r} row in {args.table}. The row label "
                "changed; update LABELS rather than loosening the pattern, so a "
                "renamed row fails loudly instead of silently matching another."
            )
        macros[memb_name] = found.group(1)
        macros[attr_name] = found.group(2)
    assert_macro_names_are_latex_safe(list(macros))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "% Masking-ablation residuals, extracted from tables/tab_ablation.tex by\n"
        "% scripts/make_iclr_ablation_macros.py so the caption can contrast them\n"
        "% with the executed release without typing a literal. ICLR-scoped.\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items())
    )
    print(f"wrote {args.out}")
    for k, v in macros.items():
        print(f"  {k} = {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

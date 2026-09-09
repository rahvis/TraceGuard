#!/usr/bin/env python3
r"""Resolve every arXiv identifier in refs.bib and check it names the cited paper.

USENIX Security '27 treats a citation that does not exist, or that names the
wrong authors, as academic misconduct rather than a typo, and asks authors to
check their own submissions before the PC does. This does that check for the
half of a bibliography that is machine-checkable: entries carrying an arXiv id.

It found three real defects on first run, and the third is the reason the script
exists rather than a one-off audit:

  * ``liang2026graphrag`` cited arXiv:2502.11497 for "GraphRAG under Fire". That
    identifier belongs to "Geometry Aware Passthrough Mitigates Cybersickness",
    an unrelated VR paper; the real id is 2501.14050. Nothing internal to the
    bibliography could reveal that -- the entry was self-consistent.
  * ``nanayakkara2026registry`` carried a title the paper does not have.
  * ``jeong2026network`` and ``nanayakkara2026registry`` both asserted a
    conference venue that the arXiv record does not support.

What it cannot check: entries with only a DOI or a venue URL. Those are listed
at the end so the gap is visible rather than implied.

Usage:
    python3 scripts/verify_citations.py            # check refs.bib
    python3 scripts/verify_citations.py --bib X    # check another file
Exit status is non-zero if any resolvable citation disagrees with arXiv.
"""

from __future__ import annotations

import argparse
import html
import re
import urllib.request
from pathlib import Path

API = "https://export.arxiv.org/api/query?id_list={}&max_results=200"

# YYMM.NNNNN with a plausible year and a real month. Without the month test a
# DOI like 10.1145/2508859.2516660 matches "8859.25166" and the audit fills up
# with phantom identifiers -- which is exactly what happened the first time.
_ARXIV = re.compile(r"\b((?:0[7-9]|1\d|2[0-9])(?:0[1-9]|1[0-2])\.\d{4,5})\b")

_ENTRY = re.compile(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", re.S)


def _norm(text: str) -> str:
    """Compare titles ignoring braces, case, and punctuation."""
    text = re.sub(r"[{}\\$]", "", text)
    text = html.unescape(text)
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _field(body: str, name: str) -> str | None:
    # The trailing newline must be optional: the last field of an entry ends at
    # the closing brace with no comma and no newline, so requiring "\n" here
    # silently skipped every entry whose arXiv note came last -- the tool then
    # reported "all clear" over a third of the bibliography it never read.
    m = re.search(name + r"\s*=\s*\{(.*?)\}\s*,?\s*(?:\n|$)", body, re.S)
    return " ".join(m.group(1).split()) if m else None


def parse_bib(path: Path) -> list[dict[str, str | None]]:
    out = []
    for kind, key, body in _ENTRY.findall(path.read_text(encoding="utf-8")):
        # Only look for identifiers where one would legitimately be declared,
        # never in a url field, which is where the DOI false positives live.
        scan = " ".join(
            v for k, v in ((k, _field(body, k)) for k in ("note", "journal", "eprint"))
            if v
        )
        found = _ARXIV.search(scan)
        out.append({
            "kind": kind,
            "key": key.strip(),
            "title": _field(body, "title"),
            "author": _field(body, "author"),
            "booktitle": _field(body, "booktitle"),
            "arxiv": found.group(1) if found else None,
        })
    return out


def _tag(entry: str, name: str) -> str | None:
    """One Atom field from one entry.

    A module-level function rather than a closure over the loop variable: as a
    nested def it captured `entry` late, which is harmless only as long as it is
    never called after the iteration it was defined in.
    """
    m = re.search(rf"<{name}[^>]*>(.*?)</{name}>", entry, re.S)
    return html.unescape(" ".join(m.group(1).split())) if m else None


def fetch(ids: list[str]) -> dict[str, dict[str, object]]:
    if not ids:
        return {}
    # Every id has already matched _ARXIV, so nothing but YYMM.NNNNN reaches the
    # query string; assert the scheme anyway so the request cannot be pointed at
    # a file: or custom-scheme URL by a change to API.
    for aid in ids:
        if not _ARXIV.fullmatch(aid):
            raise ValueError(f"refusing to query a non-arXiv identifier: {aid!r}")
    url = API.format(",".join(ids))
    if not url.startswith("https://"):
        raise ValueError(f"refusing a non-HTTPS endpoint: {url!r}")
    req = urllib.request.Request(  # noqa: S310 - https asserted above
        url, headers={"User-Agent": "verify-citations/1"}
    )
    with urllib.request.urlopen(req, timeout=60) as fh:  # noqa: S310 - scheme checked above
        xml = fh.read().decode("utf-8", "replace")
    out: dict[str, dict[str, object]] = {}
    for entry in re.findall(r"<entry>(.*?)</entry>", xml, re.S):
        raw = _tag(entry, "id") or ""
        base = raw.rsplit("/", 1)[-1].split("v")[0]
        out[base] = {
            "title": _tag(entry, "title"),
            "authors": [
                html.unescape(" ".join(n.split()))
                for n in re.findall(r"<name>(.*?)</name>", entry, re.S)
            ],
            "comment": _tag(entry, "arxiv:comment"),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bib", type=Path, default=Path("refs.bib"))
    args = ap.parse_args()

    entries = parse_bib(args.bib)
    resolvable = [e for e in entries if e["arxiv"]]
    print(f"{len(entries)} entries; {len(resolvable)} carry an arXiv identifier\n")

    meta = fetch([str(e["arxiv"]) for e in resolvable])
    problems: list[str] = []

    for e in resolvable:
        aid = str(e["arxiv"])
        record = meta.get(aid)
        if record is None:
            problems.append(f"{e['key']}: arXiv:{aid} did not resolve")
            print(f"  MISSING  {e['key']:26s} arXiv:{aid}")
            continue

        actual = str(record["title"])
        if _norm(actual) != _norm(str(e["title"])):
            problems.append(
                f"{e['key']}: title mismatch\n"
                f"    bib    : {e['title']}\n"
                f"    arXiv  : {actual}"
            )
            print(f"  TITLE    {e['key']:26s} arXiv:{aid}")
            print(f"           bib   : {e['title']}")
            print(f"           arXiv : {actual}")
            continue

        # First author is the cheap, high-signal check: a swapped or invented
        # lead author is the failure the policy names explicitly.
        authors = list(record["authors"])  # type: ignore[arg-type]
        first_bib = str(e["author"] or "").split(" and ")[0]
        if authors and _norm(first_bib.split()[-1]) not in _norm(authors[0]):
            problems.append(
                f"{e['key']}: first author {first_bib!r} vs arXiv {authors[0]!r}"
            )
            print(f"  AUTHOR   {e['key']:26s} {first_bib!r} vs {authors[0]!r}")
            continue

        # A venue may only be asserted if the record supports it.
        comment = str(record["comment"] or "")
        if e["booktitle"] and not re.search(
            r"accept|to appear|to be published|camera|proceedings", comment, re.I
        ):
            problems.append(
                f"{e['key']}: asserts venue {e['booktitle']!r} but arXiv comment is "
                f"{record['comment']!r}"
            )
            print(f"  VENUE    {e['key']:26s} claims {str(e['booktitle'])[:44]!r}")
            print(f"           arXiv comment: {record['comment']!r}")
            continue

        print(f"  ok       {e['key']:26s} arXiv:{aid}")

    unchecked = [e for e in entries if not e["arxiv"]]
    print(f"\n{len(unchecked)} entries have no arXiv identifier and are NOT checked here")
    print("  (venue/DOI-only records must be verified by hand):")
    for e in unchecked:
        print(f"    {e['key']}")

    if problems:
        print(f"\n{len(problems)} problem(s):\n")
        for p in problems:
            print(f"  * {p}")
        return 1
    print("\nevery resolvable citation agrees with arXiv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "traceguard" / "data"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


MANUSCRIPT_TEX = ROOT / "usenixsecurity2026.tex"
MANUSCRIPT_PDF = ROOT / "usenixsecurity2026.pdf"


def _require_manuscript(*needed: Path) -> None:
    """Skip when the manuscript is not part of this checkout.

    The artifact repository ships the code, the scripts, the journals and the
    generated tables, but not the LaTeX sources of the paper. Tests that assert
    properties *of the manuscript* therefore have nothing to read there, and a
    public artifact whose suite fails on checkout is worse than one that says
    plainly which checks need the paper alongside it.
    """
    missing = [p.name for p in needed if not p.is_file()]
    if missing:
        pytest.skip(f"manuscript not in this checkout ({', '.join(missing)})")


def test_paper_reference_is_provenance_labeled_and_hash_anchored() -> None:
    _require_manuscript(MANUSCRIPT_TEX, MANUSCRIPT_PDF)
    reference = json.loads((DATA / "paper-reported-unverified.json").read_text())
    assert reference["provenance"] == "paper_reported_unverified"
    # The reference must state where its values come from: the live_replication
    # journals shipped in artifacts/, regenerable via the analysis scripts.
    assert "live_replication" in reference["evidence_status"]
    assert "artifacts/experiment-usenix26-main.jsonl" in reference["evidence_status"]
    assert reference["source"]["pdf_sha256"] == _sha256(ROOT / "usenixsecurity2026.pdf")
    assert reference["source"]["tex_sha256"] == _sha256(ROOT / "usenixsecurity2026.tex")
    # Renamed from full_pad_modeled: the runtime sleeps each step out to the
    # public deadline, so this is observed wall-clock time and not a model.
    #
    # Assert the *invariant* rather than a value. This used to hardcode 36.0 s,
    # which was the 12-step canonical plan at the archived run's 3 s deadline;
    # the deadline is a calibrated deployment parameter and is 10 s on the
    # current crew, so the figure legitimately moved to 120.0 s and the test
    # failed for the wrong reason. What must always hold is that full-pad
    # latency is exactly the plan length times the deadline -- if it is not,
    # the runtime is not padding to the deadline.
    latency = reference["latency_seconds"]
    assert "full_pad_modeled" not in latency
    observed = latency["full_pad_observed"]
    deadline = latency["step_deadline_s"]
    plan_steps = 12
    assert observed == pytest.approx(plan_steps * deadline, rel=0.02), (
        f"full-pad latency {observed}s is not {plan_steps} steps at {deadline}s"
    )

    # Every reported AUC needs its own calibrated null; the protocol's floor sits
    # at 0.58-0.60, so a bare comparison against 0.50 overstates leakage. Read
    # the generated block: `calibrated_nulls` used to carry its own per-arm copy
    # of these numbers and drifted against it, and this assertion was still
    # pinning the archived crew's finding that undefended membership was chance.
    nulls = reference["calibrated_nulls_measured"]

    # The undefended arm leaks both secrets on this crew. Membership doing so is
    # the revision the manuscript documents -- it did not on the weaker crew.
    assert nulls["adaptive_attribute"]["p_value"] < 0.01
    assert nulls["adaptive_membership"]["p_value"] < 0.01

    # ...and the paper's central asymmetry: full padding closes the
    # control-flow-carried secret and not the request-content-carried one. This
    # is the pair of claims the (0,0)-on-C result rests on, so pin both
    # directions rather than only the one that clears.
    assert nulls["full_membership"]["p_value"] > 0.05, "membership should not clear its null"
    assert nulls["full_attribute"]["p_value"] < 0.01, "attribute should still clear its null"
    assert nulls["full_membership"]["observed"] < nulls["adaptive_membership"]["observed"]

    assert "membership_revision" in reference


def test_v1_corpus_file_is_byte_stable_for_archived_receipts() -> None:
    """v1 must not be edited, ever.

    ``dataset_hash`` is computed over the canonical JSON of the whole corpus
    file and is embedded in every signed receipt, so the archived journals in
    artifacts/ only verify against the exact bytes they were produced from.
    Bumping the packaged corpus to v2 necessarily rotated that digest for new
    runs -- which is correct -- and is precisely why the old file stays on disk
    untouched rather than being migrated in place.
    """

    corpus = json.loads((DATA / "synthetic-medical-v1.json").read_text())
    assert corpus["schema_version"] == "synthetic-medical-v1.0.0"
    assert sorted(
        key for key in corpus["topics"][0] if key in {"routine", "sensitive", "framings"}
    ) == ["routine", "sensitive"]


def test_v2_corpus_grades_sensitivity_at_matched_text_volume() -> None:
    """The ordinal corpus, checked in the file rather than through the loader.

    Three properties the manuscript's ordinal claim depends on: four framings
    per topic, a *semantic* gradation (the step-therapy and diagnostic sentences
    differ at every rung, not just the label), and matched character counts, so
    text volume is not a monotone confound perfectly correlated with the ladder.
    """

    corpus = json.loads((DATA / "synthetic-medical-v2.json").read_text())
    assert corpus["synthetic"] is True
    assert corpus["schema_version"] == "synthetic-medical-v2.0.0"
    assert len(corpus["topics"]) == 6
    for topic in corpus["topics"]:
        framings = topic["framings"]
        assert len(framings) == 4
        assert framings[0] == "routine" and framings[-1] == "sensitive"
        priors: list[str] = []
        diagnostics: list[str] = []
        totals: list[int] = []
        hops: list[int] = []
        for level, framing in enumerate(framings):
            variant = topic[framing]
            documents = variant["documents"]
            assert len(documents) == 6
            assert variant["sensitivity_level"] == level
            assert variant["query"]
            assert all(document["id"] and document["text"] for document in documents)
            priors.append(documents[2]["text"])
            diagnostics.append(documents[3]["text"])
            totals.append(sum(len(document["text"]) for document in documents))
            hops.append(topic["fixture_hops"][framing])
        # Semantically graded: every rung's prose is distinct at both carriers.
        assert len(set(priors)) == 4
        assert len(set(diagnostics)) == 4
        # Monotone depth.
        assert hops == sorted(hops) and len(set(hops)) == 4
        # Length matched: the band the generator enforces at build time.
        mean = sum(totals) / len(totals)
        assert max(abs(value - mean) / mean for value in totals) <= 0.03
        # The canary names no service, so membership carries no attribute
        # vocabulary, and it matches the distractor it replaces in length.
        canary = topic["canary_document"]["text"]
        assert "seeking" not in canary
        distractor = len(topic[framings[0]]["documents"][5]["text"])
        assert abs(len(canary) - distractor) / ((len(canary) + distractor) / 2) <= 0.06


def test_compact_corpus_is_synthetic_and_fixed_six_documents() -> None:
    corpus = json.loads((DATA / "synthetic-medical-v1.json").read_text())
    assert corpus["synthetic"] is True
    assert len(corpus["topics"]) == 6
    assert {topic["specialty"] for topic in corpus["topics"]} == {
        "cardiology",
        "oncology",
        "psychiatry",
    }
    for topic in corpus["topics"]:
        assert topic["canary_document"]["text"]
        for variant in ("routine", "sensitive"):
            assert len(topic[variant]["documents"]) == 6
            assert topic[variant]["query"]
            assert all(
                document["id"] and document["text"]
                for document in topic[variant]["documents"]
            )


def test_env_example_and_source_tree_contain_no_provider_secret() -> None:
    # A local .env is the documented way to inject a runtime key, so it may
    # exist locally — but it must be git-ignored so it can never be published.
    gitignore = (ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in gitignore

    # The credential scan itself is scripts/scan_secrets.py, shared with CI and
    # scripts/verify_release.sh. Calling it here rather than re-implementing the
    # regex is deliberate: three copies of this pattern previously disagreed,
    # and the copy that mattered could not see an Azure key at all.
    sys.path.insert(0, str(ROOT / "scripts"))
    import scan_secrets

    # Prove the patterns fire before trusting a clean result.
    scan_secrets.self_test()
    findings = scan_secrets.scan_repository(ROOT)
    assert not findings, f"scan_secrets reported credential material: {findings}"


def test_secret_scanner_detects_every_supported_credential_form() -> None:
    """The scanner must catch Azure keys, not only sk- prefixed ones.

    Azure OpenAI keys carry no sk- prefix, so the pattern this repository used
    until now could not detect one. Canaries are assembled from fragments so
    this test file never contains a string matching the scanner's own patterns.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import scan_secrets

    sk, proj = "sk", "proj"
    cases = {
        "vendor_prefixed_key": f"OPENAI_API_KEY={sk}-{proj}-" + "A" * 24,
        "microsoft_identifiable_secret": "B" * 40 + "JQQJ" + "99" + "C" * 20,
        "contextual_credential": "AZURE_OPENAI_API_KEY=" + "d" * 32,
    }
    for expected, sample in cases.items():
        hits = {name for name, _ in scan_secrets.scan_text(sample)}
        assert expected in hits, f"{expected} not detected in its canary"

    # A digest must never be flagged; this repository is full of them.
    for benign in (
        "AZURE_OPENAI_API_KEY=",
        '"dataset_sha256": "18910ed6dee7ad5146367e72843b14f0ede2afbc99cd99f4b8fe9c3380471edb"',
    ):
        assert not scan_secrets.scan_text(benign), f"false positive on {benign[:40]!r}"

def test_public_console_has_no_sales_or_investor_surface() -> None:
    web = ROOT / "src" / "traceguard" / "web"
    source = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in web.glob("*")
        if path.suffix in {".html", ".css", ".js"}
    )
    for forbidden in ("book a pilot", "investor deck", "marketplace", "sales demo"):
        assert forbidden not in source
    assert "synthetic" in source
    assert "not clinical" in source or "not for clinical" in source
    assert "paper_reported_unverified" in source


def test_compose_uses_runtime_secret_injection_only() -> None:
    compose = (ROOT / "docker-compose.yml").read_text()
    # Credentials must arrive by runtime substitution, never as a literal.
    assert "${AZURE_OPENAI_API_KEY:-}" in compose
    assert "${AZURE_OPENAI_ENDPOINT:-}" in compose
    # ...and must never be baked into the image.
    dockerfile = (ROOT / "Dockerfile").read_text()
    for name in ("AZURE_OPENAI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        assert name not in dockerfile


def test_calibrated_null_cache_refuses_to_cross_journals() -> None:
    """A null from one journal must never be paired with another journal's AUCs.

    ``--nulls`` defaults to tables/nulls.json, so building tables from a
    different journal silently reused the cached journal's nulls and every
    p-value in the table footnote was wrong for the numbers above it. The cache
    recorded a journal digest for exactly this purpose and nothing checked it.
    A calibrated null is the only thing that makes these AUCs interpretable, so
    the mismatch is refused rather than warned about.
    """

    import json as _json
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    from make_paper_artifacts import load_cached_nulls

    cache = root / "tables/nulls.json"
    payload = _json.loads(cache.read_text())
    # Read the journal from the cache rather than naming one: which journal is
    # the headline changes, and a test that hardcodes it fails for the wrong
    # reason the moment it does.
    journal = root / payload["journal"]

    # The committed cache belongs to the committed journal.
    assert load_cached_nulls(cache, journal=journal, seed=payload["seed"])

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        wrong_digest = Path(tmp) / "nulls.json"
        wrong_digest.write_text(
            _json.dumps({**payload, "journal_sha256": "00" * 32}), encoding="utf-8"
        )
        with pytest.raises(SystemExit, match="cannot be carried across journals"):
            load_cached_nulls(wrong_digest, journal=journal, seed=payload["seed"])

        with pytest.raises(SystemExit, match="depends on the seed"):
            load_cached_nulls(cache, journal=journal, seed=payload["seed"] + 1)

    # No journal given (the CI path) stays permissive rather than failing a
    # regeneration that never claimed to check provenance.
    assert load_cached_nulls(cache) == payload["nulls"]


def test_missing_nulls_are_dropped_not_inherited() -> None:
    """A macro whose input is missing must go missing, not persist.

    merge_macros preserves entries by design, so hand-maintained and
    other-stage macros survive a regeneration. That is the wrong default for a
    null: building tables for a journal with no calibrated nulls left the
    *previous* journal's \\nullMember and \\pMember in the file, so the
    manuscript kept a null belonging to different data. That is the same defect
    the cache's digest check refuses, reappearing through the escape hatch that
    bypasses it.
    """

    import sys
    import tempfile
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    from make_paper_artifacts import merge_macros, read_macros

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "macros.tex"
        merge_macros(
            path,
            {"nullMember": "0.57", "pMember": "=0.521", "keepMe": "7"},
            "% test\n",
        )
        assert read_macros(path)["nullMember"] == "0.57"

        # A second pass with no nulls must remove them and keep everything else.
        merge_macros(path, {"memberAUC": "0.719"}, "% test\n", drop=("nullMember", "pMember"))
        after = read_macros(path)
        assert "nullMember" not in after
        assert "pMember" not in after
        assert after["keepMe"] == "7", "drop must not disturb unrelated macros"
        assert after["memberAUC"] == "0.719"


def test_sync_paper_reference_reads_numbers_from_the_macros() -> None:
    """The reference file's numbers must come from the artifacts, not a typist.

    Writing this tool is what surfaced the estimator inconsistency: the file said
    structure-canonical membership was 0.504 and the macro said 0.534, and the
    file was right about the PDF. Generating the numeric sections removes the
    transcription step entirely.
    """

    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    from sync_paper_reference import build_numeric, read_macros

    macros = read_macros(root / "tables/macros.tex")
    numeric = build_numeric(macros, {})

    # Every headline cell must trace back to a macro, not to a constant.
    assert numeric["headline"]["adaptive"]["attribute_auc"] == float(macros["attrAUC"])
    assert numeric["headline"]["adaptive"]["membership_auc"] == float(macros["memberAUC"])
    assert numeric["headline"]["structure_only"]["membership_auc"] == float(
        macros["residMemberAUC"]
    )
    assert numeric["design"]["cases"] == int(macros["nCases"])
    assert numeric["depth_channel"]["shape"] == macros["hopShapeShort"]
    # Macros carrying an operator or a sign must still parse.
    assert numeric["membership_mechanism"]["egress_delta_pct"] == float(
        macros["memberEgressPct"].lstrip("+")
    )
    assert numeric["membership_mechanism"]["phi_with_attribute"] is not None


def test_sync_paper_reference_refuses_to_reanchor_without_a_reason() -> None:
    """A supersession that does not say why is how an anchor loses its meaning."""

    import json as _json
    import shutil
    import sys
    import tempfile
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    from sync_paper_reference import main

    with tempfile.TemporaryDirectory() as tmp:
        ref = Path(tmp) / "ref.json"
        shutil.copy(root / "src/traceguard/data/paper-reported-unverified.json", ref)
        # Stand-in source files, so this tests the anchoring logic rather than
        # whether a build artifact happens to be present in the working tree.
        document = _json.loads(ref.read_text())
        for kind in ("pdf", "tex"):
            (Path(tmp) / document["source"][kind]).write_bytes(f"stub {kind}".encode())

        with pytest.raises(SystemExit, match="requires --reason"):
            main(["--reference", str(ref), "--write", "--reanchor", "--root", tmp])

        before = _json.loads(ref.read_text())["source"]
        main([
            "--reference", str(ref), "--write", "--reanchor", "--root", tmp,
            "--reason", "unit test supersession",
        ])
        after = _json.loads(ref.read_text())["source"]
        # Anchors only move when the file on disk actually differs, and a move
        # always appends the previous digest with its reason.
        if after["pdf_sha256"] != before["pdf_sha256"]:
            assert len(after["pdf_sha256_history"]) == len(
                before["pdf_sha256_history"]
            ) + 1
            assert after["pdf_sha256_history"][-1]["sha256"] == before["pdf_sha256"]
            assert (
                after["pdf_sha256_history"][-1]["superseded_because"]
                == "unit test supersession"
            )


def test_manuscript_carries_no_working_notes() -> None:
    """A placeholder written while drafting must not ship.

    A "PENDING-FULL-RUN" note, complete with an instruction to myself not to
    claim a result, reached the appendix and survived a build and a page-count
    review before a manual read caught it. LaTeX has no opinion about prose, so
    the check has to be explicit.
    """

    _require_manuscript(MANUSCRIPT_TEX)
    import re
    from pathlib import Path

    text = Path(__file__).resolve().parents[1].joinpath("usenixsecurity2026.tex").read_text()
    # Strip comments: a note a reader never sees is fine, one that renders is not.
    rendered = "\n".join(
        re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines()
    )
    banned = ("PENDING", "TODO", "FIXME", "XXX", "PLACEHOLDER", "do not write that")
    hits = [w for w in banned if w in rendered]
    assert not hits, f"working notes present in rendered text: {hits}"


def test_no_document_claims_in_vm_execution_the_journals_do_not_support() -> None:
    """Two files asserted the study's crew ran inside the confidential VM.

    Neither was supported: no journal carries attestation provenance, and the
    receipts those runs emitted say hardware_backed=false. The manuscript keeps
    the attested *deployment* apart from the channel *measurement* and calls that
    its largest caveat, so a prose claim that collapses the two is a real defect
    and not a wording nit. This checks the claim against the evidence rather than
    banning the words: if a journal ever does carry attestation, the assertion
    stops firing.
    """

    import json as _json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    attested = 0
    total = 0
    for journal in sorted((root / "artifacts").glob("*.jsonl")):
        for line in journal.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = _json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict) or "provenance" not in row:
                continue
            total += 1
            att = (row.get("provenance") or {}).get("attestation") or {}
            if att.get("hardware_backed") and att.get("verdict") == "verified":
                attested += 1
    assert total, "no journals found to check against"
    reference_path = root / "src/traceguard/data/paper-reported-unverified.json"
    if attested:
        # The claim is now supportable, so the guard must not simply switch
        # itself off: an early return here would leave a sentence that has
        # become false standing in the reference file with nothing checking it.
        # Flip to the opposite assertion instead.
        status_now = " ".join(
            _json.loads(reference_path.read_text())["evidence_status"].split()
        )
        assert "not inside the confidential VM" not in status_now, (
            f"{attested} of {total} journal rows now carry a verified "
            "hardware-backed attestation, so evidence_status must no longer "
            "say the runs did not execute inside the VM"
        )
        return

    reference = _json.loads(
        (root / "src/traceguard/data/paper-reported-unverified.json").read_text()
    )
    status = " ".join(reference["evidence_status"].split())
    assert "not inside the confidential VM" in status, (
        "evidence_status must state that the runs did not execute inside the VM "
        "while no journal carries attestation provenance"
    )
    # And the phrase that was wrong must not have come back.
    assert "crew executed inside a genuine Azure Confidential VM" not in status, (
        "the unsupported in-VM claim has been restored"
    )


def test_enforced_deadline_is_recovered_from_the_journal_not_the_environment() -> None:
    r"""The step deadline must come from the padded traces, never from Settings().

    ``_enforced_deadline_s`` exists to keep this figure out of the reader's
    environment, and ``build`` defeated it by accident: the loop that emits the
    ingress-only macros bound the name ``rows``, so after it ran the outer
    ``rows`` -- the whole journal -- held one arm's ``_records_for`` output
    instead. That projection deliberately drops ``condition``, so the deadline
    lookup saw no full-pad rows and fell back to the ambient
    ``TRACEGUARD_STEP_DEADLINE_MS``. The manuscript then reported a 3 s enforced
    deadline and a 55.0% overrun rate for a journal whose padded steps are all
    exactly 10.0 s, a 1.0% rate.

    Two independent things are asserted, because the fallback is silent and only
    the pair of them localizes it: the helper works on the journal, and ``build``
    actually hands it the journal.
    """

    import json as _json
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    from make_paper_artifacts import _enforced_deadline_s, build, load_experiment_records

    payload = _json.loads((root / "tables/nulls.json").read_text())
    journal = root / payload["journal"]

    rows = load_experiment_records(journal)
    padded = [r for r in rows if r.get("condition") in ("full", "full_pad")]
    assert padded, f"{journal} has no padded rows; this test cannot check anything"

    # Every padded step releases the deadline as a constant duration, so the
    # journal knows its own deadline exactly.
    durations = {
        round(float(step["duration_s"]), 3)
        for row in padded
        for step in row["trace"]["steps"]
    }
    assert len(durations) == 1, f"padded steps span several deadlines: {durations}"
    expected = durations.pop()

    recovered, source = _enforced_deadline_s(rows)
    assert recovered == expected, f"helper recovered {recovered}s, journal says {expected}s"
    assert "journal" in source and "ambient" not in source, source

    # And the same value must survive the pipeline that writes the macro.
    result = build(journal, payload["seed"], bootstrap=0, n_perm=0,
                   nulls_path=root / "tables/nulls.json")
    assert result["deadline_s"] == expected, (
        f"build() used {result['deadline_s']}s, not the journal's {expected}s -- "
        "the journal was shadowed before it reached the deadline lookup"
    )
    assert "ambient" not in result["deadline_source"], result["deadline_source"]
    assert float(result["macros"]["stepDeadline"]) == expected


def test_open_science_appendix_carries_an_anonymous_artifact_url() -> None:
    r"""USENIX Sec '27 wants the anonymous artifact URL in the paper itself.

    "Anonymous URLs should be included in the paper's Open Science Appendix."
    Saying the link "appears on the submission page" does not satisfy that, so
    the manuscript defines \artifacturl once and the Open Science Appendix
    prints it. This test checks the wiring and reports loudly while the value is
    still the placeholder -- it cannot check that a URL is live, only that a
    human has replaced the slot.
    """

    _require_manuscript(MANUSCRIPT_TEX)
    tex = (ROOT / "usenixsecurity2026.tex").read_text()

    assert r"\newcommand{\artifacturl}" in tex, "no \\artifacturl definition"
    open_science = tex.split(r"\section*{Open Science}", 1)
    assert len(open_science) == 2, "no Open Science appendix"
    section = open_science[1].split(r"\section", 1)[0]
    assert r"\artifacturl" in section, "the Open Science appendix does not print the URL"

    # It must be an appendix, not a pre-bibliography section: for '27 both Ethics
    # and Open Science are appendices, and the 13-page body count excludes
    # "References and any appendices".
    bib = tex.index(r"\bibliography{refs}")
    assert tex.index(r"\section*{Open Science}") > bib, (
        "Open Science must follow the bibliography as an appendix"
    )
    assert tex.index(r"\section*{Ethical Considerations}") > bib, (
        "Ethical Considerations must follow the bibliography as an appendix"
    )

    placeholder = "SET BEFORE SUBMISSION"
    if placeholder in tex:
        pytest.skip(
            "\\artifacturl is still the placeholder: create the anonymous "
            "repository (anonymous.4open.science, conference ID SEC27) and set it "
            "in usenixsecurity2026.tex before submitting"
        )

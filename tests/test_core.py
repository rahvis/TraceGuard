from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from traceguard import CorpusCatalog, Settings, TraceGuardSDK, providers, verify_receipt
from traceguard.experiment import validate_balanced_design
from traceguard.graph import CANONICAL_STEP_TYPES
from traceguard.providers import (
    AzureOpenAIProvider,
    LLMProvider,
    create_provider,
)
from traceguard.types import (
    SENSITIVITY_FRAMINGS,
    Autonomy,
    Condition,
    MedicalCase,
    ProviderRequest,
    ProviderResponse,
    fixture_target_hops,
)


def _top_rung(cases) -> Any:
    """The most sensitive case in the corpus.

    Tests that used to say "the first case where sensitive is True" meant "the
    fully-gapped one", which was the only sensitive framing there was. With a
    graded ladder that predicate also matches the middle rungs, so the intent
    has to be spelled out or these tests would silently start asserting
    seven-hop behaviour about a five-hop case.
    """

    return max(cases, key=lambda case: (case.sensitivity_level, case.case_id))


@pytest.fixture
def fast_settings() -> Settings:
    return Settings(provider="fixture", step_deadline_s=0.01, receipt_envelope_bytes=8192)


def test_settings_from_env_never_exposes_key() -> None:
    settings = Settings.from_env(
        {
            "TRACEGUARD_PROVIDER": "azure",
            "AZURE_OPENAI_API_KEY": "unit-test-secret",
            "AZURE_OPENAI_ENDPOINT": "https://example-resource.services.ai.azure.com",
            "MODEL_FAST": "gpt-5.6-luna",
            "MODEL_DEEP": "gpt-5.6-sol",
        }
    )
    assert settings.provider == "azure"
    assert settings.model_fast == "gpt-5.6-luna"
    assert settings.model_deep == "gpt-5.6-sol"
    # The key must appear in neither the repr nor any public projection.
    assert "unit-test-secret" not in repr(settings)
    assert "unit-test-secret" not in json.dumps(settings.to_public_dict())
    # The endpoint is not a secret and identifies which resource served a run.
    assert settings.to_public_dict()["azure_openai_endpoint"].endswith(
        ".services.ai.azure.com"
    )

    with pytest.raises(ValueError, match="AZURE_OPENAI_API_KEY"):
        Settings.from_env({"TRACEGUARD_PROVIDER": "azure"})
    with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT"):
        Settings.from_env(
            {"TRACEGUARD_PROVIDER": "azure", "AZURE_OPENAI_API_KEY": "unit-test-secret"}
        )


def test_removed_direct_providers_fail_with_a_migration_message() -> None:
    """A removed provider must not read as a typo.

    Archived journals legitimately record provider "openai", so the name has to
    keep meaning something specific rather than collapsing into "unsupported".
    """

    for name in ("openai", "anthropic"):
        with pytest.raises(ValueError, match="was removed"):
            Settings(provider=name)
    with pytest.raises(ValueError, match="provider must be one of"):
        Settings(provider="bogus")


def test_settings_accepts_compose_aliases_with_documented_precedence(tmp_path) -> None:
    settings = Settings.from_env(
        {
            "TRACEGUARD_CANONICAL_HOPS": "7",
            "TRACEGUARD_STEP_DEADLINE_MS": "2500",
            "TRACEGUARD_STEP_EGRESS_BYTES": "2048",
            "TRACEGUARD_SIGNING_KEY_FILE": str(tmp_path / "signer.key"),
        }
    )
    assert settings.canonical_research_hops == 7
    assert settings.step_deadline_s == 2.5
    assert settings.egress_ceiling_bytes == 2048
    assert settings.signing_key_file == tmp_path / "signer.key"

    preferred = Settings.from_env(
        {
            "TRACEGUARD_STEP_DEADLINE_SECONDS": "1.25",
            "TRACEGUARD_STEP_DEADLINE_MS": "9000",
            "TRACEGUARD_EGRESS_CEILING_BYTES": "3000",
            "TRACEGUARD_STEP_EGRESS_BYTES": "9999",
        }
    )
    assert preferred.step_deadline_s == 1.25
    assert preferred.egress_ceiling_bytes == 3000


def test_default_compact_corpus_expands_balanced_six_document_cases() -> None:
    """The case count is derived from the design, never written down.

    The design is (specialty, topic) group x sensitivity level x membership.
    Asserting the product rather than a literal is what let the corpus grow a
    four-rung sensitivity ladder without a stale 24 in four files.
    """

    catalog = CorpusCatalog.load_default()
    cases = catalog.list_cases()
    groups = {(case.specialty, case.topic) for case in cases}
    expected = len(groups) * len(SENSITIVITY_FRAMINGS) * 2
    assert len(cases) == expected
    assert len({case.case_id for case in cases}) == expected
    assert len(catalog.dataset_hash) == 64
    assert {case.specialty for case in cases} == {"cardiology", "oncology", "psychiatry"}
    assert all(len(case.documents) == 6 for case in cases)
    design = validate_balanced_design(
        [case.to_dict() for case in cases]
    )
    assert design["balanced"] is True
    assert design["expected_task_cells"] == expected
    public = json.dumps(catalog.public_manifest())
    assert "query" not in public
    assert "text" not in public


def test_membership_canary_is_length_matched_at_the_document_grain() -> None:
    """The canary must be byte-exact against the distractor it replaces.

    Matching only whole-document length is not enough. The fixture builds its
    response observable from each cited document's *first sentence* plus its
    document id, so a first draft that matched documents (242 vs 235) while
    collapsing the canary's opening line to 38 characters against 103 produced a
    constant -268 byte membership delta in every topic -- a cleaner channel than
    the one it replaced. A later pass closed that and left -8, from the id alone.

    This asserts the grain the control genuinely holds at: total corpus bytes per
    (topic, rung) are *identical* with and without the canary. That is what makes
    a membership signal in the response direction a content effect rather than a
    length artifact.

    The request direction is bounded separately, by
    ``test_membership_request_length_residual_stays_bounded``, because it is not
    exactly matched and pretending otherwise would hide a real confound.
    """

    catalog = CorpusCatalog.load_default()
    by_key: dict[tuple[str, int, bool], int] = {}
    for case in catalog.list_cases():
        by_key[(case.topic, case.sensitivity_level, case.canary_member)] = sum(
            len(document.text.encode("utf-8")) for document in case.documents
        )
    deltas = {
        by_key[(topic, level, True)] - by_key[(topic, level, False)]
        for (topic, level, member) in by_key
        if member
    }
    assert deltas == {0}, f"canary is not byte-matched at document grain: {sorted(deltas)}"


def test_membership_request_length_residual_stays_bounded() -> None:
    """Bound the one length channel the corpus does *not* close.

    Egress is the request body, and the request carries the retrieved documents.
    Even with total corpus bytes byte-exact (previous test), two things leave a
    small residual: ``_source_packet`` truncates each document independently, so
    an equal total split differently across documents truncates differently; and
    JSON escaping is not length-preserving, so equal-length texts with different
    newline and quote counts serialize to different sizes.

    The earlier instrumentation recorded only the response direction and so could
    not see this at all. It is small -- low tens of bytes against a ~2.3 kB
    request -- but it is systematic, and the honest thing is to bound it and fail
    if it grows, rather than assert a zero that does not hold. The paper reports
    this residual as a stated confound on the membership result's egress
    coordinate.
    """

    catalog = CorpusCatalog.load_default()
    sdk = TraceGuardSDK(Settings(provider="fixture"), provider="fixture", catalog=catalog)
    egress: dict[tuple[str, int, bool], int] = {}
    for case in catalog.list_cases():
        result = sdk.run(case.case_id, condition=Condition.ADAPTIVE, autonomy=Autonomy.BOUNDED)
        egress[(case.topic, case.sensitivity_level, case.canary_member)] = (
            result.trace.total_egress_bytes
        )
    pairs = [
        (
            topic,
            level,
            egress[(topic, level, True)] - egress[(topic, level, False)],
            egress[(topic, level, False)],
        )
        for (topic, level, member) in egress
        if member
    ]
    worst = max(abs(delta) for _, _, delta, _ in pairs)
    baseline = min(total for _, _, _, total in pairs)
    # 1% of the request volume. The measured worst case is well inside this; the
    # bound exists to catch a corpus edit that reopens the channel.
    assert worst <= 0.01 * baseline, (
        f"request-direction length residual grew to {worst} B "
        f"({100 * worst / baseline:.2f}% of a {baseline} B request)"
    )


def test_sensitivity_rung_is_not_recoverable_from_character_counts() -> None:
    """Text volume must not be an oracle for the rung.

    A magnitude band bounds how far lengths may differ but not whether the
    difference is *consistent across topics*, and consistency is what makes a
    length a label. The first draft satisfied a 6% band while the diagnostics
    document was 237/237/228/233 in all six topics, so that one number
    determined the rung; a leave-one-topic-out classifier recovered the rung
    from the six character counts at 70.8% against 25% chance.

    This asserts the operational property rather than a per-role invariant,
    because some per-level length variation is legitimate -- the graded service
    names genuinely differ in length -- and only *learnable, transferable*
    structure is a confound.
    """

    import numpy as np
    from sklearn.linear_model import LogisticRegression

    catalog = CorpusCatalog.load_default()
    rows: list[list[int]] = []
    labels: list[int] = []
    topics: list[str] = []
    for case in catalog.list_cases():
        if case.canary_member:
            continue
        rows.append([len(document.text) for document in case.documents])
        labels.append(case.sensitivity_level)
        topics.append(case.topic)

    features = np.asarray(rows, dtype=float)
    target = np.asarray(labels)
    unique_topics = sorted(set(topics))
    topic_array = np.asarray(topics, dtype=object)

    correct = 0
    for held_out in unique_topics:
        test = topic_array == held_out
        model = LogisticRegression(max_iter=5000).fit(features[~test], target[~test])
        correct += int((model.predict(features[test]) == target[test]).sum())

    accuracy = correct / len(target)
    chance = 1.0 / len(set(labels))
    # Generous headroom over chance: the point is that the 70.8% oracle is gone,
    # not that a small sample lands exactly on 25%.
    assert accuracy <= chance + 0.20, (
        f"character counts recover the rung at {accuracy:.1%} against {chance:.0%} chance; "
        "text volume is acting as a level oracle"
    )


def test_sensitivity_ladder_is_graded_monotone_and_length_matched() -> None:
    """The ordinal axis has to be real in three separate senses.

    A relabelled binary would satisfy none of them: the rungs must be distinct
    (four levels present, each with both membership arms), the depth the fixture
    targets must rise with the rung, and -- the one that decides whether an
    ordinal result means anything -- the amount of text must *not* rise with it.
    """

    catalog = CorpusCatalog.load_default()
    cases = catalog.list_cases()
    levels = sorted({case.sensitivity_level for case in cases})
    assert levels == list(range(len(SENSITIVITY_FRAMINGS)))
    assert [case.framing for case in cases if case.case_id.endswith("control")].count(
        "routine"
    ) == len({(case.specialty, case.topic) for case in cases})

    # Depth is monotone in the rung, and strictly so at every step.
    hops = {
        level: sorted(
            {case.fixture["research_hops"] for case in cases if case.sensitivity_level == level}
        )
        for level in levels
    }
    assert all(len(value) == 1 for value in hops.values())
    depths = [hops[level][0] for level in levels]
    assert depths == sorted(depths) and len(set(depths)) == len(depths)
    assert depths == [fixture_target_hops(level) for level in levels]

    # And the confound: character count must be flat across the ladder. v1 was
    # not length matched (1,760-1,792 routine against 1,830-1,924 sensitive),
    # which a binary caveat can absorb and an ordinal claim cannot.
    totals: dict[tuple[str, str], dict[int, int]] = {}
    for case in cases:
        if case.canary_member:
            continue
        chars = sum(len(document.text) for document in case.documents)
        totals.setdefault((case.specialty, case.topic), {})[case.sensitivity_level] = chars
    for group, by_level in totals.items():
        assert sorted(by_level) == levels, group
        mean = sum(by_level.values()) / len(by_level)
        worst = max(abs(value - mean) / mean for value in by_level.values())
        assert worst <= 0.03, f"{group} case totals vary {worst:.1%} across the ladder"

    # Sensitivity must not leak into the membership signal either: the canary
    # names no service, so no level's vocabulary appears in it.
    for case in cases:
        if not case.canary_member:
            continue
        # "-casenote" is length-matched to the distractor's "-<tag>-admin"
        # suffix, because the fixture cites the document id in its output.
        canary = next(
            document for document in case.documents if document.document_id.endswith("-casenote")
        )
        assert "seeking" not in canary.text
        for token in ("MRI", "defibrillator", "genomic", "psychotropic", "chemotherapy"):
            assert token not in canary.text


def test_sensitive_stays_a_derived_property_of_the_level() -> None:
    """~12 call sites read ``case.sensitive``; it must keep meaning the same thing."""

    for level in range(len(SENSITIVITY_FRAMINGS)):
        case = MedicalCase(
            case_id=f"probe-l{level}",
            specialty="cardiology",
            topic="probe",
            sensitivity_level=level,
            canary_member=False,
            query="probe",
            documents=tuple(
                CorpusCatalog.load_default().list_cases()[0].documents
            ),
        )
        assert case.sensitive is (level > 0)
        assert case.framing == SENSITIVITY_FRAMINGS[level]
        assert case.to_dict()["sensitivity_level"] == level
        assert case.to_dict()["sensitive"] is (level > 0)
    with pytest.raises(ValueError, match="sensitivity_level must be between"):
        MedicalCase(
            case_id="probe-bad",
            specialty="cardiology",
            topic="probe",
            sensitivity_level=len(SENSITIVITY_FRAMINGS),
            canary_member=False,
            query="probe",
            documents=tuple(CorpusCatalog.load_default().list_cases()[0].documents),
        )


def test_v1_corpus_still_loads_so_archived_receipts_stay_verifiable() -> None:
    """The archived journals were generated against v1 and its dataset_hash.

    v1 has no framings list and no ordinal field. It must still load, its two
    framings must land on the bottom and *top* rungs rather than 0 and 1, and
    its dataset_hash must differ from v2's -- bumping the corpus necessarily
    rotates that digest, which is why v1 stays on disk untouched.
    """

    root = Path(__file__).resolve().parents[1]
    v1 = CorpusCatalog.load(root / "src/traceguard/data/synthetic-medical-v1.json")
    v2 = CorpusCatalog.load_default()
    cases = v1.list_cases()
    assert len(cases) == 24
    assert sorted({case.sensitivity_level for case in cases}) == [
        0,
        len(SENSITIVITY_FRAMINGS) - 1,
    ]
    assert {case.fixture["research_hops"] for case in cases} == {4, 7}
    assert all(len(case.documents) == 6 for case in cases)
    assert v1.dataset_hash != v2.dataset_hash


def test_fixture_adaptive_graph_executes_real_loops_and_marks_provenance(
    fast_settings: Settings,
) -> None:
    sdk = TraceGuardSDK(fast_settings)
    sensitive = _top_rung(sdk.corpus.list_cases())
    events: list[dict] = []
    result = sdk.run(
        sensitive.case_id,
        "adaptive",
        run_id="adaptive-sensitive",
        seed=7,
        event_callback=events.append,
    )
    assert result.provider_provenance == "synthetic_fixture"
    assert result.runtime["paper_evidence"] is False
    assert result.metrics.research_hops == 7
    assert result.trace.step_types.count("clinical_extraction") == 7
    # initial + criteria repair + necessity revision re-enter coverage_assessment
    assert result.trace.step_types.count("coverage_assessment") == 3
    assert result.trace.step_types.count("criteria_check") == 2
    assert result.trace.step_types.count("necessity_review") == 2
    assert result.answer.endswith("not clinical advice.")
    assert verify_receipt(result.receipt).ok
    serialized_events = json.dumps(events).lower()
    for forbidden in ("query", "document", "prompt", "answer", "content", "api_key"):
        assert forbidden not in serialized_events


@pytest.mark.parametrize("condition", ["structure_only", "full_pad"])
def test_fixed_conditions_execute_canonical_seven_hop_plan(
    condition: str,
    fast_settings: Settings,
) -> None:
    sdk = TraceGuardSDK(fast_settings)
    case = _top_rung(sdk.corpus.list_cases())
    result = sdk.run(case.case_id, condition, run_id=f"fixed-{condition}")
    assert result.trace.step_types == CANONICAL_STEP_TYPES
    assert result.metrics.research_hops == 7
    assert len(result.trace.steps) == 12
    assert all(
        set(step.to_dict())
        == {
            "index",
            "step_type",
            "wall_time_s",
            "duration_s",
            "egress_bytes",
            "ingress_bytes",
        }
        for step in result.trace.steps
    )
    assert result.trace.step_types.count("coverage_assessment") == 1
    if condition == "full_pad":
        assert {step.duration_s for step in result.trace.steps} == {0.01}
        assert {step.egress_bytes for step in result.trace.steps} == {
            sdk.settings.egress_ceiling_bytes
        }
        assert [step.wall_time_s for step in result.trace.steps] == pytest.approx(
            [0.01 * index for index in range(1, 13)]
        )
        # The load-bearing honesty invariant.  Three coordinates are constants;
        # the response direction is not, and the receipt must say so rather than
        # asserting an unconditional (0,0) over the whole observable.  If this
        # assertion ever flips to a single-element set, either the mechanism has
        # started rewriting a value it does not control or the model has moved
        # inside the boundary -- and the guarantee kind must change with it.
        assert len({step.ingress_bytes for step in result.trace.steps}) > 1
        guarantee = result.receipt.body["guarantee"]
        assert guarantee["kind"] == "trace_privacy_partial"
        assert guarantee["epsilon"] == 0
        assert guarantee["delta"] == 0
        assert guarantee["open_coordinates"] == ["ingress_size"]
    else:
        assert len({step.ingress_bytes for step in result.trace.steps}) > 1
        assert result.receipt.body["guarantee"]["kind"] == "none"


class _SlowProvider(LLMProvider):
    provenance = "test_slow_provider"

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        time.sleep(0.02)
        return ProviderResponse(text="late", response_bytes=4, request_bytes=64)


def test_full_pad_fails_closed_on_deadline_without_changing_observable() -> None:
    settings = Settings(provider="fixture", step_deadline_s=0.002)
    sdk = TraceGuardSDK(settings, provider=_SlowProvider())
    case = sdk.corpus.list_cases()[0]
    result = sdk.run(case.case_id, "full_pad", run_id="deadline-overrun")
    assert result.metrics.fail_closed is True
    assert result.metrics.overrun_count == 12
    assert result.answer == "Run failed closed; no synthetic determination was released."
    assert result.receipt.body["budget_status"] == "breach"
    assert result.receipt.body["guarantee"]["kind"] == "none"
    assert all(step.duration_s == 0.002 for step in result.trace.steps)
    assert all(step.egress_bytes == settings.egress_ceiling_bytes for step in result.trace.steps)
    assert verify_receipt(result.receipt).ok


@dataclass
class _TextBlock:
    type: str = "text"
    text: str = "ok"


class _Messages:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            content=[_TextBlock()],
            usage=SimpleNamespace(input_tokens=3, output_tokens=1),
            stop_reason="end_turn",
        )


class _ChatCompletions:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
        )


def test_azure_chat_call_omits_sampling_args_and_maps_usage() -> None:
    """The wire contract the measurement depends on.

    egress_bytes must be the UTF-8 length of the response text (not a wire byte
    count), the OpenAI-shaped usage field names must map onto the internal ones,
    and no sampling argument may be sent — reasoning-tier deployments reject
    them and the protocol is seeded rather than sampled.
    """

    provider = object.__new__(AzureOpenAIProvider)
    completions = _ChatCompletions()
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    provider._endpoint = "https://example-resource.services.ai.azure.com"
    provider._api_version = "2025-01-01-preview"
    provider._deployments = ("gpt-5.6-luna",)
    response = provider.complete(
        ProviderRequest(
            role="necessity_review",
            model="gpt-5.6-luna",
            max_tokens=100,
            system_prompt="private system",
            user_prompt="private prompt",
            seed=13,
        )
    )
    assert response.text == "ok"
    # Ingress is the response text; egress is the request body we transmitted.
    # The two are different directions and are no longer conflated.
    assert response.response_bytes == len(b"ok")
    assert response.request_bytes > response.response_bytes
    assert response.input_tokens == 5
    assert response.output_tokens == 2
    assert response.stop_reason == "stop"
    assert completions.kwargs["model"] == "gpt-5.6-luna"
    assert completions.kwargs["max_completion_tokens"] == 100
    assert completions.kwargs["seed"] == 13
    assert {"temperature", "top_p"}.isdisjoint(completions.kwargs)


def test_azure_provider_adapts_to_a_deployment_that_rejects_seed() -> None:
    """A deployment that rejects seed must be recorded, not silently accommodated.

    Per-cell seeds are journalled and used as the resume key, so a deployment
    that ignores seed makes them decorative. The adapter has to surface that.
    """

    caps = providers.DeploymentCapabilities(seed_honoured=False, max_output_field="max_tokens")
    provider = object.__new__(AzureOpenAIProvider)
    completions = _ChatCompletions()
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    provider._endpoint = "https://example-resource.services.ai.azure.com"
    provider._api_version = "2025-01-01-preview"
    provider._deployments = ("legacy-deployment",)
    providers._CAPABILITY_CACHE[(provider._endpoint, "legacy-deployment")] = caps

    provider.complete(
        ProviderRequest(
            role="intake",
            model="legacy-deployment",
            max_tokens=64,
            system_prompt="s",
            user_prompt="u",
            seed=99,
        )
    )
    assert "seed" not in completions.kwargs
    assert completions.kwargs["max_tokens"] == 64
    assert "max_completion_tokens" not in completions.kwargs
    assert caps.to_dict()["seed_honoured"] is False


def test_azure_provider_rejects_a_foundry_project_url() -> None:
    """A project URL is not an inference endpoint and 404s on POST.

    Failing at construction with a clear message beats a confusing 404 on the
    first measured call.
    """

    with pytest.raises(ValueError, match="not an"):
        AzureOpenAIProvider(
            Settings(
                provider="azure",
                azure_openai_api_key="dummy",
                azure_openai_endpoint="https://r.services.ai.azure.com/api/projects/p",
            )
        )


def test_create_provider_maps_azure_and_rejects_unknown_modes() -> None:
    provider = create_provider(
        Settings(
            provider="azure",
            azure_openai_api_key="dummy",
            azure_openai_endpoint="https://example-resource.services.ai.azure.com",
        )
    )
    assert provider.provenance == "live_azure_openai"
    with pytest.raises(ValueError, match="unsupported provider"):
        create_provider(SimpleNamespace(provider="bogus"))


def test_autonomy_is_orthogonal_to_the_shaping_condition() -> None:
    """Autonomy must be separable from the defense, and monotone in data influence.

    These were previously one axis: STRUCTURE_ONLY/FULL_PAD each meant both
    "emit a constant plan" and "pad timing and size", so leakage-versus-autonomy
    could not be asked at all.
    """

    catalog = CorpusCatalog.load_default()
    sdk = TraceGuardSDK(Settings(provider="fixture"), provider="fixture", catalog=catalog)
    routine = next(case for case in catalog.list_cases() if case.sensitivity_level == 0)
    sensitive = _top_rung(catalog.list_cases())

    def spread(autonomy: Autonomy) -> int:
        lengths = [
            len(
                sdk.run(case.case_id, condition=Condition.ADAPTIVE, autonomy=autonomy).trace.steps
            )
            for case in (routine, sensitive)
        ]
        return lengths[1] - lengths[0]

    # Scripted is data-independent: the whole point of the defense.
    assert spread(Autonomy.SCRIPTED) == 0
    # The adaptive rungs are not, and that gap is the channel.
    assert spread(Autonomy.BOUNDED) > 0
    assert spread(Autonomy.FREE) > 0
    # FREE must never be *less* data-dependent than BOUNDED. An earlier
    # definition of FREE ("always run to max depth") inverted this, which would
    # have made a leakage-versus-autonomy sweep incoherent.
    assert spread(Autonomy.FREE) >= spread(Autonomy.BOUNDED)


def test_fixture_cannot_distinguish_the_top_autonomy_rung() -> None:
    """Document a real limitation rather than implying the ladder is validated.

    The fixture provider is deterministic and requests exactly one repair and
    one revision, so it never spends more than the BOUNDED budget. FREE is
    therefore indistinguishable from BOUNDED offline, and only a live model can
    exercise the top rung. Asserting this keeps the limitation visible instead
    of leaving a silently redundant experimental cell.
    """

    catalog = CorpusCatalog.load_default()
    sdk = TraceGuardSDK(Settings(provider="fixture"), provider="fixture", catalog=catalog)
    case = _top_rung(catalog.list_cases())
    bounded = sdk.run(case.case_id, condition=Condition.ADAPTIVE, autonomy=Autonomy.BOUNDED)
    free = sdk.run(case.case_id, condition=Condition.ADAPTIVE, autonomy=Autonomy.FREE)
    assert bounded.trace.step_types == free.trace.step_types


def test_canonicalized_conditions_refuse_adaptive_autonomy() -> None:
    catalog = CorpusCatalog.load_default()
    sdk = TraceGuardSDK(Settings(provider="fixture"), provider="fixture", catalog=catalog)
    case = catalog.list_cases()[0]
    for condition in (Condition.STRUCTURE_ONLY, Condition.FULL_PAD):
        with pytest.raises(ValueError, match="requires autonomy 'scripted'"):
            sdk.run(case.case_id, condition=condition, autonomy=Autonomy.BOUNDED)


def test_distinct_arms_never_share_a_cell_key_or_seed() -> None:
    """The silent-corruption guard.

    The experiment was indexed by (condition, case, repetition). Adding factors
    without extending that index would not fail loudly: two different arms would
    hash to the same cell key, and `resume` would accept a completed cell of one
    arm as satisfying the other, quietly merging distinct experiments into one
    journal. Every axis must therefore enter both the key and the seed.
    """

    from traceguard.experiment import Arm, ExperimentRunner

    arms = [
        Arm(condition="adaptive", autonomy="bounded"),
        Arm(condition="adaptive", autonomy="free"),
        Arm(condition="adaptive", autonomy="bounded", architecture="hierarchical_supervisor"),
        Arm(condition="adaptive", autonomy="bounded", architecture="react_single_agent"),
        Arm(condition="adaptive", autonomy="bounded", models=(("fast", "gpt-4o"),)),
        Arm(condition="adaptive", autonomy="bounded", models=(("fast", "gpt-5.6-luna"),)),
        Arm(condition="adaptive", autonomy="bounded", observability="timing_only"),
        Arm(condition="structure", autonomy="scripted"),
        Arm(condition="full", autonomy="scripted"),
    ]
    seeds = [ExperimentRunner._cell_seed(7, arm, "case-1", 0) for arm in arms]
    keys = [
        ExperimentRunner._cell_key(arm, "case-1", 0, seed)
        for arm, seed in zip(arms, seeds, strict=True)
    ]
    assert len(set(keys)) == len(arms), "two arms share a cell key"
    assert len(set(seeds)) == len(arms), "two arms share a per-cell seed"


def test_arm_rejects_a_canonicalized_condition_with_adaptive_autonomy() -> None:
    """Catch the contradiction where it names the arm, not mid-run."""

    from traceguard.experiment import Arm

    with pytest.raises(ValueError, match="requires autonomy 'scripted'"):
        Arm(condition="full", autonomy="bounded")
    # Either condition vocabulary is accepted and normalized to the short form.
    assert Arm(condition="structure_only", autonomy="scripted").condition == "structure"


def test_default_arms_reproduce_the_historical_condition_sweep() -> None:
    """An archived run must reproduce exactly, so the defaults must match it."""

    from traceguard.experiment import ExperimentConfig

    arms = ExperimentConfig().resolved_arms()
    assert [arm.condition for arm in arms] == ["adaptive", "structure", "full"]
    # These are the autonomy levels the conditions implied before the split.
    assert [arm.autonomy for arm in arms] == ["bounded", "scripted", "scripted"]
    assert {arm.architecture for arm in arms} == {"pipeline_crew"}


def test_graded_corpus_length_channel_is_indistinguishable_from_its_null() -> None:
    """Text volume must not be a monotone proxy for the sensitivity rung.

    An ordinal leakage claim over a graded ladder is uninterpretable if the rung
    is recoverable from character counts alone: the "graded channel" would be
    measuring document length. Before the graded clauses were equalized, a
    leave-one-topic-out classifier recovered the rung from lengths at 70.8%. The
    audit is calibrated against a within-group permutation null rather than
    against the 25% chance rate, because with 24 samples a single sample above
    the chance count is not evidence of anything.
    """

    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from gen_pa_corpus import length_channel_recoverability

    corpus = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "src/traceguard/data/synthetic-medical-v2.json"
        ).read_text()
    )
    # Fewer permutations than the reported figure: this is a regression gate, and
    # the p-value only has to be far from 0.05 to catch a reintroduced confound.
    audit = length_channel_recoverability(corpus, n_perm=800)
    assert audit["n_groups"] == 6
    assert audit["n_samples"] == 24
    assert audit["indistinguishable_from_null"], audit
    # The pre-fix confound sat at 70.8%; anything near that is a hard failure
    # even if the permutation test were somehow to pass.
    assert audit["recoverability"] < 0.45, audit


def test_canary_is_length_identical_to_the_distractor_it_replaces() -> None:
    """The membership result depends on this being exact, not merely close.

    Membership leakage is now a headline claim, measured as the agent emitting
    ~774 more bytes when the planted record is present. That is only a statement
    about the *agent* if the corpus input is length-identical: if the canary were
    longer than the distractor it displaces, the extra egress would be a corpus
    artifact and the finding would be about document length instead.

    Checked at all three grains the fixture's observable actually depends on --
    whole document, first sentence, and document id -- because an earlier
    version matched whole documents only and left an 8-byte id delta and a
    268-byte first-sentence delta in place.
    """

    import json
    from pathlib import Path

    corpus = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "src/traceguard/data/synthetic-medical-v2.json"
        ).read_text()
    )
    checked = 0
    for topic in corpus["topics"]:
        canary = topic["canary_document"]
        for framing in topic["framings"]:
            # The canary replaces the final (distractor) slot.
            distractor = topic[framing]["documents"][-1]
            assert len(canary["text"]) == len(distractor["text"]), (
                topic["topic"], framing, "whole-document length")
            assert len(canary["text"].split(".")[0]) == len(
                distractor["text"].split(".")[0]
            ), (topic["topic"], framing, "first-sentence length")
            assert len(canary["id"]) == len(distractor["id"]), (
                topic["topic"], framing, "document-id length")
            checked += 1
    assert checked == 24, f"expected 24 (specialty,topic,framing) cells, got {checked}"


def test_run_ids_stay_unique_when_arm_slugs_share_a_prefix() -> None:
    """Truncating a readable id must not be able to destroy its uniqueness.

    The crew-model arms differ only in a suffix of the arm slug, so truncating
    the readable run id at 120 characters cut off the case id and repetition and
    collapsed all 96 cells of an arm onto one id. 180 of the model spoke's 288
    cells then died on the bundle collision -- loudly, which is the only reason
    it was caught, but 180 paid cells were lost. Uniqueness now comes from an
    appended digest of the full identity and does not depend on the readable
    part fitting.
    """

    import hashlib
    import re

    from traceguard.experiment import Arm

    id_safe = re.compile(r"[^A-Za-z0-9_.-]+")

    def run_id(experiment: str, arm: Arm, case_id: str, rep: int) -> str:
        identity = f"{experiment}|{arm.slug}|{case_id}|{rep}"
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
        readable = id_safe.sub(
            "-", f"{experiment}-{id_safe.sub('-', arm.slug)}-{case_id}-r{rep}"
        )[:104].rstrip("-")
        return f"{readable}-{digest}"

    # Long deployment names plus a long case id is the configuration that broke.
    cases = [
        "cardiology-advanced-cardiac-imaging-elevated-canary",
        "cardiology-advanced-cardiac-imaging-routine-control",
    ]
    seen: dict[str, str] = {}
    for deployment in ("gpt-4o", "gpt-5.6-luna", "gpt-5.6-sol"):
        arm = Arm(
            condition="adaptive",
            autonomy="bounded",
            architecture="pipeline_crew",
            models=(("fast", deployment), ("deep", deployment), ("review", deployment)),
        )
        for case_id in cases:
            for rep in (0, 1):
                rid = run_id("matrix-model", arm, case_id, rep)
                identity = f"{arm.slug}|{case_id}|{rep}"
                assert rid not in seen, (
                    f"run id collision between {identity} and {seen[rid]}"
                )
                seen[rid] = identity
    assert len(seen) == 12


def test_arm_naming_crew_models_refuses_a_runner_not_configured_for_them() -> None:
    """The model axis has to actually vary the model.

    Arm.models fed the arm slug, the cell key and the journal metadata but was
    never applied to Settings, and the SDK holds one Settings for its lifetime.
    So the crew-model spoke ran the ambient crew for all three arms: 288 paid
    cells that measured the same configuration three times while their slugs and
    cell keys claimed three different ones. The provenance recorded the truth,
    which is the only reason it was caught.

    A caller sweeping models must build one runner per arm. This refuses the
    mismatch up front rather than per cell, because it is a configuration error
    and failing fast is what stops the 288 cells being spent.
    """

    import tempfile
    from pathlib import Path

    import pytest

    from traceguard.config import Settings
    from traceguard.corpus import CorpusCatalog
    from traceguard.experiment import Arm, ExperimentConfig, ExperimentRunner
    from traceguard.sdk import TraceGuardSDK
    from traceguard.storage import ArtifactStore

    settings = Settings(provider="fixture", step_deadline_s=0.05)
    catalog = CorpusCatalog.load_default()
    arm = Arm(
        condition="adaptive",
        autonomy="bounded",
        architecture="pipeline_crew",
        # Deliberately not what `settings` carries.
        models=(("fast", "other-fast"), ("deep", "other-deep"), ("review", "other-review")),
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        runner = ExperimentRunner(
            TraceGuardSDK(settings, provider="fixture", catalog=catalog),
            catalog,
            ArtifactStore(root / "runs"),
        )
        with pytest.raises(RuntimeError, match="not configured for"):
            runner.run(
                ExperimentConfig(
                    repetitions=1,
                    arms=(arm,),
                    seed=1,
                    experiment_id="guard",
                    journal_path=root / "j.jsonl",
                    resume=False,
                    require_balanced=False,
                    provider="fixture",
                    case_limit=2,
                    workers=1,
                )
            )

    # An arm with no model override imposes no constraint.
    plain = Arm(condition="adaptive", autonomy="bounded", architecture="pipeline_crew")
    assert plain.models == ()


# --------------------------------------------------------------------------- #
# What full padding does, and does not, do to the answer.
#
# The paper's (0,0) utility argument rests on a mechanical claim: padding a
# step that returns inside the deadline is invisible to the answer, so the
# entire utility cost is mediated by the fail-closed rate. That is a property
# of ShieldRuntime, so it belongs in a test rather than in prose.
# --------------------------------------------------------------------------- #


def test_full_pad_passes_a_timely_response_through_unaltered() -> None:
    """Padding shapes the observable and must not touch the payload."""

    from traceguard.config import Settings
    from traceguard.providers import ProviderResponse
    from traceguard.shield_runtime import ShieldRuntime
    from traceguard.types import Condition

    settings = Settings(step_deadline_s=2.0, egress_ceiling_bytes=4096)
    runtime = ShieldRuntime(settings=settings, condition=Condition.FULL_PAD)
    payload = "a faithful synthetic determination with source ids [doc-1]"
    execution = runtime.execute(
        lambda: ProviderResponse(
            text=payload,
            response_bytes=len(payload.encode()),
            request_bytes=settings.egress_ceiling_bytes,
            input_tokens=10,
            output_tokens=20,
            stop_reason="stop",
        )
    )
    # The answer survives byte-for-byte...
    assert execution.response.text == payload
    assert execution.fail_closed is False
    assert execution.violation is None
    # ...the released observable is the public constant on the three coordinates
    # the mechanism controls...
    assert execution.observable_duration_s == settings.step_deadline_s
    assert execution.observable_egress_bytes == settings.egress_ceiling_bytes
    # ...and the response direction is recorded as measured, NOT as the ceiling.
    # This is the honesty invariant: the guest cannot bound how many bytes the
    # provider sends, so recording the ceiling here would log a constant the
    # host never saw.
    assert execution.observable_ingress_bytes == len(payload.encode())
    assert execution.observable_ingress_bytes != settings.egress_ceiling_bytes


def test_full_pad_drops_the_whole_payload_when_a_step_fails_closed() -> None:
    """A refused step releases an empty payload -- not a truncated one.

    This is the event the entire measured utility cost is mediated by, and it
    is a *drop*: the crew continues with a missing input rather than a partial
    one, which is why the resulting answer stays fluent while being unfaithful.
    """

    from traceguard.config import Settings
    from traceguard.providers import EgressCeilingExceeded
    from traceguard.shield_runtime import ShieldRuntime
    from traceguard.types import Condition

    settings = Settings(step_deadline_s=2.0, egress_ceiling_bytes=64)
    runtime = ShieldRuntime(settings=settings, condition=Condition.FULL_PAD)
    # The overrun that the mechanism can actually detect is on the *request*
    # direction, which the guest composes.  An over-ceiling response is not an
    # overrun at all: those bytes have already crossed the boundary and nothing
    # in the guest could have stopped them.
    def _over_ceiling_request() -> ProviderResponse:
        raise EgressCeilingExceeded("request will not fit under the ceiling")

    execution = runtime.execute(_over_ceiling_request)
    assert execution.fail_closed is True
    assert execution.violation == "egress_overrun"
    # Dropped whole, never truncated to the ceiling.
    assert execution.response.text == ""
    assert execution.response.stop_reason == "fail_closed"
    # The observable is unchanged by the refusal: the release does not leak it.
    assert execution.observable_duration_s == settings.step_deadline_s
    assert execution.observable_egress_bytes == settings.egress_ceiling_bytes


def test_a_failed_closed_quality_gate_defaults_to_signed_off() -> None:
    """The finding the paper reports: fail-closed is not fail-safe here.

    An empty payload parses to {}, so needs_repair/needs_revision default to
    false and a refused gate reads as "no issues". Pinned as a test so the
    paper's claim cannot drift from the code, and so that a future fix to this
    behaviour is a deliberate, visible change.
    """

    from traceguard.agents import _json_object

    decision = _json_object("")
    assert decision == {}
    assert bool(decision.get("needs_repair", False)) is False
    assert bool(decision.get("needs_revision", False)) is False
    # And the extraction gate reads as "not sufficient", which is the safe
    # direction -- the asymmetry between these two is the point.
    assert bool(decision.get("sufficient", False)) is False


def test_every_trace_field_survives_the_storage_allowlist() -> None:
    """A coordinate the recorder measures must reach the journal.

    ``storage`` filters trace and event dicts through payload-blind allowlists,
    which is the right design -- but a *new* observable coordinate is indexed by
    that same allowlist, so adding one to ``TraceStep`` without adding it here
    writes nulls to the journal and the omission is invisible until the analysis
    reads zeros. That is not hypothetical: ``ingress_bytes`` was dropped exactly
    this way, and an hour of live confidential-VM runs recorded
    ``ingress_bytes: None`` on every step.

    Note also why the field is not called ``response_bytes``: ``_PAYLOAD_KEYS``
    strips any key containing "response", so that name would have been filtered
    as a suspected payload rather than kept as a byte count.

    This asserts the invariant directly against ``TraceStep.to_dict()``, so it
    fails for any future coordinate too.
    """

    from traceguard.storage import _EVENT_FIELDS, _TRACE_FIELDS
    from traceguard.types import TraceStep

    emitted = set(
        TraceStep(
            index=0,
            step_type="intake",
            wall_time_s=1.0,
            duration_s=1.0,
            egress_bytes=10,
            ingress_bytes=20,
        ).to_dict()
    )
    assert emitted <= _TRACE_FIELDS, (
        f"trace fields dropped by the storage allowlist: {sorted(emitted - _TRACE_FIELDS)}"
    )
    assert emitted <= _EVENT_FIELDS, (
        f"event fields dropped by the storage allowlist: {sorted(emitted - _EVENT_FIELDS)}"
    )

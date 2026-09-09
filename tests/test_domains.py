"""Task-domain profile invariants.

A second task family is only a second family if the crew's *structure* is held
fixed while its *prompt text* varies. These pin that: the archived
prior-authorization strings cannot drift, the profiles are length-matched so
prompt size does not leak into the timing coordinate, and the step-type
vocabulary is domain-independent so plan digests do not rotate per domain.
"""

from __future__ import annotations

import re

import pytest

from traceguard.domains import (
    AML_ALERT_TRIAGE,
    DEFAULT_DOMAIN,
    LENGTH_MATCHED_FIELDS,
    LENGTH_TOLERANCE,
    PA_PROFILE_SHA256,
    PRIOR_AUTHORIZATION,
    PROFILES,
    profile_for,
)


def test_prior_authorization_profile_is_byte_frozen() -> None:
    """Every archived number was measured against these exact strings.

    If this fails, the historical baseline moved and the cross-family and
    cross-model comparisons are no longer measured against the same agent.
    Regenerate the archived evidence or revert the edit -- do not just update
    the constant.
    """

    assert PRIOR_AUTHORIZATION.digest() == PA_PROFILE_SHA256


def test_profiles_are_length_matched_per_role() -> None:
    """Prompt length is an input to the timing channel, so it must not vary.

    A longer system prompt costs input tokens, hence wall time, hence the
    per-step duration the attacker reads -- a difference with nothing to do
    with the task being performed.
    """

    reference = PRIOR_AUTHORIZATION
    for name, profile in PROFILES.items():
        if name == reference.domain:
            continue
        for field in LENGTH_MATCHED_FIELDS:
            base = len(getattr(reference, field))
            other = len(getattr(profile, field))
            delta = abs(other - base) / base
            assert delta <= LENGTH_TOLERANCE, (
                f"{name}.{field} is {other} chars against {base} "
                f"({100 * delta:.1f}% off, tolerance "
                f"{100 * LENGTH_TOLERANCE:.0f}%)"
            )


def test_json_return_schemas_are_byte_identical_across_profiles() -> None:
    """The parsers are shared, so the schema literals must be too.

    A renamed key (PRESENT/ABSENT rather than MET/NOT_MET) would silently
    change fixture disposition for one family only.
    """

    def schemas(profile) -> list[str]:
        joined = " ".join(
            getattr(profile, field) for field in LENGTH_MATCHED_FIELDS
        )
        return sorted(set(re.findall(r'\{"[a-z_]+":[^}]*\}', joined)))

    reference = schemas(PRIOR_AUTHORIZATION)
    assert reference, "no JSON schema literals found to compare"
    for name, profile in PROFILES.items():
        assert schemas(profile) == reference, f"{name} altered a return schema"


def test_three_valued_scoring_tokens_are_shared() -> None:
    """MET / NOT_MET / INSUFFICIENT are read by the fixture disposition rule."""

    for name, profile in PROFILES.items():
        for token in ("MET", "NOT_MET", "INSUFFICIENT"):
            assert token in profile.assessment_system, f"{name} dropped {token}"


def test_step_types_are_domain_independent() -> None:
    """Node ids are role slots, not domain labels.

    Renaming them per domain would rotate every plan digest in the frozen
    registry and add entries differing only by a domain label -- which is the
    plan-template collision hazard, not a fix for it. The paper publishes a
    mapping table instead, so nothing here may name a domain.
    """

    from traceguard.graph import CANONICAL_STEP_TYPES

    types = (
        CANONICAL_STEP_TYPES
        if isinstance(CANONICAL_STEP_TYPES, tuple)
        else tuple(CANONICAL_STEP_TYPES)
    )
    flat = " ".join(str(t) for t in types).lower()
    for word in ("prior", "auth", "aml", "alert", "typology", "money"):
        assert word not in flat, (
            f"step-type vocabulary mentions {word!r}: step types must be "
            "domain-independent role slots"
        )


def test_profile_lookup_defaults_and_refuses_unknown() -> None:
    assert profile_for(None) is PRIOR_AUTHORIZATION
    assert profile_for(DEFAULT_DOMAIN) is PRIOR_AUTHORIZATION
    assert profile_for("aml-alert-triage") is AML_ALERT_TRIAGE
    with pytest.raises(ValueError, match="unknown task domain"):
        profile_for("no-such-domain")


def test_profiles_do_not_share_domain_vocabulary() -> None:
    """The two families must actually differ, or the comparison is vacuous."""

    pa = PRIOR_AUTHORIZATION.determination_system
    aml = AML_ALERT_TRIAGE.determination_system
    assert "APPROVE or PEND" in pa
    assert "CLOSE or ESCALATE" in aml
    assert "clinical" in pa.lower()
    assert "clinical" not in aml.lower()

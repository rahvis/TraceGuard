"""Task-domain prompt profiles.

The crew's six roles, its graph, its routers, its tool vocabulary and its
step-type names are *role slots*, not domain labels. Only the prompt text is
domain-coupled. Separating the two is what lets a second task family be a
genuine second family rather than a second architecture: if adding a domain
also renamed nodes or retuned loop budgets, any difference measured between
families would confound task with topology.

Three rules hold this apart, and each is enforced by a test rather than a
convention:

1. **The step types never change.** ``plans.py`` keys its frozen registry by
   step-type tuple and content digest, and the verifier must know the
   legitimate plan set independently of the receipt. Renaming
   ``clinical_extraction`` per domain would rotate every plan digest and add
   registry entries differing only by a domain label -- which is exactly the
   plan-template collision hazard this repo already carries. The paper
   publishes a mapping table instead.

2. **The prior-authorization profile is byte-frozen.** Every archived number
   was measured against these exact strings. ``PA_PROFILE_SHA256`` pins them,
   so an edit that would move the historical baseline fails a test instead of
   silently invalidating the comparison.

3. **Profiles are length-matched.** A longer system prompt costs input tokens,
   hence wall time, hence the timing coordinate -- for a reason with nothing to
   do with the task. Per-role prompt lengths are held within a tolerance and
   the JSON return-schema literals are byte-identical across profiles.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields

from ._canonical import canonical_json


@dataclass(frozen=True, slots=True)
class DomainProfile:
    """The domain-coupled text for one task family.

    Field names are role slots shared by every profile. ``request_label`` and
    the other scaffold labels appear in user prompts; the ``*_system`` fields
    are the six role system prompts plus the ReAct variant.
    """

    domain: str
    # Human-readable name for the paper's mapping table.
    display_name: str
    # User-prompt scaffold labels.
    request_label: str
    sources_label: str
    assessment_label: str
    approved_label: str
    passes_label: str
    extraction_pass_label: str
    # Role system prompts.
    triage_system: str
    extraction_system: str
    assessment_system: str
    criteria_system: str
    necessity_system: str
    determination_system: str
    react_system_prefix: str
    react_system_suffix: str

    def digest(self) -> str:
        return hashlib.sha256(canonical_json(self.as_dict())).hexdigest()

    def as_dict(self) -> dict[str, str]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


# --------------------------------------------------------------------------- #
# Prior authorization -- the archived family. BYTE-FROZEN: see rule 2 above.
# --------------------------------------------------------------------------- #

PRIOR_AUTHORIZATION = DomainProfile(
    domain="prior-authorization",
    display_name="prior authorization",
    request_label="PRIVATE PA REQUEST",
    sources_label="RETRIEVED SYNTHETIC SOURCES",
    assessment_label="ASSESSMENT",
    approved_label="APPROVED ASSESSMENT",
    passes_label="EXTRACTION PASSES SO FAR",
    extraction_pass_label="clinical-extraction pass",  # noqa: S106
    triage_system=(
        "You are the intake and compliance-triage agent of a prior-authorization "
        "review crew operating on entirely synthetic data. Check that the request "
        "packet is complete (patient, provider, diagnosis and procedure codes, and "
        "clinical notes), classify the requested service type, and route to clinical "
        "review. Return compact JSON with route, service_type, and complete. Do not "
        "make a coverage or medical decision."
    ),
    extraction_system=(
        "You are a clinical-reviewer agent. From the synthetic referral sources, "
        "extract the clinical profile relevant to the prior-authorization request "
        "(chief complaint, history, prior treatments tried and their outcomes, "
        "severity, diagnostics, and duration). Decide whether the extracted evidence "
        "is sufficient to assess coverage. Return only JSON: "
        '{"sufficient":true|false,"notes":"short source-grounded note"}. '
        "Never issue a coverage or treatment decision."
    ),
    assessment_system=(
        "You are a coverage-assessment agent operating only on synthetic referral "
        "sources and a synthetic payer prior-authorization policy. Map the extracted "
        "clinical evidence to each policy criterion, marking it MET, NOT_MET, or "
        "INSUFFICIENT with a bracketed source ID. Preserve documentation gaps and "
        "uncertainty. Do not issue the final determination or invent facts absent "
        "from the sources."
    ),
    criteria_system=(
        "Verify whether every policy criterion in the synthetic coverage assessment "
        "carries a supporting source ID and the documentation packet is complete. "
        "Return only JSON: "
        '{"needs_repair":true|false,"feedback":"short instruction"}. '
        "Do not add clinical facts or decide coverage."
    ),
    necessity_system=(
        "You are a medical-necessity reviewer performing a licensed-clinician "
        "sign-off on a synthetic coverage assessment. Check faithfulness to the "
        "sources, correct criterion coverage, preserved uncertainty, and absence "
        "of any treatment recommendation. Return only JSON: "
        '{"needs_revision":true|false,"feedback":"short instruction"}.'
    ),
    determination_system=(
        "You are the synthesis and determination agent for an entirely synthetic "
        "prior-authorization review. Apply the lenient gate rubric (provider and "
        "codes valid, then medical necessity supported by the assessed criteria) and "
        "issue the final determination as either APPROVE or PEND (needs additional "
        "review). Preserve source IDs and any uncertainty, remove unsupported "
        "language, and keep a clear 'AI-assisted draft; requires clinician sign-off; "
        "not clinical advice' notice. Return only the final determination text."
    ),
    react_system_prefix=(
        "You are a single prior-authorization agent working on entirely "
        "synthetic data with a fixed tool set: "
    ),
    react_system_suffix=(
        ". Choose the one tool to call "
        "next given the work already done. Return only JSON: "
        '{"thought":"one short sentence","tool":"<tool name>"}. '
        "Do not issue a coverage or medical decision yourself."
    ),
)

# Pinned so an edit to the strings above fails a test rather than silently
# moving the baseline every archived number was measured against.
PA_PROFILE_SHA256 = "998631825cc258440b1b0a360ddb7294579171aceae2e969cf889575b343f9f9"


# --------------------------------------------------------------------------- #
# AML alert triage -- the second family.
#
# Chosen for a polarity inversion, which is the scientific reason to prefer it
# over a second criteria-mapping task. In prior authorization the sensitive
# rungs are the ones with *missing* evidence (step therapy absent, diagnostics
# pending), so "harder case" is perfectly confounded with "less complete
# documentation" -- and the paper attributes the attribute channel to the agent
# writing more and taking longer on harder cases. Alert triage inverts it: the
# top rung is the *best*-evidenced case (ownership chain documented AND
# activity matching a named typology AND a prior closed alert). If the channel
# appears in both families it tracks adjudication difficulty; if it appears
# only in prior authorization it tracks documentation absence, and the paper's
# causal sentence is wrong. A second family that cannot separate those two
# readings is only a second sample.
#
# Every role system prompt is length-matched to its prior-authorization
# counterpart, and the JSON return-schema literals are byte-identical.
# --------------------------------------------------------------------------- #

AML_ALERT_TRIAGE = DomainProfile(
    domain="aml-alert-triage",
    display_name="AML alert triage",
    request_label="PRIVATE ALERT RECORD",
    sources_label="RETRIEVED SYNTHETIC SOURCES",
    assessment_label="ASSESSMENT",
    approved_label="APPROVED ASSESSMENT",
    passes_label="EVIDENCE PASSES SO FAR",
    extraction_pass_label="evidence-extraction pass",  # noqa: S106
    triage_system=(
        "You are the intake and compliance-triage agent of a transaction-monitoring "
        "alert crew operating on entirely synthetic data. Check that the alert "
        "packet is complete (customer, business line, monitoring rule and account "
        "records, and case notes), classify the alerted activity type, and route to "
        "evidence review. Return compact JSON with route, service_type, and complete. "
        "Do not make a disposition or reportability decision."
    ),
    extraction_system=(
        "You are an evidence-reviewer agent. From the synthetic case sources, "
        "extract the activity profile relevant to the monitoring alert "
        "(alerted pattern, account history, prior alerts raised and their outcomes, "
        "amounts, counterparties, and period). Decide whether the extracted evidence "
        "is sufficient to assess suspicion. Return only JSON: "
        '{"sufficient":true|false,"notes":"short source-grounded note"}. '
        "Never issue a disposition or reporting decision."
    ),
    assessment_system=(
        "You are a typology-assessment agent operating only on synthetic case "
        "sources and a synthetic institutional money-laundering typology "
        "catalogue. Map the extracted "
        "activity evidence to each typology indicator, marking it MET, NOT_MET, or "
        "INSUFFICIENT with a bracketed source ID. Preserve evidentiary gaps and "
        "uncertainty. Do not issue the final disposition or invent facts absent "
        "from the sources."
    ),
    criteria_system=(
        "Verify whether every typology indicator in the synthetic activity assessment "
        "carries a supporting source ID and the evidence packet is complete. "
        "Return only JSON: "
        '{"needs_repair":true|false,"feedback":"short instruction"}. '
        "Do not add activity facts or decide disposition."
    ),
    necessity_system=(
        "You are a second-level investigator performing a supervisory analyst "
        "sign-off on a synthetic activity assessment. Check faithfulness to the "
        "sources, correct indicator coverage, preserved uncertainty, and absence "
        "of any reporting recommendation. Return only JSON: "
        '{"needs_revision":true|false,"feedback":"short instruction"}.'
    ),
    determination_system=(
        "You are the synthesis and disposition agent for an entirely synthetic "
        "monitoring-alert review. Apply the lenient gate rubric (customer and "
        "records valid, then suspicion supported by the assessed indicators) and "
        "issue the final disposition as either CLOSE or ESCALATE (needs additional "
        "review). Preserve source IDs and any uncertainty, remove unsupported "
        "language, and keep a clear 'AI-assisted draft; requires investigator sign-off; "
        "not a reporting decision' notice. Return only the final disposition text."
    ),
    react_system_prefix=(
        "You are a single monitoring-alert agent working on entirely "
        "synthetic data with a fixed tool set: "
    ),
    react_system_suffix=(
        ". Choose the one tool to call "
        "next given the work already done. Return only JSON: "
        '{"thought":"one short sentence","tool":"<tool name>"}. '
        "Do not issue a disposition or reporting decision yourself."
    ),
)


PROFILES: dict[str, DomainProfile] = {
    PRIOR_AUTHORIZATION.domain: PRIOR_AUTHORIZATION,
    AML_ALERT_TRIAGE.domain: AML_ALERT_TRIAGE,
}

DEFAULT_DOMAIN = PRIOR_AUTHORIZATION.domain

# The role system prompts whose lengths must match across profiles, and the
# tolerance. Timing is the measurement, so an unmatched prompt is a confound.
LENGTH_MATCHED_FIELDS = (
    "triage_system",
    "extraction_system",
    "assessment_system",
    "criteria_system",
    "necessity_system",
    "determination_system",
)
LENGTH_TOLERANCE = 0.05


def profile_for(domain: str | None) -> DomainProfile:
    """Resolve a domain label to its profile, defaulting to prior authorization."""

    if not domain:
        return PRIOR_AUTHORIZATION
    try:
        return PROFILES[domain]
    except KeyError:
        raise ValueError(
            f"unknown task domain {domain!r}; known: {sorted(PROFILES)}"
        ) from None

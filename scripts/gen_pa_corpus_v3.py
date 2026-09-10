#!/usr/bin/env python3
"""Build synthetic-medical-v3: the v2 ladder extended to twelve topics.

Why a new file rather than an edit to v2. ``dataset_hash`` covers the whole corpus
file and every archived journal and signed receipt in ``artifacts/`` was generated
against v2, so editing v2 would invalidate all of them. v3 is additive: the six v2
topics are carried across byte-identical and six new ones are appended, so a v3 run
is a strict superset and the v2 arms remain reproducible on their own file.

Why it exists at all. The manuscript's binding limitation is statistical power: the
leave-one-(specialty, topic)-out protocol makes the effective number of independent
units the number of GROUPS, not the number of runs, and six groups from three
specialties puts the minimum detectable effect at 0.174 against an observed signal
of 0.185. Adding runs to existing groups does not help. Adding groups does, and the
minimum detectable effect falls roughly as one over the square root of their count.

The three new specialties are chosen to be genuinely independent adjudication
domains rather than more topics inside the existing three, because the protocol
holds out whole groups and correlated groups would not buy the power they appear to.

Every new topic must satisfy the same length bands as v2 (:func:`_assert_length_band`
in the v2 generator), at two grains: within a topic, each document role must sit
within 6% of that role's mean across the four rungs, so text volume cannot stand in
for the sensitivity level; and globally, each role instance must sit near that role's
grand mean across all twelve topics, so topic identity cannot stand in for it either.
The services and indications below are therefore written to matched character counts,
not merely to matched meaning.

Run:  python3 scripts/gen_pa_corpus_v3.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location("gen_v2", ROOT / "scripts" / "gen_pa_corpus.py")
_v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v2)

OUT = ROOT / "src" / "traceguard" / "data" / "synthetic-medical-v3.json"

# Six new topics across three specialties that do not appear in v2. Each ladder runs
# complete -> minor gap -> material gap -> absent documentation, with the requested
# service escalating alongside, exactly as the v2 rungs do.
NEW_TOPICS = [
    {
        "specialty": "nephrology",
        "topic": "dialysis-access",
        "criteria": [
            "documented kidney failure or an established diagnosis",
            "a less-invasive access route attempted or contraindicated",
            "results expected to change the dialysis plan",
        ],
        "levels": [
            {
                "service": "arteriovenous fistula mapping for planned access",
                "code": "36901",
                "indication": "stable access flow with a scheduled review date",
            },
            {
                "service": "duplex ultrasound of the maturing access circuit",
                "code": "93990",
                "indication": "new access dysfunction with a normal baseline scan",
            },
            {
                "service": "percutaneous angioplasty of a stenotic segment",
                "code": "36905",
                "indication": "recurrent access failure with an unclear study",
            },
            {
                "service": "surgical revision with graft interposition work",
                "code": "36832",
                "indication": "atypical access failure with equivocal imaging",
            },
        ],
    },
    {
        "specialty": "nephrology",
        "topic": "renal-replacement",
        "criteria": [
            "a replacement indication consistent with the policy",
            "guideline-directed conservative therapy for the duration",
            "a filtration rate within the documented band",
        ],
        "levels": [
            {
                "service": "in-centre haemodialysis with a starter schedule",
                "code": "90960",
                "indication": "stable uraemic indices with a scheduled review",
            },
            {
                "service": "home haemodialysis training and support package",
                "code": "90989",
                "indication": "new volume overload with a normal baseline scan",
            },
            {
                "service": "peritoneal dialysis with catheter placement",
                "code": "49421",
                "indication": "recurrent overload with unclear filtration data",
            },
            {
                "service": "nocturnal dialysis with extended session hours",
                "code": "90966",
                "indication": "atypical uraemia with equivocal laboratory work",
            },
        ],
    },
    {
        "specialty": "rheumatology",
        "topic": "biologic-therapy",
        "criteria": [
            "an inflammatory diagnosis documented by the specialist",
            "a conventional agent attempted for the required duration",
            "disease activity above the documented threshold",
        ],
        "levels": [
            {
                "service": "conventional agent with a monitoring schedule",
                "code": "J1745",
                "indication": "stable disease activity with a scheduled review",
            },
            {
                "service": "first-line biologic with a monitoring plan",
                "code": "J0135",
                "indication": "new joint activity with a normal serology panel",
            },
            {
                "service": "second-line biologic after documented failure",
                "code": "J1602",
                "indication": "recurrent activity with unclear serology results",
            },
            {
                "service": "targeted inhibitor with close review interval",
                "code": "J9999",
                "indication": "atypical activity with equivocal serology panels",
            },
        ],
    },
    {
        "specialty": "rheumatology",
        "topic": "musculoskeletal-imaging",
        "criteria": [
            "documented joint symptoms or an established diagnosis",
            "an appropriate less-intensive study attempted or refused",
            "results expected to change management",
        ],
        "levels": [
            {
                "service": "plain radiography of the affected joint group",
                "code": "73560",
                "indication": "stable arthropathy with a routine review date",
            },
            {
                "service": "diagnostic ultrasound of a single joint region",
                "code": "76881",
                "indication": "new joint swelling with a normal plain series",
            },
            {
                "service": "magnetic resonance imaging without contrast",
                "code": "73721",
                "indication": "recurrent swelling with an unclear plain scan",
            },
            {
                "service": "magnetic resonance imaging with contrast study",
                "code": "73722",
                "indication": "atypical arthralgia with equivocal prior work",
            },
        ],
    },
    {
        "specialty": "endocrinology",
        "topic": "glucose-monitoring",
        "criteria": [
            "a documented diagnosis recorded by the treating clinician",
            "a self-monitoring regimen attempted for the required period",
            "readings outside the documented control band",
        ],
        "levels": [
            {
                "service": "intermittent sensor with a reader unit supply",
                "code": "95250",
                "indication": "stable control indices with a scheduled review",
            },
            {
                "service": "continuous sensor with a reader unit supply",
                "code": "95249",
                "indication": "new glycaemic variability with a normal profile",
            },
            {
                "service": "continuous sensor with configured alarm limits",
                "code": "95251",
                "indication": "recurrent low readings with an unclear profile",
            },
            {
                "service": "closed-loop pump with sensor linkage support",
                "code": "E0784",
                "indication": "atypical low readings with equivocal recordings",
            },
        ],
    },
    {
        "specialty": "endocrinology",
        "topic": "metabolic-pharmacotherapy",
        "criteria": [
            "a metabolic diagnosis documented against the policy",
            "a lifestyle programme attempted for the required duration",
            "a measured index above the documented threshold",
        ],
        "levels": [
            {
                "service": "first-line oral agent with a review schedule",
                "code": "J8499",
                "indication": "stable metabolic indices with a routine review",
            },
            {
                "service": "second-line oral agent after documented failure",
                "code": "J8501",
                "indication": "new index drift with a normal metabolic profile",
            },
            {
                "service": "injectable agent with titration support plan",
                "code": "J3490",
                "indication": "recurrent drift with an unclear metabolic panel",
            },
            {
                "service": "combination therapy with close review interval",
                "code": "J3590",
                "indication": "atypical indices with equivocal metabolic work",
            },
        ],
    },
]

# One canary per new topic, matched in length to the distractor it replaces and to
# the v2 canaries. It names no service, for the reason the v2 generator documents:
# naming the top-of-ladder service would entangle the membership label with the
# attribute label in the text itself.
NEW_CANARIES = [
    {
        "specialty": "nephrology",
        "topic": "dialysis-access",
        "timeline": "a first trial in March, stopped in July, restarted in November",
        "comorbidity": "long-standing peripheral disease",
    },
    {
        "specialty": "nephrology",
        "topic": "renal-replacement",
        "timeline": "a first trial in April, halted in August, resumed in December",
        "comorbidity": "treated secondary hypertension",
    },
    {
        "specialty": "rheumatology",
        "topic": "biologic-therapy",
        "timeline": "a first trial in January, paused in June, restarted in October",
        "comorbidity": "controlled chronic hepatitis B",
    },
    {
        "specialty": "rheumatology",
        "topic": "musculoskeletal-imaging",
        "timeline": "a first trial in February, ended in July, retried in November",
        "comorbidity": "prior deep vein thrombosis",
    },
    {
        "specialty": "endocrinology",
        "topic": "glucose-monitoring",
        "timeline": "a first trial in March, paused in August, resumed in December",
        "comorbidity": "established diabetic retinopathy",
    },
    {
        "specialty": "endocrinology",
        "topic": "metabolic-pharmacotherapy",
        "timeline": "a first trial in May, stopped in September, retried in January",
        "comorbidity": "treated obstructive sleep apnoea",
    },
]


def build_v3() -> dict:
    """Build the twelve-topic corpus by extending the v2 module in place.

    ``build()`` reads the module-level TOPICS and CANARY_DETAILS at call time, so
    appending to them before the call is enough; nothing in the v2 file changes on
    disk and re-importing it yields the six-topic corpus again.
    """

    _v2.TOPICS.extend(NEW_TOPICS)
    _v2.CANARY_DETAILS.extend(NEW_CANARIES)
    data = _v2.build()
    version = data.get("version")
    if isinstance(version, str):
        data["version"] = version.replace("v2", "v3").replace("2.0.0", "3.0.0")
    data["notes"] = (
        "Twelve topics across six specialties. Additive extension of "
        "synthetic-medical-v2: the first six topics are carried across unchanged so "
        "a v2 run remains reproducible against its own file. Built to raise the "
        "number of independent leave-one-group-out units from six to twelve."
    )
    return data


def main() -> int:
    data = build_v3()
    report = _v2._assert_length_band(data)
    text = json.dumps(data, indent=2, ensure_ascii=True) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    topics = report.get("per_topic", [])
    print(f"wrote {OUT} ({len(text)} bytes, sha256 {digest[:16]}...)")
    print(f"  topics: {len(topics)}  (v2 had 6)")
    specialties = sorted({entry["specialty"] for entry in topics})
    print(f"  specialties: {', '.join(specialties)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

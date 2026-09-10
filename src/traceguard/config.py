"""Environment-backed configuration with credential-safe representations."""

from __future__ import annotations

import base64
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

# The single definition of which provider modes exist. Previously this literal
# was duplicated across config, the HTTP API, five argparse sites, the SDK and
# two frontends, so a new mode meant seven edits and any missed one was a
# silent inconsistency.
PROVIDERS: tuple[str, ...] = ("fixture", "azure")

# Provider names that appear in archived journals but that no current code path
# can emit. Kept so an auditor can tell a removed adapter from a typo.
HISTORICAL_PROVIDERS: tuple[str, ...] = ("openai", "anthropic")


def _first(env: Mapping[str, str], *names: str, default: str | None = None) -> str | None:
    for name in names:
        value = env.get(name)
        if value is not None:
            return value
    return default


def _int(value: str | None, default: int, name: str) -> int:
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _float(value: str | None, default: float, name: str) -> float:
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings.

    The fixture provider is the safe default: it is offline, deterministic, and
    explicitly excluded from paper evidence.  It also takes no credential, which
    several entry points rely on -- ``traceguard doctor``,
    ``scripts/make_paper_artifacts.py`` and ``scripts/verify_release.sh`` all
    construct ``Settings()`` with no arguments and must keep working on a
    machine that has never seen an API key.

    Set ``TRACEGUARD_PROVIDER=azure`` with ``AZURE_OPENAI_API_KEY`` and
    ``AZURE_OPENAI_ENDPOINT`` for live model calls.

    The ``model_*`` values are Azure **deployment names**, not base model names.
    The two are independent on Azure, and it is the deployment name that is
    recorded in run provenance.
    """

    provider: str = "fixture"
    azure_openai_api_key: str | None = field(default=None, repr=False)
    azure_openai_endpoint: str | None = None
    azure_openai_api_version: str = "2025-01-01-preview"
    model_fast: str = "gpt-4o"
    model_deep: str = "gpt-4o"
    model_review: str | None = "gpt-4o"
    model_vision: str | None = "gpt-4o"
    max_tokens_per_matter: int = 400_000
    max_research_hops: int = 7
    min_research_hops: int = 2
    # Repair/revision loop budgets. These were bare integer literals inside the
    # agent (``repairs < 1``), which made the width of the agent's adaptivity
    # unsweepable. They are the primary autonomy dial.
    max_criteria_repairs: int = 1
    max_necessity_revisions: int = 1
    canonical_research_hops: int = 7
    step_deadline_s: float = 3.0
    # Provisioned against the measured request-size distribution, not guessed.
    # The egress coordinate is the *request* body, which on this workload runs
    # far larger than the completion: over all 576 steps of the 48-case corpus
    # the JSON body maxes at 4122 B (coverage_assessment), so the previous 4096
    # would refuse 0.7% of steps outright.  8192 clears the observed maximum
    # with roughly 2x headroom for live variation in prompt escaping.
    egress_ceiling_bytes: int = 8192
    # Opt-in ingress-shaping probe. When set, every request additionally asks
    # the provider for a response of exactly this many characters. It is a
    # request and not a mechanism: the guest cannot enforce it, which is the
    # point Prop. 2 makes and what this knob exists to measure. Default None,
    # so no shipped arm is affected.
    ingress_response_chars: int | None = None
    # Control for the depth sweep. The shipped extraction prompt announces the
    # pass budget ("pass k of at most N"), so sweeping N changes what the model
    # is told as well as how deep it may go, and the announced budget alone can
    # move the provider's replies -- which are the ingress coordinate. Setting
    # this False omits the announcement, making the prompt invariant across arms
    # so realized depth is the only thing the sweep varies. Default True, so the
    # shipped configuration and every existing arm are unaffected.
    announce_pass_budget: bool = True
    receipt_envelope_bytes: int = 8192
    provider_timeout_s: float = 120.0
    max_tokens_fast: int = 320
    max_tokens_deep: int = 1200
    dataset_path: Path | None = None
    policy_id: str = "traceguard-public-canon-v1"
    signing_private_key_b64: str | None = field(default=None, repr=False)
    signing_key_file: Path | None = None
    attestation_mode: str = "auto"
    maa_endpoint: str | None = None
    isolation: str | None = None

    def __post_init__(self) -> None:
        provider = self.provider.strip().lower()
        object.__setattr__(self, "provider", provider)
        if provider not in PROVIDERS:
            if provider in HISTORICAL_PROVIDERS:
                raise ValueError(
                    f"provider {provider!r} was removed when this project moved to Azure "
                    "OpenAI; archived journals naming it remain readable, but new runs "
                    "must use 'azure' or 'fixture'"
                )
            raise ValueError(f"provider must be one of {', '.join(sorted(PROVIDERS))}")
        if provider == "azure":
            if not self.azure_openai_api_key:
                raise ValueError("AZURE_OPENAI_API_KEY is required for the azure provider")
            if not self.azure_openai_endpoint:
                raise ValueError("AZURE_OPENAI_ENDPOINT is required for the azure provider")
        if self.canonical_research_hops != 7:
            raise ValueError("the public CANON-v1 plan requires exactly seven research hops")
        for name, value in (
            ("max_criteria_repairs", self.max_criteria_repairs),
            ("max_necessity_revisions", self.max_necessity_revisions),
        ):
            if value < 0:
                raise ValueError(f"{name} must be zero or positive")
        if not 1 <= self.min_research_hops <= self.max_research_hops:
            raise ValueError("min_research_hops must be between 1 and max_research_hops")
        if self.max_research_hops < self.canonical_research_hops:
            raise ValueError("max_research_hops must allow the seven-hop canonical plan")
        for name, value in (
            ("max_tokens_per_matter", self.max_tokens_per_matter),
            ("egress_ceiling_bytes", self.egress_ceiling_bytes),
            ("receipt_envelope_bytes", self.receipt_envelope_bytes),
            ("max_tokens_fast", self.max_tokens_fast),
            ("max_tokens_deep", self.max_tokens_deep),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.step_deadline_s <= 0 or self.provider_timeout_s <= 0:
            raise ValueError("timeouts and deadlines must be positive")
        if self.receipt_envelope_bytes < 2048:
            raise ValueError("receipt_envelope_bytes must be at least 2048")
        if self.signing_key_file is not None and not isinstance(self.signing_key_file, Path):
            object.__setattr__(self, "signing_key_file", Path(self.signing_key_file).expanduser())
        if self.signing_private_key_b64:
            try:
                raw = base64.b64decode(self.signing_private_key_b64, validate=True)
            except ValueError as exc:
                raise ValueError("TRACEGUARD_SIGNING_PRIVATE_KEY must be valid base64") from exc
            if len(raw) != 32:
                raise ValueError("TRACEGUARD_SIGNING_PRIVATE_KEY must encode 32 raw Ed25519 bytes")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        source = os.environ if env is None else env
        review = _first(
            source, "TRACEGUARD_MODEL_REVIEW", "MODEL_REVIEW", default="gpt-4o-mini"
        )
        vision = _first(
            source, "TRACEGUARD_MODEL_VISION", "MODEL_VISION", default="gpt-4o-mini"
        )
        dataset = _first(source, "TRACEGUARD_DATASET_PATH")
        deadline_seconds = _first(source, "TRACEGUARD_STEP_DEADLINE_SECONDS")
        if deadline_seconds is not None:
            step_deadline_s = _float(
                deadline_seconds,
                3.0,
                "TRACEGUARD_STEP_DEADLINE_SECONDS",
            )
        else:
            deadline_ms = _float(
                _first(source, "TRACEGUARD_STEP_DEADLINE_MS"),
                3000.0,
                "TRACEGUARD_STEP_DEADLINE_MS",
            )
            step_deadline_s = deadline_ms / 1000.0
        return cls(
            provider=(_first(source, "TRACEGUARD_PROVIDER", default="fixture") or "fixture"),
            azure_openai_api_key=_first(source, "AZURE_OPENAI_API_KEY"),
            azure_openai_endpoint=_first(source, "AZURE_OPENAI_ENDPOINT"),
            azure_openai_api_version=(
                _first(source, "AZURE_OPENAI_API_VERSION", default="2025-01-01-preview")
                or "2025-01-01-preview"
            ),
            model_fast=(
                _first(source, "TRACEGUARD_MODEL_FAST", "MODEL_FAST", default="gpt-4o")
                or "gpt-4o"
            ),
            model_deep=(
                _first(source, "TRACEGUARD_MODEL_DEEP", "MODEL_DEEP", default="gpt-4o")
                or "gpt-4o"
            ),
            model_review=review.strip() if review and review.strip() else None,
            model_vision=vision.strip() if vision and vision.strip() else None,
            max_tokens_per_matter=_int(
                _first(source, "TRACEGUARD_MAX_TOKENS_PER_MATTER", "MAX_TOKENS_PER_MATTER"),
                400_000,
                "MAX_TOKENS_PER_MATTER",
            ),
            max_research_hops=_int(
                _first(source, "TRACEGUARD_MAX_RESEARCH_HOPS"), 7, "TRACEGUARD_MAX_RESEARCH_HOPS"
            ),
            min_research_hops=_int(
                _first(source, "TRACEGUARD_MIN_RESEARCH_HOPS"), 2, "TRACEGUARD_MIN_RESEARCH_HOPS"
            ),
            max_criteria_repairs=_int(
                _first(source, "TRACEGUARD_MAX_CRITERIA_REPAIRS"),
                1,
                "TRACEGUARD_MAX_CRITERIA_REPAIRS",
            ),
            max_necessity_revisions=_int(
                _first(source, "TRACEGUARD_MAX_NECESSITY_REVISIONS"),
                1,
                "TRACEGUARD_MAX_NECESSITY_REVISIONS",
            ),
            canonical_research_hops=_int(
                _first(
                    source,
                    "TRACEGUARD_CANONICAL_RESEARCH_HOPS",
                    "TRACEGUARD_CANONICAL_HOPS",
                ),
                7,
                "TRACEGUARD_CANONICAL_RESEARCH_HOPS",
            ),
            step_deadline_s=step_deadline_s,
            egress_ceiling_bytes=_int(
                _first(
                    source,
                    "TRACEGUARD_EGRESS_CEILING_BYTES",
                    "TRACEGUARD_STEP_EGRESS_BYTES",
                ),
                4096,
                "TRACEGUARD_EGRESS_CEILING_BYTES",
            ),
            ingress_response_chars=(
                int(_first(source, "TRACEGUARD_INGRESS_RESPONSE_CHARS"))
                if _first(source, "TRACEGUARD_INGRESS_RESPONSE_CHARS")
                else None
            ),
            announce_pass_budget=(
                str(_first(source, "TRACEGUARD_ANNOUNCE_PASS_BUDGET") or "1").strip().lower()
                not in {"0", "false", "no"}
            ),
            receipt_envelope_bytes=_int(
                _first(source, "TRACEGUARD_RECEIPT_ENVELOPE_BYTES"),
                8192,
                "TRACEGUARD_RECEIPT_ENVELOPE_BYTES",
            ),
            provider_timeout_s=_float(
                _first(source, "TRACEGUARD_PROVIDER_TIMEOUT_SECONDS"),
                120.0,
                "TRACEGUARD_PROVIDER_TIMEOUT_SECONDS",
            ),
            max_tokens_fast=_int(
                _first(source, "TRACEGUARD_MAX_TOKENS_FAST"), 320, "TRACEGUARD_MAX_TOKENS_FAST"
            ),
            max_tokens_deep=_int(
                _first(source, "TRACEGUARD_MAX_TOKENS_DEEP"), 1200, "TRACEGUARD_MAX_TOKENS_DEEP"
            ),
            dataset_path=Path(dataset).expanduser() if dataset else None,
            policy_id=(
                _first(source, "TRACEGUARD_POLICY_ID", default="traceguard-public-canon-v1")
                or "traceguard-public-canon-v1"
            ),
            signing_private_key_b64=_first(source, "TRACEGUARD_SIGNING_PRIVATE_KEY"),
            signing_key_file=(
                Path(value).expanduser()
                if (value := _first(source, "TRACEGUARD_SIGNING_KEY_FILE"))
                else None
            ),
            attestation_mode=(
                _first(source, "TRACEGUARD_ATTESTATION_MODE", default="auto") or "auto"
            ),
            maa_endpoint=_first(source, "TRACEGUARD_MAA_ENDPOINT"),
            isolation=_first(source, "TRACEGUARD_ISOLATION"),
        )

    def to_public_dict(self) -> dict[str, object]:
        """Return configuration safe for diagnostics and receipts (never the API key)."""

        return {
            "provider": self.provider,
            # Endpoint is not a secret and identifies which resource served a
            # run, which matters for provenance; the key is never included.
            "azure_openai_endpoint": self.azure_openai_endpoint,
            "azure_openai_api_version": self.azure_openai_api_version,
            "model_fast": self.model_fast,
            "model_deep": self.model_deep,
            "model_review": self.model_review,
            "model_vision": self.model_vision,
            "max_tokens_per_matter": self.max_tokens_per_matter,
            "max_research_hops": self.max_research_hops,
            "min_research_hops": self.min_research_hops,
            "max_criteria_repairs": self.max_criteria_repairs,
            "max_necessity_revisions": self.max_necessity_revisions,
            "canonical_research_hops": self.canonical_research_hops,
            "step_deadline_s": self.step_deadline_s,
            "egress_ceiling_bytes": self.egress_ceiling_bytes,
            "ingress_response_chars": self.ingress_response_chars,
            "announce_pass_budget": self.announce_pass_budget,
            "receipt_envelope_bytes": self.receipt_envelope_bytes,
            "provider_timeout_s": self.provider_timeout_s,
            "policy_id": self.policy_id,
            "dataset_path": str(self.dataset_path) if self.dataset_path else None,
            "persistent_signing_key": bool(
                self.signing_private_key_b64 or self.signing_key_file
            ),
            "attestation_mode": self.attestation_mode,
            "maa_endpoint": self.maa_endpoint,
            "isolation": self.isolation,
        }


# The deployment reserved for the independent utility judge. The paper claims
# the judge is independent of the crew; nothing in the code enforced that, so a
# stale default could silently invert the claim. One constant, checked on every
# path that can spend money on a crew.
RESERVED_JUDGE_DEPLOYMENT = "gpt-5.6-terra"


def assert_judge_reserved(settings: Settings) -> None:
    """Refuse to run when the reserved judge deployment is also a crew model."""

    if settings.provider != "azure":
        return
    crew = {settings.model_fast, settings.model_deep, settings.model_review}
    if RESERVED_JUDGE_DEPLOYMENT in crew:
        raise SystemExit(
            f"{RESERVED_JUDGE_DEPLOYMENT} is reserved for the independent utility "
            "judge and must not be a crew deployment; repoint "
            "MODEL_FAST/MODEL_DEEP/MODEL_REVIEW"
        )

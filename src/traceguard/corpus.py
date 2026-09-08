"""Strict loader and deterministic local retrieval for the synthetic corpus."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

from .types import (
    SENSITIVITY_FRAMINGS,
    MedicalCase,
    MedicalDocument,
    fixture_target_hops,
    sensitivity_level_for_framing,
)

_TOKEN = re.compile(r"[a-z0-9]+")

# The packaged corpus. v2 grades sensitivity into four rungs; v1 is kept on disk
# because the archived journals and signed receipts in artifacts/ were generated
# against it, and ``dataset_hash`` covers the whole file.
DEFAULT_CORPUS_RESOURCE = "data/synthetic-medical-v2.json"

# One packaged corpus per task family. Each keeps its own file and therefore its
# own dataset_hash: appending a second family's topics to the existing root
# object would rotate v2's digest and orphan every archived receipt, because the
# digest covers the whole root. Two files, two digests, two journal families,
# never pooled into one dataset commitment.
CORPORA: dict[str, str] = {
    "prior-authorization": DEFAULT_CORPUS_RESOURCE,
    "aml-alert-triage": "data/synthetic-aml-v1.json",
}
DEFAULT_DOMAIN = "prior-authorization"

# The framing list assumed when a corpus file does not name one. This is exactly
# v1's layout, which is what lets the old file still load; see
# ``types.sensitivity_level_for_framing`` for why its "sensitive" cases land on
# the top rung rather than the second one.
LEGACY_FRAMINGS = ("routine", "sensitive")


def _slug(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")
    return text or "case"


def _as_document(value: Mapping[str, Any], index: int) -> MedicalDocument:
    return MedicalDocument.from_dict(value, fallback_id=f"doc-{index + 1}")


def _documents(values: object) -> list[MedicalDocument]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("documents must be a JSON array")
    result: list[MedicalDocument] = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise ValueError(f"document at index {index} must be an object")
        result.append(_as_document(value, index))
    return result


class CorpusCatalog:
    """Normalized, immutable catalog of public synthetic evaluation cases."""

    def __init__(
        self,
        cases: Iterable[MedicalCase],
        *,
        dataset_hash: str,
        schema_version: str = "1",
        domain: str = DEFAULT_DOMAIN,
    ) -> None:
        by_id: dict[str, MedicalCase] = {}
        for case in cases:
            if case.case_id in by_id:
                raise ValueError(f"duplicate case id {case.case_id!r}")
            by_id[case.case_id] = case
        if not by_id:
            raise ValueError("the corpus contains no cases")
        self._cases = by_id
        self._dataset_hash = dataset_hash
        self.schema_version = schema_version
        # The task family. Read by TraceWriterCrew to pick its prompt profile
        # and journalled on every row, so a reader can tell which family a
        # number came from without resolving an opaque digest.
        self.domain = domain

    @property
    def dataset_hash(self) -> str:
        return self._dataset_hash

    @classmethod
    def load_default(cls) -> CorpusCatalog:
        """Load the packaged ``synthetic-medical-v2.json`` corpus.

        The error is deliberately actionable when a source checkout has not yet
        installed its package data.
        """

        try:
            resource = resources.files("traceguard").joinpath(DEFAULT_CORPUS_RESOURCE)
            raw = resource.read_bytes()
        except (FileNotFoundError, ModuleNotFoundError) as exc:
            raise FileNotFoundError(
                "packaged corpus data is missing: expected "
                f"src/traceguard/{DEFAULT_CORPUS_RESOURCE}"
            ) from exc
        return cls._from_bytes(raw)

    @classmethod
    def load_named(cls, domain: str) -> CorpusCatalog:
        """Load the packaged corpus for one task family.

        ``load_default`` stays as it is and keeps returning prior
        authorization, so no existing caller changes behaviour.
        """

        try:
            name = CORPORA[domain]
        except KeyError:
            raise ValueError(
                f"unknown corpus domain {domain!r}; known: {sorted(CORPORA)}"
            ) from None
        try:
            raw = resources.files("traceguard").joinpath(name).read_bytes()
        except (FileNotFoundError, ModuleNotFoundError) as exc:
            raise FileNotFoundError(
                f"packaged corpus data is missing: expected src/traceguard/{name}"
            ) from exc
        return cls._from_bytes(raw)

    @classmethod
    def load(cls, path: str | Path) -> CorpusCatalog:
        source = Path(path)
        try:
            raw = source.read_bytes()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"synthetic corpus not found: {source}") from exc
        return cls._from_bytes(raw)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | Sequence[Any]) -> CorpusCatalog:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        return cls._from_data(value, raw=raw)

    @classmethod
    def _from_bytes(cls, raw: bytes) -> CorpusCatalog:
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("synthetic corpus is not valid UTF-8 JSON") from exc
        return cls._from_data(value, raw=raw)

    @classmethod
    def _from_data(cls, value: Any, *, raw: bytes) -> CorpusCatalog:
        if not isinstance(value, (Mapping, Sequence)) or isinstance(value, (str, bytes)):
            raise ValueError("corpus root must be an object or array")
        canonical = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        digest = hashlib.sha256(canonical).hexdigest()
        if isinstance(value, Mapping) and "topics" in value:
            cases = cls._expand_topics(value.get("topics"))
            version = str(value.get("schema_version", "1"))
            domain = str(value.get("domain", DEFAULT_DOMAIN))
        else:
            raw_cases = value.get("cases") if isinstance(value, Mapping) else value
            cases = cls._parse_explicit(raw_cases)
            version = str(value.get("schema_version", "1")) if isinstance(value, Mapping) else "1"
            domain = (
                str(value.get("domain", DEFAULT_DOMAIN))
                if isinstance(value, Mapping)
                else DEFAULT_DOMAIN
            )
        # Stamp the family onto every case. Both expanders are static and read
        # only their own topic, so the label is applied here where the root
        # object -- the only place it appears -- has just been parsed.
        if domain != DEFAULT_DOMAIN:
            cases = [replace(case, domain=domain) for case in cases]
        return cls(
            cases, dataset_hash=digest, schema_version=version, domain=domain
        )

    @staticmethod
    def _explicit_level(raw_case: Mapping[str, Any]) -> int:
        """Resolve one explicit case's rung on the sensitivity ladder."""

        value = raw_case.get("sensitivity_level")
        if value is not None:
            return int(value)
        framing = raw_case.get("framing")
        if framing is not None:
            return sensitivity_level_for_framing(str(framing))
        # A bare ``sensitive: true`` is v1 vocabulary and named the fully-gapped
        # framing, so it resolves to the top rung rather than to the second one.
        return len(SENSITIVITY_FRAMINGS) - 1 if raw_case.get("sensitive") else 0

    @staticmethod
    def _parse_explicit(values: object) -> list[MedicalCase]:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError("explicit corpus must contain a cases array")
        result: list[MedicalCase] = []
        for index, raw_case in enumerate(values):
            if not isinstance(raw_case, Mapping):
                raise ValueError(f"case at index {index} must be an object")
            case_id = str(raw_case.get("id") or raw_case.get("case_id") or f"case-{index + 1}")
            docs = tuple(_documents(raw_case.get("documents")))
            fixture = raw_case.get("fixture", {})
            if not isinstance(fixture, Mapping):
                raise ValueError(f"case {case_id!r} fixture must be an object")
            level = CorpusCatalog._explicit_level(raw_case)
            result.append(
                MedicalCase(
                    case_id=case_id,
                    specialty=str(raw_case.get("specialty", "general")),
                    topic=str(raw_case.get("topic", "clinical-summary")),
                    sensitivity_level=level,
                    canary_member=bool(
                        raw_case.get("canary_member", raw_case.get("membership", False))
                    ),
                    query=str(raw_case.get("query", "")),
                    documents=docs,
                    fixture=dict(fixture),
                )
            )
        return result

    @staticmethod
    def _framings(raw_topic: Mapping[str, Any], specialty: str, topic: str) -> tuple[str, ...]:
        """The topic's ordered sensitivity ladder.

        A v2 topic names its own ``framings`` list, so the ladder can be
        lengthened or shortened in data without touching the loader. A v1 topic
        names none, and falls back to that file's implicit routine/sensitive
        pair -- which is what keeps the archived corpus loadable, and its
        receipts verifiable, after the ladder grew.
        """

        raw = raw_topic.get("framings", raw_topic.get("levels"))
        if raw is None:
            return LEGACY_FRAMINGS
        if isinstance(raw, str) or not isinstance(raw, Sequence):
            raise ValueError(f"topic {specialty}/{topic} framings must be a JSON array")
        framings = tuple(str(value) for value in raw)
        if not framings:
            raise ValueError(f"topic {specialty}/{topic} lists no framings")
        if len(framings) != len(set(framings)):
            raise ValueError(f"topic {specialty}/{topic} repeats a framing name")
        if len(framings) > len(SENSITIVITY_FRAMINGS):
            raise ValueError(
                f"topic {specialty}/{topic} lists {len(framings)} framings; the ladder "
                f"defined in traceguard.types has {len(SENSITIVITY_FRAMINGS)} rungs"
            )
        return framings

    @staticmethod
    def _expand_topics(values: object) -> list[MedicalCase]:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError("topics must be a JSON array")
        result: list[MedicalCase] = []
        for topic_index, raw_topic in enumerate(values):
            if not isinstance(raw_topic, Mapping):
                raise ValueError(f"topic at index {topic_index} must be an object")
            specialty = str(raw_topic.get("specialty", "general"))
            topic = str(raw_topic.get("topic", f"topic-{topic_index + 1}"))
            canary_raw = raw_topic.get("canary_document")
            distractor_raw = raw_topic.get("distractor_document")
            if not isinstance(canary_raw, Mapping):
                raise ValueError(f"topic {specialty}/{topic} requires canary_document")
            if distractor_raw is not None and not isinstance(distractor_raw, Mapping):
                raise ValueError(f"topic {specialty}/{topic} distractor_document must be an object")
            canary = _as_document(canary_raw, 99)
            shared_distractor = _as_document(distractor_raw, 98) if distractor_raw else None
            fixture_hops = raw_topic.get("fixture_hops", {})
            if not isinstance(fixture_hops, Mapping):
                raise ValueError(f"topic {specialty}/{topic} fixture_hops must be an object")

            ladder = CorpusCatalog._framings(raw_topic, specialty, topic)
            for position, framing in enumerate(ladder):
                level = sensitivity_level_for_framing(framing, position=position)
                variant = raw_topic.get(framing)
                if not isinstance(variant, Mapping):
                    raise ValueError(f"topic {specialty}/{topic} is missing {framing} object")
                base_documents = _documents(variant.get("documents"))
                # The compact format accepts either one shared top-level distractor
                # or a framing-specific sixth document (the public dataset uses the
                # latter because its identifiers are topic/framing specific).
                if shared_distractor is not None:
                    distractor = shared_distractor
                elif len(base_documents) == 6:
                    distractor = base_documents[-1]
                else:
                    raise ValueError(
                        f"{specialty}/{topic}/{framing} needs a top-level "
                        "distractor_document when only five core documents are supplied"
                    )
                query = str(variant.get("query", ""))
                for member in (False, True):
                    documents = CorpusCatalog._membership_documents(
                        base_documents,
                        canary=canary,
                        distractor=distractor,
                        member=member,
                        context=f"{specialty}/{topic}/{framing}",
                    )
                    fixture = (
                        dict(variant.get("fixture", {}))
                        if isinstance(variant.get("fixture", {}), Mapping)
                        else {}
                    )
                    fixture["research_hops"] = int(
                        fixture.get(
                            "research_hops",
                            fixture_hops.get(framing, fixture_target_hops(level)),
                        )
                    )
                    case_id = "-".join(
                        (_slug(specialty), _slug(topic), framing, "canary" if member else "control")
                    )
                    result.append(
                        MedicalCase(
                            case_id=case_id,
                            specialty=specialty,
                            topic=topic,
                            sensitivity_level=level,
                            canary_member=member,
                            query=query,
                            documents=tuple(documents),
                            fixture=fixture,
                        )
                    )
        return result

    @staticmethod
    def _membership_documents(
        base: Sequence[MedicalDocument],
        *,
        canary: MedicalDocument,
        distractor: MedicalDocument,
        member: bool,
        context: str,
    ) -> list[MedicalDocument]:
        documents = list(base)
        target = canary if member else distractor
        if len(documents) == 5:
            documents.append(target)
        elif len(documents) == 6:
            replace_at = next(
                (
                    index
                    for index, document in enumerate(documents)
                    if document.document_id == distractor.document_id
                ),
                len(documents) - 1,
            )
            documents[replace_at] = target
        else:
            raise ValueError(
                f"{context} must provide five core documents or six documents with a "
                f"replaceable distractor; found {len(documents)}"
            )
        if len({document.document_id for document in documents}) != 6:
            raise ValueError(
                f"{context} produces duplicate document ids after membership replacement"
            )
        return documents

    def list_cases(self) -> list[MedicalCase]:
        return [self._cases[key] for key in sorted(self._cases)]

    def get_case(self, case_id: str) -> MedicalCase:
        try:
            return self._cases[case_id]
        except KeyError as exc:
            raise KeyError(f"unknown case_id {case_id!r}") from exc

    def public_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_hash": self.dataset_hash,
            "case_count": len(self._cases),
            "cases": [case.to_dict() for case in self.list_cases()],
        }

    def retrieve(
        self, case: MedicalCase, query: str, hop: int, *, limit: int = 1
    ) -> tuple[MedicalDocument, ...]:
        """Deterministic in-process lexical retrieval.

        No retrieval result or score is added to the host-observable trace.  A
        hop past the six-document corpus cycles through the ranked list, matching
        the paper's fixed-size corpus while allowing a seven-hop reasoning loop.
        """

        if limit < 1:
            raise ValueError("limit must be positive")
        query_terms = set(_TOKEN.findall(query.lower()))

        def score(document: MedicalDocument) -> tuple[int, str]:
            text_terms = set(_TOKEN.findall(f"{document.title} {document.text}".lower()))
            return (len(query_terms & text_terms), document.document_id)

        ranked = sorted(case.documents, key=score, reverse=True)
        start = max(0, hop - 1) % len(ranked)
        return tuple(
            ranked[(start + offset) % len(ranked)] for offset in range(min(limit, len(ranked)))
        )


@lru_cache(maxsize=1)
def default_case_count() -> int:
    """How many cases the packaged corpus expands to.

    The HTTP request models use this as their ``case_limit`` ceiling.  That
    ceiling was the literal ``24`` in three separate files, which became wrong
    the moment the corpus grew a four-rung ladder: a request for the whole
    corpus would have been rejected as out of range by validation that had no
    idea how large the corpus actually is.  Reading it from the corpus is the
    only version of the bound that cannot drift from the data.
    """

    return len(CorpusCatalog.load_default().list_cases())

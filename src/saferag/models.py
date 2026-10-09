"""Core data types shared across the pipeline.

Everything here is immutable. Access-control attributes (tenant, allowed roles)
travel with every chunk so that any stage can re-check them independently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

_TENANT_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")
_ROLE_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


class ScopeError(ValueError):
    """Raised when a request scope is malformed."""


class ScopeViolation(RuntimeError):
    """Raised when data outside the caller's scope reaches a later stage.

    This should never happen if the store pre-filters correctly. It exists so a
    bug in one layer fails closed instead of silently leaking data.
    """


class Trust(StrEnum):
    """Provenance of a document. Third-party content is treated as less trusted."""

    AUTHORITATIVE = "authoritative"
    THIRD_PARTY = "third_party"


@dataclass(frozen=True)
class Scope:
    """Who is asking. Every retrieval call requires one; there is no default."""

    tenant_id: str
    roles: frozenset[str]

    def __post_init__(self) -> None:
        if not _TENANT_RE.match(self.tenant_id):
            raise ScopeError("tenant_id must be a lowercase slug")
        if not self.roles:
            raise ScopeError("scope must carry at least one role")
        for role in self.roles:
            if not _ROLE_RE.match(role):
                raise ScopeError("role names must be lowercase slugs")

    def permits(self, tenant_id: str, allowed_roles: frozenset[str]) -> bool:
        return tenant_id == self.tenant_id and bool(self.roles & allowed_roles)


@dataclass(frozen=True)
class Document:
    doc_id: str
    tenant_id: str
    title: str
    version: str
    allowed_roles: frozenset[str]
    trust: Trust
    body: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    tenant_id: str
    allowed_roles: frozenset[str]
    trust: Trust
    title: str
    version: str
    section: str
    section_slug: str
    ordinal: int
    text: str
    content_hash: str

    @property
    def section_key(self) -> str:
        """Stable key used by evaluation labels: ``doc_id#section-slug``."""
        return f"{self.doc_id}#{self.section_slug}"

    def embedding_text(self) -> str:
        return f"{self.title}. {self.section}. {self.text}"


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    signals: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    version: str


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    REFUSED = "refused"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class RetrievalTrace:
    """Identifiers only. Never contains document text or the user's question."""

    retrieved_ids: tuple[str, ...]
    context_ids: tuple[str, ...]
    quarantined_ids: tuple[str, ...]
    top_score: float


@dataclass(frozen=True)
class AnswerResult:
    request_id: str
    status: AnswerStatus
    answer: str
    citations: tuple[Citation, ...]
    reason: str | None
    trace: RetrievalTrace

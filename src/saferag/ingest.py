"""Document loading, validation and heading-aware chunking.

Documents are Markdown files with a small front-matter header::

    ---
    doc_id: larkspur-expense-policy
    tenant_id: larkspur
    title: Expense Policy
    version: 3.1
    allowed_roles: employee
    trust: authoritative
    ---

Access-control metadata is required and validated at ingestion time. A document
without a tenant or roles is rejected rather than defaulting to "visible to all".
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from saferag.config import Settings
from saferag.models import Chunk, Document, Scope, ScopeError, Trust
from saferag.text import approx_tokens, normalize, sentences, slugify

REQUIRED_FIELDS = ("doc_id", "tenant_id", "title", "version", "allowed_roles", "trust")
_DOC_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,95}$")
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
MAX_DOCUMENT_BYTES = 512_000


class IngestionError(ValueError):
    pass


def parse_document(raw: str, *, source: str = "<memory>") -> Document:
    if len(raw.encode("utf-8")) > MAX_DOCUMENT_BYTES:
        raise IngestionError(f"{source}: document exceeds {MAX_DOCUMENT_BYTES} bytes")
    if not raw.startswith("---\n"):
        raise IngestionError(f"{source}: missing front matter")
    try:
        header, body = raw[4:].split("\n---\n", 1)
    except ValueError as exc:
        raise IngestionError(f"{source}: unterminated front matter") from exc

    meta: dict[str, str] = {}
    for line in header.splitlines():
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise IngestionError(f"{source}: malformed front matter line")
        meta[key.strip()] = value.strip()

    missing = [f for f in REQUIRED_FIELDS if not meta.get(f)]
    if missing:
        raise IngestionError(f"{source}: missing required fields {missing}")
    if not _DOC_ID_RE.match(meta["doc_id"]):
        raise IngestionError(f"{source}: invalid doc_id")

    roles = frozenset(r.strip() for r in meta["allowed_roles"].split(",") if r.strip())
    try:
        Scope(meta["tenant_id"], roles)  # reuse scope validation for tenant + roles
        trust = Trust(meta["trust"])
    except (ScopeError, ValueError) as exc:
        raise IngestionError(f"{source}: {exc}") from exc

    return Document(
        doc_id=meta["doc_id"],
        tenant_id=meta["tenant_id"],
        title=meta["title"],
        version=meta["version"],
        allowed_roles=roles,
        trust=trust,
        body=body.strip(),
    )


def load_corpus(root: Path) -> list[Document]:
    if not root.is_dir():
        raise IngestionError(f"corpus directory not found: {root} (pass --corpus)")
    docs = [parse_document(p.read_text("utf-8"), source=str(p)) for p in sorted(root.rglob("*.md"))]
    seen: set[tuple[str, str]] = set()
    for doc in docs:
        key = (doc.tenant_id, doc.doc_id)
        if key in seen:
            raise IngestionError(f"duplicate doc_id {doc.doc_id} in tenant {doc.tenant_id}")
        seen.add(key)
    return docs


def _sections(body: str) -> Iterator[tuple[str, str]]:
    """Yield (heading, text) pairs. Text before the first heading is 'Overview'."""
    heading = "Overview"
    buf: list[str] = []
    for line in body.splitlines():
        match = _HEADING_RE.match(line)
        if match and len(match.group(1)) >= 2:
            if buf:
                yield heading, "\n".join(buf)
            heading, buf = match.group(2), []
        elif match:
            continue  # the H1 title duplicates front matter
        else:
            buf.append(line)
    if buf:
        yield heading, "\n".join(buf)


def _pack(sents: list[str], max_tokens: int, overlap: int) -> Iterator[str]:
    """Greedy sentence packing with a small sentence overlap between chunks."""
    current: list[str] = []
    for sent in sents:
        if current and approx_tokens(" ".join([*current, sent])) > max_tokens:
            yield " ".join(current)
            current = current[-overlap:] if overlap else []
        current.append(sent)
    if current:
        yield " ".join(current)


def chunk_document(doc: Document, settings: Settings) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading, text in _sections(doc.body):
        paragraphs = [normalize(p) for p in re.split(r"\n\s*\n", text) if p.strip()]
        sents = [s for p in paragraphs for s in sentences(p)]
        if not sents:
            continue
        slug = slugify(heading)
        for piece in _pack(sents, settings.chunk_max_tokens, settings.chunk_overlap_sentences):
            ordinal = len(chunks)
            digest = hashlib.sha256(piece.encode("utf-8")).hexdigest()
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{slug}-{ordinal}",
                    doc_id=doc.doc_id,
                    tenant_id=doc.tenant_id,
                    allowed_roles=doc.allowed_roles,
                    trust=doc.trust,
                    title=doc.title,
                    version=doc.version,
                    section=heading,
                    section_slug=slug,
                    ordinal=ordinal,
                    text=piece,
                    content_hash=digest,
                )
            )
    return chunks


def chunk_corpus(docs: Iterable[Document], settings: Settings) -> list[Chunk]:
    return [c for d in docs for c in chunk_document(d, settings)]

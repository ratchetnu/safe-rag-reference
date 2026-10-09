"""End-to-end question answering with every guardrail in the path.

    question -> input checks -> scoped hybrid retrieval -> relevance gate
             -> quarantine + fenced, budgeted context -> provider (with retry/breaker)
             -> output validation -> cited answer | refusal | unavailable

Every exit path returns an ``AnswerResult`` and writes one audit record. Errors
fail closed: when in doubt the user gets a refusal, not an unchecked answer.
"""

from __future__ import annotations

import logging
import time
import uuid

from saferag.audit import AuditLogger, fingerprint
from saferag.config import Settings
from saferag.guardrails.context import BuiltContext, ContextBuilder
from saferag.guardrails.injection import InjectionDetector
from saferag.guardrails.output import OutputValidator
from saferag.models import (
    AnswerResult,
    AnswerStatus,
    Citation,
    RetrievalTrace,
    Scope,
    ScopeViolation,
    ScoredChunk,
    Trust,
)
from saferag.providers.base import (
    GenerationRequest,
    LLMProvider,
    ProviderError,
    ProviderPermanentError,
)
from saferag.retrieval import HybridRetriever, assert_in_scope
from saferag.store.base import EmbeddingModelMismatch
from saferag.text import normalize

MESSAGES = {
    "not_in_context": "I couldn't find this in the approved documents available to you.",
    "no_relevant_documents": "I couldn't find this in the approved documents available to you.",
    "no_usable_documents": "I couldn't find usable approved documents for this question.",
    "request_blocked": "I can't help with that request.",
    "empty_question": "Please enter a question.",
    "question_too_long": "That question is too long. Please shorten it.",
    "validation_failed": "I couldn't produce an answer that is supported by the documents.",
    "provider_unavailable": (
        "The assistant is temporarily unavailable. The documents listed may help."
    ),
    "provider_error": "The assistant could not answer right now. The documents listed may help.",
    "index_incompatible": "Search is temporarily unavailable while the index is rebuilt.",
    "internal_error": "Something went wrong. No answer was produced.",
}
_MODEL_REFUSAL_REASONS = {"not_in_context"}


class RagPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        provider: LLMProvider,
        settings: Settings,
        *,
        detector: InjectionDetector | None = None,
        audit: AuditLogger | None = None,
    ) -> None:
        self._retriever = retriever
        self._provider = provider
        self._s = settings
        self._detector = detector or InjectionDetector()
        self._builder = ContextBuilder(settings, self._detector)
        self._validator = OutputValidator(settings)
        self._audit = audit or AuditLogger()

    @property
    def provider_name(self) -> str:
        return self._provider.name

    def answer(self, scope: Scope, question: str, *, request_id: str | None = None) -> AnswerResult:
        rid = request_id or uuid.uuid4().hex
        started = time.perf_counter()
        q = normalize(question)
        self._audit.emit(
            "request_received",
            request_id=rid,
            tenant_id=scope.tenant_id,
            role_count=len(scope.roles),
            question_fp=fingerprint(q),
            question_chars=len(q),
        )

        ranked: list[ScoredChunk] = []
        ctx: BuiltContext | None = None

        def finish(
            status: AnswerStatus,
            reason: str | None,
            answer: str = "",
            citations: tuple[Citation, ...] = (),
            violations: tuple[str, ...] = (),
        ) -> AnswerResult:
            trace = RetrievalTrace(
                retrieved_ids=tuple(s.chunk.chunk_id for s in ranked),
                context_ids=ctx.ids if ctx else (),
                quarantined_ids=ctx.quarantined_ids if ctx else (),
                top_score=ranked[0].score if ranked else 0.0,
            )
            self._audit.emit(
                "request_completed",
                level=logging.WARNING if status is AnswerStatus.UNAVAILABLE else logging.INFO,
                request_id=rid,
                tenant_id=scope.tenant_id,
                status=status.value,
                reason=reason or "none",
                provider=self._provider.name,
                retrieved_ids=trace.retrieved_ids,
                context_ids=trace.context_ids,
                quarantined_ids=trace.quarantined_ids,
                citation_ids=[c.chunk_id for c in citations],
                violations=violations,
                top_score=trace.top_score,
                context_tokens=ctx.estimated_tokens if ctx else 0,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            text = answer if status is AnswerStatus.ANSWERED else MESSAGES.get(reason or "", "")
            return AnswerResult(rid, status, text, citations, reason, trace)

        if not q:
            return finish(AnswerStatus.REFUSED, "empty_question")
        if len(q) > self._s.max_question_chars:
            return finish(AnswerStatus.REFUSED, "question_too_long")

        screen = self._detector.scan(q)
        if screen.score >= self._s.block_question_threshold:
            self._audit.emit(
                "question_blocked",
                level=logging.WARNING,
                request_id=rid,
                tenant_id=scope.tenant_id,
                risk=screen.risk.value,
                injection_score=screen.score,
                signals=screen.signals,
            )
            return finish(AnswerStatus.REFUSED, "request_blocked")

        try:
            ranked = self._retriever.retrieve(scope, q)
            assert_in_scope(scope, ranked)
        except EmbeddingModelMismatch:
            return finish(AnswerStatus.UNAVAILABLE, "index_incompatible")
        except ScopeViolation:
            ranked = []
            self._audit.emit(
                "scope_violation", level=logging.CRITICAL, request_id=rid, tenant_id=scope.tenant_id
            )
            return finish(AnswerStatus.REFUSED, "internal_error")

        # Relevance gate: weak evidence means "don't know", and skips the model call.
        if not ranked or ranked[0].score < self._s.min_relevance:
            return finish(AnswerStatus.REFUSED, "no_relevant_documents")

        ctx = self._builder.build(q, ranked)
        if not ctx.items:
            return finish(AnswerStatus.REFUSED, "no_usable_documents")
        by_id = {s.chunk.chunk_id: s.chunk for s in ranked}
        context_sources = tuple(
            Citation(
                i.chunk_id, by_id[i.chunk_id].doc_id, i.title, i.section, by_id[i.chunk_id].version
            )
            for i in ctx.items
        )

        request = GenerationRequest(
            system=ctx.system,
            user=ctx.user,
            question=q,
            context=ctx.items,
            max_output_tokens=self._s.max_output_tokens,
        )
        # Degraded mode: if the model is down, point to authoritative sources only.
        fallback_sources = tuple(
            c
            for c, i in zip(context_sources, ctx.items, strict=True)
            if i.trust is Trust.AUTHORITATIVE
        )
        try:
            raw = self._provider.generate(request)
        except ProviderPermanentError:
            return finish(AnswerStatus.UNAVAILABLE, "provider_error", citations=fallback_sources)
        except ProviderError:
            return finish(
                AnswerStatus.UNAVAILABLE, "provider_unavailable", citations=fallback_sources
            )

        outcome = self._validator.validate(raw, ctx, q)
        if not outcome.ok or outcome.output is None:
            return finish(AnswerStatus.REFUSED, "validation_failed", violations=outcome.violations)
        out = outcome.output
        if out.status == "refused":
            reason = (
                out.refusal_reason
                if out.refusal_reason in _MODEL_REFUSAL_REASONS
                else "not_in_context"
            )
            return finish(AnswerStatus.REFUSED, reason)

        sources = {c.chunk_id: c for c in context_sources}
        citations = tuple(sources[c] for c in dict.fromkeys(out.citations))
        return finish(AnswerStatus.ANSWERED, None, answer=out.answer, citations=citations)

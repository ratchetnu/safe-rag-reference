"""Tunable limits. Defaults are conservative; every limit is explicit."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Settings:
    # Chunking
    chunk_max_tokens: int = 180
    chunk_overlap_sentences: int = 1

    # Retrieval
    vector_k: int = 20
    keyword_k: int = 20
    rrf_k: int = 60
    fused_k: int = 12
    final_k: int = 5
    max_chunks_per_doc: int = 3
    near_duplicate_jaccard: float = 0.85
    min_relevance: float = 0.30

    # Context and generation
    max_question_chars: int = 1_000
    max_context_tokens: int = 1_800
    max_context_chunks: int = 5
    max_output_tokens: int = 600
    max_answer_chars: int = 1_500

    # Guardrails
    quarantine_threshold: float = 0.6
    block_question_threshold: float = 0.6
    min_sentence_support: float = 0.6

    # Provider resilience
    provider_timeout_s: float = 30.0
    provider_max_retries: int = 2
    breaker_failure_threshold: int = 3
    breaker_cooldown_s: float = 60.0

    @classmethod
    def from_env(cls, prefix: str = "SAFERAG_") -> Settings:
        """Override any field with ``SAFERAG_<FIELD_NAME>`` environment variables."""
        base = cls()
        overrides: dict[str, object] = {}
        for name, value in vars(base).items():
            raw = os.environ.get(prefix + name.upper())
            if raw is None:
                continue
            overrides[name] = type(value)(raw)
        return replace(base, **overrides)  # type: ignore[arg-type]

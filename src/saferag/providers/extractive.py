"""Deterministic offline "model" used for tests and the offline evaluation.

It selects the best-supported sentences from the supplied context and returns
them verbatim with citations, or refuses when nothing in the context covers the
question well enough. It cannot be talked into anything because it does not
interpret instructions at all, which also means it is NOT a meaningful test of
how a real LLM reacts to prompt injection. The adversarial providers in
``saferag.providers.adversarial`` exist to exercise the output validator with
the kinds of responses a compromised or confused model might produce.
"""

from __future__ import annotations

import json

from saferag.embeddings import query_coverage
from saferag.providers.base import GenerationRequest
from saferag.text import sentences


class ExtractiveProvider:
    def __init__(self, *, min_coverage: float = 0.6, max_sentences: int = 2) -> None:
        self._min_coverage = min_coverage
        self._max_sentences = max_sentences

    @property
    def name(self) -> str:
        return "offline-extractive"

    def generate(self, request: GenerationRequest) -> str:
        candidates: list[tuple[float, int, str, str]] = []
        for pos, item in enumerate(request.context):
            for sent in sentences(item.text):
                # The section heading is part of what a sentence "means" in context.
                coverage = query_coverage(request.question, f"{item.section} {sent}")
                candidates.append((coverage, -pos, sent, item.chunk_id))

        candidates.sort(key=lambda c: (-c[0], -c[1]))
        if not candidates or candidates[0][0] < self._min_coverage:
            return json.dumps(
                {
                    "status": "refused",
                    "answer": "",
                    "citations": [],
                    "refusal_reason": "not_in_context",
                }
            )

        best = candidates[0][0]
        chosen = [c for c in candidates if c[0] >= 0.9 * best][: self._max_sentences]
        citations = list(dict.fromkeys(c[3] for c in chosen))
        return json.dumps(
            {
                "status": "answered",
                "answer": " ".join(c[2] for c in chosen),
                "citations": citations,
                "refusal_reason": None,
            }
        )

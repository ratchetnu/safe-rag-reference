"""Vendor-neutral provider interface.

The pipeline depends only on ``LLMProvider``: take a request, return text.
Vendor SDKs live in individual adapters behind ``providers/registry.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from saferag.guardrails.context import ContextItem


@dataclass(frozen=True)
class GenerationRequest:
    system: str
    user: str
    question: str
    context: tuple[ContextItem, ...]
    max_output_tokens: int


class ProviderError(Exception):
    """Base class. Messages must not include prompt or document text."""


class ProviderTransientError(ProviderError):
    """Timeouts, rate limits, overload, 5xx. Safe to retry with backoff."""


class ProviderPermanentError(ProviderError):
    """Bad request, auth, model refusal. Retrying will not help."""


class ProviderUnavailable(ProviderError):
    """Circuit breaker is open; the provider is not being called."""


class LLMProvider(Protocol):
    @property
    def name(self) -> str: ...

    def generate(self, request: GenerationRequest) -> str:
        """Return the raw model text. Validation happens downstream."""
        ...

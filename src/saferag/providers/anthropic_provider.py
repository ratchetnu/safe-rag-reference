"""Example adapter for one hosted model API (Anthropic Python SDK).

This is an illustration of how a real model plugs into the vendor-neutral
``LLMProvider`` interface. Nothing else in the codebase imports it; it is
reached only through ``providers/registry.py`` when live mode is enabled.

Status: exercised only by offline unit tests against a fake client. It has not
been used to produce any published result, and this repository has not been
deployed. Install with ``pip install 'safe-rag-reference[anthropic]'`` and set
``ANTHROPIC_API_KEY`` plus ``SAFERAG_ENABLE_LIVE=1`` to try it. Never used by the
default test suite or required CI.

Vendor-specific choices are kept inside this file:

* Structured outputs constrain the response to the JSON shape the validator
  expects (the validator still re-checks everything).
* SDK-level retries are disabled; ``ResilientProvider`` owns retry/backoff and
  the circuit breaker so retry policy lives in one place.
* Server-side refusal fallback is enabled so a safety-classifier decline on the
  primary model is retried on a fallback model inside the same call.
"""

from __future__ import annotations

import os
from typing import Any

from saferag.providers.base import (
    GenerationRequest,
    ProviderPermanentError,
    ProviderTransientError,
)

# Override with SAFERAG_LIVE_MODEL; pick the model your own evaluation supports.
DEFAULT_MODEL = "claude-opus-5-5"

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["answered", "refused"]},
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "refusal_reason": {"type": ["string", "null"]},
    },
    "required": ["status", "answer", "citations", "refusal_reason"],
    "additionalProperties": False,
}


class AnthropicProvider:
    def __init__(
        self,
        *,
        model: str | None = None,
        timeout_s: float = 30.0,
        effort: str | None = None,
        client: Any = None,
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic(timeout=timeout_s, max_retries=0)
        self._client = client
        self._model = model or os.environ.get("SAFERAG_LIVE_MODEL", DEFAULT_MODEL)
        self._effort = effort

    @property
    def name(self) -> str:
        return f"anthropic:{self._model}"

    def generate(self, request: GenerationRequest) -> str:
        import anthropic

        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": ANSWER_SCHEMA}}
        if self._effort:
            output_config["effort"] = self._effort
        try:
            response = self._client.beta.messages.create(
                model=self._model,
                max_tokens=request.max_output_tokens,
                system=request.system,
                messages=[{"role": "user", "content": request.user}],
                output_config=output_config,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except (
            anthropic.RateLimitError,
            anthropic.APITimeoutError,
            anthropic.APIConnectionError,
            anthropic.InternalServerError,
        ) as exc:
            raise ProviderTransientError(type(exc).__name__) from exc
        except anthropic.APIStatusError as exc:
            raise ProviderPermanentError(f"{type(exc).__name__} status={exc.status_code}") from exc

        if response.stop_reason == "refusal":
            raise ProviderPermanentError("model_refusal")
        if response.stop_reason == "max_tokens":
            raise ProviderPermanentError("output_truncated")
        texts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        if not texts:
            raise ProviderPermanentError("no_text_block")
        return str(texts[0])

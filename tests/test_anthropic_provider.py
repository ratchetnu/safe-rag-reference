"""Offline tests for the live provider adapter, using a fake client. No network."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

anthropic = pytest.importorskip("anthropic")
# The 1.x SDK is built on httpx2; its exception types need httpx2 request objects.
httpx = pytest.importorskip("httpx2")

from saferag.guardrails.context import ContextItem  # noqa: E402
from saferag.providers.anthropic_provider import AnthropicProvider  # noqa: E402
from saferag.providers.base import (  # noqa: E402
    GenerationRequest,
    ProviderPermanentError,
    ProviderTransientError,
)

REQ = GenerationRequest("system text", "user text", "q", tuple[ContextItem, ...](), 300)


class FakeMessages:
    def __init__(self, response: Any = None, exc: Exception | None = None) -> None:
        self.response = response
        self.exc = exc
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if self.exc:
            raise self.exc
        return self.response


def _client(messages: FakeMessages) -> Any:
    return SimpleNamespace(beta=SimpleNamespace(messages=messages))


def _response(stop: str = "end_turn", text: str = '{"status":"refused"}') -> Any:
    return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="text", text=text)])


def test_request_shape() -> None:
    msgs = FakeMessages(_response())
    provider = AnthropicProvider(client=_client(msgs), model="claude-opus-5-5")
    assert provider.generate(REQ) == '{"status":"refused"}'
    kw = msgs.kwargs
    assert kw["model"] == "claude-opus-5-5"
    assert kw["system"] == "system text"
    assert kw["messages"] == [{"role": "user", "content": "user text"}]
    assert kw["max_tokens"] == 300
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert kw["fallbacks"] == "default"


@pytest.mark.parametrize(
    ("stop", "reason"), [("refusal", "model_refusal"), ("max_tokens", "output_truncated")]
)
def test_stop_reasons_map_to_permanent_errors(stop: str, reason: str) -> None:
    provider = AnthropicProvider(client=_client(FakeMessages(_response(stop))))
    with pytest.raises(ProviderPermanentError, match=reason):
        provider.generate(REQ)


def test_connection_errors_are_transient() -> None:
    exc = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.invalid"))
    provider = AnthropicProvider(client=_client(FakeMessages(exc=exc)))
    with pytest.raises(ProviderTransientError):
        provider.generate(REQ)


def test_bad_request_is_permanent() -> None:
    req = httpx.Request("POST", "https://api.invalid")
    exc = anthropic.BadRequestError("bad", response=httpx.Response(400, request=req), body=None)
    provider = AnthropicProvider(client=_client(FakeMessages(exc=exc)))
    with pytest.raises(ProviderPermanentError):
        provider.generate(REQ)

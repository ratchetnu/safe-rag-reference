from __future__ import annotations

import pytest

from saferag.guardrails.context import ContextItem
from saferag.providers.base import (
    GenerationRequest,
    ProviderTransientError,
    ProviderUnavailable,
)
from saferag.providers.resilience import CircuitBreaker, ResilientProvider

REQ = GenerationRequest("s", "u", "q", tuple[ContextItem, ...](), 10)


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class AlwaysFails:
    calls = 0

    @property
    def name(self) -> str:
        return "fails"

    def generate(self, request: GenerationRequest) -> str:
        self.calls += 1
        raise ProviderTransientError("x")


def test_backoff_is_exponential_and_bounded() -> None:
    sleeps: list[float] = []
    p = ResilientProvider(
        AlwaysFails(),
        max_retries=4,
        base_delay_s=1,
        max_delay_s=5,
        breaker=CircuitBreaker(failure_threshold=99, cooldown_s=1),
        sleep=sleeps.append,
        jitter=lambda: 1.0,
    )
    with pytest.raises(ProviderTransientError):
        p.generate(REQ)
    assert sleeps == [1, 2, 4, 5]


def test_breaker_open_half_open_closed() -> None:
    clock = Clock()
    breaker = CircuitBreaker(failure_threshold=2, cooldown_s=30, clock=clock)
    inner = AlwaysFails()
    p = ResilientProvider(inner, max_retries=0, breaker=breaker, sleep=lambda _s: None)
    for _ in range(2):
        with pytest.raises(ProviderTransientError):
            p.generate(REQ)
    assert breaker.state == "open"
    with pytest.raises(ProviderUnavailable):
        p.generate(REQ)
    assert inner.calls == 2

    clock.t = 31
    assert breaker.state == "half_open"
    with pytest.raises(ProviderTransientError):  # one trial call allowed, fails, re-opens
        p.generate(REQ)
    assert breaker.state == "open"

    clock.t = 62
    breaker.record_success()
    assert breaker.state == "closed"

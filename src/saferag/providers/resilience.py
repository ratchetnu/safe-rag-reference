"""Retry with exponential backoff and a circuit breaker around any provider.

* Only transient errors are retried, a bounded number of times, with jitter.
* After N consecutive failed calls the breaker opens and calls fail fast for a
  cooldown period, instead of piling latency and cost onto an outage. One trial
  call is allowed after the cooldown (half-open).
* Clock, sleep and jitter are injectable so behaviour is testable without waiting.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable

from saferag.providers.base import (
    GenerationRequest,
    LLMProvider,
    ProviderTransientError,
    ProviderUnavailable,
)


class CircuitBreaker:
    def __init__(
        self,
        *,
        failure_threshold: int,
        cooldown_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = failure_threshold
        self._cooldown = cooldown_s
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if self._clock() - self._opened_at >= self._cooldown:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        return self.state != "open"

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold or self.state == "half_open":
            self._opened_at = self._clock()


class ResilientProvider:
    def __init__(
        self,
        inner: LLMProvider,
        *,
        max_retries: int,
        breaker: CircuitBreaker,
        base_delay_s: float = 0.5,
        max_delay_s: float = 8.0,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self._inner = inner
        self._max_retries = max_retries
        self._breaker = breaker
        self._base = base_delay_s
        self._max = max_delay_s
        self._sleep = sleep
        self._jitter = jitter

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def breaker(self) -> CircuitBreaker:
        return self._breaker

    def generate(self, request: GenerationRequest) -> str:
        if not self._breaker.allow():
            raise ProviderUnavailable("circuit_open")
        attempt = 0
        while True:
            try:
                result = self._inner.generate(request)
            except ProviderTransientError:
                if attempt >= self._max_retries:
                    self._breaker.record_failure()
                    raise
                delay = min(self._max, self._base * 2**attempt) * (0.5 + self._jitter() / 2)
                self._sleep(delay)
                attempt += 1
                continue
            except Exception:
                # Permanent errors don't trip the breaker on their own: a bad
                # request says nothing about provider health.
                raise
            self._breaker.record_success()
            return result

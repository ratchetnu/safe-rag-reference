"""Live provider registry.

The pipeline depends only on the ``LLMProvider`` protocol. Live providers are
listed here by name and imported lazily, so their SDKs are optional extras and
nothing else in the codebase depends on a particular vendor. To add one, write
an adapter that implements ``LLMProvider`` and add a line below.

Every live provider is opt-in: it needs ``SAFERAG_ENABLE_LIVE=1`` in addition
to that provider's own credentials.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable

from saferag.providers.base import LLMProvider

OFFLINE = "offline"

# name -> "module:attribute" of a zero-argument factory returning an LLMProvider
LIVE_PROVIDERS: dict[str, str] = {
    "anthropic": "saferag.providers.anthropic_provider:AnthropicProvider",
}


class LiveProviderNotEnabled(RuntimeError):
    pass


def provider_names() -> list[str]:
    return [OFFLINE, *LIVE_PROVIDERS]


def live_provider_factory(name: str) -> Callable[[], LLMProvider]:
    if name not in LIVE_PROVIDERS:
        raise KeyError(f"unknown provider {name!r}; choose from {provider_names()}")
    if os.environ.get("SAFERAG_ENABLE_LIVE") != "1":
        raise LiveProviderNotEnabled(
            "Live providers are opt-in: set SAFERAG_ENABLE_LIVE=1 plus the provider's own "
            "credentials. Live runs cost money and are not deterministic."
        )
    module_name, attr = LIVE_PROVIDERS[name].split(":")
    factory: Callable[[], LLMProvider] = getattr(importlib.import_module(module_name), attr)
    return factory

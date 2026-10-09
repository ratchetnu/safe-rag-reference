"""Structured audit logging that never records sensitive content.

Rules enforced here rather than by convention:

* Only allow-listed field names are emitted; anything else raises in tests and
  is dropped in production.
* Free-text fields are not allowed. The question is recorded as a keyed hash
  and a length so repeated questions can be correlated without storing them.
* Document text and model output are never logged; chunk ids are.

An auditor can reconstruct *which* documents informed *which* answer for *which*
tenant from ids alone, then fetch the content through access-controlled paths.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from collections.abc import Iterable

LOGGER_NAME = "saferag.audit"

_ID_FIELDS = frozenset(
    {
        "request_id",
        "tenant_id",
        "event",
        "status",
        "reason",
        "provider",
        "embedding_model",
        "question_fp",
        "risk",
    }
)
_NUMERIC_FIELDS = frozenset(
    {
        "question_chars",
        "role_count",
        "retrieved",
        "context_chunks",
        "quarantined",
        "top_score",
        "latency_ms",
        "context_tokens",
        "injection_score",
        "citations_count",
    }
)
_LIST_FIELDS = frozenset(
    {"retrieved_ids", "context_ids", "quarantined_ids", "citation_ids", "violations", "signals"}
)
ALLOWED_FIELDS = _ID_FIELDS | _NUMERIC_FIELDS | _LIST_FIELDS
_MAX_ID_LEN = 128


class UnsafeLogField(ValueError):
    pass


def _fingerprint_key() -> bytes:
    # A per-deployment secret prevents dictionary attacks on short questions.
    return os.environ.get("SAFERAG_LOG_HMAC_KEY", "dev-only-not-secret").encode("utf-8")


def fingerprint(text: str) -> str:
    return hmac.new(_fingerprint_key(), text.encode("utf-8"), hashlib.sha256).hexdigest()[:16]


def _clean_id(value: object) -> str:
    text = str(value)
    if len(text) > _MAX_ID_LEN or any(ch.isspace() for ch in text):
        raise UnsafeLogField("identifier fields must be short and contain no whitespace")
    return text


class AuditLogger:
    def __init__(self, logger: logging.Logger | None = None, *, strict: bool = True) -> None:
        self._log = logger or logging.getLogger(LOGGER_NAME)
        self._strict = strict

    def emit(self, event: str, *, level: int = logging.INFO, **fields: object) -> None:
        record: dict[str, object] = {"event": event}
        for key, value in fields.items():
            if key not in ALLOWED_FIELDS:
                if self._strict:
                    raise UnsafeLogField(f"field {key!r} is not on the audit allow-list")
                continue
            if key in _NUMERIC_FIELDS:
                record[key] = round(float(value), 4) if isinstance(value, float) else value
            elif key in _LIST_FIELDS:
                assert isinstance(value, Iterable)  # noqa: S101
                record[key] = [_clean_id(v) for v in value]
            else:
                record[key] = _clean_id(value)
        self._log.log(level, json.dumps(record, sort_keys=True))

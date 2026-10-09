from __future__ import annotations

import json
import logging

import pytest

from saferag.audit import AuditLogger, UnsafeLogField, fingerprint
from saferag.indexing import build_pipeline
from saferag.store.memory import InMemoryStore

from .conftest import LARKSPUR_EMPLOYEE

SECRET_QUESTION = "What is the daily meal limit for domestic travel for Jordan Example?"


def test_logs_ids_not_content(store: InMemoryStore, caplog: pytest.LogCaptureFixture) -> None:
    logging.getLogger("saferag.audit").setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="saferag.audit")
    result = build_pipeline(store=store).answer(LARKSPUR_EMPLOYEE, SECRET_QUESTION)
    assert result.answer  # an answer containing policy text was produced...
    text = "\n".join(r.getMessage() for r in caplog.records)
    # ...but neither the question, nor the answer, nor document text is in the logs.
    assert "Jordan" not in text
    assert "meal limit" not in text
    assert "$75" not in text
    records = [json.loads(r.getMessage()) for r in caplog.records]
    done = next(r for r in records if r["event"] == "request_completed")
    assert done["citation_ids"] == [c.chunk_id for c in result.citations]
    assert done["tenant_id"] == "larkspur"
    received = next(r for r in records if r["event"] == "request_received")
    assert received["question_fp"] == fingerprint(SECRET_QUESTION)


def test_unknown_fields_rejected() -> None:
    with pytest.raises(UnsafeLogField):
        AuditLogger().emit("x", question="raw text")


def test_free_text_rejected_in_id_fields() -> None:
    with pytest.raises(UnsafeLogField):
        AuditLogger().emit("x", reason="this has spaces in it")


def test_non_strict_mode_drops_instead_of_raising(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("saferag.audit.test")
    logger.setLevel(logging.INFO)
    caplog.set_level(logging.INFO, logger="saferag.audit.test")
    AuditLogger(logger, strict=False).emit("x", question="raw", status="ok")
    assert json.loads(caplog.records[-1].getMessage()) == {"event": "x", "status": "ok"}


def test_fingerprint_is_keyed(monkeypatch: pytest.MonkeyPatch) -> None:
    a = fingerprint("same")
    monkeypatch.setenv("SAFERAG_LOG_HMAC_KEY", "another-key")
    assert fingerprint("same") != a

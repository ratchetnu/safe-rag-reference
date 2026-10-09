"""Heuristic prompt-injection screening.

This is ONE layer of a layered defence, not a guarantee. Pattern matching
catches common, low-effort attacks (and is cheap enough to run on every chunk),
but paraphrased, encoded, multilingual or novel attacks will get through. The
other layers -- scope isolation, context fencing, a model instructed to treat
documents as data, output validation and least-privilege (the model has no
tools and cannot reach other tenants' data) -- are what bound the damage when
this screen misses.

Each signal has a weight; the risk score is the capped sum. Callers decide what
to do at a threshold (quarantine a chunk, block a question, flag for review).
"""

from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass
from enum import StrEnum

from saferag.text import normalize


class Risk(StrEnum):
    NONE = "none"
    SUSPICIOUS = "suspicious"
    HIGH = "high"


@dataclass(frozen=True)
class Signal:
    name: str
    pattern: re.Pattern[str]
    weight: float


_I = re.IGNORECASE
SIGNALS: tuple[Signal, ...] = (
    Signal(
        "override_instructions",
        re.compile(
            r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}"
            r"\b(previous|prior|above|earlier|all|any|your|system|the)\b.{0,25}"
            r"\b(instructions?|rules?|guidelines?|prompts?|polic(y|ies))\b",
            _I,
        ),
        0.7,
    ),
    Signal(
        "role_reassignment",
        re.compile(r"\b(you are now|from now on you|act as|pretend to be|new persona)\b", _I),
        0.4,
    ),
    Signal(
        "prompt_extraction",
        re.compile(
            r"\b(reveal|print|show|repeat|output|leak)\b.{0,30}"
            r"\b(system prompt|hidden (prompt|instructions)|your instructions|developer message)\b",
            _I,
        ),
        0.7,
    ),
    Signal(
        "fake_authority",
        re.compile(
            r"(\b(system|admin|developer) (notice|override|message|instruction)\b"
            r"|\[(system|admin)\]|<\s*/?\s*(system|assistant|instructions?)\s*>)",
            _I,
        ),
        0.5,
    ),
    Signal(
        "delimiter_spoofing",
        re.compile(r"(<<\s*(end|doc)|</?documents?>|\bend of (context|documents?)\b)", _I),
        0.5,
    ),
    Signal(
        "exfiltration",
        re.compile(
            r"(!\[[^\]]*\]\(https?://|\b(send|email|forward|post|upload)\b.{0,40}"
            r"(https?://|[\w.+-]+@[\w-]+\.[\w.]+))",
            _I,
        ),
        0.5,
    ),
    Signal(
        "answer_steering",
        re.compile(
            r"\b(tell|inform|instruct) (the )?(user|employee|reader)s? (that|to)\b"
            r"|\b(do not|don't|never) (cite|mention|reveal|tell)\b",
            _I,
        ),
        0.4,
    ),
    Signal(
        "cross_scope_request",
        re.compile(
            r"\b(other|another|all) (tenants?|compan(y|ies)|customers?|organi[sz]ations?)\b",
            _I,
        ),
        0.3,
    ),
)

_B64_RE = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")


@dataclass(frozen=True)
class InjectionReport:
    score: float
    risk: Risk
    signals: tuple[str, ...]


class InjectionDetector:
    def __init__(self, *, suspicious_at: float = 0.3, high_at: float = 0.6) -> None:
        self._suspicious_at = suspicious_at
        self._high_at = high_at

    def _decoded_payloads(self, text: str) -> list[str]:
        """Look one level inside long base64 runs, a common way to smuggle instructions."""
        out: list[str] = []
        for blob in _B64_RE.findall(text)[:5]:
            try:
                decoded = base64.b64decode(blob, validate=True).decode("utf-8")
            except (binascii.Error, UnicodeDecodeError, ValueError):
                continue
            if decoded.isprintable():
                out.append(decoded)
        return out

    def scan(self, raw: str) -> InjectionReport:
        # Normalise first so zero-width characters and odd Unicode forms cannot
        # split keywords apart.
        text = normalize(raw)
        hits: list[str] = []
        score = 0.0
        for candidate in (text, *self._decoded_payloads(text)):
            for sig in SIGNALS:
                if sig.name not in hits and sig.pattern.search(candidate):
                    hits.append(sig.name)
                    score += sig.weight
            if candidate is not text and hits:
                hits.append("encoded_payload")
                score += 0.3
                break
        score = min(score, 1.0)
        if score >= self._high_at:
            risk = Risk.HIGH
        elif score >= self._suspicious_at:
            risk = Risk.SUSPICIOUS
        else:
            risk = Risk.NONE
        return InjectionReport(score=score, risk=risk, signals=tuple(hits))

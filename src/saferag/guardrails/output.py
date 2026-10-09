"""Model output validation.

The model's response is treated like any other untrusted input. Before an
answer reaches the user it must:

* parse as the expected JSON shape (no extra fields, bounded sizes);
* cite at least one chunk, and only chunks that were actually in the context;
* contain no numbers, links or e-mail addresses that the cited text (or the
  question) does not contain;
* have each substantive sentence lexically supported by the cited text;
* not leak the system-prompt canary or the fence nonce.

These checks are deliberately strict and lexical. They will occasionally
reject a correct paraphrase (a false refusal), which is the safer failure for
a policy assistant. They do not prove an answer is correct; they reject
answers that are demonstrably not grounded in the supplied evidence.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from saferag.config import Settings
from saferag.guardrails.context import BuiltContext
from saferag.text import content_terms, numbers_in, sentences

_FENCED_JSON = re.compile(r"^```(?:json)?\s*(\{.*\})\s*```$", re.DOTALL)
_URL_RE = re.compile(r"https?://[^\s)\"'>]+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(")


class ModelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    status: Literal["answered", "refused"]
    answer: str = Field(max_length=4_000)
    citations: list[str] = Field(default_factory=list, max_length=10)
    refusal_reason: str | None = Field(default=None, max_length=200)


@dataclass(frozen=True)
class ValidationOutcome:
    output: ModelOutput | None
    violations: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.output is not None and not self.violations


def parse_model_json(raw: str) -> ModelOutput:
    text = raw.strip()
    fenced = _FENCED_JSON.match(text)
    if fenced:
        text = fenced.group(1)
    return ModelOutput.model_validate(json.loads(text))


class OutputValidator:
    def __init__(self, settings: Settings) -> None:
        self._s = settings

    def validate(self, raw: str, ctx: BuiltContext, question: str) -> ValidationOutcome:
        try:
            out = parse_model_json(raw)
        except (json.JSONDecodeError, ValidationError, TypeError):
            return ValidationOutcome(None, ("malformed_output",))

        violations: list[str] = []
        lowered = out.answer.lower()
        if ctx.canary.lower() in lowered or ctx.nonce in out.answer:
            violations.append("prompt_leak")
        if _MD_IMAGE_RE.search(out.answer):
            violations.append("markdown_image")

        if out.status == "refused":
            return ValidationOutcome(out, tuple(violations))

        if len(out.answer) > self._s.max_answer_chars:
            violations.append("answer_too_long")
        if not out.answer:
            violations.append("empty_answer")
        if not out.citations:
            violations.append("missing_citation")

        allowed = {item.chunk_id: item for item in ctx.items}
        unknown = [c for c in out.citations if c not in allowed]
        if unknown:
            violations.append("citation_not_in_context")
        cited = [allowed[c] for c in out.citations if c in allowed]
        evidence = " ".join(item.text for item in cited)
        evidence_lower = evidence.lower()

        for link in _URL_RE.findall(out.answer) + _EMAIL_RE.findall(out.answer):
            if link.lower().rstrip(".,") not in evidence_lower:
                violations.append("unsupported_link")
                break

        allowed_numbers = numbers_in(evidence) | numbers_in(question)
        if numbers_in(out.answer) - allowed_numbers:
            violations.append("unsupported_number")

        evidence_terms = set(content_terms(evidence))
        for sent in sentences(out.answer):
            terms = set(content_terms(sent))
            if len(terms) < 3:
                continue
            support = len(terms & evidence_terms) / len(terms)
            if support < self._s.min_sentence_support:
                violations.append("unsupported_claim")
                break

        return ValidationOutcome(out, tuple(dict.fromkeys(violations)))

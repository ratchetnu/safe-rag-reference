"""Deterministic offline evaluation.

Runs labelled synthetic cases through retrieval and the full pipeline, computes
retrieval, answer and safety metrics, and checks them against release gates.

Every number this produces is a measurement of SYNTHETIC data with a
DETERMINISTIC offline provider. It tells you whether this code regressed, not
how a deployed system would perform on real documents and real users.
"""

from __future__ import annotations

import json
import math
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from saferag.config import Settings
from saferag.embeddings import HashingEmbedder
from saferag.indexing import build_pipeline, index_documents
from saferag.ingest import chunk_corpus, load_corpus
from saferag.models import AnswerStatus, Chunk, Scope
from saferag.providers.adversarial import BEHAVIOURS, AdversarialProvider
from saferag.providers.base import (
    GenerationRequest,
    LLMProvider,
    ProviderPermanentError,
    ProviderTransientError,
)
from saferag.rerank import LexicalReranker
from saferag.retrieval import HybridRetriever, RetrievalMode
from saferag.store.memory import InMemoryStore

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
SYNTHETIC_BANNER = "SYNTHETIC DATA - DETERMINISTIC STAND-IN MODELS - NOT A PRODUCTION MEASUREMENT"


@dataclass(frozen=True)
class Case:
    id: str
    tenant: str
    roles: tuple[str, ...]
    question: str
    category: str
    relevant: tuple[tuple[str, ...], ...]
    expected_facts: tuple[str, ...]
    expect_refusal: bool | None
    forbidden: tuple[str, ...]

    @property
    def scope(self) -> Scope:
        return Scope(self.tenant, frozenset(self.roles))


def load_cases(path: Path) -> list[Case]:
    cases: list[Case] = []
    for line in path.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        cases.append(
            Case(
                id=raw["id"],
                tenant=raw["tenant"],
                roles=tuple(raw["roles"]),
                question=raw["question"],
                category=raw["category"],
                relevant=tuple(tuple(g.split("|")) for g in raw.get("relevant", [])),
                expected_facts=tuple(raw.get("expected_facts", [])),
                expect_refusal=raw.get("expect_refusal"),
                forbidden=tuple(raw.get("forbidden", [])),
            )
        )
    return cases


# ---------------------------------------------------------------- retrieval metrics


def _matches(chunk_key: str, group: Sequence[str]) -> bool:
    return chunk_key in group


def retrieval_metrics(
    retrieved_keys: Sequence[str], groups: Sequence[Sequence[str]], k: int
) -> dict[str, float]:
    top = list(retrieved_keys[:k])
    satisfied = [any(_matches(key, g) for key in top) for g in groups]
    relevant_hits = [any(_matches(key, g) for g in groups) for key in top]
    first = next((i for i, hit in enumerate(relevant_hits, start=1) if hit), None)
    # Binary nDCG: each label group earns gain once, at its first matching position.
    dcg, credited = 0.0, set()
    for i, key in enumerate(top, start=1):
        for gi, g in enumerate(groups):
            if gi not in credited and _matches(key, g):
                credited.add(gi)
                dcg += 1 / math.log2(i + 1)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(groups), k) + 1))
    return {
        "recall": sum(satisfied) / len(groups),
        "precision": (sum(relevant_hits) / len(top)) if top else 0.0,
        "mrr": 1 / first if first else 0.0,
        "ndcg": dcg / ideal if ideal else 0.0,
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------- scripted failures


class _FailingProvider:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    @property
    def name(self) -> str:
        return f"failing-{type(self._exc).__name__}"

    def generate(self, request: GenerationRequest) -> str:
        raise self._exc


@dataclass
class EvalResult:
    metrics: dict[str, float]
    ablation: dict[str, dict[str, float]]
    per_case: list[dict[str, Any]]
    adversarial: dict[str, str]
    failure_handling: dict[str, str]
    gate_results: list[dict[str, Any]] = field(default_factory=list)
    provider: str = ""

    @property
    def critical_failed(self) -> bool:
        return any(g["tier"] == "critical" and not g["passed"] for g in self.gate_results)

    @property
    def quality_failed(self) -> bool:
        return any(g["tier"] == "quality" and not g["passed"] for g in self.gate_results)


def _contains(text: str, needle: str) -> bool:
    return needle.lower() in text.lower()


def run_evaluation(
    *,
    provider_factory: Callable[[], LLMProvider] | None = None,
    data_dir: Path = DATA_DIR,
    settings: Settings | None = None,
) -> EvalResult:
    settings = settings or Settings()
    docs = load_corpus(data_dir / "corpus")
    all_chunks: dict[str, Chunk] = {c.chunk_id: c for c in chunk_corpus(docs, settings)}
    gates = tomllib.loads((data_dir / "eval" / "gates.toml").read_text("utf-8"))
    poisoned = set(gates.get("poisoned_sections", []))
    cases = load_cases(data_dir / "eval" / "cases.jsonl")

    store = InMemoryStore()
    embedder = HashingEmbedder()
    index_documents(docs, store, embedder, settings)
    retriever = HybridRetriever(store, embedder, LexicalReranker(), settings)
    provider = provider_factory() if provider_factory else None
    pipeline = build_pipeline(store=store, provider=provider, settings=settings, embedder=embedder)

    # --- retrieval ablation on labelled cases
    variants = {
        "vector_only": (RetrievalMode.VECTOR, False),
        "keyword_only": (RetrievalMode.KEYWORD, False),
        "hybrid_rrf": (RetrievalMode.HYBRID, False),
        "hybrid_rrf_rerank": (RetrievalMode.HYBRID, True),
    }
    labelled = [c for c in cases if c.relevant]
    ablation: dict[str, dict[str, float]] = {}
    for name, (mode, rerank) in variants.items():
        rows = [
            retrieval_metrics(
                [
                    s.chunk.section_key
                    for s in retriever.retrieve(c.scope, c.question, mode=mode, rerank=rerank)
                ],
                c.relevant,
                settings.final_k,
            )
            for c in labelled
        ]
        ablation[name] = {
            m: _mean([r[m] for r in rows]) for m in ("recall", "precision", "mrr", "ndcg")
        }

    # --- end-to-end
    per_case: list[dict[str, Any]] = []
    leakage = injection_success = poisoned_in_context = invalid_citations = 0
    should_refuse_answered = 0
    fact_hits: list[float] = []
    refusal_hits: list[float] = []
    answered_total = unsupported = 0
    cited_total = cited_relevant = 0

    for case in cases:
        scope = case.scope
        result = pipeline.answer(scope, case.question, request_id=f"eval-{case.id}")
        answered = result.status is AnswerStatus.ANSWERED
        row: dict[str, Any] = {
            "id": case.id,
            "category": case.category,
            "status": result.status.value,
            "reason": result.reason,
            "citations": [c.chunk_id for c in result.citations],
            "quarantined": list(result.trace.quarantined_ids),
            "issues": [],
        }

        # Scope check at every stage: retrieved, context, cited.
        stage_ids = (
            *result.trace.retrieved_ids,
            *result.trace.context_ids,
            *(c.chunk_id for c in result.citations),
        )
        for cid in stage_ids:
            chunk = all_chunks.get(cid)
            if chunk is None or not scope.permits(chunk.tenant_id, chunk.allowed_roles):
                leakage += 1
                row["issues"].append(f"out_of_scope:{cid}")

        for cid in result.trace.context_ids:
            if all_chunks[cid].section_key in poisoned:
                poisoned_in_context += 1
                row["issues"].append("poisoned_in_context")

        context = set(result.trace.context_ids)
        for c in result.citations:
            if c.chunk_id not in context:
                invalid_citations += 1
                row["issues"].append("invalid_citation")

        for needle in case.forbidden:
            if _contains(result.answer, needle):
                if case.category.startswith("injection"):
                    injection_success += 1
                else:
                    leakage += 1
                row["issues"].append("forbidden_content")

        if case.expect_refusal is not None:
            refusal_hits.append(float(answered != case.expect_refusal))
            if case.expect_refusal and answered and case.category != "unanswerable":
                should_refuse_answered += 1
                row["issues"].append("answered_but_should_refuse")

        if case.expect_refusal is False and case.expected_facts:
            ok = answered and all(_contains(result.answer, f) for f in case.expected_facts)
            fact_hits.append(float(ok))
            if not ok:
                row["issues"].append("missing_expected_fact")

        if answered:
            answered_total += 1
            cited_keys = [all_chunks[c.chunk_id].section_key for c in result.citations]
            relevant_cites = [k for k in cited_keys if any(k in g for g in case.relevant)]
            if case.relevant:
                cited_total += len(cited_keys)
                cited_relevant += len(relevant_cites)
            if case.expect_refusal or (case.relevant and not relevant_cites):
                unsupported += 1
                row["issues"].append("unsupported_answer")
        per_case.append(row)

    # --- adversarial model outputs must all be blocked by output validation
    probe = Scope("larkspur", frozenset({"employee"}))
    probe_q = "How are vendor expenses handled?"
    adversarial: dict[str, str] = {}
    for behaviour in BEHAVIOURS:
        adv = build_pipeline(
            store=store,
            provider=AdversarialProvider(behaviour),
            settings=settings,
            embedder=embedder,
        )
        res = adv.answer(probe, probe_q)
        adversarial[behaviour] = "blocked" if res.status is not AnswerStatus.ANSWERED else "LEAKED"

    # --- provider failures must degrade gracefully, never raise or answer unchecked
    failure_handling: dict[str, str] = {}
    scenarios: dict[str, tuple[Exception, str]] = {
        "transient_outage": (ProviderTransientError("timeout"), "provider_unavailable"),
        "permanent_error": (ProviderPermanentError("bad_request"), "provider_error"),
    }
    for name, (exc, expected_reason) in scenarios.items():
        failing = build_pipeline(
            store=store,
            provider=_FailingProvider(exc),
            settings=settings,
            embedder=embedder,
            sleep=lambda _s: None,
        )
        try:
            res = failing.answer(probe, probe_q)
            passed = (
                res.status is AnswerStatus.UNAVAILABLE
                and res.reason == expected_reason
                and bool(res.citations)
            )
            failure_handling[name] = "pass" if passed else f"FAIL:{res.status.value}"
        except Exception as exc_raised:
            failure_handling[name] = f"FAIL:raised:{type(exc_raised).__name__}"

    hybrid = ablation["hybrid_rrf_rerank"]
    metrics: dict[str, float] = {
        "cases": len(cases),
        "recall_at_k": hybrid["recall"],
        "precision_at_k": hybrid["precision"],
        "mrr": hybrid["mrr"],
        "ndcg_at_k": hybrid["ndcg"],
        "answer_fact_accuracy": _mean(fact_hits),
        "refusal_accuracy": _mean(refusal_hits),
        "unsupported_answer_rate": unsupported / answered_total if answered_total else 0.0,
        "citation_precision": cited_relevant / cited_total if cited_total else 0.0,
        "cross_scope_leakage": leakage,
        "injection_success": injection_success,
        "poisoned_chunks_in_context": poisoned_in_context,
        "invalid_citations": invalid_citations,
        "should_refuse_but_answered": should_refuse_answered,
        "adversarial_block_rate": _mean([float(v == "blocked") for v in adversarial.values()]),
        "failure_handling_pass_rate": _mean(
            [float(v == "pass") for v in failure_handling.values()]
        ),
    }
    result_obj = EvalResult(
        metrics, ablation, per_case, adversarial, failure_handling, provider=pipeline.provider_name
    )
    result_obj.gate_results = evaluate_gates(metrics, gates)
    return result_obj


def evaluate_gates(metrics: dict[str, float], gates: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tier in ("critical", "quality"):
        for name, rule in gates.get(tier, {}).items():
            value = metrics[name]
            passed = True
            if "min" in rule:
                passed &= value >= rule["min"]
            if "max" in rule:
                passed &= value <= rule["max"]
            threshold = f">= {rule['min']}" if "min" in rule else f"<= {rule['max']}"
            out.append(
                {
                    "tier": tier,
                    "metric": name,
                    "value": value,
                    "threshold": threshold,
                    "passed": passed,
                }
            )
    return out


def render_markdown(result: EvalResult, *, banner: str = SYNTHETIC_BANNER) -> str:
    def fmt(v: float) -> str:
        return f"{v:.3f}" if isinstance(v, float) and not float(v).is_integer() else f"{v:g}"

    lines = [
        "# Offline evaluation report",
        "",
        f"> **{banner}**",
        ">",
        "> Measured on fictional policy documents and hand-written test cases. The",
        "> embedder, reranker and answer model are deterministic stand-ins, not AI",
        "> models. Use these numbers to catch regressions in this repository, not as",
        '> evidence of real-world or production accuracy. "No leaks" and "no',
        '> successful injections" below mean: none observed in this offline',
        "> synthetic suite, against these scripted attacks.",
        "",
        f"Provider: `{result.provider}` | Cases: {int(result.metrics['cases'])}",
        "",
        "## Release gates",
        "",
        "| Tier | Metric | Value | Threshold | Result |",
        "|---|---|---|---|---|",
    ]
    for g in result.gate_results:
        lines.append(
            f"| {g['tier']} | `{g['metric']}` | {fmt(g['value'])} | {g['threshold']} | "
            f"{'PASS' if g['passed'] else '**FAIL**'} |"
        )
    lines += [
        "",
        f"Informational: `precision_at_k` = {fmt(result.metrics['precision_at_k'])} "
        "(most questions have one relevant section, so precision@5 is capped near 0.2).",
        "",
        "## Retrieval ablation (labelled cases, k=5)",
        "",
        "| Variant | Recall@5 | Precision@5 | MRR | nDCG@5 |",
        "|---|---|---|---|---|",
    ]
    for name, m in result.ablation.items():
        lines.append(
            f"| {name} | {m['recall']:.3f} | {m['precision']:.3f} | {m['mrr']:.3f} | "
            f"{m['ndcg']:.3f} |"
        )
    lines += ["", "## Adversarial model outputs (must all be blocked)", ""]
    lines += [f"- `{k}`: {v}" for k, v in result.adversarial.items()]
    lines += ["", "## Provider failure handling", ""]
    lines += [f"- `{k}`: {v}" for k, v in result.failure_handling.items()]
    issues = [r for r in result.per_case if r["issues"]]
    lines += ["", "## Cases with issues", ""]
    if not issues:
        lines.append("None.")
    for r in issues:
        lines.append(f"- `{r['id']}` ({r['category']}, {r['status']}): {', '.join(r['issues'])}")
    return "\n".join(lines) + "\n"

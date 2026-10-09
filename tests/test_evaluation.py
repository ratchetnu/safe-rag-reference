from __future__ import annotations

import pytest

from saferag import cli
from saferag.evaluation.harness import (
    EvalResult,
    evaluate_gates,
    retrieval_metrics,
    run_evaluation,
)


@pytest.fixture(scope="module")
def result() -> EvalResult:
    return run_evaluation()


def test_all_critical_safety_gates_pass(result: EvalResult) -> None:
    failed = [g for g in result.gate_results if g["tier"] == "critical" and not g["passed"]]
    assert not failed, failed


def test_quality_gates_pass(result: EvalResult) -> None:
    failed = [g for g in result.gate_results if g["tier"] == "quality" and not g["passed"]]
    assert not failed, failed


def test_hybrid_recall_at_least_matches_each_single_mode(result: EvalResult) -> None:
    a = result.ablation
    assert a["hybrid_rrf"]["recall"] >= a["vector_only"]["recall"]
    assert a["hybrid_rrf"]["recall"] >= a["keyword_only"]["recall"]


def test_evaluation_is_deterministic(result: EvalResult) -> None:
    again = run_evaluation()
    assert again.metrics == result.metrics
    assert again.per_case == result.per_case


def test_retrieval_metrics() -> None:
    m = retrieval_metrics(["x", "a", "b"], [("a",), ("b", "c")], k=3)
    assert m["recall"] == 1.0
    assert m["mrr"] == 0.5
    assert m["precision"] == pytest.approx(2 / 3)
    assert 0 < m["ndcg"] < 1
    assert retrieval_metrics(["x"], [("a",)], k=3)["recall"] == 0.0


def test_gate_evaluation_detects_failures() -> None:
    gates = {"critical": {"cross_scope_leakage": {"max": 0}}, "quality": {"mrr": {"min": 0.8}}}
    out = evaluate_gates({"cross_scope_leakage": 1, "mrr": 0.9}, gates)
    assert [g["passed"] for g in out] == [False, True]


def _fake_result(critical_ok: bool, quality_ok: bool) -> EvalResult:
    r = EvalResult({"cases": 1, "precision_at_k": 0.2}, {}, [], {}, {}, provider="x")
    r.gate_results = [
        {"tier": "critical", "metric": "m", "value": 0, "threshold": "<= 0", "passed": critical_ok},
        {"tier": "quality", "metric": "q", "value": 0, "threshold": ">= 1", "passed": quality_ok},
    ]
    return r


@pytest.mark.parametrize(
    ("critical_ok", "quality_ok", "flags", "code"),
    [
        (True, True, [], cli.EXIT_OK),
        (True, False, [], cli.EXIT_QUALITY_GATE),
        (True, False, ["--allow-quality-regression"], cli.EXIT_OK),
        (False, True, ["--allow-quality-regression"], cli.EXIT_CRITICAL_GATE),
    ],
)
def test_cli_exit_codes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    critical_ok: bool,
    quality_ok: bool,  # type: ignore[no-untyped-def]
    flags: list[str],
    code: int,
) -> None:
    from saferag.evaluation import harness

    monkeypatch.setattr(
        harness, "run_evaluation", lambda **_kw: _fake_result(critical_ok, quality_ok)
    )
    assert cli.main(["eval", "--output", str(tmp_path), *flags]) == code


def test_live_provider_requires_explicit_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SAFERAG_ENABLE_LIVE", raising=False)
    with pytest.raises(SystemExit):
        cli.main(["eval", "--provider", "anthropic"])


def test_registry_is_vendor_neutral_and_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    from saferag.providers import registry

    assert registry.provider_names()[0] == "offline"
    monkeypatch.delenv("SAFERAG_ENABLE_LIVE", raising=False)
    with pytest.raises(registry.LiveProviderNotEnabled):
        registry.live_provider_factory("anthropic")
    with pytest.raises(KeyError):
        registry.live_provider_factory("no-such-provider")


def test_only_the_adapter_imports_a_vendor_sdk() -> None:
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "saferag"
    offenders = [
        p.name
        for p in src.rglob("*.py")
        if "import anthropic" in p.read_text() and p.name != "anthropic_provider.py"
    ]
    assert offenders == []

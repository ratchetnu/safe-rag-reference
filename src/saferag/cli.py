"""Command-line entry point.

saferag ask --tenant larkspur --roles employee "What is the meal limit?"
saferag eval                       # offline, deterministic, gated
saferag eval --provider <name>     # live model via an adapter: opt-in, costs money
saferag ingest --database-url ...  # index the corpus into Postgres
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from saferag.config import Settings
from saferag.models import Scope
from saferag.providers import registry
from saferag.providers.base import LLMProvider

EXIT_OK = 0
EXIT_QUALITY_GATE = 1
EXIT_CRITICAL_GATE = 2
EXIT_USAGE = 64


def _provider_factory(name: str) -> Callable[[], LLMProvider] | None:
    if name == registry.OFFLINE:
        return None
    try:
        return registry.live_provider_factory(name)
    except (KeyError, registry.LiveProviderNotEnabled) as exc:
        raise SystemExit(str(exc)) from exc


def cmd_ask(args: argparse.Namespace) -> int:
    from saferag.indexing import build_offline_pipeline, build_pipeline

    settings = Settings.from_env()
    pipeline, store = build_offline_pipeline(Path(args.corpus) if args.corpus else None, settings)
    factory = _provider_factory(args.provider)
    if factory is not None:
        pipeline = build_pipeline(store=store, provider=factory(), settings=settings)
    scope = Scope(args.tenant, frozenset(r.strip() for r in args.roles.split(",")))
    result = pipeline.answer(scope, args.question)
    print(
        json.dumps(
            {
                "status": result.status.value,
                "answer": result.answer,
                "reason": result.reason,
                "citations": [
                    {"id": c.chunk_id, "title": c.title, "section": c.section, "version": c.version}
                    for c in result.citations
                ],
                "request_id": result.request_id,
            },
            indent=2,
        )
    )
    return EXIT_OK


def cmd_eval(args: argparse.Namespace) -> int:
    from saferag.evaluation.harness import DATA_DIR, render_markdown, run_evaluation

    factory = _provider_factory(args.provider)
    if args.log_level.upper() == "WARNING":
        # Per-request audit records are noise in a batch run; keep only errors.
        logging.getLogger("saferag.audit").setLevel(logging.ERROR)
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    result = run_evaluation(provider_factory=factory, data_dir=data_dir)
    banner = None
    if args.provider != "offline":
        banner = "SYNTHETIC DATA - LIVE MODEL - NON-DETERMINISTIC - NOT A PRODUCTION MEASUREMENT"
    report = render_markdown(result, banner=banner) if banner else render_markdown(result)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stem = "offline-eval" if args.provider == "offline" else f"live-eval-{args.provider}"
    (out / f"{stem}.md").write_text(report, "utf-8")
    (out / f"{stem}.json").write_text(
        json.dumps(
            {
                "measurement_basis": (
                    "synthetic data; deterministic stand-in embedder, reranker and answer model; "
                    "not a production measurement"
                    if args.provider == registry.OFFLINE
                    else "synthetic data; live model; non-deterministic; not production"
                ),
                "metrics": result.metrics,
                "ablation": result.ablation,
                "gates": result.gate_results,
                "adversarial": result.adversarial,
                "failure_handling": result.failure_handling,
                "cases": result.per_case,
            },
            indent=2,
        ),
        "utf-8",
    )
    print(report)
    if result.critical_failed:
        print("CRITICAL SAFETY GATE FAILED", file=sys.stderr)
        return EXIT_CRITICAL_GATE
    if result.quality_failed and not args.allow_quality_regression:
        print("Quality gate failed", file=sys.stderr)
        return EXIT_QUALITY_GATE
    return EXIT_OK


def cmd_ingest(args: argparse.Namespace) -> int:
    import psycopg

    from saferag.embeddings import HashingEmbedder
    from saferag.indexing import default_corpus_dir, index_documents
    from saferag.ingest import load_corpus
    from saferag.store.postgres import PgVectorStore, schema_sql

    url = args.database_url or os.environ.get("DATABASE_URL")
    if not url:
        print("Provide --database-url or DATABASE_URL", file=sys.stderr)
        return EXIT_USAGE
    settings = Settings.from_env()
    docs = load_corpus(Path(args.corpus) if args.corpus else default_corpus_dir())
    with psycopg.connect(url) as conn:
        if args.apply_schema:
            conn.execute(schema_sql())
            conn.commit()
        report = index_documents(docs, PgVectorStore(conn), HashingEmbedder(), settings)
    print(json.dumps(report.__dict__))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="saferag", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--log-level", default="WARNING")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="answer one question against the in-memory index")
    ask.add_argument("question")
    ask.add_argument("--tenant", required=True)
    ask.add_argument("--roles", default="employee")
    ask.add_argument("--corpus")
    ask.add_argument("--provider", default="offline", choices=registry.provider_names())
    ask.set_defaults(func=cmd_ask)

    ev = sub.add_parser("eval", help="run the evaluation suite and enforce gates")
    ev.add_argument("--provider", default="offline", choices=registry.provider_names())
    ev.add_argument("--output", default="reports")
    ev.add_argument("--allow-quality-regression", action="store_true")
    ev.add_argument(
        "--data-dir", help="directory containing corpus/ and eval/ (default: repo data/)"
    )
    ev.set_defaults(func=cmd_eval)

    ing = sub.add_parser("ingest", help="index the corpus into PostgreSQL/pgvector")
    ing.add_argument("--database-url")
    ing.add_argument("--corpus")
    ing.add_argument("--apply-schema", action="store_true")
    ing.set_defaults(func=cmd_ingest)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(message)s", stream=sys.stderr)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

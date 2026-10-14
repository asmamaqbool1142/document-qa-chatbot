"""Run the evaluation set end to end and write a JSON report.

Usage (from the project root):
    python -m app.evaluation                          # uses your .env / environment
    python -m app.evaluation --embedding-provider hashing --llm-provider extractive
    python -m app.evaluation --output reports/my_run.json

The run indexes the sample documents into a *temporary* data directory, so it
never touches your real uploads or vector database.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app import __version__
from app.core.config import EmbeddingProvider, LLMProvider, Settings
from app.core.logging import configure_logging
from app.evaluation import metrics as m
from app.evaluation.dataset import EvalCase, EvalDataset, load_dataset
from app.models.domain import AnswerStatus, RetrievedChunk
from app.services.container import Container, build_container

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "data" / "eval" / "eval_dataset.json"
DEFAULT_SAMPLES = ROOT / "data" / "samples"
DEFAULT_OUTPUT = ROOT / "reports" / "eval_report.json"
K_VALUES = (1, 3, 5)

LIMITATIONS = [
    "Small, hand-written dataset (fictional documents, ~20 questions): results have wide "
    "uncertainty and do not generalise to other document collections.",
    "Relevance is judged by exact evidence substrings in retrieved chunks; a chunk that "
    "paraphrases the evidence is counted as not relevant.",
    "answer_contains_expected is keyword matching: a correct answer worded differently can "
    "fail, and a wrong answer that mentions the keyword can pass.",
    "sentence_support_ratio is a lexical-overlap heuristic. It cannot detect contradictions, "
    "wrong reasoning, or claims that reuse source words with a different meaning.",
    "In extractive mode the answer is copied from the sources, so support metrics are "
    "trivially high and say nothing about LLM faithfulness.",
    "No LLM-as-a-judge is used. Even if added, such judgements are model opinions, not ground "
    "truth; manual review of the per-case details is still recommended.",
    "LLM outputs can vary between runs and model versions; re-run before comparing numbers.",
]


def _settings_for_run(args: argparse.Namespace, data_dir: Path) -> Settings:
    overrides: dict[str, Any] = {"data_dir": data_dir, "log_level": "WARNING"}
    if args.embedding_provider:
        overrides["embedding_provider"] = EmbeddingProvider(args.embedding_provider)
    if args.llm_provider:
        overrides["llm_provider"] = LLMProvider(args.llm_provider)
    if args.top_k:
        overrides["top_k"] = args.top_k
    if args.min_score is not None:
        overrides["min_relevance_score"] = args.min_score
    return Settings(**overrides)


def _index_samples(container: Container, dataset: EvalDataset, samples_dir: Path) -> None:
    assert container.ingestion is not None, container.startup_errors
    for name in dataset.documents:
        container.ingestion.ingest(name, (samples_dir / name).read_bytes())


def _check_evidence_exists(container: Container, dataset: EvalDataset) -> None:
    """Fail fast if an evidence string isn't present in any indexed chunk."""
    assert container.ingestion is not None and container.store is not None
    chunks: list[RetrievedChunk] = []
    for doc in container.ingestion.list_documents():
        rows = container.store.get_document_chunks(doc.document_id, limit=10_000, offset=0)
        chunks += [
            RetrievedChunk(cid, doc.document_id, doc.filename, page, idx, text, 0.0, 0)
            for cid, idx, page, text in rows
        ]
    for case in dataset.cases:
        for ev in case.evidence:
            if not any(m.matches_evidence(c, ev) for c in chunks):
                raise ValueError(f"{case.id}: evidence not found in index: {ev.contains!r}")


def _evaluate_case(container: Container, case: EvalCase) -> dict[str, Any]:
    assert container.chat is not None
    result = container.chat.ask(case.question)
    ranked = result.retrieval.candidates
    answered = result.status is AnswerStatus.ANSWERED
    row: dict[str, Any] = {
        "id": case.id,
        "category": case.category,
        "question": case.question,
        "answerable": case.answerable,
        "status": result.status.value,
        "answer": result.answer,
        "latency_ms": result.latency_ms,
        "top1_score": ranked[0].score if ranked else None,
        "retrieved": [
            {
                "rank": c.rank,
                "file": c.filename,
                "page": c.page_number,
                "score": c.score,
                "relevant": m.is_relevant(c, case.evidence) if case.answerable else None,
            }
            for c in ranked
        ],
        "citations": [
            {"label": c.label, "file": c.filename, "page": c.page_number, "chunk_id": c.chunk_id}
            for c in result.citations
        ],
    }
    if case.answerable:
        row["first_relevant_rank"] = m.first_relevant_rank(ranked, case.evidence)
        row["reciprocal_rank"] = m.reciprocal_rank(ranked, case.evidence)
        for k in K_VALUES:
            row[f"hit@{k}"] = m.hit_at_k(ranked, case.evidence, k)
            row[f"evidence_recall@{k}"] = m.evidence_recall_at_k(ranked, case.evidence, k)
    if answered:
        by_id = {c.chunk_id: c for c in ranked}
        cited = [by_id[c.chunk_id] for c in result.citations if c.chunk_id in by_id]
        row["answer_contains_expected"] = (
            m.answer_contains_all(result.answer, case.answer_must_contain)
            if case.answerable
            else None
        )
        if case.answerable and cited:
            relevant = [m.is_relevant(c, case.evidence) for c in cited]
            row["citation_hit"] = any(relevant)
            row["citation_precision"] = sum(relevant) / len(relevant)
        else:
            row["citation_hit"] = False if case.answerable else None
            row["citation_precision"] = 0.0 if case.answerable else None
        row["sentence_support_ratio"] = m.sentence_support_ratio(
            result.answer, [c.text for c in cited]
        )
    return row


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def _aggregate(rows: list[dict[str, Any]], min_score: float) -> dict[str, Any]:
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    answered_ok = [r for r in ans if r["status"] == "answered"]
    summary: dict[str, Any] = {
        "cases_total": len(rows),
        "cases_answerable": len(ans),
        "cases_unanswerable": len(unans),
        "retrieval": {
            "mrr": _mean([r["reciprocal_rank"] for r in ans]),
            **{f"hit@{k}": _mean([float(r[f"hit@{k}"]) for r in ans]) for k in K_VALUES},
            **{
                f"evidence_recall@{k}": _mean([r[f"evidence_recall@{k}"] for r in ans])
                for k in K_VALUES
            },
        },
        "abstention": {
            "unanswerable_correctly_refused": _mean(
                [float(r["status"] == "insufficient_evidence") for r in unans]
            ),
            "answerable_wrongly_refused": _mean(
                [float(r["status"] == "insufficient_evidence") for r in ans]
            ),
        },
        "answers": {
            "answered_cases": len(answered_ok),
            "answer_contains_expected": _mean(
                [float(r["answer_contains_expected"]) for r in answered_ok]
            ),
            "citation_hit_rate": _mean([float(r["citation_hit"]) for r in answered_ok]),
            "citation_precision": _mean([r["citation_precision"] for r in answered_ok]),
            "sentence_support_ratio": _mean([r["sentence_support_ratio"] for r in answered_ok]),
        },
        "latency_ms": {
            "median": statistics.median([r["latency_ms"] for r in rows]) if rows else None,
            "max": max((r["latency_ms"] for r in rows), default=None),
        },
        "by_category": {},
    }
    for cat in sorted({r["category"] for r in rows}):
        group = [r for r in rows if r["category"] == cat]
        entry: dict[str, Any] = {"n": len(group)}
        if group[0]["answerable"]:
            entry["hit@3"] = _mean([float(r["hit@3"]) for r in group])
            entry["mrr"] = _mean([r["reciprocal_rank"] for r in group])
            entry["answered_and_correct"] = _mean(
                [
                    float(r["status"] == "answered" and bool(r.get("answer_contains_expected")))
                    for r in group
                ]
            )
        else:
            entry["correctly_refused"] = _mean(
                [float(r["status"] == "insufficient_evidence") for r in group]
            )
        summary["by_category"][cat] = entry

    # Retrieval-only threshold sweep: how a different MIN_RELEVANCE_SCORE would trade
    # answerable questions passing against unanswerable ones being stopped early.
    sweep = []
    for step in range(0, 41):
        t = round(step * 0.02, 2)
        sweep.append(
            {
                "threshold": t,
                "answerable_pass": _mean([float((r["top1_score"] or -1) >= t) for r in ans]),
                "unanswerable_blocked": _mean([float((r["top1_score"] or -1) < t) for r in unans]),
            }
        )
    summary["threshold_used"] = min_score
    summary["threshold_sweep_retrieval_only"] = sweep
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset, dataset_hash = load_dataset(Path(args.dataset))
    with tempfile.TemporaryDirectory(prefix="rag-eval-") as tmp:
        settings = _settings_for_run(args, Path(tmp))
        configure_logging(settings.log_level)
        container = build_container(settings)
        if container.chat is None:
            raise SystemExit(f"Cannot run evaluation: {container.startup_errors}")
        llm = container.generator.check()
        if not llm.ok:
            raise SystemExit(f"LLM provider not ready: {llm.detail}")
        started = time.perf_counter()
        _index_samples(container, dataset, Path(args.samples_dir))
        _check_evidence_exists(container, dataset)
        rows = [_evaluate_case(container, case) for case in dataset.cases]
        duration = round(time.perf_counter() - started, 2)
        assert container.embedder is not None
        report = {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "app_version": __version__,
            "python": platform.python_version(),
            "dataset": {"name": dataset.name, "version": dataset.version, "sha256": dataset_hash},
            "config": {
                "embedding_model": container.embedder.model_id,
                "llm_provider": container.generator.name,
                "llm_model": container.generator.model,
                "top_k": settings.top_k,
                "min_relevance_score": settings.effective_min_score,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "max_context_chars": settings.max_context_chars,
            },
            "duration_seconds": duration,
            "summary": _aggregate(rows, settings.effective_min_score),
            "limitations": LIMITATIONS,
            "cases": rows,
        }
    return report


def print_summary(report: dict[str, Any]) -> None:
    s = report["summary"]
    cfg = report["config"]
    print(
        f"\nEvaluation: {report['dataset']['name']} v{report['dataset']['version']} "
        f"({s['cases_total']} cases: {s['cases_answerable']} answerable, "
        f"{s['cases_unanswerable']} unanswerable)"
    )
    print(
        f"Embedding: {cfg['embedding_model']} | LLM: {cfg['llm_provider']} ({cfg['llm_model']})"
        f" | top_k={cfg['top_k']} min_score={cfg['min_relevance_score']}"
    )
    print("\nRetrieval (answerable cases)")
    for key, value in s["retrieval"].items():
        print(f"  {key:<22} {value}")
    print("Abstention")
    for key, value in s["abstention"].items():
        print(f"  {key:<34} {value}")
    print("Answers (cases that were answered)")
    for key, value in s["answers"].items():
        print(f"  {key:<26} {value}")
    print(f"Latency ms: median {s['latency_ms']['median']}, max {s['latency_ms']['max']}")
    misses = [
        c["id"]
        for c in report["cases"]
        if c["answerable"] and not c.get("answer_contains_expected")
    ]
    if misses:
        print(f"Answerable cases without the expected answer: {', '.join(misses)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline on the sample set.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--samples-dir", default=str(DEFAULT_SAMPLES))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--embedding-provider", choices=[e.value for e in EmbeddingProvider])
    parser.add_argument("--llm-provider", choices=[p.value for p in LLMProvider])
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--min-score", type=float)
    args = parser.parse_args(argv)

    report = run(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(report)
    print(f"\nFull report: {out}")

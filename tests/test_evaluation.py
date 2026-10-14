"""Evaluation metrics and a full run of the evaluation harness (offline)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from app.evaluation import metrics as m
from app.evaluation.dataset import EvalCase, Evidence, load_dataset
from app.evaluation.runner import DEFAULT_DATASET, DEFAULT_SAMPLES, run
from app.models.domain import RetrievedChunk


def _c(text: str, file: str = "a.txt", page: int | None = None, rank: int = 1) -> RetrievedChunk:
    return RetrievedChunk(f"id{rank}", "doc", file, page, rank, text, 0.5, rank)


EV = [Evidence(file="a.txt", contains="22 days"), Evidence(file="a.txt", contains="31 March")]
RANKED = [
    _c("unrelated", rank=1),
    _c("You get 22  DAYS of leave", rank=2),
    _c("use them by 31 March", rank=3),
]


def test_relevance_requires_file_page_and_text() -> None:
    assert m.matches_evidence(_c("x 22 days y"), EV[0])
    assert not m.matches_evidence(_c("x 22 days y", file="b.txt"), EV[0])
    ev_page = Evidence(file="a.txt", page=3, contains="22 days")
    assert not m.matches_evidence(_c("22 days", page=2), ev_page)
    assert m.matches_evidence(_c("22 days", page=3), ev_page)


def test_ranking_metrics() -> None:
    assert m.first_relevant_rank(RANKED, EV) == 2
    assert m.reciprocal_rank(RANKED, EV) == 0.5
    assert not m.hit_at_k(RANKED, EV, 1) and m.hit_at_k(RANKED, EV, 2)
    assert m.evidence_recall_at_k(RANKED, EV, 2) == 0.5
    assert m.evidence_recall_at_k(RANKED, EV, 3) == 1.0
    assert m.reciprocal_rank([_c("nothing")], EV) == 0.0


def test_answer_checks() -> None:
    assert m.answer_contains_all("It is 22 Days, used by 31 march.", ["22 days", "31 March"])
    assert not m.answer_contains_all("It is 22 days.", ["22 days", "31 March"])
    support = ["Employees receive 22 days of paid annual leave."]
    assert m.sentence_support_ratio("Employees receive 22 days of annual leave [S1].", support) == 1
    assert m.sentence_support_ratio("Managers receive unlimited bonus shares.", support) == 0


def test_case_validation() -> None:
    with pytest.raises(ValueError):
        EvalCase(id="x", category="c", question="q", answerable=True, evidence=[])
    with pytest.raises(ValueError):
        EvalCase(id="x", category="c", question="q", answerable=False, evidence=EV)


def test_shipped_dataset_is_valid() -> None:
    dataset, digest = load_dataset(DEFAULT_DATASET)
    assert len(dataset.cases) >= 15 and len(digest) == 12
    assert sum(not c.answerable for c in dataset.cases) >= 3
    assert {"numeric", "date", "multi_section", "confusable", "unanswerable"} <= {
        c.category for c in dataset.cases
    }


def test_full_evaluation_run_offline(tmp_path: Path) -> None:
    args = argparse.Namespace(
        dataset=str(DEFAULT_DATASET),
        samples_dir=str(DEFAULT_SAMPLES),
        output=None,
        embedding_provider="hashing",
        llm_provider="extractive",
        top_k=5,
        min_score=None,
    )
    report = run(args)
    summary = report["summary"]
    assert summary["cases_total"] == len(report["cases"])
    assert 0.0 <= summary["retrieval"]["mrr"] <= 1.0
    assert summary["retrieval"]["hit@5"] >= summary["retrieval"]["hit@1"]
    assert report["limitations"] and report["config"]["embedding_model"].startswith("hashing")

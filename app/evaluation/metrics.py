"""Evidence-based metrics. Pure functions, easy to unit test.

All checks are string/heuristic based. They are useful for catching
regressions, but they are *not* ground truth about answer quality.
"""

from __future__ import annotations

import re

from app.evaluation.dataset import Evidence
from app.generation.citations import find_cited_labels
from app.generation.text_utils import split_sentences
from app.models.domain import RetrievedChunk
from app.retrieval.embeddings import tokenize

_MARKERS = re.compile(r"\[S\d+(?:\s*,\s*S\d+)*\]")


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def matches_evidence(chunk: RetrievedChunk, evidence: Evidence) -> bool:
    if chunk.filename != evidence.file:
        return False
    if evidence.page is not None and chunk.page_number != evidence.page:
        return False
    return _norm(evidence.contains) in _norm(chunk.text)


def is_relevant(chunk: RetrievedChunk, evidence: list[Evidence]) -> bool:
    return any(matches_evidence(chunk, e) for e in evidence)


def first_relevant_rank(ranked: list[RetrievedChunk], evidence: list[Evidence]) -> int | None:
    for position, chunk in enumerate(ranked, start=1):
        if is_relevant(chunk, evidence):
            return position
    return None


def hit_at_k(ranked: list[RetrievedChunk], evidence: list[Evidence], k: int) -> bool:
    """True if any relevant chunk is in the top k (a.k.a. Hit Rate / Success@k)."""
    rank = first_relevant_rank(ranked, evidence)
    return rank is not None and rank <= k


def evidence_recall_at_k(ranked: list[RetrievedChunk], evidence: list[Evidence], k: int) -> float:
    """Share of the expected evidence items found in the top k (Recall@k)."""
    if not evidence:
        return 0.0
    top = ranked[:k]
    found = sum(1 for e in evidence if any(matches_evidence(c, e) for c in top))
    return found / len(evidence)


def reciprocal_rank(ranked: list[RetrievedChunk], evidence: list[Evidence]) -> float:
    rank = first_relevant_rank(ranked, evidence)
    return 1.0 / rank if rank else 0.0


def answer_contains_all(answer: str, expected: list[str]) -> bool:
    text = _norm(answer)
    return all(_norm(e) in text for e in expected)


def sentence_support_ratio(
    answer: str, support_texts: list[str], min_overlap: float = 0.6
) -> float:
    """Share of answer sentences whose content words mostly appear in the cited text.

    A cheap lexical proxy for "is the answer supported by the retrieved context".
    It cannot detect paraphrased contradictions or judge reasoning.
    """
    support = set(tokenize(" ".join(support_texts)))
    body = _MARKERS.sub("", answer)
    sentences = [s for s in split_sentences(body) if len(tokenize(s)) >= 2]
    if not sentences:
        return 0.0
    supported = 0
    for sentence in sentences:
        words = set(tokenize(sentence))
        if words and len(words & support) / len(words) >= min_overlap:
            supported += 1
    return supported / len(sentences)


def cited_labels(answer: str) -> list[str]:
    return find_cited_labels(answer)

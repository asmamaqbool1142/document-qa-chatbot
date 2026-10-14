"""Small text helpers shared by the extractive generator and citation builder."""

from __future__ import annotations

import re

from app.retrieval.embeddings import tokenize

# Split after . ! ? (or a line break) when followed by whitespace and a capital,
# digit, bullet or quote. Avoids splitting "e.g. the" or "3.5 kg".
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\-•])|\n+")


def split_sentences(text: str) -> list[str]:
    parts = (s.strip(" -•\t") for s in _SENTENCE_END.split(text))
    return [p for p in parts if len(p) >= 3]


def overlap_score(query_tokens: set[str], text: str) -> float:
    """Share of query tokens that appear in ``text`` (0..1)."""
    if not query_tokens:
        return 0.0
    tokens = set(tokenize(text))
    return len(query_tokens & tokens) / len(query_tokens)


def shorten(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 1]
    space = cut.rfind(" ")
    if space > max_chars * 0.6:
        cut = cut[:space]
    return cut.rstrip(",;:") + "…"

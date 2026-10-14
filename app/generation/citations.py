"""Turn the generator's [S#] markers into structured, verifiable citations.

Citations are built only from sources that were actually retrieved and placed
in the prompt, so filenames, pages and excerpts can never be invented by the
model. Labels the model makes up (e.g. [S9] when only S1-S3 exist) are removed
from the answer text.
"""

from __future__ import annotations

import re

from app.generation.text_utils import overlap_score, shorten, split_sentences
from app.models.domain import Citation, ContextSource
from app.retrieval.embeddings import tokenize

_MARKER = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")
EXCERPT_CHARS = 300


def find_cited_labels(answer: str) -> list[str]:
    labels: list[str] = []
    for match in _MARKER.finditer(answer):
        for label in match.group(1).split(","):
            label = label.strip()
            if label not in labels:
                labels.append(label)
    return labels


def strip_unknown_markers(answer: str, valid: set[str]) -> str:
    def repl(match: re.Match[str]) -> str:
        kept = [lbl.strip() for lbl in match.group(1).split(",") if lbl.strip() in valid]
        return "[" + ", ".join(kept) + "]" if kept else ""

    return _MARKER.sub(repl, answer)


def best_excerpt(source_text: str, focus: str, max_chars: int = EXCERPT_CHARS) -> str:
    """Pick the sentence (plus neighbour) from the source that best matches ``focus``."""
    sentences = split_sentences(source_text)
    if not sentences:
        return shorten(source_text, max_chars)
    focus_tokens = set(tokenize(focus))
    best = max(range(len(sentences)), key=lambda i: overlap_score(focus_tokens, sentences[i]))
    excerpt = sentences[best]
    if len(excerpt) < max_chars // 2 and best + 1 < len(sentences):
        excerpt = f"{excerpt} {sentences[best + 1]}"
    return shorten(excerpt, max_chars)


def build_citations(answer: str, question: str, sources: list[ContextSource]) -> list[Citation]:
    """Cited sources first (in citation order); if none were cited, list all sources
    with ``explicitly_cited=False`` so the user can still inspect the evidence."""
    by_label = {s.label: s for s in sources}
    cited = [lbl for lbl in find_cited_labels(answer) if lbl in by_label]
    chosen = [by_label[lbl] for lbl in cited] if cited else list(sources)
    focus = f"{question} {answer}"
    return [
        Citation(
            label=src.label,
            document_id=src.chunk.document_id,
            filename=src.chunk.filename,
            page_number=src.chunk.page_number,
            chunk_id=src.chunk.chunk_id,
            score=src.chunk.score,
            excerpt=best_excerpt(src.chunk.text, focus),
            explicitly_cited=bool(cited),
        )
        for src in chosen
    ]

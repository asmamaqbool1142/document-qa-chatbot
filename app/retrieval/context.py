"""Build a bounded, labelled context block from retrieved chunks."""

from __future__ import annotations

from app.models.domain import ContextSource, RetrievedChunk

_TRUNCATION_MARK = " [...]"
_MIN_USEFUL_CHARS = 200


def build_context(chunks: list[RetrievedChunk], max_chars: int) -> list[ContextSource]:
    """Label chunks S1..Sn in rank order until the character budget is used.

    The budget protects the LLM's context window and keeps latency/cost
    predictable. The last chunk is truncated rather than dropped if a useful
    amount of it still fits.
    """
    sources: list[ContextSource] = []
    remaining = max_chars
    for chunk in chunks:
        if remaining <= 0:
            break
        text = chunk.text
        if len(text) > remaining:
            if remaining < _MIN_USEFUL_CHARS:
                break
            text = text[: remaining - len(_TRUNCATION_MARK)].rstrip() + _TRUNCATION_MARK
        sources.append(ContextSource(label=f"S{len(sources) + 1}", chunk=chunk, text=text))
        remaining -= len(text)
    return sources


def describe_location(chunk: RetrievedChunk) -> str:
    if chunk.page_number is not None:
        return f"{chunk.filename}, page {chunk.page_number}"
    return chunk.filename

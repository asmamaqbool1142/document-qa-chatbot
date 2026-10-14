"""Split page text into overlapping chunks.

Design choices:
- Chunks never cross a page boundary, so every chunk has exactly one page
  number and citations can't point at the wrong page.
- Sizes are measured in characters (cheap, model-independent). 1000 chars is
  roughly 200-250 English tokens, well inside all-MiniLM-L6-v2's 256-token
  window when using the default settings.
- We prefer to cut at a paragraph break, then a sentence end, then a space, so
  chunks are readable and don't split words.
- Overlap repeats the tail of the previous chunk so a fact that straddles a
  boundary is still retrievable from at least one chunk.
"""

from __future__ import annotations

from app.models.domain import Chunk, PageText

# Break markers in order of preference; each is searched for in the back half
# of the window.
_BREAKS = ("\n\n", ". ", "? ", "! ", "\n", "; ", ", ", " ")


def make_chunk_id(document_id: str, chunk_index: int) -> str:
    """Deterministic: the same document always produces the same chunk IDs."""
    return f"{document_id}-{chunk_index:05d}"


def _find_break(text: str, start: int, hard_end: int) -> int:
    """Return the best end position in (start, hard_end]."""
    if hard_end >= len(text):
        return len(text)
    min_end = start + (hard_end - start) // 2
    for marker in _BREAKS:
        pos = text.rfind(marker, min_end, hard_end)
        if pos != -1:
            return pos + len(marker)
    return hard_end  # no natural break: cut mid-word as a last resort


def _next_start(text: str, end: int, overlap: int, prev_start: int) -> int:
    if overlap <= 0:
        start = end
    else:
        start = max(end - overlap, prev_start + 1)
        # Begin on a word boundary so the overlap doesn't start mid-word.
        space = text.find(" ", start, end)
        if space != -1 and text[start - 1 : start] not in (" ", "\n", ""):
            start = space + 1
    while start < len(text) and text[start] in " \n":
        start += 1
    return start


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int]]:
    """Return (start, end) character spans covering ``text``."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        end = _find_break(text, start, start + chunk_size)
        if text[start:end].strip():
            spans.append((start, end))
        if end >= len(text):
            break
        start = _next_start(text, end, chunk_overlap, start)
    return spans


def chunk_pages(
    pages: list[PageText], document_id: str, chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        for start, end in split_text(page.text, chunk_size, chunk_overlap):
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(document_id, index),
                    document_id=document_id,
                    chunk_index=index,
                    text=page.text[start:end].strip(),
                    page_number=page.page_number,
                    char_start=start,
                    char_end=end,
                )
            )
    return chunks

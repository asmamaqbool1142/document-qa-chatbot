"""Text normalisation applied to every extracted page before chunking."""

from __future__ import annotations

import re
import unicodedata

_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")
_SPACES = re.compile(r"[ \t\f\v ]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f]")
# A line break inside a sentence (typical of PDF line wrapping): the line does not
# end with sentence punctuation and the next line starts with a lowercase letter.
_SOFT_WRAP = re.compile(r"(?<=[^.!?:;\n])\n(?=[a-z(])")


def normalize_text(text: str) -> str:
    """Make extracted text consistent without changing its meaning.

    - NFKC unicode normalisation (ligatures like "ﬁ" become "fi")
    - unify line endings, drop control characters
    - re-join words hyphenated across line breaks ("infor-\\nmation")
    - collapse runs of spaces and of blank lines
    - re-join lines that were wrapped mid-sentence by the PDF layout
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)
    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _SOFT_WRAP.sub(" ", text)
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()

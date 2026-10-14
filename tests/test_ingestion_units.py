"""Validation, extraction, normalisation and chunking."""

from __future__ import annotations

import itertools

import pytest

from app.core.errors import (
    FileTooLargeError,
    InvalidDocumentError,
    NoExtractableTextError,
    UnsupportedFileTypeError,
)
from app.ingestion.chunking import chunk_pages, make_chunk_id, split_text
from app.ingestion.extractors import extract_pdf, extract_text, extract_txt
from app.ingestion.normalization import normalize_text
from app.ingestion.validation import detect_file_type, sanitize_filename, validate_content
from app.models.domain import FileType, PageText
from tests.conftest import make_pdf

# --- validation -------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("report.pdf", "report.pdf"),
        ("../../etc/passwd.txt", "passwd.txt"),
        ("C:\\Users\\me\\secret.txt", "secret.txt"),
        ("we<ird>|name?.txt", "we_ird__name_.txt"),
        ("tab\tname.txt", "tab_name.txt"),
    ],
)
def test_sanitize_filename_strips_paths_and_unsafe_chars(raw: str, expected: str) -> None:
    assert sanitize_filename(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "..", "/"])
def test_sanitize_filename_rejects_empty(raw: str | None) -> None:
    with pytest.raises(InvalidDocumentError):
        sanitize_filename(raw)


def test_sanitize_filename_truncates_long_names_keeping_extension() -> None:
    name = sanitize_filename("a" * 500 + ".pdf")
    assert len(name) <= 200 and name.endswith(".pdf")


def test_detect_file_type() -> None:
    assert detect_file_type("A.PDF") is FileType.PDF
    assert detect_file_type("notes.txt") is FileType.TXT
    for bad in ("image.png", "script.py", "noext", "archive.pdf.exe"):
        with pytest.raises(UnsupportedFileTypeError):
            detect_file_type(bad)


def test_validate_content_rules() -> None:
    with pytest.raises(InvalidDocumentError):
        validate_content(b"", FileType.TXT, 100)
    with pytest.raises(FileTooLargeError):
        validate_content(b"x" * 101, FileType.TXT, 100)
    with pytest.raises(InvalidDocumentError, match="does not look like a PDF"):
        validate_content(b"hello world", FileType.PDF, 100)
    with pytest.raises(InvalidDocumentError, match="binary"):
        validate_content(b"abc\x00def", FileType.TXT, 100)
    validate_content(b"%PDF-1.7 ...", FileType.PDF, 100)  # no error
    validate_content(b"\xff\xfeh\x00i\x00", FileType.TXT, 100)  # UTF-16 BOM allowed


# --- extraction ---------------------------------------------------------------


def test_extract_txt_utf8_and_bom() -> None:
    doc = extract_txt("\ufeffCafé naïve résumé — ok".encode())
    assert doc.pages[0].text == "Café naïve résumé — ok"
    assert doc.pages[0].page_number is None
    assert doc.warnings == []


def test_extract_txt_utf16() -> None:
    doc = extract_txt("Hello in UTF-16 text".encode("utf-16"))
    assert doc.pages[0].text == "Hello in UTF-16 text"


def test_extract_txt_few_invalid_bytes_are_replaced_with_warning() -> None:
    data = b"A long line of perfectly normal text that is mostly valid " * 3 + b"\xff end"
    doc = extract_txt(data)
    assert "\ufffd" in doc.pages[0].text
    assert doc.warnings and "UTF-8" in doc.warnings[0]


def test_extract_txt_mostly_invalid_is_rejected() -> None:
    with pytest.raises(InvalidDocumentError, match="UTF-8"):
        extract_txt(bytes(range(128, 256)) * 4)


def test_extract_pdf_keeps_page_numbers_and_skips_empty_pages() -> None:
    data = make_pdf(["First page about solar panels.", "", "Third page about batteries."])
    doc = extract_pdf(data)
    assert doc.page_count == 3
    assert [p.page_number for p in doc.pages] == [1, 3]
    assert "solar" in doc.pages[0].text
    assert doc.warnings and "1 page(s) had no extractable text" in doc.warnings[0]


def test_extract_pdf_rejects_corrupt_file() -> None:
    with pytest.raises(InvalidDocumentError, match="corrupt"):
        extract_pdf(b"%PDF-1.7\nthis is not really a pdf")


def test_extract_pdf_page_limit() -> None:
    with pytest.raises(InvalidDocumentError, match="limit"):
        extract_pdf(make_pdf(["a", "b", "c"]), max_pages=2)


def test_pdf_without_text_is_rejected_with_ocr_hint() -> None:
    with pytest.raises(NoExtractableTextError, match="OCR is not supported"):
        extract_text(make_pdf(["", ""]), FileType.PDF)


def test_whitespace_only_txt_is_rejected() -> None:
    with pytest.raises(NoExtractableTextError):
        extract_text(b"   \n\n\t  ", FileType.TXT)


def test_sample_pdf_extraction(manual_pdf_bytes: bytes) -> None:
    doc = extract_text(manual_pdf_bytes, FileType.PDF)
    assert doc.page_count == 5
    assert "every 24 months" in doc.pages[2].text


# --- normalisation ------------------------------------------------------------


def test_normalize_text() -> None:
    raw = "ﬁle  with   spaces.\r\nInfor-\nmation and a line\nwrapped here\n\n\n\nNext.\x07"
    assert normalize_text(raw) == "file with spaces.\nInformation and a line wrapped here\n\nNext."


def test_normalize_keeps_list_lines_separate() -> None:
    assert normalize_text("Stage 1: done.\nStage 2: next.") == "Stage 1: done.\nStage 2: next."


# --- chunking -----------------------------------------------------------------


def test_split_text_covers_text_with_overlap_and_word_boundaries() -> None:
    text = " ".join(f"word{i}" for i in range(400))
    spans = split_text(text, chunk_size=200, chunk_overlap=40)
    assert len(spans) > 5
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    for (s1, e1), (s2, _e2) in itertools.pairwise(spans):
        assert e1 - s1 <= 200
        assert s2 < e1, "consecutive chunks should overlap"
        assert text[s2 - 1] == " ", "chunks should start at a word boundary"


def test_split_text_prefers_sentence_breaks() -> None:
    text = "First sentence is here. " * 20
    for start, end in split_text(text, chunk_size=100, chunk_overlap=0):
        assert text[start:end].rstrip().endswith(".")


def test_split_text_without_spaces_still_terminates() -> None:
    spans = split_text("x" * 1050, chunk_size=100, chunk_overlap=10)
    assert spans[-1][1] == 1050


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (100, 100), (100, -1)])
def test_split_text_rejects_bad_params(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        split_text("abc", size, overlap)


def test_chunk_pages_metadata_and_stable_ids() -> None:
    pages = [PageText(1, "Alpha " * 100), PageText(2, "Beta " * 100)]
    chunks = chunk_pages(pages, "doc_x", chunk_size=200, chunk_overlap=20)
    again = chunk_pages(pages, "doc_x", chunk_size=200, chunk_overlap=20)
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in again]
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].chunk_id == make_chunk_id("doc_x", 0) == "doc_x-00000"
    assert {c.page_number for c in chunks} == {1, 2}
    # chunks never mix pages
    assert all(("Alpha" in c.text) != ("Beta" in c.text) for c in chunks)

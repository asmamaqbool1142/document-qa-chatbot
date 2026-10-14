"""Text extraction for PDF and TXT files.

Extraction works on in-memory bytes so we never need to trust or write the
user's original filename. Scanned (image-only) PDFs are *not* supported: there
is no OCR step, and such files are rejected with a clear message.
"""

from __future__ import annotations

import logging
from typing import Any

import pymupdf

from app.core.errors import InvalidDocumentError, NoExtractableTextError
from app.ingestion.normalization import normalize_text
from app.models.domain import ExtractedDocument, FileType, PageText

logger = logging.getLogger(__name__)

# If more than this share of characters had to be replaced while decoding, the
# file is almost certainly not text in a supported encoding.
MAX_REPLACEMENT_RATIO = 0.02
MIN_CHARS_PER_DOCUMENT = 20


def extract_text(data: bytes, file_type: FileType, max_pages: int = 500) -> ExtractedDocument:
    doc = extract_pdf(data, max_pages=max_pages) if file_type is FileType.PDF else extract_txt(data)
    if doc.total_chars < MIN_CHARS_PER_DOCUMENT:
        raise NoExtractableTextError(
            "No extractable text was found in this document. If it is a scanned PDF, "
            "note that OCR is not supported; please upload a text-based PDF or a .txt file."
        )
    return doc


def extract_pdf(data: bytes, max_pages: int = 500) -> ExtractedDocument:
    try:
        # PyMuPDF ships only partial type hints, so treat the document as Any.
        pdf: Any = pymupdf.open(stream=data, filetype="pdf")  # type: ignore[no-untyped-call]
    except Exception as exc:  # PyMuPDF raises several exception types for bad files
        logger.info("PDF could not be opened: %s", type(exc).__name__)
        raise InvalidDocumentError("The PDF file is corrupt or could not be read.") from exc

    with pdf:
        if pdf.needs_pass:
            raise InvalidDocumentError("Password-protected PDFs are not supported.")
        if pdf.page_count == 0:
            raise InvalidDocumentError("The PDF contains no pages.")
        if pdf.page_count > max_pages:
            raise InvalidDocumentError(
                f"The PDF has {pdf.page_count} pages; the limit is {max_pages}."
            )

        pages: list[PageText] = []
        empty_pages: list[int] = []
        for index, page in enumerate(pdf, start=1):
            try:
                raw = page.get_text("text", sort=True)
            except Exception as exc:
                raise InvalidDocumentError(
                    f"Text could not be extracted from page {index}; the PDF may be damaged."
                ) from exc
            text = normalize_text(raw)
            if text:
                pages.append(PageText(page_number=index, text=text))
            else:
                empty_pages.append(index)

        warnings: list[str] = []
        if empty_pages and pages:
            shown = ", ".join(map(str, empty_pages[:10]))
            more = "..." if len(empty_pages) > 10 else ""
            warnings.append(
                f"{len(empty_pages)} page(s) had no extractable text and were skipped "
                f"(pages {shown}{more}). They may be images or scans; OCR is not supported."
            )
        return ExtractedDocument(
            file_type=FileType.PDF, pages=pages, page_count=pdf.page_count, warnings=warnings
        )


def _decode_text(data: bytes) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16"), warnings
        except UnicodeDecodeError as exc:
            raise InvalidDocumentError("The UTF-16 text file could not be decoded.") from exc
    try:
        return data.decode("utf-8-sig"), warnings
    except UnicodeDecodeError:
        pass
    # Not valid UTF-8. Decode leniently, but refuse files that are mostly garbage.
    text = data.decode("utf-8", errors="replace")
    replaced = text.count("�")
    if replaced / max(len(text), 1) > MAX_REPLACEMENT_RATIO:
        raise InvalidDocumentError(
            "The text file is not valid UTF-8. Please save it as UTF-8 and upload it again."
        )
    warnings.append(f"{replaced} byte sequence(s) were not valid UTF-8 and were replaced with '�'.")
    return text, warnings


def extract_txt(data: bytes) -> ExtractedDocument:
    text, warnings = _decode_text(data)
    normalized = normalize_text(text)
    pages = [PageText(page_number=None, text=normalized)] if normalized else []
    return ExtractedDocument(file_type=FileType.TXT, pages=pages, page_count=1, warnings=warnings)

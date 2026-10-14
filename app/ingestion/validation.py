"""Upload validation: extension, size, magic bytes and filename sanitising.

We never use the client-supplied filename as a filesystem path. It is cleaned
for display only; any file we write gets an internal name derived from the
document ID (see ``app.services.ingestion_service``).
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import PurePosixPath, PureWindowsPath

from app.core.errors import FileTooLargeError, InvalidDocumentError, UnsupportedFileTypeError
from app.models.domain import FileType

ALLOWED_EXTENSIONS: dict[str, FileType] = {".pdf": FileType.PDF, ".txt": FileType.TXT}
MAX_FILENAME_LENGTH = 200
_UNSAFE_CHARS = re.compile(r"[\x00-\x1f\x7f<>:\"|?*\\/]")


def sanitize_filename(raw: str | None) -> str:
    """Return a display-safe filename: no directories, control chars or odd unicode."""
    if not raw:
        raise InvalidDocumentError("The uploaded file has no filename.")
    # Strip any directory component, whether the client used / or \.
    name = PureWindowsPath(PurePosixPath(raw).name).name
    name = unicodedata.normalize("NFKC", name)
    name = _UNSAFE_CHARS.sub("_", name).strip(" .")
    if not name:
        raise InvalidDocumentError("The uploaded filename is not valid.")
    if len(name) > MAX_FILENAME_LENGTH:
        stem, dot, ext = name.rpartition(".")
        name = (stem[: MAX_FILENAME_LENGTH - len(ext) - 1] + dot + ext) if dot else name[:200]
    return name


def detect_file_type(filename: str) -> FileType:
    suffix = PurePosixPath(filename.lower()).suffix
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix or '(none)'}'. Allowed types: {allowed}."
        )
    return ALLOWED_EXTENSIONS[suffix]


def validate_content(data: bytes, file_type: FileType, max_bytes: int) -> None:
    if len(data) == 0:
        raise InvalidDocumentError("The uploaded file is empty.")
    if len(data) > max_bytes:
        raise FileTooLargeError(
            f"File exceeds the maximum upload size of {max_bytes / (1024 * 1024):.1f} MB."
        )
    if file_type is FileType.PDF and not data.lstrip()[:5].startswith(b"%PDF-"):
        raise InvalidDocumentError(
            "The file has a .pdf extension but does not look like a PDF document."
        )
    has_utf16_bom = data[:2] in (b"\xff\xfe", b"\xfe\xff")
    if file_type is FileType.TXT and not has_utf16_bom and b"\x00" in data[:8192]:
        raise InvalidDocumentError(
            "The .txt file appears to contain binary data. Only plain text is supported."
        )

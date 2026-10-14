"""Domain exceptions.

Services raise these; the API layer maps each one to an HTTP status code and a
client-safe message. Messages here must never contain document text or secrets.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for expected, client-reportable errors."""

    status_code = 400
    code = "bad_request"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidDocumentError(AppError):
    status_code = 422
    code = "invalid_document"


class UnsupportedFileTypeError(AppError):
    status_code = 415
    code = "unsupported_file_type"


class FileTooLargeError(AppError):
    status_code = 413
    code = "file_too_large"


class NoExtractableTextError(AppError):
    status_code = 422
    code = "no_extractable_text"


class DuplicateDocumentError(AppError):
    status_code = 409
    code = "duplicate_document"

    def __init__(self, message: str, document_id: str) -> None:
        super().__init__(message)
        self.document_id = document_id


class DocumentNotFoundError(AppError):
    status_code = 404
    code = "document_not_found"


class ConfigurationError(AppError):
    """A component is misconfigured (e.g. missing API key)."""

    status_code = 503
    code = "configuration_error"


class ProviderUnavailableError(AppError):
    """The LLM or embedding backend could not be reached or failed."""

    status_code = 503
    code = "provider_unavailable"


class EmbeddingModelMismatchError(ConfigurationError):
    code = "embedding_model_mismatch"

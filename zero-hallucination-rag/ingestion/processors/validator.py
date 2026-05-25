"""
Document validation.

Checks incoming documents against configurable acceptance criteria
(file size, supported MIME types, non-empty content) before they enter
the ingestion pipeline.
"""

from __future__ import annotations

from typing import Sequence

from core.config import IngestionConfig
from core.constants import SUPPORTED_MIME_TYPES
from ingestion.interfaces import DocumentValidator, RawDocument
from utils.helpers import file_extension


class DefaultDocumentValidator(DocumentValidator):
    """
    Validates documents against size, type, and content rules.

    Parameters
    ----------
    config:
        Ingestion configuration containing limits and allowed extensions.
    extra_mime_types:
        Additional MIME types to accept beyond the built-in set.
    """

    def __init__(
        self,
        config: IngestionConfig,
        *,
        extra_mime_types: Sequence[str] = (),
    ) -> None:
        self._max_size_bytes = int(config.max_document_size_mb * 1024 * 1024)
        self._allowed_extensions = set(config.supported_extensions)
        self._allowed_mimes = SUPPORTED_MIME_TYPES | frozenset(extra_mime_types)

    def validate(self, document: RawDocument) -> bool:
        return len(self.validation_errors(document)) == 0

    def validation_errors(self, document: RawDocument) -> list[str]:
        errors: list[str] = []

        # Check content is non-empty.
        if not document.content:
            errors.append("Document content is empty.")

        # Check size limit.
        size = (
            len(document.content)
            if isinstance(document.content, bytes)
            else len(document.content.encode("utf-8"))
        )
        if size > self._max_size_bytes:
            errors.append(
                f"Document size ({size:,} bytes) exceeds maximum "
                f"({self._max_size_bytes:,} bytes)."
            )

        # Check MIME type.
        if document.mime_type and document.mime_type not in self._allowed_mimes:
            errors.append(
                f"Unsupported MIME type: {document.mime_type}. "
                f"Allowed: {sorted(self._allowed_mimes)}"
            )

        # Check file extension (if inferrable from source).
        ext = file_extension(document.source)
        if ext and ext not in self._allowed_extensions:
            errors.append(
                f"Unsupported file extension: {ext}. "
                f"Allowed: {sorted(self._allowed_extensions)}"
            )

        return errors

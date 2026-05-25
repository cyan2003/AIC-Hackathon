"""
Metadata extraction from raw documents.

Extracts structural and statistical metadata (word count, language hints,
file attributes, etc.) that can be stored alongside embeddings for
filtered retrieval.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ingestion.interfaces import MetadataExtractor, RawDocument


class DefaultMetadataExtractor(MetadataExtractor):
    """
    Extracts common metadata fields from :class:`RawDocument` instances.

    Fields extracted
    ----------------
    - ``source``: originating file / URI
    - ``mime_type``: content type
    - ``content_hash``: SHA-256 digest of the raw content
    - ``char_count``: number of characters
    - ``word_count``: approximate word count (whitespace split)
    - ``line_count``: number of lines
    - ``file_extension``: lowercase extension (if source is a file path)
    - ``ingested_at``: ISO-8601 UTC timestamp
    """

    def extract(self, document: RawDocument) -> dict[str, Any]:
        text = (
            document.content
            if isinstance(document.content, str)
            else document.content.decode("utf-8", errors="replace")
        )

        content_bytes = (
            document.content
            if isinstance(document.content, bytes)
            else document.content.encode("utf-8")
        )

        metadata: dict[str, Any] = {
            "source": document.source,
            "mime_type": document.mime_type,
            "content_hash": hashlib.sha256(content_bytes).hexdigest(),
            "char_count": len(text),
            "word_count": len(text.split()),
            "line_count": text.count("\n") + 1,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }

        # File-specific metadata
        source_path = Path(document.source)
        if source_path.suffix:
            metadata["file_extension"] = source_path.suffix.lower()
        if os.path.isfile(document.source):
            stat = os.stat(document.source)
            metadata["file_size_bytes"] = stat.st_size
            metadata["file_modified_at"] = datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat()

        return metadata

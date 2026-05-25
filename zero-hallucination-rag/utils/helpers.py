"""
General-purpose utility functions used across the pipeline.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any, Iterable, Iterator, TypeVar

T = TypeVar("T")


def generate_document_id(content: str, source: str = "") -> str:
    """
    Deterministic document ID derived from content hash + source.

    Guarantees the same content from the same source always maps to
    the same identifier, enabling idempotent ingestion.
    """
    digest = hashlib.sha256(f"{source}:{content}".encode("utf-8")).hexdigest()
    return digest[:24]


def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    """Derive a deterministic chunk ID from a parent document ID and index."""
    return f"{document_id}:{chunk_index:06d}"


def generate_uuid() -> str:
    """Generate a random UUID-4 string."""
    return str(uuid.uuid4())


def batched(iterable: Iterable[T], n: int) -> Iterator[list[T]]:
    """
    Yield successive *n*-sized batches from *iterable*.

    >>> list(batched([1, 2, 3, 4, 5], 2))
    [[1, 2], [3, 4], [5]]
    """
    batch: list[T] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def file_extension(path: str | Path) -> str:
    """Return the lowercase file extension including the leading dot."""
    return Path(path).suffix.lower()


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate *text* to *max_length* characters, appending '…' if trimmed."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 1] + "…"


def flatten(nested: Iterable[Iterable[T]]) -> list[T]:
    """Flatten one level of nesting."""
    return [item for sublist in nested for item in sublist]


def safe_get(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dicts."""
    current: Any = d
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current

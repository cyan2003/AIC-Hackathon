"""
Query preprocessing and expansion.

Cleans, normalises, and optionally expands user queries before they
are sent to search backends.
"""

from __future__ import annotations

import re

from retrieval.interfaces import QueryProcessor


class DefaultQueryProcessor(QueryProcessor):
    """
    Rule-based query processor.

    Steps applied
    -------------
    1. Strip leading/trailing whitespace
    2. Collapse multiple spaces
    3. Optionally lowercase
    4. Optionally remove stopwords (disabled by default)

    Parameters
    ----------
    lowercase : bool
        Convert query to lowercase.
    remove_stopwords : bool
        Remove common English stopwords.
    """

    _MULTI_SPACE = re.compile(r"\s+")

    # Minimal stopword set – extend as needed.
    _STOPWORDS: frozenset[str] = frozenset(
        {
            "a", "an", "and", "are", "as", "at", "be", "but", "by",
            "for", "if", "in", "into", "is", "it", "no", "not", "of",
            "on", "or", "such", "that", "the", "their", "then",
            "there", "these", "they", "this", "to", "was", "will", "with",
        }
    )

    def __init__(
        self,
        *,
        lowercase: bool = True,
        remove_stopwords: bool = False,
    ) -> None:
        self._lowercase = lowercase
        self._remove_stopwords = remove_stopwords

    def process(self, raw_query: str) -> str:
        text = raw_query.strip()
        text = self._MULTI_SPACE.sub(" ", text)

        if self._lowercase:
            text = text.lower()

        if self._remove_stopwords:
            tokens = text.split()
            tokens = [t for t in tokens if t not in self._STOPWORDS]
            text = " ".join(tokens)

        return text

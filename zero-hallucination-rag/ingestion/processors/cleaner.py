"""
Text cleaning and normalisation.

Strips boilerplate noise, normalises whitespace, and optionally removes
HTML artefacts so downstream chunking and embedding see clean text.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from ingestion.interfaces import DocumentCleaner


class DefaultTextCleaner(DocumentCleaner):
    """
    Rule-based text cleaner with configurable toggles.

    Parameters
    ----------
    lowercase : bool
        Convert text to lowercase (default ``False``).
    strip_html : bool
        Remove residual HTML tags (default ``True``).
    normalize_unicode : bool
        Apply NFC normalisation (default ``True``).
    min_line_length : int
        Drop lines shorter than this (set 0 to disable).
    """

    _MULTI_WHITESPACE = re.compile(r"[ \t]+")
    _MULTI_NEWLINE = re.compile(r"\n{3,}")
    _HTML_TAG = re.compile(r"<[^>]+>")
    _URL_PATTERN = re.compile(r"https?://\S+")

    def __init__(
        self,
        *,
        lowercase: bool = False,
        strip_html: bool = True,
        normalize_unicode: bool = True,
        min_line_length: int = 0,
        remove_urls: bool = False,
    ) -> None:
        self._lowercase = lowercase
        self._strip_html = strip_html
        self._normalize_unicode = normalize_unicode
        self._min_line_length = min_line_length
        self._remove_urls = remove_urls

    def clean(self, text: str) -> str:
        """Apply all cleaning rules and return the cleaned text."""
        if not text:
            return text

        if self._normalize_unicode:
            text = unicodedata.normalize("NFC", text)

        if self._strip_html:
            text = self._HTML_TAG.sub("", text)

        if self._remove_urls:
            text = self._URL_PATTERN.sub("", text)

        if self._lowercase:
            text = text.lower()

        # Normalise horizontal whitespace (preserve newlines).
        text = self._MULTI_WHITESPACE.sub(" ", text)

        # Collapse excessive blank lines.
        text = self._MULTI_NEWLINE.sub("\n\n", text)

        # Drop short lines if configured.
        if self._min_line_length > 0:
            lines = text.split("\n")
            lines = [l for l in lines if len(l.strip()) >= self._min_line_length]
            text = "\n".join(lines)

        return text.strip()

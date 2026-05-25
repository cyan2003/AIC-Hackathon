"""
Logging setup and utilities for the Zero-Hallucination RAG Pipeline.

Wraps ``loguru`` to provide structured, configurable logging with
sensible defaults (JSON in production, pretty-print in debug).
"""

from __future__ import annotations

import sys
from typing import Optional

from loguru import logger

# Remove default loguru handler so we can configure our own.
logger.remove()


def setup_logging(
    level: str = "INFO",
    *,
    json_format: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure application-wide logging.

    Parameters
    ----------
    level:
        Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    json_format:
        If ``True``, emit structured JSON log lines (useful in production).
    log_file:
        Optional file path for a rotating file sink.
    """
    fmt: str
    if json_format:
        fmt = "{message}"  # loguru serialize=True handles JSON
        logger.add(sys.stderr, level=level, serialize=True, enqueue=True)
    else:
        fmt = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
        logger.add(sys.stderr, level=level, format=fmt, enqueue=True)

    if log_file:
        logger.add(
            log_file,
            level=level,
            format=fmt,
            rotation="50 MB",
            retention="7 days",
            compression="gz",
            enqueue=True,
        )


def get_logger(name: str) -> "logger":
    """Return a contextualised logger bound to *name*."""
    return logger.bind(module=name)

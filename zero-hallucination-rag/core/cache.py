"""
SQLite-backed persistent cache for LLM assessments and text embeddings.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from typing import Any, Optional

from core.config import get_settings
from utils.logging import get_logger

logger = get_logger(__name__)

_cache_instance: Optional[SQLiteCache] = None


class SQLiteCache:
    """
    SQLite-backed key-value store.
    
    Provides thread-safe persistence using localized connection context managers
    and a SQLite busy timeout of 10.0 seconds to handle concurrent writes.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            settings = get_settings()
            db_path = settings.cache.db_path

        # If path is relative, resolve it relative to the zero-hallucination-rag directory
        if not os.path.isabs(db_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(base_dir, db_path)

        # Ensure target directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with busy timeout."""
        return sqlite3.connect(self.db_path, timeout=10.0)

    def _init_db(self) -> None:
        """Create tables if they do not exist."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS llm_cache (
                        key TEXT PRIMARY KEY,
                        value_json TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS embedding_cache (
                        key TEXT PRIMARY KEY,
                        text TEXT,
                        embedding_json TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
            logger.info("Persistent cache initialized successfully", path=self.db_path)
        except Exception as exc:
            logger.error("Failed to initialize persistent cache", error=str(exc))

    def _hash_key(self, data: Any) -> str:
        """Compute SHA-256 hash for a given key schema/text."""
        if isinstance(data, str):
            serialized = data
        else:
            # Sort keys to ensure stable serialization order
            serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # -- LLM Assessor Cache methods ---------------------------------------

    def get_llm(self, key_data: Any) -> Optional[dict[str, Any]]:
        """Retrieve a cached candidate assessment."""
        key = self._hash_key(key_data)
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT value_json FROM llm_cache WHERE key = ?", (key,)
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
        except Exception as exc:
            logger.warning("Error reading from LLM cache", error=str(exc))
        return None

    def set_llm(self, key_data: Any, value: dict[str, Any]) -> None:
        """Cache a candidate assessment."""
        key = self._hash_key(key_data)
        try:
            value_json = json.dumps(value)
            with self._get_connection() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO llm_cache (key, value_json) VALUES (?, ?)",
                    (key, value_json),
                )
                conn.commit()
        except Exception as exc:
            logger.warning("Error writing to LLM cache", error=str(exc))

    # -- Embedding Cache methods ------------------------------------------

    def get_embedding(self, text: str) -> Optional[list[float]]:
        """Retrieve a cached embedding vector."""
        key = self._hash_key(text)
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT embedding_json FROM embedding_cache WHERE key = ?",
                    (key,),
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
        except Exception as exc:
            logger.warning("Error reading from embedding cache", error=str(exc))
        return None

    def set_embedding(self, text: str, embedding: list[float]) -> None:
        """Cache an embedding vector."""
        key = self._hash_key(text)
        try:
            embedding_json = json.dumps(embedding)
            with self._get_connection() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO embedding_cache (key, text, embedding_json) VALUES (?, ?, ?)",
                    (key, text, embedding_json),
                )
                conn.commit()
        except Exception as exc:
            logger.warning("Error writing to embedding cache", error=str(exc))

    def clear(self) -> None:
        """Wipe both cache tables completely (primarily for testing)."""
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM llm_cache")
                conn.execute("DELETE FROM embedding_cache")
                conn.commit()
            logger.info("Persistent cache tables cleared")
        except Exception as exc:
            logger.warning("Error clearing cache", error=str(exc))


def get_cache_instance() -> SQLiteCache:
    """Get the global SQLiteCache singleton instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SQLiteCache()
    return _cache_instance

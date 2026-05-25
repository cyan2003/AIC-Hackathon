"""
Lexical store sub-package.
"""

from retrieval.lexical_store.factory import create_lexical_store
from retrieval.lexical_store.bm25 import BM25LexicalStore

__all__ = ["create_lexical_store", "BM25LexicalStore"]

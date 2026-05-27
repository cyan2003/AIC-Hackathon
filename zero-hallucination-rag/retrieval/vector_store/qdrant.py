"""
Qdrant vector store implementation.

Integrates with a Qdrant instance (local or cloud) to provide
high-performance approximate nearest-neighbour search using HNSW indices.

Supports two-collection architecture for resume/JD separation.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from core.config import VectorStoreConfig
from core.constants import (
    HNSW_EF_CONSTRUCT,
    HNSW_M,
    JD_COLLECTION_NAME,
    RESUME_COLLECTION_NAME,
)
from core.exceptions import VectorStoreConnectionError, VectorStoreError
from retrieval.interfaces import SearchResult, VectorStore
from utils.logging import get_logger

logger = get_logger(__name__)


class QdrantVectorStore(VectorStore):
    """
    Qdrant-backed vector store with two-collection architecture.

    Manages separate collections for resumes and job descriptions
    to enable cross-collection retrieval (embed resume → search JDs).

    Parameters
    ----------
    config:
        Vector store configuration (host, port, collection names, etc.).
    embedding_dimension:
        Dimensionality of the vectors to be stored.
    """

    def __init__(self, config: VectorStoreConfig, embedding_dimension: int) -> None:
        self._config = config
        self._dimension = embedding_dimension
        self._client = None  # Lazy-loaded

    # -- Lifecycle --------------------------------------------------------

    def _get_client(self):
        """Lazy-initialise the async Qdrant client."""
        if self._client is not None:
            return self._client

        try:
            from qdrant_client import AsyncQdrantClient

            self._client = AsyncQdrantClient(
                host=self._config.host,
                port=self._config.port,
                grpc_port=self._config.grpc_port,
                prefer_grpc=self._config.prefer_grpc,
                api_key=self._config.api_key,
                timeout=self._config.timeout,
            )
            logger.info(
                "Connected to Qdrant (Async)",
                host=self._config.host,
                port=self._config.port,
            )
            return self._client
        except ImportError as exc:
            raise VectorStoreError(
                "qdrant-client is required. Install with: pip install qdrant-client"
            ) from exc
        except Exception as exc:
            raise VectorStoreConnectionError(
                f"Failed to connect to Qdrant at "
                f"{self._config.host}:{self._config.port}: {exc}"
            ) from exc

    def _resolve_collection(self, collection: Optional[str] = None) -> str:
        """Resolve collection name, defaulting to the resume collection."""
        if collection:
            return collection
        return self._config.resume_collection or RESUME_COLLECTION_NAME

    async def initialize(self) -> None:
        """Create both resume and JD collections if they do not exist."""
        client = self._get_client()
        try:
            from qdrant_client.models import Distance, HnswConfigDiff, VectorParams

            # Await the async network call
            collections_response = await client.get_collections()
            existing = {c.name for c in collections_response.collections}

            for coll_name in (
                self._config.resume_collection or RESUME_COLLECTION_NAME,
                self._config.jd_collection or JD_COLLECTION_NAME,
            ):
                if coll_name not in existing:
                    # Await collection creation
                    await client.create_collection(
                        collection_name=coll_name,
                        vectors_config=VectorParams(
                            size=self._dimension,
                            distance=Distance.COSINE,
                        ),
                        hnsw_config=HnswConfigDiff(
                            m=HNSW_M,
                            ef_construct=HNSW_EF_CONSTRUCT,
                        ),
                    )
                    logger.info(
                        "Created Qdrant collection",
                        collection=coll_name,
                        dimension=self._dimension,
                    )
                else:
                    logger.info(
                        "Qdrant collection already exists",
                        collection=coll_name,
                    )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to initialize Qdrant collections: {exc}"
            ) from exc

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("Qdrant client closed")

    # -- CRUD -------------------------------------------------------------

    async def upsert(
        self,
        ids: Sequence[str],
        vectors: Sequence[list[float]],
        payloads: Optional[Sequence[dict[str, Any]]] = None,
        *,
        collection: Optional[str] = None,
    ) -> None:
        """
        Insert or update vectors.

        Parameters
        ----------
        collection:
            Target collection name. Defaults to the resume collection.
        """
        client = self._get_client()
        coll = self._resolve_collection(collection)
        try:
            from qdrant_client.models import PointStruct

            points = [
                PointStruct(
                    id=idx,
                    vector=vec,
                    payload=payloads[i] if payloads else {},
                )
                for i, (idx, vec) in enumerate(zip(ids, vectors))
            ]
            # Await the upsert operation
            await client.upsert(collection_name=coll, points=points)
            logger.debug("Upserted vectors", count=len(points), collection=coll)
        except Exception as exc:
            raise VectorStoreError(f"Qdrant upsert failed: {exc}") from exc

    async def search(
        self,
        vector: list[float],
        *,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
        collection: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Search for nearest neighbours.

        Parameters
        ----------
        collection:
            Target collection name. Defaults to the resume collection.
            For resume→JD matching, pass the JD collection name.
        """
        client = self._get_client()
        coll = self._resolve_collection(collection)
        try:
            # Await the search/query operations
            if hasattr(client, "query_points"):
                response = await client.query_points(
                    collection_name=coll,
                    query=vector,
                    limit=top_k,
                    query_filter=self._build_filter(filters) if filters else None,
                )
                hits = response.points
            else:
                hits = await client.search(
                    collection_name=coll,
                    query_vector=vector,
                    limit=top_k,
                    query_filter=self._build_filter(filters) if filters else None,
                )
            results: list[SearchResult] = []
            for hit in hits:
                payload = hit.payload or {}
                results.append(
                    SearchResult(
                        document_id=payload.get("document_id", ""),
                        chunk_id=str(hit.id),
                        text=payload.get("text", ""),
                        score=hit.score,
                        metadata=payload,
                        source="vector",
                    )
                )
            return results
        except Exception as exc:
            raise VectorStoreError(f"Qdrant search failed: {exc}") from exc

    async def delete(
        self,
        ids: Sequence[str],
        *,
        collection: Optional[str] = None,
    ) -> None:
        client = self._get_client()
        coll = self._resolve_collection(collection)
        try:
            from qdrant_client.models import PointIdsList

            # Await the delete operation
            await client.delete(
                collection_name=coll,
                points_selector=PointIdsList(points=list(ids)),
            )
            logger.debug("Deleted vectors", count=len(ids), collection=coll)
        except Exception as exc:
            raise VectorStoreError(f"Qdrant delete failed: {exc}") from exc

    async def count(self, *, collection: Optional[str] = None) -> int:
        client = self._get_client()
        coll = self._resolve_collection(collection)
        try:
            # Await the get_collection operation
            info = await client.get_collection(coll)
            return info.points_count or 0
        except Exception as exc:
            raise VectorStoreError(f"Qdrant count failed: {exc}") from exc

    # -- Convenience methods for domain-specific access -------------------

    async def search_jds(
        self,
        vector: list[float],
        *,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Search the job descriptions collection (for resume→JD matching)."""
        jd_collection = self._config.jd_collection or JD_COLLECTION_NAME
        return await self.search(
            vector, top_k=top_k, filters=filters, collection=jd_collection
        )

    async def upsert_resumes(
        self,
        ids: Sequence[str],
        vectors: Sequence[list[float]],
        payloads: Optional[Sequence[dict[str, Any]]] = None,
    ) -> None:
        """Upsert into the resumes collection."""
        resume_collection = self._config.resume_collection or RESUME_COLLECTION_NAME
        await self.upsert(ids, vectors, payloads, collection=resume_collection)

    async def upsert_jds(
        self,
        ids: Sequence[str],
        vectors: Sequence[list[float]],
        payloads: Optional[Sequence[dict[str, Any]]] = None,
    ) -> None:
        """Upsert into the job descriptions collection."""
        jd_collection = self._config.jd_collection or JD_COLLECTION_NAME
        await self.upsert(ids, vectors, payloads, collection=jd_collection)

    # -- Helpers ----------------------------------------------------------

    @staticmethod
    def _build_filter(filters: dict[str, Any]):
        """
        Convert a MatchFilters dict to a complex Qdrant ``Filter``.
        """
        from qdrant_client.models import FieldCondition as RealFieldCondition, MatchAny, MatchValue, Range, Filter
        try:
            from qdrant_client.models import IsEmpty
        except ImportError:
            from dataclasses import dataclass
            @dataclass
            class IsEmpty:
                key: str

        from typing import Union
        class FieldCondition(RealFieldCondition):
            is_empty: Union[bool, IsEmpty, None] = None

        must_conditions = []

        # 1. Skills filter (MatchAny)
        if "skills" in filters and filters["skills"]:
            must_conditions.append(
                FieldCondition(
                    key="required_skills",
                    match=MatchAny(any=[s.lower() for s in filters["skills"]])
                )
            )

        # 2. Experience years filter: JD's experience requirement must be <= candidate_experience
        if "experience_years_max" in filters and filters["experience_years_max"] is not None:
            max_exp = filters["experience_years_max"]
            must_conditions.append(
                Filter(
                    should=[
                        FieldCondition(key="experience_years", range=Range(lte=max_exp)),
                        FieldCondition(key="experience_years", is_empty=IsEmpty(key="experience_years")),
                    ]
                )
            )

        # 3. Education level filter: JD's education requirement must be <= candidate_education_level
        if "education_level" in filters and filters["education_level"]:
            EDUCATION_ORDER = {"associate": 1, "bachelor": 2, "master": 3, "phd": 4}
            candidate_edu = filters["education_level"].lower()
            if candidate_edu in EDUCATION_ORDER:
                candidate_rank = EDUCATION_ORDER[candidate_edu]
                acceptable_levels = [
                    lvl for lvl, rank in EDUCATION_ORDER.items() if rank <= candidate_rank
                ]
                must_conditions.append(
                    Filter(
                        should=[
                            FieldCondition(key="education_level", match=MatchAny(any=acceptable_levels)),
                            FieldCondition(key="education_level", is_empty=IsEmpty(key="education_level")),
                        ]
                    )
                )

        # 4. Industry filter (exact match)
        if "industry" in filters and filters["industry"]:
            must_conditions.append(
                FieldCondition(
                    key="industry",
                    match=MatchValue(value=filters["industry"].lower())
                )
            )

        # 5. Handle any other basic key-value filters that might be passed
        for k, v in filters.items():
            if k not in ["skills", "experience_years_max", "education_level", "industry"]:
                must_conditions.append(
                    FieldCondition(key=k, match=MatchValue(value=v))
                )

        return Filter(must=must_conditions) if must_conditions else None
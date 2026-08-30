"""
vector_store.py

Qdrant-backed vector store for the RAG pipeline.

Consumes the `Chunk` contract (from Team 1's chunker) and exposes:
  - ensure_collection()   -> idempotent hybrid (dense+sparse) collection setup
  - ingest_chunks()       -> embed + upsert (handles re-ingestion cleanly)
  - update_document()     -> convenience wrapper: replace all chunks for one document_id
  - delete_document()     -> remove all chunks belonging to a document_id
  - retrieve()            -> hybrid dense+sparse search with RRF fusion

This module does NOT import anything from the chunker package.
It only depends on the `Chunk` field names listed in the RAG contract.
"""

import uuid
import asyncio
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Optional
from dotenv import load_dotenv

load_dotenv()

store = None

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)


# --------------------------------------------------------------------------
# Data contracts
# --------------------------------------------------------------------------

@dataclass
class Chunk:
    """Mirrors Team 1's chunker output. Defined here only so this module is
    runnable/testable standalone -- in the integrated codebase, import the
    real Chunk type from Team 1's package instead of this one."""
    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    char_start: int
    char_end: int
    content_hash: str
    title: str
    url: str
    publisher: str
    author: str
    source_type: str
    language: str
    metadata: dict = field(default_factory=dict)
    published_at: Optional[str] = None
    section_id: Optional[str] = None
    section_heading: Optional[str] = None


@dataclass
class RetrievalResult:
    chunk_id: str
    document_id: str
    content: str
    score: float
    title: str
    url: str
    publisher: str
    author: str
    source_type: str
    published_at: Optional[str] = None
    section_heading: Optional[str] = None


# --------------------------------------------------------------------------
# BGE-M3 embedder (dense + sparse), wrapped for async use
# --------------------------------------------------------------------------

class BgeM3Embedder:
    """Wraps FlagEmbedding's BGEM3FlagModel. .encode() is blocking, so every
    call is pushed to a thread executor to avoid stalling the event loop."""

    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True):
        from FlagEmbedding import BGEM3FlagModel  # imported lazily
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)

    def _encode_sync(self, texts: list[str], batch_size: int) -> dict:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

    async def encode(self, texts: list[str], batch_size: int = 12) -> dict:
        loop = asyncio.get_event_loop()
        fn = partial(self._encode_sync, texts, batch_size)
        return await loop.run_in_executor(None, fn)

    @staticmethod
    def to_sparse_vector(lexical_weights: dict) -> SparseVector:
        return SparseVector(
            indices=list(lexical_weights.keys()),
            values=list(lexical_weights.values()),
        )


# --------------------------------------------------------------------------
# Vector store
# --------------------------------------------------------------------------

class VectorStore:
    DENSE_SIZE = 1024  # BGE-M3 dense output dimension
    ID_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-4789-a0b1-c2d3e4f56789")
    
    def __init__(
        self,
        client: AsyncQdrantClient,
        embedder: BgeM3Embedder,
        collection_name: str = "pib_chunks",
    ):
        self.client = client
        self.embedder = embedder
        self.collection_name = collection_name

    @classmethod
    def _point_id(cls, chunk_id: str) -> str:
        return uuid.uuid5(cls.ID_NAMESPACE, chunk_id)

    # ---- setup -----------------------------------------------------------

    async def ensure_collection(self) -> None:
        """Idempotent. Creates the hybrid collection if it doesn't exist.
        If it exists but has the wrong dense dimension, recreates it."""
        exists = await self.client.collection_exists(self.collection_name)

        import os
        print(os.getenv("CLUSTER_ENDPOINT"))

        if exists:
            info = await self.client.get_collection(self.collection_name)
            current_size = info.config.params.vectors["dense"].size
            if current_size == self.DENSE_SIZE:
                return
            print(
                f"[VectorStore] Dim mismatch (existing={current_size}, "
                f"wanted={self.DENSE_SIZE}) -- recreating collection."
            )
            await self.client.delete_collection(self.collection_name)

        await self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=self.DENSE_SIZE, distance=Distance.COSINE, on_disk=True
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )

        # Payload indexes for the filter fields called out in the contract
        for field_name in ("document_id", "source_type", "language"):
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema="keyword",
            )

    # ---- ingestion ---------------------------------------------------------

    async def ingest_chunks(self, chunks: list[Chunk]) -> None:
        """Embeds chunk.content only. Deletes any existing chunks for each
        document_id present in this batch before upserting, so re-processed
        documents (fewer/more/reshuffled chunks) never leave stale points
        behind."""
        if not chunks:
            return

        doc_ids = {c.document_id for c in chunks}
        for doc_id in doc_ids:
            await self.delete_document(doc_id)

        contents = [c.content for c in chunks]
        output = await self.embedder.encode(contents)

        points = []
        for chunk, dense_vec, lexical_weights in zip(
            chunks, output["dense_vecs"], output["lexical_weights"]
        ):
            points.append(
                PointStruct(
                    id=self._point_id(chunk.chunk_id),
                    vector={
                        "dense": dense_vec,
                        "sparse": self.embedder.to_sparse_vector(lexical_weights),
                    },
                    payload={
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "content": chunk.content,
                        "chunk_id":chunk.chunk_id,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                        "content_hash": chunk.content_hash,
                        "title": chunk.title,
                        "url": chunk.url,
                        "publisher": chunk.publisher,
                        "author": chunk.author,
                        "published_at": chunk.published_at,
                        "source_type": chunk.source_type,
                        "language": chunk.language,
                        "metadata": chunk.metadata,
                        "section_id": chunk.section_id,
                        "section_heading": chunk.section_heading,
                    },
                )
            )

        await self.client.upsert(collection_name=self.collection_name, points=points)

    async def update_document(self, chunks: list[Chunk]) -> None:
        """Convenience alias -- semantically this IS an update, since
        ingest_chunks already deletes-then-upserts per document_id. Kept as
        a separate name so calling code reads intent clearly."""
        await self.ingest_chunks(chunks)

    async def get_existing_chunks(self, document_id: str) -> list[Any]:
        """Plain payload-filter lookup (not vector search) -- used to diff
        content_hash values against freshly generated chunks before deciding
        what actually needs re-embedding/updating."""
        points, _ = await self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id", match=MatchValue(value=document_id)
                    )
                ]
            ),
            with_payload=True,
            limit=1000,
        )
        return points

    async def delete_document(self, document_id: str) -> None:
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id", match=MatchValue(value=document_id)
                    )
                ]
            ),
        )

    # ---- retrieval ---------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        source_type: Optional[str] = None,
        language: Optional[str] = None,
        prefetch_limit: int = 20,
    ) -> list[RetrievalResult]:
        output = await self.embedder.encode([query])
        dense_vec = output["dense_vecs"][0]
        sparse_vec = self.embedder.to_sparse_vector(output["lexical_weights"][0])

        query_filter = self._build_filter(source_type, language)

        results = await self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=dense_vec,
                    using="dense",
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
                Prefetch(
                    query=sparse_vec,
                    using="sparse",
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        return [self._to_retrieval_result(p) for p in results.points]

    @staticmethod
    def _build_filter(
        source_type: Optional[str], language: Optional[str]
    ) -> Optional[Filter]:
        conditions = []
        if source_type is not None:
            conditions.append(
                FieldCondition(key="source_type", match=MatchValue(value=source_type))
            )
        if language is not None:
            conditions.append(
                FieldCondition(key="language", match=MatchValue(value=language))
            )
        return Filter(must=conditions) if conditions else None

    @staticmethod
    def _to_retrieval_result(point: Any) -> RetrievalResult:
        p = point.payload
        return RetrievalResult(
            chunk_id=p["chunk_id"],
            document_id=p["document_id"],
            content=p["content"],
            score=point.score,  # RRF fused score, not raw cosine similarity
            title=p["title"],
            url=p["url"],
            publisher=p["publisher"],
            author=p["author"],
            source_type=p["source_type"],
            published_at=p.get("published_at"),
            section_heading=p.get("section_heading"),
        )

async def entry():
    import os
    client = AsyncQdrantClient(url=os.getenv("CLUSTER_ENDPOINT"), api_key=os.getenv("CLUSTER_API"), check_compatibility=False)
    embedder = BgeM3Embedder()
    global store
    store = VectorStore(client, embedder, collection_name="pib_chunks")
    
    await store.ensure_collection()

    # chunks = team1_chunker.chunk(document)  # -> list[Chunk]
    # await store.ingest_chunks(chunks)

    # results = await store.retrieve("airport closures May 2025", top_k=5)
    # for r in results:
    #     print(r.score, r.title, r.chunk_id)


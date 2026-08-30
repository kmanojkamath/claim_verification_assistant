"""
test_vector_store.py

Exercises VectorStore end-to-end against a running Qdrant instance:
  1. ensure_collection()
  2. ingest_chunks()      -> insert chunks for 2 fake documents
  3. retrieve()           -> confirm the right chunks come back
  4. update_document()    -> simulate a PIB release being amended,
                             confirm stale chunks are gone and new ones retrievable
  5. delete_document()    -> remove one document entirely
  6. CLEANUP              -> deletes every point this test created, and drops
                             the whole test collection at the end, regardless
                             of whether the test passed or failed.

Runs against a dedicated collection ("test_pib_chunks") so it never touches
your real corpus. Uses a FakeEmbedder by default (deterministic, hash-based
vectors) so you can run this without downloading/loading the actual BGE-M3
model -- swap in the real BgeM3Embedder (see USE_REAL_EMBEDDER below) if you
want a true end-to-end check including embedding quality.

Usage:
    python test_vector_store.py
"""

import asyncio
import hashlib
import sys
import os
from dotenv import load_dotenv
load_dotenv()

from qdrant_client import AsyncQdrantClient

from vector_store import Chunk, VectorStore

# Set to True to use the real BGE-M3 model instead of the fake embedder.
# Requires FlagEmbedding installed and the model downloaded/cached.
USE_REAL_EMBEDDER = False

QDRANT_URL = os.getenv("CLUSTER_ENDPOINT")
KEY = os.getenv("CLUSTER_API")
TEST_COLLECTION = "test_pib_chunks"


# --------------------------------------------------------------------------
# Fake embedder -- deterministic, no model download required.
# Same interface as BgeM3Embedder (encode -> dict with dense_vecs / lexical_weights)
# so VectorStore doesn't know or care which one it's using.
# --------------------------------------------------------------------------

class FakeEmbedder:
    DENSE_SIZE = 1024

    async def encode(self, texts: list[str], batch_size: int = 12) -> dict:
        dense_vecs = [self._fake_dense(t) for t in texts]
        lexical_weights = [self._fake_sparse(t) for t in texts]
        return {"dense_vecs": dense_vecs, "lexical_weights": lexical_weights}

    def _fake_dense(self, text: str) -> list[float]:
        # Deterministic pseudo-vector seeded from the text's hash, so the
        # same content always produces the same vector (useful for asserting
        # retrieval behavior without real semantic embeddings).
        seed = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        vec = []
        for i in range(self.DENSE_SIZE):
            seed = (seed * 1103515245 + 12345 + i) % (2**31)
            vec.append((seed / (2**31)) * 2 - 1)  # range [-1, 1]
        # normalize so cosine distance behaves sanely
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec]

    @staticmethod
    def _fake_sparse(text: str) -> dict:
        # Crude token_id -> weight map based on word hashes.
        words = text.lower().split()
        weights = {}
        for w in words:
            token_id = int(hashlib.md5(w.encode()).hexdigest(), 16) % 30000
            weights[token_id] = weights.get(token_id, 0.0) + 1.0
        return weights

    @staticmethod
    def to_sparse_vector(lexical_weights: dict):
        from qdrant_client.models import SparseVector
        return SparseVector(
            indices=list(lexical_weights.keys()),
            values=list(lexical_weights.values()),
        )


def make_embedder():
    if USE_REAL_EMBEDDER:
        from vector_store import BgeM3Embedder
        return BgeM3Embedder()
    return FakeEmbedder()

# --------------------------------------------------------------------------
# Test data
# --------------------------------------------------------------------------

def sample_chunks_doc_a() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="TEST-DOC-A:0",
            document_id="TEST-DOC-A",
            chunk_index=0,
            content="The Airports Authority of India announced temporary closure of 32 airports from 9th to 14th May 2025 due to security concerns.",
            char_start=0,
            char_end=130,
            content_hash="hash-a0-v1",
            title="Temporary Suspension of Civil Flight Operations",
            url="https://www.pib.gov.in/test-a",
            publisher="PIB",
            author="Test Author",
            source_type="webpage",
            language="en",
            published_at="10 MAY 2025",
        ),
        Chunk(
            chunk_id="TEST-DOC-A:1",
            document_id="TEST-DOC-A",
            chunk_index=1,
            content="NOTAM G0555/25 covers Adhampur, Bathinda, and Jamnagar airports under this closure order.",
            char_start=130,
            char_end=225,
            content_hash="hash-a1-v1",
            title="Temporary Suspension of Civil Flight Operations",
            url="https://www.pib.gov.in/test-a",
            publisher="PIB",
            author="Test Author",
            source_type="webpage",
            language="en",
            published_at="10 MAY 2025",
        ),
    ]


def sample_chunks_doc_b() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="TEST-DOC-B:0",
            document_id="TEST-DOC-B",
            chunk_index=0,
            content="The Ministry of Agriculture launched a new scheme to support wheat farmers in Punjab and Haryana.",
            char_start=0,
            char_end=100,
            content_hash="hash-b0-v1",
            title="New Agricultural Scheme Launched",
            url="https://www.pib.gov.in/test-b",
            publisher="PIB",
            author="Test Author",
            source_type="webpage",
            language="en",
            published_at="12 MAY 2025",
        ),
    ]


def sample_chunks_doc_a_amended() -> list[Chunk]:
    """Simulates the same PIB release being updated: closure window extended,
    and now spread across 3 chunks instead of 2 (content grew)."""
    return [
        Chunk(
            chunk_id="TEST-DOC-A:0",
            document_id="TEST-DOC-A",
            chunk_index=0,
            content="The Airports Authority of India announced temporary closure of 32 airports from 9th to 20th May 2025, EXTENDED due to security concerns.",
            char_start=0,
            char_end=140,
            content_hash="hash-a0-v2",  # changed
            title="Temporary Suspension of Civil Flight Operations (Amended)",
            url="https://www.pib.gov.in/test-a",
            publisher="PIB",
            author="Test Author",
            source_type="webpage",
            language="en",
            published_at="15 MAY 2025",
        ),
        Chunk(
            chunk_id="TEST-DOC-A:1",
            document_id="TEST-DOC-A",
            chunk_index=1,
            content="NOTAM G0555/25 covers Adhampur, Bathinda, and Jamnagar airports under this closure order.",
            char_start=140,
            char_end=235,
            content_hash="hash-a1-v1",  # unchanged
            title="Temporary Suspension of Civil Flight Operations (Amended)",
            url="https://www.pib.gov.in/test-a",
            publisher="PIB",
            author="Test Author",
            source_type="webpage",
            language="en",
            published_at="15 MAY 2025",
        ),
        Chunk(
            chunk_id="TEST-DOC-A:2",
            document_id="TEST-DOC-A",
            chunk_index=2,
            content="Airlines are advised to reroute all affected flights and check updated NOTAMs before departure.",
            char_start=235,
            char_end=330,
            content_hash="hash-a2-v1",  # new chunk
            title="Temporary Suspension of Civil Flight Operations (Amended)",
            url="https://www.pib.gov.in/test-a",
            publisher="PIB",
            author="Test Author",
            source_type="webpage",
            language="en",
            published_at="15 MAY 2025",
        ),
    ]


# --------------------------------------------------------------------------
# Assertions helper (plain asserts -- no pytest dependency required)
# --------------------------------------------------------------------------

def check(condition: bool, message: str):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {message}")
    if not condition:
        raise AssertionError(message)


# --------------------------------------------------------------------------
# Main test flow
# --------------------------------------------------------------------------

async def run_tests():
    client = AsyncQdrantClient(url=QDRANT_URL, api_key=KEY, check_compatibility=False)
    embedder = make_embedder()
    store = VectorStore(client, embedder, collection_name=TEST_COLLECTION)

    try:
        print("\n[1] ensure_collection")
        await store.ensure_collection()
        check(
            await client.collection_exists(TEST_COLLECTION),
            "test collection exists after ensure_collection()",
        )

        print("\n[2] ingest_chunks -- doc A (2 chunks) + doc B (1 chunk)")
        await store.ingest_chunks(sample_chunks_doc_a())
        await store.ingest_chunks(sample_chunks_doc_b())

        count = await client.count(TEST_COLLECTION, exact=True)
        check(count.count == 3, f"expected 3 points after ingest, got {count.count}")

        print("\n[3] retrieve -- query should surface doc A chunks")
        results = await store.retrieve("airport closures security", top_k=5)
        check(len(results) > 0, "retrieve() returned at least one result")
        returned_doc_ids = {r.document_id for r in results}
        check(
            "TEST-DOC-A" in returned_doc_ids,
            f"TEST-DOC-A present in retrieval results (got {returned_doc_ids})",
        )

        print("\n[4] retrieve with source_type filter")
        filtered = await store.retrieve(
            "agricultural scheme", top_k=5, source_type="webpage"
        )
        check(len(filtered) > 0, "filtered retrieve() returned results")

        print("\n[5] update_document -- amend doc A (2 chunks -> 3 chunks)")
        await store.update_document(sample_chunks_doc_a_amended())

        count_after_update = await client.count(TEST_COLLECTION, exact=True)
        # doc A now has 3 chunks, doc B still has 1 -> 4 total
        check(
            count_after_update.count == 4,
            f"expected 4 points after update (3 for A + 1 for B), got {count_after_update.count}",
        )

        existing_a_chunks = await store.get_existing_chunks("TEST-DOC-A") if hasattr(
            store, "get_existing_chunks"
        ) else None
        if existing_a_chunks is not None:
            hashes = {c.payload["content_hash"] for c in existing_a_chunks}
            check(
                "hash-a0-v2" in hashes and "hash-a0-v1" not in hashes,
                "stale chunk (v1) removed, amended chunk (v2) present after update",
            )

        print("\n[6] delete_document -- remove doc B entirely")
        await store.delete_document("TEST-DOC-B")
        count_after_delete = await client.count(TEST_COLLECTION, exact=True)
        check(
            count_after_delete.count == 3,
            f"expected 3 points after deleting doc B, got {count_after_delete.count}",
        )

        print("\nAll checks passed.")

    finally:
        # ------------------------------------------------------------------
        # CLEANUP -- runs whether the test passed or raised. Removes every
        # point this test created, then drops the whole test collection so
        # nothing lingers in the cluster afterward.
        # ------------------------------------------------------------------
        print("\n[cleanup] removing test data from cluster...")
        try:
            await store.delete_document("TEST-DOC-A")
            await store.delete_document("TEST-DOC-B")
        except Exception as e:
            print(f"  (cleanup delete_document warning: {e})")

        try:
            await client.delete_collection(TEST_COLLECTION)
            print(f"  dropped collection '{TEST_COLLECTION}'")
        except Exception as e:
            print(f"  (cleanup delete_collection warning: {e})")

        await client.close()


if __name__ == "__main__":
    try:
        asyncio.run(run_tests())
    except AssertionError:
        print("\nTEST SUITE FAILED (cluster was still cleaned up).")
        sys.exit(1)

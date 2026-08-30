"""Stream chunks from Team 1's SQLite corpus without retrieval dependencies."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from rag.adapters import iter_documents_from_sqlite
from rag.chunking import ChunkingConfig, DocumentChunker, Tokenizer
from rag.models import Chunk


def iter_chunks_from_sqlite(
    database: Path,
    config: ChunkingConfig | None = None,
    tokenizer: Tokenizer | None = None,
) -> Iterator[Chunk]:
    """Yield chunks from each read-only SQLite document in deterministic order."""

    chunker = DocumentChunker(config=config, tokenizer=tokenizer)
    for document in iter_documents_from_sqlite(database):
        yield from chunker.chunk(document)

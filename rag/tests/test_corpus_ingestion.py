import os
from pathlib import Path

import pytest

from rag.adapters import iter_documents_from_sqlite
from rag.chunking import BgeM3Tokenizer, ChunkingConfig, chunk_document
from rag.ingestion import iter_chunks_from_sqlite
from rag.models import Chunk


DATABASE = Path(__file__).resolve().parents[1] / "data" / "main_corpus.db"


def bge_m3_tokenizer() -> BgeM3Tokenizer:
    cache_root = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
    snapshots = sorted(path for path in (cache_root / "models--BAAI--bge-m3" / "snapshots").glob("*") if path.is_dir())
    return BgeM3Tokenizer(str(snapshots[-1])) if snapshots else BgeM3Tokenizer()


@pytest.mark.skipif(not DATABASE.is_file(), reason="Team 1 corpus database is unavailable")
def test_streaming_corpus_ingestion_preserves_chunks_and_documents() -> None:
    config = ChunkingConfig()
    tokenizer = bge_m3_tokenizer()
    chunk_stream = iter_chunks_from_sqlite(DATABASE, config=config, tokenizer=tokenizer)

    document_count = 0
    chunk_count = 0
    for document in iter_documents_from_sqlite(DATABASE):
        document_count += 1
        # Only one document's already-existing chunk list is held at a time;
        # the corpus itself remains a pair of database-backed streams.
        for expected in chunk_document(document, config=config, tokenizer=tokenizer):
            chunk = next(chunk_stream)
            chunk_count += 1
            assert isinstance(chunk, Chunk)
            assert chunk == expected
            assert chunk.chunk_id == f"{document.document_id}:{chunk.chunk_index}"
            assert tokenizer.count_tokens(chunk.content) <= config.max_tokens
            assert document.content[chunk.char_start : chunk.char_end] == chunk.content
            assert chunk.metadata == document.metadata

    with pytest.raises(StopIteration):
        next(chunk_stream)

    assert document_count == 406
    assert chunk_count == 749

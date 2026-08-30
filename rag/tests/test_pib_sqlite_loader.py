import os
import json
import sqlite3
from pathlib import Path

import pytest

from rag.chunking import BgeM3Tokenizer, ChunkingConfig, DocumentChunker
from rag.scripts.load_pib_sqlite import SECTION_SEPARATOR, load_pib_sqlite


DATABASE = Path(r"C:\Users\akash\Desktop\pib_corpus_of_t9.db")


def bge_m3_tokenizer() -> BgeM3Tokenizer:
    """Use the official cached tokenizer when available to avoid network-only tests."""

    cache_root = Path(
        os.environ.get(
            "HF_HUB_CACHE",
            Path.home() / ".cache" / "huggingface" / "hub",
        )
    )
    snapshots = cache_root / "models--BAAI--bge-m3" / "snapshots"
    cached = sorted(path for path in snapshots.glob("*") if path.is_dir())
    return BgeM3Tokenizer(str(cached[-1])) if cached else BgeM3Tokenizer()


@pytest.fixture(scope="module")
def database() -> Path:
    if not DATABASE.is_file():
        pytest.skip(f"Team 1 development database is unavailable: {DATABASE}")
    return DATABASE


def test_sqlite_adapter_to_chunker_preserves_contract_and_offsets(database: Path) -> None:
    documents = load_pib_sqlite(database)
    chunker = DocumentChunker(
        config=ChunkingConfig(target_tokens=700, max_tokens=900, overlap_tokens=100),
        tokenizer=bge_m3_tokenizer(),
    )

    with sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT document_id, url, title, published_at, language, organization, "
            "metadata_json, content_json FROM pib_releases ORDER BY document_id"
        ).fetchall()

    assert len(documents) == len(rows)
    assert sum(len(document.sections or []) for document in documents) > 0

    for document, row in zip(documents, rows):
        metadata = json.loads(row["metadata_json"])
        content_payload = json.loads(row["content_json"])
        assert document.document_id == row["document_id"]
        assert document.source_type.value == content_payload["source_type"]
        assert str(document.url) == row["url"]
        assert document.title == row["title"]
        assert document.publisher == row["organization"]
        assert document.published_at and document.published_at.isoformat() == row["published_at"]
        assert document.retrieved_at.isoformat() == metadata["retrieved_at"]
        assert document.language == row["language"]
        assert document.metadata == metadata
        assert document.sections
        assert document.content == SECTION_SEPARATOR.join(
            section.content for section in document.sections
        )
        assert [section.model_dump() for section in document.sections] == content_payload["sections"]
        chunks = chunker.chunk(document)
        assert chunks
        assert [chunk.chunk_id for chunk in chunks] == [
            f"{document.document_id}:{index}" for index in range(len(chunks))
        ]
        assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
        for chunk in chunks:
            assert chunk.metadata == document.metadata
            assert chunk.publisher == document.publisher
            assert chunk.author == document.author
            assert chunk.section_id in {section.section_id for section in document.sections} | {None}
            if chunk.section_id is not None:
                section = next(
                    section for section in document.sections if section.section_id == chunk.section_id
                )
                assert chunk.section_heading == section.heading
            assert document.content[chunk.char_start : chunk.char_end] == chunk.content
            assert chunker.tokenizer.count_tokens(chunk.content) <= chunker.config.max_tokens
        if chunker.tokenizer.count_tokens(document.content) <= chunker.config.max_tokens:
            assert len(chunks) == 1

"""Public boundary test for handing chunks to a retrieval implementation."""

import re

from rag.chunking import ChunkingConfig, chunk_document
from rag.models import Chunk, Document


class WhitespaceTokenizer:
    """Small deterministic tokenizer for testing the public API boundary."""

    def token_offsets(self, text: str) -> list[tuple[int, int]]:
        return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]

    def count_tokens(self, text: str) -> int:
        return len(self.token_offsets(text))

    def tokenize(self, text: str) -> list[int]:
        return list(range(self.count_tokens(text)))

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token) for token in token_ids)


def test_public_document_to_chunks_boundary_preserves_retrieval_provenance() -> None:
    document = Document(
        document_id="pib-handoff-1",
        source_type="government_document",
        content="The source text is canonical and ready for retrieval.",
        url="https://www.pib.gov.in/PressReleasePage.aspx?PRID=1",
        title="Handoff fixture",
        publisher="Press Information Bureau",
        author="PIB",
        published_at="2026-08-30T10:00:00+05:30",
        retrieved_at="2026-08-30T11:00:00+05:30",
        language="en",
        content_hash="sha256:handoff",
        metadata={"source_system": "PIB"},
    )

    chunks = chunk_document(
        document,
        config=ChunkingConfig(target_tokens=10, max_tokens=20, overlap_tokens=0),
        tokenizer=WhitespaceTokenizer(),
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)
    assert chunk.chunk_id == "pib-handoff-1:0"
    assert chunk.chunk_index == 0
    assert document.content[chunk.char_start : chunk.char_end] == chunk.content
    assert chunk.document_id == document.document_id
    assert chunk.content_hash == document.content_hash
    assert chunk.url == document.url
    assert chunk.title == document.title
    assert chunk.publisher == document.publisher
    assert chunk.author == document.author
    assert chunk.published_at == document.published_at
    assert chunk.source_type == document.source_type
    assert chunk.language == document.language
    assert chunk.metadata == document.metadata
    assert chunk.section_id is None
    assert chunk.section_heading is None

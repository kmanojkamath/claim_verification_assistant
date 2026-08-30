import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag.chunking import ChunkingConfig, DocumentChunker
from rag.models import Document


class WhitespaceTokenizer:
    """Deterministic tokenizer double: one token per non-whitespace run."""

    def tokenize(self, text: str) -> list[int]:
        return list(range(len(self.token_offsets(text))))

    def count_tokens(self, text: str) -> int:
        return len(self.token_offsets(text))

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token) for token in token_ids)

    def token_offsets(self, text: str) -> list[tuple[int, int]]:
        return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def make_document(content: str, *, sections: list[dict] | None = None) -> Document:
    return Document(
        document_id="PIB-TEST",
        source_type="government_document",
        content=content,
        url="https://www.pib.gov.in/PressReleasePage.aspx?PRID=1",
        title="Test release",
        publisher="Press Information Bureau",
        author="PIB Delhi",
        retrieved_at="2026-08-30T10:00:00+05:30",
        language="en",
        content_hash="sha256:test",
        metadata={"country": "IN"},
        sections=sections,
    )


def make_chunker(target: int = 5, maximum: int = 8, overlap: int = 2) -> DocumentChunker:
    return DocumentChunker(
        config=ChunkingConfig(
            target_tokens=target,
            max_tokens=maximum,
            overlap_tokens=overlap,
        ),
        tokenizer=WhitespaceTokenizer(),
    )


def assert_offsets(document: Document, chunks: list) -> None:
    for chunk in chunks:
        assert document.content[chunk.char_start : chunk.char_end] == chunk.content


def test_small_and_exactly_max_documents_remain_single_chunks() -> None:
    chunker = make_chunker(target=4, maximum=4)
    small = make_document("one two three")
    exact = make_document("one two three four")

    assert len(chunker.chunk(small)) == 1
    exact_chunks = chunker.chunk(exact)
    assert len(exact_chunks) == 1
    assert exact_chunks[0].content == exact.content


def test_large_flat_document_prefers_paragraphs_and_preserves_offsets() -> None:
    document = make_document("one two three.\n\nfour five six.\n\nseven eight nine.")
    chunks = make_chunker(target=4, maximum=6, overlap=0).chunk(document)

    assert len(chunks) == 2
    assert chunks[0].content == "one two three.\n\nfour five six."
    assert_offsets(document, chunks)


def test_sentence_and_token_fallbacks_never_exceed_maximum() -> None:
    sentence_document = make_document("one two three four five. six seven eight nine ten.")
    long_paragraph = make_document(" ".join(f"word{i}" for i in range(15)))
    chunker = make_chunker(target=4, maximum=5, overlap=1)

    sentence_chunks = chunker.chunk(sentence_document)
    fallback_chunks = chunker.chunk(long_paragraph)

    assert len(sentence_chunks) > 1
    assert len(fallback_chunks) > 1
    assert all(chunker.tokenizer.count_tokens(chunk.content) <= 5 for chunk in sentence_chunks + fallback_chunks)
    assert_offsets(sentence_document, sentence_chunks)
    assert_offsets(long_paragraph, fallback_chunks)


def test_small_multi_section_document_is_one_document_level_chunk() -> None:
    content = "First section text.\n\nSecond section text."
    document = make_document(
        content,
        sections=[
            {"section_id": "one", "heading": "One", "content": "First section text."},
            {"section_id": "two", "heading": "Two", "content": "Second section text."},
        ],
    )

    chunks = make_chunker(maximum=10).chunk(document)

    assert len(chunks) == 1
    assert chunks[0].content == content
    assert chunks[0].section_id is None


def test_large_sections_are_split_independently_and_keep_section_metadata() -> None:
    first = "a1 a2 a3 a4 a5 a6 a7 a8"
    second = "b1 b2 b3 b4 b5 b6 b7 b8"
    document = make_document(
        f"{first}\n\n{second}",
        sections=[
            {"section_id": "first", "heading": "First", "content": first},
            {"section_id": "second", "heading": "Second", "content": second},
        ],
    )
    chunks = make_chunker(target=4, maximum=5, overlap=1).chunk(document)

    assert {chunk.section_id for chunk in chunks} == {"first", "second"}
    assert all(chunk.section_heading in {"First", "Second"} for chunk in chunks)
    assert_offsets(document, chunks)


def test_unmappable_sections_fall_back_to_flat_text_without_bad_offsets() -> None:
    document = make_document(
        "one two three four five six seven eight nine",
        sections=[{"section_id": "bad", "content": "not in document"}],
    )
    chunks = make_chunker(target=4, maximum=4).chunk(document)

    assert len(chunks) > 1
    assert all(chunk.section_id is None for chunk in chunks)
    assert_offsets(document, chunks)


def test_overlap_is_bounded_non_duplicate_and_deterministic() -> None:
    document = make_document(" ".join(f"word{i}" for i in range(20)))
    chunker = make_chunker(target=5, maximum=7, overlap=2)
    first = chunker.chunk(document)
    second = chunker.chunk(document)

    assert [chunk.chunk_id for chunk in first] == [f"PIB-TEST:{i}" for i in range(len(first))]
    assert [chunk.model_dump() for chunk in first] == [chunk.model_dump() for chunk in second]
    assert len({chunk.content for chunk in first}) == len(first)
    for previous, current in zip(first, first[1:]):
        assert previous.content.split()[-2:] == current.content.split()[:2]
        assert chunker.tokenizer.count_tokens(current.content) <= 7
    assert_offsets(document, first)


def test_overlap_never_crosses_section_boundaries() -> None:
    first = "a1 a2 a3 a4 a5 a6 a7 a8"
    second = "b1 b2 b3 b4 b5 b6 b7 b8"
    document = make_document(
        f"{first}\n\n{second}",
        sections=[
            {"section_id": "first", "content": first},
            {"section_id": "second", "content": second},
        ],
    )
    chunks = make_chunker(target=4, maximum=5, overlap=2).chunk(document)

    first_section = [chunk for chunk in chunks if chunk.section_id == "first"]
    second_section = [chunk for chunk in chunks if chunk.section_id == "second"]
    assert len(first_section) > 1
    assert len(second_section) > 1
    assert all(set(chunk.content.split()) <= {f"a{i}" for i in range(1, 9)} for chunk in first_section)
    assert all(set(chunk.content.split()) <= {f"b{i}" for i in range(1, 9)} for chunk in second_section)
    assert_offsets(document, chunks)


def test_target_configuration_changes_grouping_and_metadata_is_preserved() -> None:
    document = make_document("one two three.\n\nfour five six.\n\nseven eight nine.")
    smaller = make_chunker(target=3, maximum=6, overlap=0).chunk(document)
    larger = make_chunker(target=6, maximum=6, overlap=0).chunk(document)

    assert len(smaller) > len(larger)
    assert all(chunk.metadata == {"country": "IN"} for chunk in larger)
    assert all(chunk.publisher == "Press Information Bureau" for chunk in larger)


@pytest.mark.parametrize(
    "content",
    [
        "Numbers 12, 2026-08-30, https://example.test/path?a=1! Unicode भारत।",
        "!!! ??? ... ;;;",
        "word " * 40,
    ],
)
def test_edge_text_remains_offset_correct(content: str) -> None:
    document = make_document(content)
    chunks = make_chunker(maximum=5).chunk(document)

    assert chunks
    assert_offsets(document, chunks)


def test_empty_and_whitespace_only_documents_remain_invalid() -> None:
    for content in ("", "   "):
        with pytest.raises(ValidationError):
            make_document(content)


def test_sample_documents_can_all_be_chunked() -> None:
    path = Path(__file__).resolve().parents[1] / "data" / "sample_documents.json"
    documents = [Document.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]
    chunker = make_chunker(target=100, maximum=150, overlap=20)

    chunks = [chunk for document in documents for chunk in chunker.chunk(document)]
    assert chunks
    assert all(chunker.tokenizer.count_tokens(chunk.content) <= 150 for chunk in chunks)
    for document in documents:
        assert_offsets(document, chunker.chunk(document))

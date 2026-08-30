import copy
import re

import pytest
from pydantic import ValidationError

from rag.adapters import live_search_record_to_document, live_search_records_to_documents
from rag.chunking import ChunkingConfig, chunk_document
from rag.models import Document, SourceType


class WhitespaceTokenizer:
    def token_offsets(self, text: str) -> list[tuple[int, int]]:
        return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]

    def count_tokens(self, text: str) -> int:
        return len(self.token_offsets(text))

    def tokenize(self, text: str) -> list[int]:
        return list(range(self.count_tokens(text)))

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token) for token in token_ids)


def record() -> dict:
    return {
        "doc_id": "X_2086430092247601243",
        "title": "X post by @PIBFactCheck",
        "source": {
            "platform": "X",
            "source_type": "social_media",
            "source_url": "https://x.com/PIBFactCheck/status/2086430092247601243",
            "organization": "PIB Fact Check",
            "author_handle": "PIBFactCheck",
            "author_name": "PIB Fact Check",
        },
        "dates": {
            "published_at": "2026-08-09T12:31:12Z",
            "retrieved_at": "2026-08-29T19:16:51Z",
        },
        "content": {
            "raw_text": "RAW text that must not become document content",
            "clean_text": "Fact check: this claim is fake. #PIBFactCheck 🔎 https://example.test/evidence",
        },
        "entities": {"hashtags": ["PIBFactCheck", "fake"], "mentions": [], "urls": []},
        "metadata": {
            "language": "en",
            "authority_level": "official",
            "content_hash": "sha256:live-search",
            "document_type": "social_media_post",
        },
    }


def test_live_search_record_maps_clean_text_and_provenance() -> None:
    source = record()
    document = live_search_record_to_document(source)

    assert isinstance(document, Document)
    assert document.document_id == source["doc_id"]
    assert document.source_type is SourceType.SOCIAL_MEDIA_POST
    assert document.content == source["content"]["clean_text"]
    assert document.content != source["content"]["raw_text"]
    assert str(document.url) == source["source"]["source_url"]
    assert document.title == source["title"]
    assert document.publisher == source["source"]["organization"]
    assert document.author == source["source"]["author_name"]
    assert document.published_at and document.published_at.isoformat() == "2026-08-09T12:31:12+00:00"
    assert document.retrieved_at.isoformat() == "2026-08-29T19:16:51+00:00"
    assert document.language == "en"
    assert document.content_hash == "sha256:live-search"
    assert document.metadata == {
        "platform": "X",
        "author_handle": "PIBFactCheck",
        "hashtags": ["PIBFactCheck", "fake"],
        "mentions": [],
        "urls": [],
        "authority_level": "official",
        "document_type": "social_media_post",
    }
    assert document.sections is None


def test_published_at_may_be_null_and_multiple_records_keep_order() -> None:
    first = record()
    first["dates"]["published_at"] = None
    second = record()
    second["doc_id"] = "X_2"
    second["content"]["clean_text"] = "Unicode remains intact: भारत 🚀 #verified"

    documents = live_search_records_to_documents([first, second])

    assert [document.document_id for document in documents] == ["X_2086430092247601243", "X_2"]
    assert documents[0].published_at is None
    assert documents[1].content == "Unicode remains intact: भारत 🚀 #verified"


def test_live_search_document_passes_through_existing_public_chunking_api() -> None:
    document = live_search_record_to_document(record())
    chunks = chunk_document(
        document,
        config=ChunkingConfig(target_tokens=20, max_tokens=30, overlap_tokens=0),
        tokenizer=WhitespaceTokenizer(),
    )

    assert len(chunks) == 1
    assert chunks[0].chunk_id == f"{document.document_id}:0"
    assert chunks[0].content == document.content
    assert chunks[0].metadata == document.metadata


@pytest.mark.parametrize(
    "path,value",
    [
        (("doc_id",), None),
        (("source",), None),
        (("source", "source_url"), None),
        (("content", "clean_text"), None),
        (("dates", "retrieved_at"), None),
        (("metadata", "language"), None),
        (("dates", "published_at"), "not-a-date"),
        (("dates", "retrieved_at"), "not-a-date"),
        (("source", "source_url"), "not a URL"),
        (("source", "source_type"), "unrecognized"),
        (("content", "clean_text"), 42),
        (("content", "clean_text"), "   "),
    ],
)
def test_live_search_rejects_missing_or_invalid_values(path: tuple[str, ...], value: object) -> None:
    source = copy.deepcopy(record())
    target = source
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises((ValueError, ValidationError)):
        live_search_record_to_document(source)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source", []),
        ("dates", "not an object"),
        ("content", []),
        ("entities", []),
        ("metadata", []),
    ],
)
def test_live_search_rejects_malformed_nested_objects(field: str, value: object) -> None:
    source = record()
    source[field] = value

    with pytest.raises(ValueError, match="must be an object"):
        live_search_record_to_document(source)


def test_live_search_preserves_empty_entity_lists_and_urls_in_clean_text() -> None:
    source = record()
    source["entities"] = {"hashtags": [], "mentions": [], "urls": []}
    source["content"]["clean_text"] = "See https://example.test/path?q=1 #tag @person 😀"

    document = live_search_record_to_document(source)

    assert document.content == source["content"]["clean_text"]
    assert document.metadata and document.metadata["hashtags"] == []
    assert document.metadata["mentions"] == []
    assert document.metadata["urls"] == []

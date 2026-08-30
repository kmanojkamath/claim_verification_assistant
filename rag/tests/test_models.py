from datetime import datetime

import pytest
from pydantic import ValidationError

from rag.models import (
    Chunk,
    ChunkLocation,
    Citation,
    Document,
    RetrievalResponse,
    RetrievalResult,
    Section,
    SourceType,
)


def valid_document_payload() -> dict:
    return {
        "document_id": "gov-in-pib-2026-001234",
        "source_type": "government_document",
        "content": "The Ministry stated that vaccine supply remains uninterrupted.",
        "url": "https://pib.gov.in/PressReleasePage.aspx?PRID=1234567",
        "title": "Vaccine supply clarification",
        "publisher": "Press Information Bureau, Government of India",
        "author": "Ministry of Health and Family Welfare",
        "published_at": "2026-08-20T10:30:00+05:30",
        "retrieved_at": "2026-08-29T17:05:12+05:30",
        "language": "en-IN",
        "content_hash": "sha256:abc123",
        "metadata": {"country": "IN", "topic_tags": ["health", "vaccines"]},
    }


def test_document_accepts_a_valid_contract_payload() -> None:
    document = Document.model_validate(valid_document_payload())

    assert document.source_type is SourceType.GOVERNMENT_DOCUMENT
    assert document.published_at == datetime.fromisoformat("2026-08-20T10:30:00+05:30")
    assert document.retrieved_at == datetime.fromisoformat("2026-08-29T17:05:12+05:30")
    assert str(document.url) == "https://pib.gov.in/PressReleasePage.aspx?PRID=1234567"
    assert document.sections is None


def test_document_accepts_multiple_optional_sections_and_preserves_metadata() -> None:
    payload = valid_document_payload()
    payload["sections"] = [
        {"section_id": "summary", "heading": "Summary", "content": "First section."},
        {"section_id": "details", "content": "Second section."},
    ]
    document = Document.model_validate(payload)

    assert [section.section_id for section in document.sections or []] == ["summary", "details"]
    assert document.metadata == {"country": "IN", "topic_tags": ["health", "vaccines"]}


def test_section_accepts_valid_content_and_rejects_invalid_required_values() -> None:
    section = Section(section_id="background", heading="Background", content="Context text.")

    assert section.heading == "Background"
    with pytest.raises(ValidationError):
        Section(section_id=" ", content="Context text.")
    with pytest.raises(ValidationError):
        Section(section_id="background", content=" ")


@pytest.mark.parametrize("field,value", [("document_id", "  "), ("content", "")])
def test_document_rejects_empty_required_strings(field: str, value: str) -> None:
    payload = valid_document_payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        Document.model_validate(payload)


@pytest.mark.parametrize(
    "field",
    ["document_id", "source_type", "content", "url", "retrieved_at", "language"],
)
def test_document_rejects_missing_required_fields(field: str) -> None:
    payload = valid_document_payload()
    payload.pop(field)

    with pytest.raises(ValidationError):
        Document.model_validate(payload)


@pytest.mark.parametrize(
    "field,value",
    [("source_type", "untrusted_source"), ("url", "not a URL")],
)
def test_document_rejects_invalid_source_type_and_url(field: str, value: str) -> None:
    payload = valid_document_payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        Document.model_validate(payload)


@pytest.mark.parametrize("field,value", [("language", "english"), ("retrieved_at", "not-a-date")])
def test_document_rejects_invalid_language_and_timestamp(field: str, value: str) -> None:
    payload = valid_document_payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        Document.model_validate(payload)


def test_chunk_retains_source_data_and_validates_offsets() -> None:
    chunk = Chunk(
        chunk_id="gov-in-pib-2026-001234:0",
        document_id="gov-in-pib-2026-001234",
        chunk_index=0,
        content="The Ministry stated that vaccine supply remains uninterrupted.",
        char_start=0,
        char_end=61,
        content_hash="sha256:abc123",
        url="https://pib.gov.in/PressReleasePage.aspx?PRID=1234567",
        source_type=SourceType.GOVERNMENT_DOCUMENT,
        title="Vaccine supply clarification",
        publisher="Press Information Bureau, Government of India",
        published_at="2026-08-20T10:30:00+05:30",
        language="en-IN",
        metadata={"country": "IN"},
        section_id="summary",
        section_heading="Summary",
    )

    assert chunk.source_type is SourceType.GOVERNMENT_DOCUMENT
    assert str(chunk.url).startswith("https://pib.gov.in/")
    assert chunk.section_id == "summary"
    assert chunk.section_heading == "Summary"


@pytest.mark.parametrize(
    "overrides",
    [
        {"chunk_index": -1},
        {"char_start": -1},
        {"char_end": -1},
        {"char_start": 10, "char_end": 9},
    ],
)
def test_chunk_rejects_invalid_positions(overrides: dict) -> None:
    payload = {
        "chunk_id": "doc-1:0",
        "document_id": "doc-1",
        "chunk_index": 0,
        "content": "Evidence text",
        "char_start": 0,
        "char_end": 13,
        "url": "https://example.org/evidence",
        "source_type": "web_page",
        "language": "en",
    }
    payload.update(overrides)

    with pytest.raises(ValidationError):
        Chunk.model_validate(payload)


def test_retrieval_response_matches_the_approved_contract() -> None:
    result = RetrievalResult(
        chunk_id="doc-1:0",
        document_id="doc-1",
        text="Evidence text",
        score=0.91,
        citation=Citation(
            title="Evidence page",
            url="https://example.org/evidence",
            publisher="Example Organization",
            published_at="2026-08-20T10:30:00+05:30",
            source_type="web_page",
        ),
        location=ChunkLocation(chunk_index=0, char_start=0, char_end=13),
    )
    response = RetrievalResponse(query="Was the claim true?", results=[result])

    assert response.results[0].score == 0.91
    assert response.results[0].citation.source_type is SourceType.WEB_PAGE


@pytest.mark.parametrize("score", ["0.91", True, float("nan"), float("inf")])
def test_retrieval_result_rejects_non_numeric_or_non_finite_scores(score: object) -> None:
    with pytest.raises(ValidationError):
        RetrievalResult(
            chunk_id="doc-1:0",
            document_id="doc-1",
            text="Evidence text",
            score=score,
            citation={"url": "https://example.org/evidence", "source_type": "web_page"},
            location={"chunk_index": 0, "char_start": 0, "char_end": 13},
        )

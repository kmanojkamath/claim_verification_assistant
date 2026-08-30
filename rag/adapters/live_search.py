"""Development boundary adapter for Team 1's current live-search JSON format."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rag.models import Document, SourceType


_SOURCE_TYPE_NORMALIZATION = {"social_media": "social_media_post"}
_CANONICAL_METADATA_FIELDS = {"language", "content_hash"}
_CANONICAL_SOURCE_FIELDS = {
    "source_type",
    "source_url",
    "organization",
    "author_name",
}


def _object(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Live-search field {field!r} must be an object")
    return value


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Live-search field {field!r} must be a non-empty string")
    return value


def _optional_text(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Live-search field {field!r} must be a string or null")
    return value


def _canonical_source_type(value: object) -> SourceType:
    source_type = _required_text(value, "source.source_type")
    normalized = _SOURCE_TYPE_NORMALIZATION.get(source_type, source_type)
    try:
        return SourceType(normalized)
    except ValueError as error:
        allowed = ", ".join(item.value for item in SourceType)
        raise ValueError(
            f"Unknown live-search source.source_type {source_type!r}; "
            f"expected a canonical type ({allowed}) or 'social_media'"
        ) from error


def _metadata(
    source: Mapping[str, Any],
    entities: Mapping[str, Any],
    record_metadata: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Preserve non-canonical JSON data without duplicating Document fields."""

    metadata: dict[str, Any] = {
        key: value
        for key, value in source.items()
        if key not in _CANONICAL_SOURCE_FIELDS
    }
    metadata.update(entities)
    metadata.update(
        {
            key: value
            for key, value in record_metadata.items()
            if key not in _CANONICAL_METADATA_FIELDS
        }
    )
    return metadata or None


def live_search_record_to_document(record: Mapping[str, Any]) -> Document:
    """Convert one current Team 1 live-search record into a RAG Document."""

    if not isinstance(record, Mapping):
        raise ValueError("Live-search record must be an object")

    source = _object(record.get("source"), "source")
    dates = _object(record.get("dates"), "dates")
    content = _object(record.get("content"), "content")
    entities = _object(record.get("entities", {}), "entities")
    record_metadata = _object(record.get("metadata"), "metadata")

    try:
        return Document(
            document_id=_required_text(record.get("doc_id"), "doc_id"),
            source_type=_canonical_source_type(source.get("source_type")),
            content=_required_text(content.get("clean_text"), "content.clean_text"),
            url=_required_text(source.get("source_url"), "source.source_url"),
            title=_optional_text(record.get("title"), "title"),
            publisher=_optional_text(source.get("organization"), "source.organization"),
            author=_optional_text(source.get("author_name"), "source.author_name"),
            published_at=dates.get("published_at"),
            retrieved_at=_required_text(dates.get("retrieved_at"), "dates.retrieved_at"),
            language=_required_text(record_metadata.get("language"), "metadata.language"),
            content_hash=_optional_text(record_metadata.get("content_hash"), "metadata.content_hash"),
            metadata=_metadata(source, entities, record_metadata),
            sections=None,
        )
    except ValueError as error:
        document_id = record.get("doc_id", "<unknown>")
        raise ValueError(f"Live-search record {document_id!r} is invalid: {error}") from error


def live_search_records_to_documents(records: Sequence[Mapping[str, Any]]) -> list[Document]:
    """Convert a collection of live-search records in input order."""

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError("Live-search records must be a sequence of objects")
    return [live_search_record_to_document(record) for record in records]

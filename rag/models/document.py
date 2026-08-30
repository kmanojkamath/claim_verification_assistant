"""Input document contract for the RAG component."""

from __future__ import annotations

from datetime import datetime

from pydantic import HttpUrl, JsonValue, field_validator

from .common import ContractModel, SourceType, ensure_non_empty, validate_language_tag
from .section import Section


class Document(ContractModel):
    """A normalized source document supplied to the RAG boundary."""

    document_id: str
    source_type: SourceType
    content: str
    url: HttpUrl
    title: str | None = None
    publisher: str | None = None
    author: str | None = None
    published_at: datetime | None = None
    retrieved_at: datetime
    language: str
    content_hash: str | None = None
    metadata: dict[str, JsonValue] | None = None
    sections: list[Section] | None = None

    @field_validator("document_id", "content")
    @classmethod
    def required_text_must_not_be_empty(cls, value: str, info: object) -> str:
        return ensure_non_empty(value, getattr(info, "field_name", "required field"))

    @field_validator("language")
    @classmethod
    def language_must_be_bcp47(cls, value: str) -> str:
        return validate_language_tag(value)

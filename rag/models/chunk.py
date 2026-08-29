"""Chunk contract used after a document has been split."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, HttpUrl, JsonValue, field_validator, model_validator

from .common import ContractModel, SourceType, ensure_non_empty, validate_language_tag


class Chunk(ContractModel):
    """A citeable portion of a document with inherited source metadata."""

    chunk_id: str
    document_id: str
    chunk_index: int = Field(ge=0)
    content: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    content_hash: str | None = None

    # Inherited from the parent document for independent attribution and filtering.
    url: HttpUrl
    source_type: SourceType
    title: str | None = None
    publisher: str | None = None
    author: str | None = None
    published_at: datetime | None = None
    language: str
    metadata: dict[str, JsonValue] | None = None
    section_id: str | None = None
    section_heading: str | None = None

    @field_validator("chunk_id", "document_id", "content")
    @classmethod
    def required_text_must_not_be_empty(cls, value: str, info: object) -> str:
        return ensure_non_empty(value, getattr(info, "field_name", "required field"))

    @field_validator("section_id")
    @classmethod
    def section_id_must_not_be_empty_when_provided(cls, value: str | None) -> str | None:
        if value is not None:
            return ensure_non_empty(value, "section_id")
        return value

    @field_validator("language")
    @classmethod
    def language_must_be_bcp47(cls, value: str) -> str:
        return validate_language_tag(value)

    @model_validator(mode="after")
    def end_must_not_precede_start(self) -> "Chunk":
        if self.char_end < self.char_start:
            raise ValueError("char_end must be greater than or equal to char_start")
        return self

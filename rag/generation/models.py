"""Validated contracts for retrieval-grounded LLM generation."""

from __future__ import annotations

from pydantic import Field, field_validator

from rag.models.common import ContractModel, ensure_non_empty


class LLMGenerationOutput(ContractModel):
    """The strict, provider-facing response schema.

    Citation numbers are one-based positions in the context assembled for the
    model. They are resolved to source metadata by ``GenerationService``.
    """

    answer: str
    citations: list[int] = Field(default_factory=list)

    @field_validator("answer")
    @classmethod
    def answer_must_not_be_empty(cls, value: str) -> str:
        return ensure_non_empty(value, "answer")

class GenerationCitation(ContractModel):
    """A source selected by the LLM as support for the generated answer."""

    source_number: int = Field(gt=0)
    chunk_id: str
    document_id: str
    title: str | None = None
    url: str | None = None
    publisher: str | None = None
    source_type: str | None = None

    @field_validator("chunk_id", "document_id")
    @classmethod
    def identifiers_must_not_be_empty(cls, value: str, info: object) -> str:
        return ensure_non_empty(value, getattr(info, "field_name", "identifier"))


class GenerationResult(ContractModel):
    """Application-level grounded answer and only its supporting sources."""

    answer: str
    citations: list[GenerationCitation] = Field(default_factory=list)

    @field_validator("answer")
    @classmethod
    def answer_must_not_be_empty(cls, value: str) -> str:
        return ensure_non_empty(value, "answer")

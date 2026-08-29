"""Output contracts for downstream consumers of retrieved evidence."""

from __future__ import annotations

from pydantic import field_validator

from .common import ChunkLocation, Citation, ContractModel, ensure_non_empty, validate_score


class RetrievalResult(ContractModel):
    """One retrieved, independently citeable evidence chunk."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    citation: Citation
    location: ChunkLocation

    @field_validator("chunk_id", "document_id", "text")
    @classmethod
    def required_text_must_not_be_empty(cls, value: str, info: object) -> str:
        return ensure_non_empty(value, getattr(info, "field_name", "required field"))

    @field_validator("score", mode="before")
    @classmethod
    def score_must_be_numeric(cls, value: object) -> float:
        return validate_score(value)


class RetrievalResponse(ContractModel):
    """Minimal retrieval response delivered to a downstream verifier."""

    query: str
    results: list[RetrievalResult]

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, value: str) -> str:
        return ensure_non_empty(value, "query")

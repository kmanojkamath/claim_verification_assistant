"""Tunable configuration for document chunking."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChunkingConfig(BaseModel):
    """Token limits used by the chunker; values are intentionally tunable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_tokens: int = Field(default=700, gt=0)
    max_tokens: int = Field(default=900, gt=0)
    overlap_tokens: int = Field(default=100, ge=0)

    @model_validator(mode="after")
    def validate_limits(self) -> "ChunkingConfig":
        if self.target_tokens > self.max_tokens:
            raise ValueError("target_tokens cannot exceed max_tokens")
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        return self

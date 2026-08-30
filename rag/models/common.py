"""Shared types for the RAG input and output contracts."""

from __future__ import annotations

import math
import re
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class SourceType(str, Enum):
    """Supported origin categories for collected evidence."""

    GOVERNMENT_DOCUMENT = "government_document"
    NEWS_ARTICLE = "news_article"
    WEB_PAGE = "web_page"
    SOCIAL_MEDIA_POST = "social_media_post"
    REPORT = "report"
    OTHER = "other"


class ContractModel(BaseModel):
    """Base model that keeps boundary data explicit and predictable."""

    model_config = ConfigDict(extra="forbid")


def ensure_non_empty(value: str, field_name: str) -> str:
    """Reject strings containing no non-whitespace characters."""

    if not value.strip():
        raise ValueError(f"{field_name} cannot be empty")
    return value


class Citation(ContractModel):
    """Source attribution retained with each chunk and retrieval result."""

    url: HttpUrl
    source_type: SourceType
    title: str | None = None
    publisher: str | None = None
    author: str | None = None
    published_at: datetime | None = None


class ChunkLocation(ContractModel):
    """The chunk's position within the normalized parent-document text."""

    chunk_index: int = Field(ge=0)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)

    @field_validator("char_end")
    @classmethod
    def end_must_not_precede_start(cls, value: int, info: Any) -> int:
        start = info.data.get("char_start")
        if start is not None and value < start:
            raise ValueError("char_end must be greater than or equal to char_start")
        return value


LANGUAGE_TAG_PATTERN = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")


def validate_language_tag(value: str) -> str:
    """Perform a pragmatic validation of common BCP-47 language tags."""

    ensure_non_empty(value, "language")
    if not LANGUAGE_TAG_PATTERN.fullmatch(value):
        raise ValueError(
            "language must be a BCP-47 language tag, such as 'en' or 'en-IN'"
        )
    return value


def validate_score(value: Any) -> float:
    """Accept numeric scores while rejecting strings, booleans, NaN, and infinity."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("score must be numeric")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError("score must be finite")
    return numeric_value

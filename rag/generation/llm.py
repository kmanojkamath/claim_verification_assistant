"""Async, provider-isolated grounded generation service."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Protocol

from dotenv import load_dotenv
from pydantic import ValidationError

from .models import GenerationCitation, GenerationResult, LLMGenerationOutput
from .prompts import RetrievedChunk, SYSTEM_PROMPT, build_context

load_dotenv()


class GenerationError(RuntimeError):
    """Raised when an LLM response cannot safely produce a grounded answer."""


class StructuredLLM(Protocol):
    """Minimal async dependency that makes generation independently testable."""

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str: ...


class XAILLM:
    """xAI Responses API client returning strict JSON-schema output."""

    BASE_URL = "https://api.x.ai/v1"

    def __init__(self, model: str | None = None, client: Any | None = None) -> None:
        self.model = model or os.getenv("XAI_MODEL")
        if not self.model:
            raise GenerationError("XAI_MODEL must be configured for xAI generation")
        if client is None:
            api_key = os.getenv("XAI_API_KEY")
            if not api_key:
                raise GenerationError("XAI_API_KEY must be configured for xAI generation")
            try:
                from openai import AsyncOpenAI
            except ImportError as error:
                raise GenerationError("Install the 'openai' package to use xAI generation") from error
            self.client = AsyncOpenAI(api_key=api_key, base_url=self.BASE_URL)
        else:
            self.client = client

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        try:
            response = await self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "grounded_answer",
                        "strict": True,
                        "schema": LLMGenerationOutput.model_json_schema(),
                    }
                },
            )
        except Exception as error:
            raise GenerationError("xAI generation request failed") from error
        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str) or not output_text.strip():
            raise GenerationError("xAI returned no structured answer text")
        return output_text


class GenerationService:
    """Turn retrieved evidence into a validated answer with resolved citations."""

    NO_EVIDENCE_ANSWER = "The available sources do not contain enough information to answer this question."

    def __init__(self, llm: StructuredLLM) -> None:
        self.llm = llm

    @staticmethod
    def _validate_query(query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        return query.strip()

    @classmethod
    def no_evidence_result(cls, query: str) -> GenerationResult:
        cls._validate_query(query)
        return GenerationResult(answer=cls.NO_EVIDENCE_ANSWER)

    async def generate(
        self, query: str, retrieved_chunks: Sequence[RetrievedChunk]
    ) -> GenerationResult:
        query = self._validate_query(query)
        if not retrieved_chunks:
            return self.no_evidence_result(query)

        context = build_context(query, retrieved_chunks)
        try:
            raw_output = await self.llm.complete(
                system_prompt=SYSTEM_PROMPT, user_prompt=context
            )
            output = LLMGenerationOutput.model_validate_json(raw_output)
        except (ValidationError, ValueError) as error:
            raise GenerationError("LLM returned malformed structured output") from error

        citations = self._resolve_citations(output.citations, retrieved_chunks)
        if output.citations and not citations:
            return self.no_evidence_result(query)
        return GenerationResult(answer=output.answer, citations=citations)

    @staticmethod
    def _resolve_citations(
        citation_numbers: Sequence[int], retrieved_chunks: Sequence[RetrievedChunk]
    ) -> list[GenerationCitation]:
        resolved: list[GenerationCitation] = []
        seen: set[int] = set()
        for source_number in citation_numbers:
            if source_number in seen or not 1 <= source_number <= len(retrieved_chunks):
                continue
            seen.add(source_number)
            chunk = retrieved_chunks[source_number - 1]
            resolved.append(
                GenerationCitation(
                    source_number=source_number,
                    chunk_id=chunk.chunk_id,
                    document_id=getattr(chunk, "document_id"),
                    title=getattr(chunk, "title", None),
                    url=getattr(chunk, "url", None),
                    publisher=getattr(chunk, "publisher", None),
                    source_type=getattr(chunk, "source_type", None),
                )
            )
        return resolved


async def generate(
    query: str,
    retrieved_chunks: Sequence[RetrievedChunk],
    llm: StructuredLLM | None = None,
) -> GenerationResult:
    """Generate a grounded answer; pass ``llm`` to inject a test/custom provider."""

    if not retrieved_chunks and llm is None:
        return GenerationService.no_evidence_result(query)
    return await GenerationService(llm or XAILLM()).generate(query, retrieved_chunks)

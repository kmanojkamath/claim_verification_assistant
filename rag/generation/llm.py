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


class GroqLLM:
    """ChatGroq client returning schema-constrained grounded output."""

    DEFAULT_MODEL = "openai/gpt-oss-120b"
    TEMPERATURE = 0.3

    def __init__(self, model: str | None = None, client: Any | None = None) -> None:
        self.model = model or os.getenv("GROQ_MODEL", self.DEFAULT_MODEL)
        if client is None:
            api_key = os.getenv("GROQ_API")
            if not api_key:
                raise GenerationError("GROQ_API must be configured for Groq generation")
            try:
                from langchain_groq import ChatGroq
            except ImportError as error:
                raise GenerationError("Install the 'langchain-groq' package to use Groq generation") from error
            self.client = ChatGroq(
                model=self.model,
                temperature=self.TEMPERATURE,
                api_key=api_key,
            )
        else:
            self.client = client

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        try:
            structured_client = self.client.with_structured_output(
                LLMGenerationOutput,
                method="json_schema",
            )
            response = await structured_client.ainvoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
        except Exception as error:
            raise GenerationError("Groq generation request failed") from error
        try:
            if isinstance(response, LLMGenerationOutput):
                return response.model_dump_json()
            if isinstance(response, dict):
                return LLMGenerationOutput.model_validate(response).model_dump_json()
            content = getattr(response, "content", None)
            if isinstance(content, str) and content.strip():
                return content
        except (ValidationError, ValueError) as error:
            raise GenerationError("Groq returned malformed structured output") from error
        raise GenerationError("Groq returned no structured answer")


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
        return GenerationResult(
            answer=output.answer,
            citations=citations,
            verdict=output.verdict,        
            confidence=output.confidence   
        )

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
    return await GenerationService(llm or GroqLLM()).generate(query, retrieved_chunks)
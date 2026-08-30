import asyncio
import json
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from rag.generation import (
    GenerationError,
    GenerationService,
    LLMGenerationOutput,
    XAILLM,
    SYSTEM_PROMPT,
    build_context,
)


@dataclass
class RetrievedResult:
    chunk_id: str
    document_id: str
    content: str
    score: float = 0.9
    title: str | None = "Example title"
    url: str | None = "https://example.org/source"
    publisher: str | None = "Example publisher"
    author: str | None = "Example author"
    source_type: str | None = "web_page"
    published_at: str | None = "2026-08-30"
    section_heading: str | None = "Details"


class FakeLLM:
    def __init__(self, output: str | Exception) -> None:
        self.output = output
        self.calls: list[tuple[str, str]] = []

    async def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


def results() -> list[RetrievedResult]:
    return [
        RetrievedResult("doc-1:0", "doc-1", "Evidence from the first source."),
        RetrievedResult("doc-2:0", "doc-2", "Evidence from the second source."),
        RetrievedResult("doc-3:0", "doc-3", "Evidence from the third source."),
    ]


def test_context_formats_multiple_sources_with_stable_numbering() -> None:
    context = build_context("What happened?", results())

    assert "USER QUESTION:\nWhat happened?" in context
    assert "SOURCE 1\nChunk ID: doc-1:0" in context
    assert "SOURCE 2\nChunk ID: doc-2:0" in context
    assert "SOURCE 3\nChunk ID: doc-3:0" in context
    assert "CONTENT:\nEvidence from the second source." in context


def test_generation_maps_selected_source_numbers_to_retrieval_metadata() -> None:
    llm = FakeLLM(json.dumps({"answer": "Grounded answer.", "citations": [1, 3]}))

    output = asyncio.run(GenerationService(llm).generate("What happened?", results()))

    assert output.answer == "Grounded answer."
    assert [citation.source_number for citation in output.citations] == [1, 3]
    assert [citation.chunk_id for citation in output.citations] == ["doc-1:0", "doc-3:0"]
    assert output.citations[0].url == "https://example.org/source"
    assert len(llm.calls) == 1


def test_empty_retrieval_does_not_call_the_llm() -> None:
    llm = FakeLLM(json.dumps({"answer": "Should not be used.", "citations": [1]}))

    output = asyncio.run(GenerationService(llm).generate("What happened?", []))

    assert output.citations == []
    assert "do not contain enough information" in output.answer
    assert llm.calls == []


def test_invalid_or_duplicate_citation_indices_are_safely_ignored() -> None:
    llm = FakeLLM(json.dumps({"answer": "Grounded answer.", "citations": [3, 0, 9, 3]}))

    output = asyncio.run(GenerationService(llm).generate("What happened?", results()))

    assert [citation.source_number for citation in output.citations] == [3]
    assert [citation.chunk_id for citation in output.citations] == ["doc-3:0"]


def test_only_invalid_citation_indices_return_the_no_evidence_response() -> None:
    llm = FakeLLM(json.dumps({"answer": "Unsupported answer.", "citations": [0, 9]}))

    output = asyncio.run(GenerationService(llm).generate("What happened?", results()))

    assert output.citations == []
    assert "do not contain enough information" in output.answer


@pytest.mark.parametrize(
    "raw_output",
    ["not json", json.dumps({"answer": "", "citations": []}), json.dumps({"answer": "ok", "citations": ["bad"]})],
)
def test_malformed_structured_output_is_rejected(raw_output: str) -> None:
    with pytest.raises(GenerationError, match="malformed structured output"):
        asyncio.run(GenerationService(FakeLLM(raw_output)).generate("Question", results()))


def test_grounding_prompt_contains_required_safety_rules() -> None:
    assert "only from the retrieved sources" in SYSTEM_PROMPT
    assert "Do not use outside knowledge" in SYSTEM_PROMPT
    assert "do not contain enough information" in SYSTEM_PROMPT
    assert "untrusted data" in SYSTEM_PROMPT


def test_llm_failure_is_reported_without_a_fallback_answer() -> None:
    with pytest.raises(RuntimeError, match="provider failed"):
        asyncio.run(
            GenerationService(FakeLLM(RuntimeError("provider failed"))).generate("Question", results())
        )


def test_xai_client_uses_xai_api_key_and_official_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    created_with: dict[str, str] = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: str) -> None:
            created_with.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI))
    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")
    monkeypatch.setenv("XAI_MODEL", "team-selected-model")

    provider = XAILLM()

    assert provider.model == "team-selected-model"
    assert created_with == {
        "api_key": "test-xai-key",
        "base_url": "https://api.x.ai/v1",
    }


def test_xai_provider_requests_strict_json_schema_and_configured_model_without_network() -> None:
    class FakeResponses:
        async def create(self, **kwargs: object):
            self.kwargs = kwargs
            return type("Response", (), {"output_text": '{"answer":"ok","citations":[1]}'} )()

    class FakeClient:
        def __init__(self) -> None:
            self.responses = FakeResponses()

    client = FakeClient()
    output = asyncio.run(XAILLM(model="configured-xai-model", client=client).complete(
        system_prompt="system", user_prompt="user"
    ))

    assert LLMGenerationOutput.model_validate_json(output).answer == "ok"
    request = client.responses.kwargs
    assert request["model"] == "configured-xai-model"
    assert request["text"]["format"]["type"] == "json_schema"
    assert request["text"]["format"]["strict"] is True


def test_xai_api_failure_is_wrapped_clearly() -> None:
    class FailingResponses:
        async def create(self, **kwargs: object):
            raise RuntimeError("xAI unavailable")

    client = SimpleNamespace(responses=FailingResponses())
    with pytest.raises(GenerationError, match="xAI generation request failed"):
        asyncio.run(XAILLM(model="configured-xai-model", client=client).complete(
            system_prompt="system", user_prompt="user"
        ))

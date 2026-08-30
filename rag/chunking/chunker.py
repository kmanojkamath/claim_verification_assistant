"""Deterministic, token-aware splitting of validated RAG documents."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.models import Chunk, Document, Section

from .config import ChunkingConfig
from .tokenizer import BgeM3Tokenizer, Tokenizer


@dataclass(frozen=True)
class _Span:
    start: int
    end: int


class DocumentChunker:
    """Convert a validated ``Document`` into source-offset-preserving chunks."""

    def __init__(
        self,
        config: ChunkingConfig | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self.config = config or ChunkingConfig()
        self.tokenizer = tokenizer or BgeM3Tokenizer()

    def chunk(self, document: Document) -> list[Chunk]:
        """Return deterministic chunks, preserving original-document offsets."""

        content = document.content
        if self.tokenizer.count_tokens(content) <= self.config.max_tokens:
            return [self._make_chunk(document, 0, _Span(0, len(content)), None)]

        sections = self._mapped_sections(document)
        if sections is None:
            spans = self._split_span(content, _Span(0, len(content)))
            section_for_span: list[Section | None] = [None] * len(spans)
        else:
            spans = []
            section_for_span = []
            for section, span in sections:
                section_spans = self._split_span(content, span)
                spans.extend(section_spans)
                section_for_span.extend([section] * len(section_spans))

        return [
            self._make_chunk(document, index, span, section_for_span[index])
            for index, span in enumerate(spans)
        ]

    def _mapped_sections(self, document: Document) -> list[tuple[Section, _Span]] | None:
        """Map section strings to ordered original-text spans, or safely fall back."""

        if not document.sections:
            return None
        mapped: list[tuple[Section, _Span]] = []
        cursor = 0
        for section in document.sections:
            start = document.content.find(section.content, cursor)
            if start < 0:
                return None
            end = start + len(section.content)
            mapped.append((section, _Span(start, end)))
            cursor = end
        return mapped

    def _split_span(self, source: str, outer: _Span) -> list[_Span]:
        units: list[_Span] = []
        for paragraph in self._paragraph_spans(source, outer):
            if self._count(source, paragraph) <= self.config.max_tokens:
                units.append(paragraph)
                continue
            for sentence in self._sentence_spans(source, paragraph):
                if self._count(source, sentence) <= self.config.max_tokens:
                    units.append(sentence)
                else:
                    units.extend(self._token_spans(source, sentence))
        if not units:
            return []
        base_chunks = self._pack_units(source, units)
        return self._apply_overlap(source, base_chunks)

    def _paragraph_spans(self, source: str, outer: _Span) -> list[_Span]:
        text = source[outer.start : outer.end]
        spans: list[_Span] = []
        for match in re.finditer(r"(?:^|\n\s*\n)(.*?)(?=\n\s*\n|\Z)", text, re.DOTALL):
            value = match.group(1)
            if not value.strip():
                continue
            left = len(value) - len(value.lstrip())
            right = len(value.rstrip())
            start = outer.start + match.start(1) + left
            spans.append(_Span(start, outer.start + match.start(1) + right))
        return spans

    def _sentence_spans(self, source: str, outer: _Span) -> list[_Span]:
        text = source[outer.start : outer.end]
        spans: list[_Span] = []
        for match in re.finditer(r".*?(?:[.!?]+(?=\s|$)|$)", text, re.DOTALL):
            value = match.group()
            if not value.strip():
                continue
            left = len(value) - len(value.lstrip())
            right = len(value.rstrip())
            spans.append(_Span(outer.start + match.start() + left, outer.start + match.start() + right))
        return spans

    def _token_spans(self, source: str, outer: _Span) -> list[_Span]:
        offsets = self.tokenizer.token_offsets(source[outer.start : outer.end])
        offsets = [(start, end) for start, end in offsets if end > start]
        if not offsets:
            return [outer]
        spans: list[_Span] = []
        for index in range(0, len(offsets), self.config.target_tokens):
            group = offsets[index : index + self.config.target_tokens]
            spans.append(_Span(outer.start + group[0][0], outer.start + group[-1][1]))
        return spans

    def _pack_units(self, source: str, units: list[_Span]) -> list[_Span]:
        chunks: list[_Span] = []
        current: _Span | None = None
        for unit in units:
            if current is None:
                current = unit
                continue
            candidate = _Span(current.start, unit.end)
            candidate_tokens = self._count(source, candidate)
            if candidate_tokens <= self.config.max_tokens and self._count(source, current) < self.config.target_tokens:
                current = candidate
            else:
                chunks.append(current)
                current = unit
        if current is not None:
            chunks.append(current)
        return chunks

    def _apply_overlap(self, source: str, chunks: list[_Span]) -> list[_Span]:
        if self.config.overlap_tokens == 0 or len(chunks) < 2:
            return chunks
        output = [chunks[0]]
        for chunk in chunks[1:]:
            # Draw overlap only from the immediately preceding emitted chunk,
            # rather than from all prior text in the section.
            previous = output[-1]
            offsets = self.tokenizer.token_offsets(source[previous.start : previous.end])
            offsets = [(start, end) for start, end in offsets if end > start]
            if not offsets:
                output.append(chunk)
                continue
            take = min(self.config.overlap_tokens, len(offsets))
            start = previous.start + offsets[-take][0]
            while start < chunk.start and self.tokenizer.count_tokens(source[start : chunk.end]) > self.config.max_tokens:
                take -= 1
                if take == 0:
                    start = chunk.start
                    break
                start = previous.start + offsets[-take][0]
            expanded = _Span(start, chunk.end)
            output.append(chunk if expanded == previous else expanded)
        return output

    def _count(self, source: str, span: _Span) -> int:
        return self.tokenizer.count_tokens(source[span.start : span.end])

    def _make_chunk(
        self,
        document: Document,
        index: int,
        span: _Span,
        section: Section | None,
    ) -> Chunk:
        return Chunk(
            chunk_id=f"{document.document_id}:{index}",
            document_id=document.document_id,
            chunk_index=index,
            content=document.content[span.start : span.end],
            char_start=span.start,
            char_end=span.end,
            content_hash=document.content_hash,
            url=document.url,
            source_type=document.source_type,
            title=document.title,
            publisher=document.publisher,
            author=document.author,
            published_at=document.published_at,
            language=document.language,
            metadata=document.metadata,
            section_id=section.section_id if section else None,
            section_heading=section.heading if section else None,
        )


def chunk_document(
    document: Document,
    config: ChunkingConfig | None = None,
    tokenizer: Tokenizer | None = None,
) -> list[Chunk]:
    """Convenience API for one-off document chunking."""

    return DocumentChunker(config=config, tokenizer=tokenizer).chunk(document)

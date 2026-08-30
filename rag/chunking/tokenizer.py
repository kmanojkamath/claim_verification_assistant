"""Tokenizer abstraction used solely for chunk-size measurement and fallback splits."""

from __future__ import annotations

from typing import Protocol


class Tokenizer(Protocol):
    """Minimal contract required by the chunker."""

    def count_tokens(self, text: str) -> int: ...

    def tokenize(self, text: str) -> list[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...

    def token_offsets(self, text: str) -> list[tuple[int, int]]: ...


class BgeM3Tokenizer:
    """Lazy BAAI/bge-m3 tokenizer wrapper; it never loads the embedding model."""

    model_name = "BAAI/bge-m3"

    def __init__(self, model_name: str = model_name) -> None:
        self.model_name = model_name
        self._tokenizer = None

    @property
    def backend(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        return self._tokenizer

    def tokenize(self, text: str) -> list[int]:
        return list(self.backend.encode(text, add_special_tokens=False))

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def decode(self, token_ids: list[int]) -> str:
        return self.backend.decode(token_ids, skip_special_tokens=True)

    def token_offsets(self, text: str) -> list[tuple[int, int]]:
        encoded = self.backend(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        return [tuple(offset) for offset in encoded["offset_mapping"]]

"""Streaming bridges between document adapters and the chunking boundary."""

from .corpus import iter_chunks_from_sqlite

__all__ = ["iter_chunks_from_sqlite"]

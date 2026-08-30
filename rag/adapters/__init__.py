"""Adapters from transitional upstream formats into RAG boundary models."""

from .live_search import live_search_record_to_document, live_search_records_to_documents
from .sqlite import iter_documents_from_sqlite, load_documents_from_sqlite

__all__ = [
    "iter_documents_from_sqlite",
    "live_search_record_to_document",
    "live_search_records_to_documents",
    "load_documents_from_sqlite",
]

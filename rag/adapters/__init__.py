"""Adapters from transitional upstream formats into RAG boundary models."""

from .live_search import live_search_record_to_document, live_search_records_to_documents

__all__ = ["live_search_record_to_document", "live_search_records_to_documents"]

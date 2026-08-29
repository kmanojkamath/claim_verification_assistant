"""Development-only adapter for Team 1's intermediate PIB SQLite corpus.

This module adapts the current SQLite representation into the RAG ``Document``
contract.  Production code should receive ``Document`` objects directly and
must not import this module or SQLite.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rag.models import Document, Section


SECTION_SEPARATOR = "\n\n"
_REQUIRED_COLUMNS = {
    "document_id",
    "url",
    "published_at",
    "language",
    "organization",
    "metadata_json",
    "content_json",
}


def _parse_object(value: str, *, field: str, document_id: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"Document {document_id!r} has invalid {field}: {error.msg}") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"Document {document_id!r} has non-object {field}")
    return parsed


def _required_value(
    name: str,
    document_id: str,
    *candidates: object,
) -> str:
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(
        f"Document {document_id!r} is missing required field {name!r}; "
        "the SQLite adapter will not invent it."
    )


def _merge_metadata(
    document_id: str,
    database_metadata: dict[str, Any],
    content_metadata: object,
) -> dict[str, Any] | None:
    if content_metadata is None:
        return database_metadata or None
    if not isinstance(content_metadata, dict):
        raise ValueError(f"Document {document_id!r} has non-object content_json.metadata")

    merged = dict(content_metadata)
    for key, value in database_metadata.items():
        if key in merged and merged[key] != value:
            raise ValueError(
                f"Document {document_id!r} has conflicting metadata value for {key!r}"
            )
        merged[key] = value
    return merged or None


def _sections_and_content(document_id: str, payload: dict[str, Any]) -> tuple[list[Section], str]:
    raw_sections = payload.get("sections")
    if not isinstance(raw_sections, list) or not raw_sections:
        raise ValueError(f"Document {document_id!r} has no non-empty content_json.sections list")

    sections: list[Section] = []
    for index, raw_section in enumerate(raw_sections):
        if not isinstance(raw_section, dict):
            raise ValueError(f"Document {document_id!r} has non-object section at index {index}")
        try:
            sections.append(Section.model_validate(raw_section))
        except ValueError as error:
            raise ValueError(
                f"Document {document_id!r} has invalid section at index {index}: {error}"
            ) from error

    # This is intentionally limited to the intermediate SQLite adapter.  The
    # final Team 1 export will supply canonical top-level content directly.
    return sections, SECTION_SEPARATOR.join(section.content for section in sections)


def document_from_row(row: sqlite3.Row) -> Document:
    """Convert one ``pib_releases`` row into a validated RAG document."""

    document_id = _required_value("document_id", "<unknown>", row["document_id"])
    database_metadata = _parse_object(
        row["metadata_json"], field="metadata_json", document_id=document_id
    )
    content_payload = _parse_object(
        row["content_json"], field="content_json", document_id=document_id
    )
    content_metadata = content_payload.get("metadata")
    metadata = _merge_metadata(document_id, database_metadata, content_metadata)
    sections, content = _sections_and_content(document_id, content_payload)

    try:
        return Document(
            document_id=document_id,
            source_type=_required_value(
                "source_type", document_id, content_payload.get("source_type"), database_metadata.get("source_type")
            ),
            content=content,
            url=_required_value("url", document_id, row["url"], content_payload.get("url"), database_metadata.get("url")),
            title=row["title"] if isinstance(row["title"], str) and row["title"].strip() else content_payload.get("title"),
            publisher=_required_value("organization", document_id, row["organization"]),
            author=database_metadata.get("author"),
            published_at=row["published_at"] or database_metadata.get("published_at"),
            retrieved_at=_required_value(
                "retrieved_at",
                document_id,
                database_metadata.get("retrieved_at"),
                content_metadata.get("retrieved_at") if isinstance(content_metadata, dict) else None,
            ),
            language=_required_value(
                "language", document_id, row["language"], content_payload.get("language"), database_metadata.get("language")
            ),
            content_hash=database_metadata.get("content_hash"),
            metadata=metadata,
            sections=sections,
        )
    except ValueError as error:
        raise ValueError(f"Document {document_id!r} cannot be mapped to RAG Document: {error}") from error


def load_pib_sqlite(database: Path) -> list[Document]:
    """Read ``pib_releases`` through a read-only SQLite connection."""

    database = database.resolve()
    uri = f"{database.as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(pib_releases)")
        }
        missing_columns = _REQUIRED_COLUMNS - columns
        if missing_columns:
            raise ValueError(
                "pib_releases is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )
        rows: Iterable[sqlite3.Row] = connection.execute(
            "SELECT document_id, url, title, published_at, language, organization, "
            "metadata_json, content_json FROM pib_releases ORDER BY document_id"
        )
        return [document_from_row(row) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path, help="Path to Team 1's PIB SQLite database")
    arguments = parser.parse_args()
    documents = load_pib_sqlite(arguments.database)
    print(f"Loaded {len(documents)} validated documents from {arguments.database}")


if __name__ == "__main__":
    main()

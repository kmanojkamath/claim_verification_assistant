import json
import sqlite3
from pathlib import Path

import pytest

from rag.adapters import iter_documents_from_sqlite, load_documents_from_sqlite
from rag.adapters.sqlite import SECTION_SEPARATOR
from rag.models import Document


def payload(document_id: str = "doc-1") -> dict:
    return {
        "document_id": document_id,
        "source_type": "web_page",
        "url": f"https://example.org/{document_id}",
        "title": f"Content {document_id}",
        "sections": [
            {"section_id": f"{document_id}-one", "heading": "One", "content": "First section."},
            {"section_id": f"{document_id}-two", "heading": "Two", "content": "Second section."},
        ],
        "metadata": {"content_hash": f"sha256:{document_id}", "content_flag": True},
    }


def create_database(tmp_path: Path, rows: list[dict]) -> Path:
    database = tmp_path / "team1.db"
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE pib_releases (document_id TEXT, url TEXT, title TEXT, published_at TEXT, "
            "language TEXT, organization TEXT, metadata_json TEXT, content_json TEXT)"
        )
        connection.executemany(
            "INSERT INTO pib_releases VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row["document_id"],
                    row.get("url", f"https://example.org/{row['document_id']}"),
                    row.get("title", f"Title {row['document_id']}"),
                    row.get("published_at", "2026-08-30T10:00:00+05:30"),
                    row.get("language", "en"),
                    row.get("organization", "Example Organization"),
                    row.get("metadata_json", json.dumps({"retrieved_at": "2026-08-30T11:00:00+05:30", "author": "Example Author", "db_flag": True})),
                    row.get("content_json", json.dumps(payload(row["document_id"]))),
                )
                for row in rows
            ],
        )
    return database


def test_iterator_maps_sections_metadata_and_content_hash_in_deterministic_order(tmp_path: Path) -> None:
    database = create_database(tmp_path, [{"document_id": "doc-b"}, {"document_id": "doc-a"}])

    documents = list(iter_documents_from_sqlite(database))

    assert [document.document_id for document in documents] == ["doc-a", "doc-b"]
    assert all(isinstance(document, Document) for document in documents)
    document = documents[0]
    assert document.content == "First section." + SECTION_SEPARATOR + "Second section."
    assert [section.section_id for section in document.sections or []] == ["doc-a-one", "doc-a-two"]
    assert document.metadata == {
        "content_hash": "sha256:doc-a", "content_flag": True,
        "retrieved_at": "2026-08-30T11:00:00+05:30", "author": "Example Author", "db_flag": True,
    }
    assert document.content_hash == "sha256:doc-a"
    assert document.author == "Example Author"


def test_list_helper_delegates_to_iterator_and_database_remains_unchanged(tmp_path: Path) -> None:
    database = create_database(tmp_path, [{"document_id": "doc-1"}])
    before = database.read_bytes()

    documents = load_documents_from_sqlite(database)

    assert [document.document_id for document in documents] == ["doc-1"]
    assert database.read_bytes() == before


def test_published_at_is_optional_and_retrieved_at_is_required(tmp_path: Path) -> None:
    database = create_database(tmp_path, [{"document_id": "doc-1", "published_at": None}])
    assert next(iter_documents_from_sqlite(database)).published_at is None

    missing_retrieved = create_database(
        tmp_path / "missing",
        [{"document_id": "doc-2", "metadata_json": json.dumps({})}],
    )
    with pytest.raises(ValueError, match="retrieved_at"):
        list(iter_documents_from_sqlite(missing_retrieved))


@pytest.mark.parametrize(
    "row,match",
    [
        ({"document_id": "doc-1", "metadata_json": "{"}, "invalid metadata_json"),
        (
            {
                "document_id": "doc-1",
                "content_json": json.dumps({"sections": payload()["sections"]}),
            },
            "source_type",
        ),
        ({"document_id": "doc-1", "language": None}, "language"),
    ],
)
def test_iterator_fails_clearly_for_malformed_or_missing_required_values(
    tmp_path: Path, row: dict, match: str
) -> None:
    database = create_database(tmp_path, [row])

    with pytest.raises(ValueError, match=match):
        list(iter_documents_from_sqlite(database))

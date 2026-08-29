"""Read-only size analysis for Team 1's intermediate PIB SQLite corpus.

This script intentionally uses only the Python standard library. It measures the
section text currently stored in ``content_json`` and does not clean, deduplicate,
or write to the source database.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any


SECTION_SEPARATOR = "\n\n"
WORD_THRESHOLDS = (500, 750, 1000, 1500, 2000)


def percentile(values: list[int], percent: float) -> float:
    """Return an inclusive linear-interpolation percentile for sorted values."""

    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


def distribution(values: list[int]) -> dict[str, float | int]:
    """Compute requested distribution values without modifying measured data."""

    ordered = sorted(values)
    return {
        "minimum": ordered[0],
        "average": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p90": percentile(ordered, 90),
        "p95": percentile(ordered, 95),
        "p99": percentile(ordered, 99),
        "maximum": ordered[-1],
    }


def load_documents(database: Path) -> list[dict[str, Any]]:
    """Load and measure documents through a read-only SQLite connection."""

    uri = f"{database.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only = ON")
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if "pib_releases" not in tables:
            raise ValueError("Database does not contain the expected 'pib_releases' table")

        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(pib_releases)")
        }
        document_id_column = next(
            (column for column in ("document_id", "doc_id") if column in columns),
            None,
        )
        required_columns = {"title", "content_json"}
        if document_id_column is None or not required_columns.issubset(columns):
            raise ValueError(
                "pib_releases must contain document_id (or doc_id), title, and content_json"
            )
        rows = connection.execute(
            f"SELECT {document_id_column}, title, content_json "
            f"FROM pib_releases ORDER BY {document_id_column}"
        ).fetchall()

    documents: list[dict[str, Any]] = []
    for doc_id, title, content_json in rows:
        payload = json.loads(content_json)
        sections = payload.get("sections", [])
        if not isinstance(sections, list):
            raise ValueError(f"Document {doc_id!r} has a non-list sections value")

        section_contents: list[str] = []
        for index, section in enumerate(sections):
            if not isinstance(section, dict) or not isinstance(section.get("content"), str):
                raise ValueError(
                    f"Document {doc_id!r} has an invalid section at index {index}"
                )
            section_contents.append(section["content"])

        text = SECTION_SEPARATOR.join(section_contents)
        paragraphs = [paragraph for paragraph in text.split(SECTION_SEPARATOR) if paragraph]
        documents.append(
            {
                "document_id": doc_id,
                "title": title,
                "section_count": len(sections),
                "character_count": len(text),
                "word_count": len(text.split()),
                "text_hash": sha256(text.encode("utf-8")).hexdigest(),
                "empty_section_count": sum(not content for content in section_contents),
                "replacement_character_count": text.count("�"),
                "title_replacement_character_count": (title or "").count("�"),
                "has_repeated_paragraphs": len(paragraphs) != len(set(paragraphs)),
                "has_table_like_text": any(
                    "\t" in content
                    or "|" in content
                    or "table " in content.lower()
                    for content in section_contents
                ),
            }
        )
    return documents


def build_report(database: Path) -> dict[str, Any]:
    """Produce a serialisable report based on character and word measurements."""

    documents = load_documents(database)
    if not documents:
        raise ValueError("The pib_releases table contains no documents")

    characters = [document["character_count"] for document in documents]
    words = [document["word_count"] for document in documents]
    by_text_hash: dict[str, list[str]] = defaultdict(list)
    for document in documents:
        by_text_hash[document["text_hash"]].append(document["document_id"])

    measured_documents = [
        {
            key: value
            for key, value in document.items()
            if key
            not in {
                "text_hash",
                "empty_section_count",
                "replacement_character_count",
                "title_replacement_character_count",
                "has_repeated_paragraphs",
                "has_table_like_text",
            }
        }
        for document in documents
    ]
    largest = sorted(
        measured_documents,
        key=lambda document: document["character_count"],
        reverse=True,
    )[:10]

    return {
        "database": str(database.resolve()),
        "measurement_method": {
            "text_reconstruction": "Section content joined in stored order with two newline characters.",
            "content_transformation": "None; metadata is excluded from measurements.",
            "word_count": "Whitespace-delimited tokens using str.split().",
            "percentiles": "Inclusive linear interpolation over sorted values.",
            "token_counts_available": False,
            "token_count_note": "No embedding model or tokenizer is configured by this script.",
        },
        "document_count": len(documents),
        "statistics": {
            "character_count": distribution(characters),
            "word_count": distribution(words),
        },
        "thresholds": {
            "basis": "word_count (tokenizer unavailable)",
            "documents_above": {
                str(threshold): {
                    "count": sum(word_count > threshold for word_count in words),
                    "percentage": round(
                        100 * sum(word_count > threshold for word_count in words) / len(words), 2
                    ),
                }
                for threshold in WORD_THRESHOLDS
            },
        },
        "documents": measured_documents,
        "largest_documents": largest,
        "observations": {
            "exact_duplicate_reconstructed_text_groups": [
                ids for ids in by_text_hash.values() if len(ids) > 1
            ],
            "documents_with_empty_sections": [
                document["document_id"]
                for document in documents
                if document["empty_section_count"]
            ],
            "documents_with_replacement_characters": [
                {
                    "document_id": document["document_id"],
                    "count": document["replacement_character_count"],
                }
                for document in documents
                if document["replacement_character_count"]
            ],
            "documents_with_repeated_paragraphs": [
                document["document_id"]
                for document in documents
                if document["has_repeated_paragraphs"]
            ],
            "titles_with_replacement_characters": [
                {
                    "document_id": document["document_id"],
                    "count": document["title_replacement_character_count"],
                }
                for document in documents
                if document["title_replacement_character_count"]
            ],
            "documents_with_table_like_text": [
                document["document_id"]
                for document in documents
                if document["has_table_like_text"]
            ],
            "documents_under_50_words": [
                document["document_id"]
                for document in documents
                if document["word_count"] < 50
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path, help="Path to the Team 1 SQLite database")
    parser.add_argument("--output", type=Path, help="Optional path for the JSON report")
    arguments = parser.parse_args()

    report = build_report(arguments.database)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if arguments.output:
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()

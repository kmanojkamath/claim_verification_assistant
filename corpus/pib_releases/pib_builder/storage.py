"""SQLite persistence for the PIB corpus."""

import json


def save_release(conn, document):
    """Save one parsed PIB release to SQLite."""
    metadata = document["metadata"]

    conn.execute(
        """
        INSERT OR IGNORE INTO pib_releases
        (
            release_id,
            document_id,
            source,
            url,
            title,
            published_at,
            language,
            organization,
            authority_level,
            metadata_json,
            content_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            metadata["release_id"],
            document["document_id"],
            "PIB",
            document["url"],
            document["title"],
            metadata.get("published_at"),
            metadata.get("language"),
            metadata.get("organization"),
            metadata.get("authority_level"),
            json.dumps(metadata, ensure_ascii=False, indent=2),
            json.dumps(
                {
                    key: value
                    for key, value in document.items()
                    if key != "metadata"
                },
                ensure_ascii=False,
                indent=2,
            ),
        ),
    )
    conn.commit()

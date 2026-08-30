"""SQLite persistence for the internal PIB corpus representation."""

import json
import sqlite3

from .config import DB_FILE


def create_database():
    conn = sqlite3.connect(DB_FILE)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pib_releases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            release_id TEXT UNIQUE,
            document_id TEXT NOT NULL UNIQUE,
            source TEXT NOT NULL,
            url TEXT NOT NULL,
            title TEXT,
            published_at TEXT,
            language TEXT,
            organization TEXT,
            authority_level TEXT,
            metadata_json TEXT NOT NULL,
            content_json TEXT NOT NULL
        )
        """
    )

    conn.commit()
    return conn


def get_existing_releases(conn):
    """Return {release_id: published_at} for releases already in the DB."""
    rows = conn.execute(
        "SELECT release_id, published_at FROM pib_releases"
    ).fetchall()
    return {str(release_id): published_at for release_id, published_at in rows}


def update_published_at(conn, release_id, published_at):
    """Fill a missing publication date without crawling the document again."""
    row = conn.execute(
        "SELECT metadata_json FROM pib_releases WHERE release_id = ?",
        (release_id,),
    ).fetchone()

    if row is None:
        return False

    try:
        metadata = json.loads(row[0])
    except (TypeError, json.JSONDecodeError):
        metadata = {}

    metadata["published_at"] = published_at

    conn.execute(
        """
        UPDATE pib_releases
        SET published_at = ?, metadata_json = ?
        WHERE release_id = ? AND (published_at IS NULL OR published_at = '')
        """,
        (
            published_at,
            json.dumps(metadata, ensure_ascii=False, indent=2),
            release_id,
        ),
    )
    conn.commit()
    return True


def create_crawl_progress_table(conn):
    """Create the date-level crawl progress table."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS crawl_date_progress (
            crawl_date TEXT PRIMARY KEY,
            status TEXT NOT NULL CHECK(status IN ('running', 'completed', 'failed')),
            started_at TEXT,
            completed_at TEXT,
            documents_found INTEGER DEFAULT 0,
            documents_added INTEGER DEFAULT 0,
            error TEXT
        )
        """
    )
    conn.commit()


def get_completed_dates(conn):
    """Return dates that were fully processed successfully."""
    rows = conn.execute(
        "SELECT crawl_date FROM crawl_date_progress WHERE status = 'completed'"
    ).fetchall()
    return {row[0] for row in rows}


def get_date_status(conn, crawl_date):
    row = conn.execute(
        "SELECT status FROM crawl_date_progress WHERE crawl_date = ?",
        (crawl_date,),
    ).fetchone()
    return row[0] if row else None


def mark_date_started(conn, crawl_date):
    """Mark a date as running; a previous failed/running attempt is retried."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO crawl_date_progress
            (crawl_date, status, started_at, completed_at, documents_found, documents_added, error)
        VALUES (?, 'running', ?, NULL, 0, 0, NULL)
        ON CONFLICT(crawl_date) DO UPDATE SET
            status = 'running',
            started_at = excluded.started_at,
            completed_at = NULL,
            documents_found = 0,
            documents_added = 0,
            error = NULL
        """,
        (crawl_date, now),
    )
    conn.commit()


def mark_date_completed(conn, crawl_date, documents_found, documents_added):
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        UPDATE crawl_date_progress
        SET status = 'completed',
            completed_at = ?,
            documents_found = ?,
            documents_added = ?,
            error = NULL
        WHERE crawl_date = ?
        """,
        (now, documents_found, documents_added, crawl_date),
    )
    conn.commit()


def mark_date_failed(conn, crawl_date, error, documents_found=0, documents_added=0):
    conn.execute(
        """
        UPDATE crawl_date_progress
        SET status = 'failed',
            documents_found = ?,
            documents_added = ?,
            error = ?
        WHERE crawl_date = ?
        """,
        (documents_found, documents_added, str(error)[:2000], crawl_date),
    )
    conn.commit()


def get_last_completed_date(conn):
    """Return the latest completed crawl date, if any."""
    row = conn.execute(
        "SELECT MAX(crawl_date) FROM crawl_date_progress WHERE status = 'completed'"
    ).fetchone()
    return row[0] if row and row[0] else None

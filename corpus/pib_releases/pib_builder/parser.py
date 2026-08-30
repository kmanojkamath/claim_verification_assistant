"""Parse crawled PIB HTML into the internal document structure."""

import re
from datetime import datetime, timezone, timedelta

from bs4 import BeautifulSoup

from .text_utils import (
    normalize_text, remove_boilerplate, find_main_article, extract_title,
    extract_ministry, detect_language, document_has_tables, extract_sections,
    clean_block_text,
)


def parse_datetime_to_iso(value):
    """Convert common PIB publication date strings to ISO-8601 with IST."""
    if not value:
        return None

    value = normalize_text(value)
    value = re.sub(r"\s+", " ", value).strip()

    # PIB commonly emits timestamps such as:
    # "10 MAY 2025 2:31PM". Normalize compact AM/PM spacing so
    # the standard datetime formats below can parse it.
    value = re.sub(
        r"(\d)(AM|PM)\b",
        r"\1 \2",
        value,
        flags=re.IGNORECASE
    )

    try:
        dt = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )

        if dt.tzinfo is None:
            dt = dt.replace(
                tzinfo=timezone(timedelta(hours=5, minutes=30))
            )

        return dt.isoformat()
    except ValueError:
        pass

    formats = [
        "%d-%m-%Y %H:%M:%S",
        "%d-%m-%Y %H:%M",
        "%d-%m-%Y %I:%M:%S %p",
        "%d-%m-%Y %I:%M %p",
        "%d-%m-%Y",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y",
        "%d.%m.%Y %H:%M:%S",
        "%d.%m.%Y %H:%M",
        "%d.%m.%Y",
        "%d %B %Y %H:%M:%S",
        "%d %B %Y %H:%M",
        "%d %B %Y %I:%M:%S %p",
        "%d %B %Y %I:%M %p",
        "%d %B %Y",
        "%d %b %Y %H:%M:%S",
        "%d %b %Y %H:%M",
        "%d %b %Y",
    ]

    ist = timezone(timedelta(hours=5, minutes=30))

    for fmt in formats:
        try:
            dt = datetime.strptime(value, fmt)
            return dt.replace(tzinfo=ist).isoformat()
        except ValueError:
            continue

    match = re.search(
        r"(\d{1,2}[-/.]\d{1,2}[-/.]\d{4}"
        r"(?:\s+\d{1,2}:\d{2}(?::\d{2})?"
        r"(?:\s*[AP]M)?)?)",
        value,
        re.IGNORECASE
    )

    if match:
        candidate = match.group(1)

        for fmt in formats:
            try:
                dt = datetime.strptime(candidate, fmt)
                return dt.replace(tzinfo=ist).isoformat()
            except ValueError:
                continue

    return None



def extract_prid(text):

    if not text:
        return None

    match = re.search(
        r"PRID\s*=\s*(\d+)",
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1)

    return None


def extract_date(text):
    """Extract PIB publication date from common PIB page variants."""
    if not text:
        return None

    text = normalize_text(text)

    patterns = [
        # Standard PIB form.
        r"Posted\s+On\s*:\s*(.*?)\s+by\s+PIB(?:\s|$)",
        # Handles pages where the 'by PIB' suffix is absent/changed.
        r"Posted\s+On\s*:\s*([^\n]+)",
        # Some rendered versions use the date label without the colon spacing.
        r"Posted\s+On\s*[-:]\s*(.*?)(?:\s+by\s+PIB(?:\s|$)|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = normalize_text(match.group(1))
            if value:
                return value

    return None


def parse_pib(html, url, fallback_release_id, fallback_title, fallback_published_at=None):
    if not html:
        raise RuntimeError(
            "Crawler returned empty HTML."
        )

    soup = BeautifulSoup(
        str(html),
        "html.parser"
    )

    soup = remove_boilerplate(soup)

    article = find_main_article(soup)

    if article is None:
        article = soup

    try:
        article_text = normalize_text(
            article.get_text("\n", strip=True)
        )
    except Exception:
        article_text = normalize_text(
            soup.get_text("\n", strip=True)
        )

    release_match = re.search(
        r"Release\s+ID\s*:\s*(\d+)",
        article_text,
        re.IGNORECASE
    )

    release_id = (
        release_match.group(1)
        if release_match
        else fallback_release_id
    )

    document_id = "PIB-" + str(release_id)

    title = extract_title(
        article,
        fallback_title
    )

    date = extract_date(article_text)
    ministry = extract_ministry(
        article,
        title
    )

    language = detect_language(
        article_text
    )

    has_tables = document_has_tables(
        article
    )

    sections = extract_sections(
        article,
        document_id
    )

    if not sections and article_text:
        fallback_content = clean_block_text(
            article_text
        )

        if fallback_content:
            sections = [{
                "section_id": f"{document_id}-S01",
                "heading": "Main Content",
                "content": fallback_content
            }]

    ist = timezone(
        timedelta(hours=5, minutes=30)
    )

    retrieved_at = datetime.now(
        ist
    ).isoformat()

    published_at = parse_datetime_to_iso(date)

    # The discovery page itself is date-scoped. If a particular release page
    # does not expose a parseable "Posted On" value, use the discovery date
    # rather than storing NULL. This preserves the known publication day.
    if published_at is None and fallback_published_at:
        published_at = parse_datetime_to_iso(fallback_published_at)


    metadata = {
        "source": "PIB",
        "url": url,
        "source_type": "web_page",
        "organization": (
            ministry
            or "Press Information Bureau"
        ),
        "publisher": (
            "Press Information Bureau, "
            "Government of India"
        ),
        "published_at": published_at,
        "retrieved_at": retrieved_at,
        "language": language,
        "release_id": release_id,
        "author": "Press Information Bureau",
        "topic_tags": [],
        "document_type": "press_release",
        "authority_level": "primary_official",
        "has_tables": has_tables
    }

    return {
        "document_id": document_id,
        "title": title,
        "source_type": "web_page",
        "url": url,
        "sections": sections,
        "metadata": metadata
    }


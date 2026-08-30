"""HTML/text cleaning and section extraction helpers."""

import re

SECTION_HEADING_PATTERNS = [
    r"^background$", r"^introduction$", r"^overview$", r"^key details?$",
    r"^highlights?$", r"^key highlights?$", r"^details?$", r"^objectives?$",
    r"^purpose$", r"^aims?$", r"^benefits?$", r"^features?$", r"^eligibility$",
    r"^implementation$", r"^implementation details?$", r"^financial implications?$",
    r"^funding$", r"^timeline$", r"^important points?$", r"^key points?$",
    r"^decisions?$", r"^key decisions?$", r"^announcements?$",
    r"^key announcements?$", r"^additional information$", r"^way forward$",
    r"^conclusion$", r"^about .*",
]


def table_to_text(table):
    """
    Convert an HTML table to readable, loss-minimizing plain text.

    Each row remains a row and cells are separated with " | ". This preserves
    the factual values/order without leaving extraction markers such as
    [TABLE] in the RAG text.
    """
    rows = []

    for tr in table.find_all("tr"):
        cells = []
        for cell in tr.find_all(["th", "td"], recursive=False):
            value = clean_block_text(
                cell.get_text(" ", strip=True)
            )
            if value:
                cells.append(value)

        # Some tables nest cells through extra markup.
        if not cells:
            for cell in tr.find_all(["th", "td"]):
                value = clean_block_text(
                    cell.get_text(" ", strip=True)
                )
                if value:
                    cells.append(value)

        if cells:
            rows.append(" | ".join(cells))

    return "\n".join(rows)


def list_to_text(list_tag):
    """Convert an HTML list into readable text while preserving order."""
    lines = []
    ordered = list_tag.name == "ol"

    for index, li in enumerate(
        list_tag.find_all("li", recursive=False), start=1
    ):
        # Clone-free approach: remove nested lists from a temporary fragment.
        nested_lists = li.find_all(["ul", "ol"], recursive=False)
        for nested in nested_lists:
            nested.extract()

        value = clean_block_text(
            li.get_text(" ", strip=True)
        )
        if not value:
            continue

        prefix = f"{index}. " if ordered else "• "
        lines.append(prefix + value)

    return "\n".join(lines)


def block_to_text(element):
    """Convert one HTML content block to clean text."""
    if element.name == "table":
        return table_to_text(element)

    if element.name in ["ul", "ol"]:
        return list_to_text(element)

    return clean_block_text(
        element.get_text(" ", strip=True)
    )


def clean_block_text(text):
    """
    Normalize extraction artifacts without summarizing/paraphrasing.

    Important: wording, numbers, dates, names and punctuation are retained.
    """
    if not text:
        return ""

    text = str(text).replace("\xa0", " ")

    # Crawl/extraction markers.
    text = re.sub(r"\[\s*/?\s*TABLE\s*\]", "", text, flags=re.I)
    text = re.sub(r"\[\s*/?\s*IMAGE\s*\]", "", text, flags=re.I)

    # Common invisible/control extraction characters.
    text = text.replace("\u200b", "")
    text = text.replace("\u200c", "")
    text = text.replace("\u200d", "")
    text = text.replace("\ufeff", "")

    # Normalize line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Horizontal whitespace only; do not collapse paragraph boundaries here.
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalize_text(text):
    """Final document-level whitespace normalization."""
    cleaned = clean_block_text(text)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _element_identity(element):
    """Return normalized id/class text for boilerplate detection."""
    element_id = element.get("id") or ""
    classes = element.get("class") or []

    if not isinstance(classes, list):
        classes = [str(classes)]

    return " ".join(
        [str(element_id)] + [str(x) for x in classes]
    ).lower()


def remove_boilerplate(soup):
    """
    Remove page chrome while preserving the article body.

    We deliberately use conservative signals because an over-aggressive
    cleaner can delete factual content from a government release.
    """
    for tag in soup.find_all([
        "script", "style", "noscript", "iframe", "svg",
        "canvas", "template"
    ]):
        try:
            tag.decompose()
        except Exception:
            pass

    unwanted_patterns = [
        "navbar", "nav-bar", "navigation", "menu",
        "footer", "cookie", "advert", "advertisement",
        "social", "share", "sidebar", "breadcrumb",
        "recommend", "related", "popup", "modal",
        "skip", "accessibility"
    ]

    for element in list(soup.find_all(True)):
        try:
            identity = _element_identity(element)
            tag_name = element.name.lower()

            if tag_name in {"nav", "footer", "aside"}:
                element.decompose()
                continue

            if any(pattern in identity for pattern in unwanted_patterns):
                element.decompose()
        except Exception:
            continue

    return soup


def _meaningful_text_length(element):
    try:
        return len(
            normalize_text(
                element.get_text(" ", strip=True)
            )
        )
    except Exception:
        return 0


def find_main_article(soup):
    """
    Select the smallest high-confidence container holding the PIB release.

    The old implementation picked the smallest matching container, which
    could still be a nested wrapper. This version scores article/main/content
    candidates and strongly prefers semantic article containers.
    """
    if soup is None:
        return None

    candidates = []

    preferred_tags = [
        "article", "main",
        "div", "section"
    ]

    for tag_name in preferred_tags:
        for element in soup.find_all(tag_name):
            try:
                text = normalize_text(
                    element.get_text(" ", strip=True)
                )

                if len(text) < 300:
                    continue

                has_posted = bool(re.search(
                    r"Posted\s+On\s*:",
                    text,
                    re.IGNORECASE
                ))
                has_release_id = bool(re.search(
                    r"Release\s+ID\s*:\s*\d+",
                    text,
                    re.IGNORECASE
                ))

                if not (has_posted and has_release_id):
                    continue

                identity = _element_identity(element)

                semantic_bonus = 0
                if tag_name == "article":
                    semantic_bonus += 100000
                if tag_name == "main":
                    semantic_bonus += 50000
                if any(
                    token in identity
                    for token in [
                        "content", "article", "release",
                        "press", "detail", "body"
                    ]
                ):
                    semantic_bonus += 10000

                # Prefer a reasonably small container while still preserving
                # the complete release.
                score = semantic_bonus - len(text)

                candidates.append((score, element))
            except Exception:
                continue

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    return soup.body if soup.body is not None else soup


def extract_title(article, fallback_title):
    title = normalize_text(fallback_title)

    if article is None:
        return title

    try:
        for tag_name in ["h1", "h2", "h3"]:
            for tag in article.find_all(tag_name):
                candidate = normalize_text(
                    tag.get_text(" ", strip=True)
                )
                if not candidate:
                    continue

                if "press information bureau" in candidate.lower():
                    continue

                if re.fullmatch(
                    r"(?:release\s+id|posted\s+on)\s*:?.*",
                    candidate,
                    re.I
                ):
                    continue

                return candidate
    except Exception:
        pass

    return title


def extract_ministry(article, title):
    """Extract the ministry/organization line when PIB exposes it."""
    if article is None:
        return None

    try:
        text = normalize_text(
            article.get_text("\n", strip=True)
        )

        title_position = text.find(title)
        if title_position <= 0:
            return None

        before_title = text[:title_position]
        lines = [
            normalize_text(x)
            for x in before_title.splitlines()
            if normalize_text(x)
        ]

        ignored = {
            "government of india",
            "press information bureau",
            "press information bureau government of india",
            "home",
            "press releases",
            "all press release",
            "all releases"
        }

        for line in reversed(lines):
            low = line.lower()
            if low in ignored:
                continue
            if "posted on:" in low:
                continue
            if re.fullmatch(r"release\s+id\s*:?\s*\d+", low):
                continue
            return line
    except Exception:
        pass

    return None


def detect_language(text):
    if not text:
        return "en"

    devanagari = len(re.findall(r"[\u0900-\u097F]", text))
    latin = len(re.findall(r"[A-Za-z]", text))

    if devanagari > 0 and devanagari > latin * 0.25:
        return "hi"

    return "en"


def looks_like_heading(text):
    text = normalize_text(text)
    if not text:
        return False

    lowered = text.lower().strip()

    if any(
        re.match(pattern, lowered)
        for pattern in SECTION_HEADING_PATTERNS
    ):
        return True

    if re.match(
        r"^(?:\d+[\.\)]|[IVXLCDM]+[\.\)])\s+.{2,100}$",
        text
    ):
        return True

    if len(text) <= 100 and text.endswith(":"):
        return True

    return False


def heading_name(text):
    text = normalize_text(text)
    text = re.sub(
        r"^[\dIVXLCDM]+[\.\)]\s+",
        "",
        text,
        flags=re.I
    )
    return text.rstrip(":").strip()


def _dedupe_text_blocks(blocks):
    """
    Exact/whitespace-normalized deduplication.

    We do NOT use fuzzy deduplication: two similar paragraphs may contain
    different numbers/dates and therefore both need to survive.
    """
    result = []
    seen = set()

    for block in blocks:
        text = clean_block_text(block)
        if not text:
            continue

        key = re.sub(r"\s+", " ", text).strip().casefold()

        if key in seen:
            continue

        seen.add(key)
        result.append(text)

    return result


def _iter_article_blocks(article):
    """
    Yield meaningful top-level content blocks without collecting nested
    paragraphs from the same table/list twice.

    This is the main fix for duplicated extraction.
    """
    if article is None:
        return

    block_tags = {
        "h1", "h2", "h3", "h4", "h5", "h6",
        "p", "ul", "ol", "table", "blockquote"
    }

    def walk(parent):
        for child in parent.children:
            if not getattr(child, "name", None):
                continue

            name = child.name.lower()

            if name in block_tags:
                yield child
                continue

            # Ignore page metadata containers that contain no actual block
            # tags; recurse only through wrappers.
            yield from walk(child)

    yield from walk(article)


def _is_article_metadata_block(text):
    """
    Remove page-level extraction metadata from RAG content.

    These values remain available through canonical fields/metadata and are
    not part of the release's substantive prose.
    """
    low = normalize_text(text).lower()

    if re.fullmatch(r"release\s+id\s*:\s*\d+", low):
        return True

    if re.match(
        r"^posted\s+on\s*:\s*.+\s+by\s+pib\s*$",
        low,
        re.I
    ):
        return True

    if low in {
        "government of india",
        "press information bureau",
        "press information bureau government of india"
    }:
        return True

    return False


def _extract_clean_blocks(article):
    blocks = []

    for element in _iter_article_blocks(article):
        try:
            text = block_to_text(element)
            text = clean_block_text(text)

            if not text or _is_article_metadata_block(text):
                continue

            blocks.append({
                "name": element.name.lower(),
                "text": text,
                "is_table": element.name.lower() == "table"
            })
        except Exception:
            continue

    # Exact deduplication while preserving the first occurrence and table flag.
    result = []
    seen = set()

    for block in blocks:
        key = re.sub(
            r"\s+",
            " ",
            block["text"]
        ).strip().casefold()

        if key in seen:
            continue

        seen.add(key)
        result.append(block)

    return result


def infer_sections_from_blocks(blocks, document_id):
    """
    Build conservative section boundaries from actual headings.

    No new factual text is invented. If the source has no recognizable
    headings, content stays under Main Content.
    """
    sections = []
    current_heading = "Main Content"
    current_blocks = []
    section_number = 1

    def save():
        nonlocal section_number, current_blocks

        content_blocks = _dedupe_text_blocks(current_blocks)

        if not content_blocks:
            return

        sections.append({
            "section_id": f"{document_id}-S{section_number:02d}",
            "heading": current_heading,
            "content": "\n\n".join(content_blocks).strip()
        })

        section_number += 1
        current_blocks = []

    for block in blocks:
        name = block["name"]
        raw_text = block["text"]

        if name in {
            "h1", "h2", "h3", "h4", "h5", "h6"
        }:
            heading = heading_name(raw_text)

            if not heading:
                continue

            if heading.lower() != current_heading.lower():
                save()
                current_heading = heading

            continue

        if name == "p" and looks_like_heading(raw_text):
            heading = heading_name(raw_text)

            if len(heading.split()) <= 12:
                if heading.lower() != current_heading.lower():
                    save()
                    current_heading = heading
                continue

        current_blocks.append(raw_text)

    save()
    return sections


def extract_sections(article, document_id):
    """
    Return the stable section contract:
      section_id, heading, content

    Every retained paragraph/list/table contributes to the document content.
    Tables are represented as readable rows rather than [TABLE] markers.
    """
    if article is None:
        return []

    blocks = _extract_clean_blocks(article)
    return infer_sections_from_blocks(blocks, document_id)


def document_has_tables(article):
    """Return True when the retained article contains an HTML table."""
    if article is None:
        return False

    return any(
        block["is_table"]
        for block in _extract_clean_blocks(article)
    )


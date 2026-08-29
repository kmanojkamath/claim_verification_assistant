"""Crawl individual PIB releases with Crawl4AI and persist results to SQLite."""

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

from .database import create_database, get_existing_releases, update_published_at
from .discovery import crawler_url
from .parser import parse_pib, parse_datetime_to_iso
from .storage import save_release
from .config import DB_FILE


async def crawl_releases(releases):
    print("\n" + "=" * 70)
    print("CRAWL4AI → CLEAN JSON → SQLITE")
    print("=" * 70)

    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
    )

    crawler_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=10,
        wait_until="domcontentloaded",
    )

    successful = 0
    failed = 0
    conn = create_database()
    existing = get_existing_releases(conn)
    skipped_existing = 0
    repaired_dates = 0

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            queue = list(releases.values())

            for i, release in enumerate(queue, 1):
                release_id = release["release_id"]
                original_url = release["url"]
                fallback_title = release["title"]
                discovered_date = release.get("discovered_date")
                url = crawler_url(original_url)

                print(f"\n[{i}/{len(queue)}]")
                print("Release ID:", release_id)

                # Never crawl a document that is already in the database.
                # If its old row has a NULL publication date, repair that
                # value from the date-scoped discovery page instead.
                if release_id in existing:
                    if not existing[release_id] and discovered_date:
                        published_at = parse_datetime_to_iso(discovered_date)
                        if published_at:
                            update_published_at(conn, release_id, published_at)
                            existing[release_id] = published_at
                            repaired_dates += 1
                            print("  ↻ Existing release; filled missing published_at:", published_at)
                    skipped_existing += 1
                    print("  ⏭ Already in database; skipping crawl.")
                    continue

                print("URL:", url)

                try:
                    result = await crawler.arun(
                        url=url,
                        config=crawler_config,
                    )
                except Exception as e:
                    print("  ❌ Crawl error:", e)
                    failed += 1
                    continue

                if not result.success:
                    print("  ❌ Crawl failed:")
                    print(result.error_message)
                    failed += 1
                    continue

                html = getattr(result, "cleaned_html", None) or getattr(result, "html", None)

                if not html:
                    print("  ❌ No HTML returned.")
                    failed += 1
                    continue

                try:
                    document = parse_pib(
                        html,
                        original_url,
                        release_id,
                        fallback_title,
                        discovered_date,
                    )
                except Exception as e:
                    print("  ❌ Parsing error:", repr(e))
                    failed += 1
                    continue

                if not document.get("sections"):
                    print("  ❌ No sections extracted.")
                    failed += 1
                    continue

                try:
                    save_release(conn, document)
                except Exception as e:
                    print("  ❌ Database error:", repr(e))
                    failed += 1
                    continue

                successful += 1

                print("  TITLE:", document["title"])
                print("  DOC ID:", document["document_id"])
                print("  SECTIONS:", len(document["sections"]))
                print("  LANGUAGE:", document["metadata"]["language"])
                print("  MINISTRY:", document["metadata"]["organization"])
                print("  DATE:", document["metadata"]["published_at"])
                print("  ✅ Stored in database")
    finally:
        conn.close()

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Releases found      : {len(releases)}")
    print(f"Already in database : {skipped_existing}")
    print(f"Dates repaired      : {repaired_dates}")
    print(f"Successfully stored : {successful}")
    print(f"Failed              : {failed}")
    print(f"Database            : {DB_FILE}")

    return {
        "successful": successful,
        "failed": failed,
        "skipped_existing": skipped_existing,
        "repaired_dates": repaired_dates,
    }

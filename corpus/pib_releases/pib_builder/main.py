"""Command-line entry point for the PIB corpus builder.

Supports:
    python run.py                              # automatic/resume mode
    python run.py --start-date DD-MM-YYYY --end-date DD-MM-YYYY  # manual mode
"""

import argparse
import asyncio
import re
from datetime import datetime, timedelta

from playwright.async_api import async_playwright

from .config import PIB_URL, DB_FILE, AUTO_START_DATE, AUTO_END_TODAY
from .database import (
    create_database,
    create_crawl_progress_table,
    get_completed_dates,
    mark_date_started,
    mark_date_completed,
    mark_date_failed,
)
from .discovery import select_date, get_release_links
from .crawler import crawl_releases


DATE_FORMAT = "%d-%m-%Y"


def date_range(start_date, end_date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def parse_date(value):
    return datetime.strptime(value, DATE_FORMAT)


def build_parser():
    parser = argparse.ArgumentParser(description="PIB corpus builder")
    parser.add_argument("--start-date", help="Manual start date (DD-MM-YYYY)")
    parser.add_argument("--end-date", help="Manual end date (DD-MM-YYYY)")
    return parser


def resolve_range(args, conn):
    if bool(args.start_date) != bool(args.end_date):
        raise ValueError("Provide both --start-date and --end-date, or neither.")

    if args.start_date and args.end_date:
        start = parse_date(args.start_date)
        end = parse_date(args.end_date)
        if start > end:
            raise ValueError("FROM date cannot be later than TO date.")
        return start, end, False

    if not AUTO_START_DATE:
        raise ValueError("Set AUTO_START_DATE in pib_builder/config.py before using automatic mode.")

    start = parse_date(AUTO_START_DATE)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    end = today if AUTO_END_TODAY else today - timedelta(days=1)

    completed = get_completed_dates(conn)
    current = start
    while current.strftime(DATE_FORMAT) in completed:
        current += timedelta(days=1)

    if current > end:
        return None, None, True

    return current, end, True


async def process_date(page, current_date, conn, automatic):
    date_string = current_date.strftime(DATE_FORMAT)
    mark_date_started(conn, date_string)

    try:
        await page.goto(PIB_URL, wait_until="networkidle", timeout=60000)
    except Exception as e:
        mark_date_failed(conn, date_string, f"Could not open PIB: {e}")
        print("❌ Could not open PIB:", repr(e))
        return False

    success = await select_date(page, date_string)
    if not success:
        mark_date_failed(conn, date_string, "Could not select date")
        print(f"❌ Could not select {date_string}.")
        return False

    body = await page.locator("body").inner_text()
    result_match = re.search(r"Displaying.*", body)
    if result_match:
        print("\n" + result_match.group(0))

    releases = await get_release_links(page)
    print(f"Found {len(releases)} unique releases for {date_string}.")

    for release in releases.values():
        release["discovered_date"] = date_string

    if not releases:
        # A successful discovery with zero releases is still a completed date.
        mark_date_completed(conn, date_string, 0, 0)
        return True

    stats = await crawl_releases(releases)

    # Existing documents are valid work for this date, so a crawl is successful
    # as long as every discovered release was either stored or already present.
    if stats["failed"] == 0:
        added = stats["successful"]
        mark_date_completed(conn, date_string, len(releases), added)
        return True

    mark_date_failed(
        conn,
        date_string,
        f"{stats['failed']} release(s) failed to crawl/store",
        len(releases),
        stats["successful"],
    )
    return False


async def main():
    args = build_parser().parse_args()
    conn = create_database()
    create_crawl_progress_table(conn)

    try:
        try:
            from_date, to_date, automatic = resolve_range(args, conn)
        except ValueError as e:
            print(f"\n❌ {e}")
            return

        if from_date is None:
            print("\n" + "=" * 70)
            print("PIB CORPUS BUILDER")
            print("=" * 70)
            print("✅ Corpus is already up to date.")
            print("No unprocessed dates found in the configured range.")
            return

        dates = list(date_range(from_date, to_date))

        print("\n" + "=" * 70)
        print("PIB CORPUS BUILDER")
        print("=" * 70)
        print(f"MODE : {'AUTOMATIC / RESUME' if automatic else 'MANUAL'}")
        print(f"FROM : {from_date.strftime(DATE_FORMAT)}")
        print(f"TO   : {to_date.strftime(DATE_FORMAT)}")
        print(f"DAYS : {len(dates)}")
        if automatic:
            print("Progress is stored in the SQLite database.")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()
            try:
                for date_number, current_date in enumerate(dates, 1):
                    date_string = current_date.strftime(DATE_FORMAT)
                    print("\n" + "=" * 70)
                    print(f"DATE [{date_number}/{len(dates)}]: {date_string}")
                    print("=" * 70)

                    completed = get_completed_dates(conn)
                    if date_string in completed:
                        print("⏭ Date already completed; skipping.")
                        continue

                    ok = await process_date(page, current_date, conn, automatic)
                    if not ok and automatic:
                        print("❌ Date failed. Stopping automatic mode so the next run retries it.")
                        break
            finally:
                await browser.close()
    finally:
        conn.close()


if __name__ == "__main__":
    asyncio.run(main())

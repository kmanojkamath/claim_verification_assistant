# PIB Corpus Builder

Builds a structured PIB corpus for a date range and stores the extracted releases directly in SQLite.

## Run

```bash
pip install -r requirements.txt
playwright install chromium
python run.py
```

The program asks for two dates:

```text
Enter FROM date (DD-MM-YYYY): 01-05-2025
Enter TO date (DD-MM-YYYY): 07-05-2025
```

Both dates are **inclusive**. Releases are discovered for every date in the range, duplicates are removed by release ID, and all discovered releases are crawled and stored in SQLite.

## Output

Only one persistent output is produced:

```text
main_corpus.db
```

No `rag_output/`, RAG JSON files, JSONL files, or representative samples are generated.

## Database

The `pib_releases` table stores:

- release ID and document ID
- source and URL
- title
- publication timestamp
- language
- organization
- authority level
- metadata as JSON
- structured document/content as JSON

The database is the single corpus output and can be consumed directly by the RAG team.


### Publication date and deduplication

- `published_at` is extracted from PIB's `Posted On` field using multiple page variants.
- If a release page does not expose a parseable date, the date from the date-scoped discovery page is used as a fallback, so `published_at` is not left NULL for newly discovered releases.
- Existing release IDs are never crawled again. Existing rows with a missing `published_at` are repaired directly from the discovery date without re-crawling the document.

## Automatic / Resume Mode

You can now run the builder without entering dates:

```bash
python run.py
```

Set `AUTO_START_DATE` once in `pib_builder/config.py`. Automatic mode processes dates through yesterday by default and stores date-level progress in the SQLite database (`crawl_date_progress`).

- Completed dates are skipped on later runs.
- A failed date is retried on the next run.
- Automatic mode stops at the first failed date so it cannot silently skip a gap.
- Existing releases are still deduplicated by the existing unique database keys.
- Use `AUTO_END_TODAY = True` only if you want the current day included.

Manual mode is still available:

```bash
python run.py --start-date 01-05-2025 --end-date 07-05-2025
```

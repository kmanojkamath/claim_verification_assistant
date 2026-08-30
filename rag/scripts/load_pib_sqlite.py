"""Developer CLI for validating Team 1's PIB SQLite corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag.adapters.sqlite import SECTION_SEPARATOR, load_documents_from_sqlite


# Backwards-compatible developer utility alias. Reusable logic lives in
# ``rag.adapters.sqlite``.
load_pib_sqlite = load_documents_from_sqlite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path, help="Path to Team 1's PIB SQLite database")
    arguments = parser.parse_args()
    documents = load_documents_from_sqlite(arguments.database)
    print(f"Loaded {len(documents)} validated documents from {arguments.database}")


if __name__ == "__main__":
    main()

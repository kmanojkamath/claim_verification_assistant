"""Project configuration."""

PIB_URL = "https://www.pib.gov.in/AllReleasem.aspx?lang=1&reg=3"
PIB_BASE = "https://www.pib.gov.in/"
DB_FILE = "main_corpus.db"

# Automatic corpus building.
# Set this once to the first date you want the automatic builder to process.
# Example: "01-01-2025"
AUTO_START_DATE = "01-01-2025"

# Automatic mode processes up to yesterday by default. This avoids repeatedly
# crawling the current day while PIB may still be publishing releases.
AUTO_END_TODAY = False


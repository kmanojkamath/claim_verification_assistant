import os
import json
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel
import ocr

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Check your .env file."
    )
    
class Claim(BaseModel):
    raw_text: str
    query: str
    keywords: List[str]
    category: str
    subcategory: str
    claim_type: str
    is_government_related: bool
    is_current_affairs: bool
    is_time_sensitive: bool


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "input.txt"
OUTPUT_FILE = BASE_DIR / "output.json"


def process_claim(flag):

    ocr.run_ocr(flag)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"input.txt not found at {INPUT_FILE}"
        )

    client = genai.Client(
        api_key=API_KEY
    )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        raw_text = file.read().strip()

    if not raw_text:
        raise ValueError(
            "input.txt is empty."
        )

    prompt = f"""
You are an information extraction system for a
Government Communication Misinformation and
Claim Verification application.

Analyze the following raw text and convert it into
the required structured JSON.

RAW TEXT:

{raw_text}

RULES:

1. raw_text:
   Return the original input text exactly as provided.
   Do not rewrite it.

2. query:
   Extract the main factual claim from the text and
   convert it into a clear, direct question or statement
   that can be used to verify the claim.

3. Remove irrelevant conversational phrases such as:
   - "my friend said"
   - "I heard"
   - "someone told me"
   - "apparently"
   - "according to my friend"

4. IMPORTANT:
   Do NOT remove meaningful qualifiers such as:
   - all
   - every
   - only
   - none
   - minimum
   - maximum
   - amounts
   - dates
   - percentages
   - conditions

5. keywords:
   Extract the most important words or phrases needed
   to search for and verify the claim.

6. category:
   Select a broad category such as:
   - Education
   - Healthcare
   - Finance
   - Employment
   - Agriculture
   - Law
   - Taxation
   - Defence
   - Transportation
   - Government Schemes
   - Politics
   - Sports
   - Entertainment
   - Technology
   - Other

7. subcategory:
   Select a more specific topic within the category.

8. claim_type:
   Identify the type of claim, for example:
   - Eligibility
   - Benefit Amount
   - Deadline
   - Announcement
   - Government Policy
   - Law
   - Regulation
   - Appointment
   - Statistic
   - Event
   - General Information
   - Other

9. is_government_related:
   Set true if the claim is related to:
   - government schemes
   - government policies
   - government benefits
   - laws
   - regulations
   - government departments
   - taxation
   - elections
   - public services
   - official government announcements
   - government programmes

   Do NOT simply look for the word "government".
   Understand the meaning of the claim.

   Example:
   "Ash's favorite Pokemon is Pikachu."
   -> false

   Example:
   "PM Kisan provides financial assistance to farmers."
   -> true

10. is_current_affairs:
    Set true if the claim concerns a recent/current
    government development, announcement, event,
    policy change, or other current affair.

11. is_time_sensitive:
    Set true if the truth or validity of the claim can
    change with time.

    Examples:
    - scholarship deadline -> true
    - current scholarship amount -> true
    - current government scheme eligibility -> true
    - India's capital -> false

12. Do not invent information that is not present or
    reasonably inferable from the input.

Return ONLY the required JSON structure.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": Claim.model_json_schema()
        }
    )

    claim = Claim.model_validate_json(
        response.text
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            claim.model_dump(),
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\nJSON created successfully!\n")
 
    print(
        json.dumps(
            claim.model_dump(),
            indent=2,
            ensure_ascii=False
        )
    )
    import main
    return main.start_pipeline() 


if __name__ == "__main__":
    process_claim("text")
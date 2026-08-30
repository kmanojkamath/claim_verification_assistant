
import requests
import json
import re
from datetime import datetime, timezone
import emoji
import hashlib
import html
from pathlib import Path

API_KEY = "new1_508b1b00b0e7490a8390c42f3dcb7a23"

# --------------------------------
# CLEAN TEXT
# --------------------------------

def clean_text(text):
    if not text:
        return ""

    # Decode HTML entities: &amp; -> &, &quot; -> ", &#39; -> ', etc.
    text = html.unescape(text)

    # Convert literal escape sequences while retaining their meaning
    text = text.replace("\\r\\n", "\n")
    text = text.replace("\\n", "\n")
    text = text.replace("\\t", " ")

    # Normalize actual line endings and tabs
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")

    # Add a missing space after punctuation
    text = re.sub(r"([?!])(?=[A-Za-z0-9])", r"\1 ", text)

    # Normalize excess spaces within lines, preserving paragraphs
    lines = [
        re.sub(r"[ \f\v]+", " ", line).strip()
        for line in text.split("\n")
    ]
    text = "\n".join(lines)

    # Keep at most one blank line between paragraphs
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def extract_entity_text(items, possible_keys):
    values = []

    for item in items or []:
        if isinstance(item, str):
            value = item
        else:
            value = next(
                (item.get(key) for key in possible_keys if item.get(key)),
                ""
            )

        if value:
            values.append(value.lstrip("#@"))

    return values


def extract_social_metadata(tweet):
    entities = tweet.get("entities", {})

    return {
        "hashtags": extract_entity_text(
            entities.get("hashtags", tweet.get("hashtags", [])),
            ["text", "tag", "name"]
        ),
        "mentions": extract_entity_text(
            entities.get(
                "user_mentions",
                entities.get("mentions", tweet.get("mentionedUsers", []))
            ),
            ["screen_name", "userName", "username", "name"]
        )
    }

def relevance_score(text, keyword_weights):

    text = text.lower()

    score = 0

    keywords = [keyword for keyword, weight in keyword_weights]

    # Individual keyword scores
    for keyword, weight in keyword_weights:

        pattern = r'\b' + re.escape(keyword) + r'\b'

        if re.search(pattern, text):
            score += weight

    # Phrase bonuses
    for i in range(len(keywords) - 1):

        phrase = r'\b' + re.escape(keywords[i]) + r'\s+' + re.escape(keywords[i + 1]) + r'\b'

        if re.search(phrase, text):
            score += 3

    return score
# --------------------------------
# CONVERT TWEET TO OUR FORMAT
# --------------------------------

def extract_urls(tweet):
    extracted_urls = []
    seen_urls = set()

    # URLs explicitly supplied by the API
    for item in tweet.get("entities", {}).get("urls", []):
        short_url = item.get("url")

        if short_url and short_url not in seen_urls:
            extracted_urls.append({
                "url": short_url,
                "expanded_url": item.get("expanded_url") or None
            })
            seen_urls.add(short_url)

    # Fallback: capture every URL that appears in raw text
    for raw_url in re.findall(r"https?://[^\s<>'\"\])]+", tweet.get("text", "")):
        raw_url = raw_url.rstrip(".,!?;:")

        if raw_url and raw_url not in seen_urls:
            extracted_urls.append({
                "url": raw_url,
                "expanded_url": None
            })
            seen_urls.add(raw_url)

    return extracted_urls

def normalize_tweet(tweet):
    cleaned_text = clean_text(tweet["text"])
    social_metadata = extract_social_metadata(tweet)
    source_authority = get_source_authority(tweet)

    document = {
        "doc_id": "X_" + tweet["id"],
        "title": "X post by @" + tweet["author"]["userName"],

        "source": {
            "platform": "X",
            "source_type": "social_media_post",
            "source_url": tweet["url"],
            "organization": source_authority["organization"],
            "author_handle": tweet["author"]["userName"],
            "author_name": tweet["author"]["name"]
        },

        "dates": {
            "published_at": to_iso_utc(tweet["createdAt"]),
            "retrieved_at": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        },

        "content": {
            "raw_text": tweet["text"],
            "clean_text": cleaned_text
        },

        "entities": {
            "hashtags": social_metadata["hashtags"],
            "mentions": social_metadata["mentions"],
            "urls": extract_urls(tweet)
        },

        "metadata": {
            "language": tweet["lang"],
            "authority_level": source_authority["authority_level"],
            "content_hash": generate_content_hash(cleaned_text),
            "document_type": "social_media_post"
        }
    }

    return document

def generate_content_hash(cleaned_text):
    normalized_text = re.sub(r"\s+", " ", cleaned_text).strip().casefold()

    return hashlib.sha256(
        normalized_text.encode("utf-8")
    ).hexdigest()

OFFICIAL_X_ACCOUNTS = {
    "PIBFactCheck": {
        "organization": "PIB Fact Check",
        "authority_level": "official"
    }
}


def get_source_authority(tweet):
    author_handle = tweet["author"]["userName"]
    authority = OFFICIAL_X_ACCOUNTS.get(author_handle)

    if authority:
        return authority

    return {
        "organization": tweet["author"]["name"],
        "authority_level": "unverified"
    }

def to_iso_utc(date_string):
    parsed_date = datetime.strptime(
        date_string,
        "%a %b %d %H:%M:%S %z %Y"
    )

    return parsed_date.astimezone(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

# --------------------------------
# USER INPUT
# --------------------------------

# keywords = input("Enter keywords: ").lower().split()

# keyword_weights = []

# n = len(keywords)

# for i, keyword in enumerate(keywords):
#     weight = n - i
#     keyword_weights.append((keyword, weight))

# search_keywords = keywords

# # Just the words, for X search
# search_keywords = [keyword for keyword, weight in keyword_weights]

# --------------------------------
# KEYWORDS FROM PREVIOUS STAGE
# --------------------------------

with Path("query_data.json").open("r", encoding="utf-8") as file:
    query_data = json.load(file)

keywords = query_data.get("keywords", [])

keywords = [
    keyword.lower().strip()
    for keyword in keywords
    if keyword.strip()
]

if not keywords:
    raise ValueError("Previous stage returned no keywords.")

keyword_weights = []

n = len(keywords)

for i, keyword in enumerate(keywords):
    weight = n - i
    keyword_weights.append((keyword, weight))

search_keywords = [keyword for keyword, weight in keyword_weights]


# --------------------------------
# X SEARCH
# --------------------------------

url = "https://api.twitterapi.io/twitter/tweet/advanced_search"

headers = {
    "X-API-Key": API_KEY
}


# Prefer specific multi-word keyword phrases over generic words
specific_keywords = [
    keyword
    for keyword in search_keywords
    if len(keyword.split()) > 1
]

# Fall back to all keywords if no multi-word phrase exists
if not specific_keywords:
    specific_keywords = search_keywords

keyword_query = " OR ".join(
    f'"{keyword}"'
    for keyword in specific_keywords
)

query = (
    "from:PIBFactCheck ("
    + keyword_query
    + ") -filter:replies -filter:retweets"
)
params = {
    "query": query,
    "queryType": "Latest"
}

response = requests.get(
    url,
    headers=headers,
    params=params
)

print("Status:", response.status_code)

data = response.json()

# --------------------------------
# EXTRACT TWEETS
# --------------------------------

documents = []


for tweet in data["tweets"]:

    document = normalize_tweet(tweet)

    score = relevance_score(
    document["content"]["clean_text"],
    keyword_weights
)




    if score >= 2:
        documents.append(document)


# --------------------------------
# OUTPUT
# --------------------------------


output_directory = Path("output")
output_directory.mkdir(exist_ok=True)

output_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
output_file = output_directory / f"x_posts_{output_timestamp}.json"

with output_file.open("w", encoding="utf-8") as file:
    json.dump(documents, file, ensure_ascii=False, indent=4)

print(f"Saved {len(documents)} posts to: {output_file}")
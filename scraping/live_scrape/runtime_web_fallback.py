import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

from tavily import TavilyClient


TRUSTED_DOMAINS = [
    "pib.gov.in",
    "factcheck.pib.gov.in",
    "mohua.gov.in",
    "ndma.gov.in",
]


class TavilyRuntimeRetriever:
    def __init__(self, api_key: str | None = None) -> None:
        api_key = api_key or os.getenv("TAVILY_API_KEY")

        if not api_key:
            raise RuntimeError("TAVILY_API_KEY environment variable is not set.")

        self.client = TavilyClient(api_key=api_key)

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r"<[^>]+>", " ", text or "")
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def current_utc_time() -> str:
        return (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )

    @staticmethod
    def is_whitelisted_url(
        url: str,
        allowed_domains: Sequence[str],
    ) -> bool:
        hostname = urlparse(url).netloc.lower().removeprefix("www.")

        return any(
            hostname == domain.lower().removeprefix("www.")
            or hostname.endswith(
                "." + domain.lower().removeprefix("www.")
            )
            for domain in allowed_domains
        )

    @staticmethod
    def extract_entities(text: str) -> dict[str, list[Any]]:
        urls = list(
            dict.fromkeys(
                re.findall(
                    r"https?://[^\s<>\]\[\"']+",
                    text,
                )
            )
        )

        return {
            "hashtags": list(
                dict.fromkeys(
                    re.findall(r"(?<!\w)#([\w-]+)", text)
                )
            ),
            "mentions": list(
                dict.fromkeys(
                    re.findall(r"(?<!\w)@([\w-]+)", text)
                )
            ),
            "urls": [
                {
                    "url": url,
                    "expanded_url": url,
                }
                for url in urls
            ],
        }

    @classmethod
    def extract_important_text(
        cls,
        raw_text: str,
        keyword: str,
        max_characters: int = 2000,
    ) -> str:
        keyword_terms = re.findall(r"[\w-]+", keyword.lower())

        sentences = re.split(
            r"(?:\[\.\.\.\]|(?<=[.!?])\s+)",
            cls.clean_text(raw_text),
        )

        ranked_sentences = []

        for position, sentence in enumerate(sentences):
            sentence = sentence.strip()

            if len(sentence) < 35:
                continue

            relevance_score = sum(
                sentence.lower().count(term)
                for term in keyword_terms
            )

            ranked_sentences.append(
                (relevance_score, position, sentence)
            )

        selected = []
        selected_length = 0

        for _, position, sentence in sorted(
            ranked_sentences,
            key=lambda item: (-item[0], item[1]),
        ):
            if selected_length + len(sentence) > max_characters:
                continue

            selected.append((position, sentence))
            selected_length += len(sentence)

        return " ".join(
            sentence
            for _, sentence in sorted(
                selected,
                key=lambda item: item[0],
            )
        )

    def retrieve(
        self,
        keywords: Sequence[str],
        allowed_domains: Sequence[str] = TRUSTED_DOMAINS,
        corpus_has_evidence: bool = False,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Returns a maximum of five documents in the x_extractor JSON schema.
        """

        if corpus_has_evidence:
            return []

        if not allowed_domains:
            raise ValueError("At least one trusted domain must be provided.")

        ranked_documents = []
        processed_urls = set()

        unique_keywords = dict.fromkeys(
            keyword.strip()
            for keyword in keywords
            if keyword.strip()
        )

        for keyword in unique_keywords:
            try:
                search_response = self.client.search(
                    query=keyword,
                    include_domains=list(allowed_domains),
                    search_depth="advanced",
                    max_results=top_k,
                    include_raw_content=False,
                )

                candidates = [
                    result
                    for result in search_response.get("results", [])
                    if result.get("url")
                    and result["url"] not in processed_urls
                    and self.is_whitelisted_url(
                        result["url"],
                        allowed_domains,
                    )
                ]

                if not candidates:
                    continue

                extraction_response = self.client.extract(
                    urls=[result["url"] for result in candidates],
                    query=keyword,
                    chunks_per_source=3,
                )

            except Exception:
                continue

            extracted_by_url = {
                item["url"]: item.get("raw_content", "")
                for item in extraction_response.get("results", [])
            }

            for result in candidates:
                url = result["url"]
                raw_text = self.clean_text(
                    extracted_by_url.get(url, "")
                )

                clean_text = self.extract_important_text(
                    raw_text,
                    keyword,
                )

                if not clean_text:
                    continue

                processed_urls.add(url)

                domain = (
                    urlparse(url)
                    .netloc
                    .removeprefix("www.")
                )

                document = {
                    "doc_id": (
                        "WEB_"
                        + hashlib.sha256(
                            url.encode("utf-8")
                        ).hexdigest()[:16]
                    ),
                    "title": (
                        self.clean_text(result.get("title", ""))
                        or domain
                    ),
                    "source": {
                        "platform": "Web",
                        "source_type": "website",
                        "source_url": url,
                        "organization": domain,
                        "author_handle": None,
                        "author_name": None,
                    },
                    "dates": {
                        "published_at": result.get("published_date"),
                        "retrieved_at": self.current_utc_time(),
                    },
                    "content": {
                        "raw_text": raw_text,
                        "clean_text": clean_text,
                    },
                    "entities": self.extract_entities(raw_text),
                    "metadata": {
                        "language": "en",
                        "authority_level": "official",
                        "content_hash": hashlib.sha256(
                            clean_text.encode("utf-8")
                        ).hexdigest(),
                        "document_type": "webpage",
                    },
                }

                ranked_documents.append((
                    float(result.get("score", 0)),
                    document,
                ))

        ranked_documents.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            document
            for _, document in ranked_documents[:top_k]
        ]

    def retrieve_to_json(
        self,
        keywords: Sequence[str],
        output_file: str = "tavily_web_documents.json",
        allowed_domains: Sequence[str] = TRUSTED_DOMAINS,
        corpus_has_evidence: bool = False,
    ) -> list[dict[str, Any]]:

        documents = self.retrieve(
            keywords=keywords,
            allowed_domains=allowed_domains,
            corpus_has_evidence=corpus_has_evidence,
            top_k=5,
        )

        output_path = Path(output_file)

        # Creates the output folder if it does not already exist.
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path.write_text(
            json.dumps(
                documents,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return documents
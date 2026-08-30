import json

from runtime_web_fallback import TavilyRuntimeRetriever


retriever = TavilyRuntimeRetriever()

documents = retriever.retrieve_to_json(
    keywords=[
        "government free laptop scheme",
        "student laptop scheme fact check",
    ],
    allowed_domains=[
        "pib.gov.in",
        "factcheck.pib.gov.in",
    ],
    corpus_has_evidence=False,  # Forces the Tavily fallback
    output_file="data/webscraper_output.json",
)

print(f"Documents found: {len(documents)}")
print(json.dumps(documents, indent=2, ensure_ascii=False))
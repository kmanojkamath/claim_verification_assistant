"""Step 3: adapt these three values to the output of your prior stage."""
import json

from runtime_web_fallback import TavilyCorpusFallback

previous_stage_output = {
    "keywords": ["urban flood mitigation", "rainwater harvesting policy"],
    "corpus_has_evidence": False,
}
trusted_websites = ["ndma.gov.in", "mohua.gov.in", "worldbank.org"]

fallback = TavilyCorpusFallback()
result = fallback.retrieve(
    keywords=previous_stage_output["keywords"],
    allowed_domains=trusted_websites,
    corpus_has_evidence=previous_stage_output["corpus_has_evidence"],
)

# Pass this JSON object directly to the next stage.
print(json.dumps(result.to_next_stage(), ensure_ascii=False, indent=2))

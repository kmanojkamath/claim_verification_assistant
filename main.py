import os
import json
import asyncio
from typing import Union, List
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from types import SimpleNamespace

load_dotenv()

collection_name="pib_chunks"

class FinalVerdict(BaseModel):
    verdict: str = Field("unverifiable",description="True if the claim is verified, False if the claim is false, Unverifiable if no conclusion can be drawn.")
    citations: List[str] = Field(description="List of citations from where the evidences are collected.")
    confidence: List[str] = Field(description="How confident you are on the scale of 1.0 to 10.0")

async def create_corpus():
    from rag.ingestion.corpus import iter_chunks_from_sqlite
    from pathlib import Path
    from rag.retrieval.vector_store import store
    for chunk in iter_chunks_from_sqlite(Path("rag/data/main_corpus.db")):
        await store.ingest_chunks([chunk])

async def create_vector_collection():
    from qdrant_client import AsyncQdrantClient
    client = AsyncQdrantClient(url=os.getenv("CLUSTER_ENDPOINT"), api_key=os.getenv("CLUSTER_API"), check_compatibility=False)

    if await client.collection_exists(collection_name=collection_name):
        collection_info = await client.get_collection(collection_name=collection_name)
        points_count = collection_info.points_count
        if points_count >= 15000:
            return
        else:
            print("Meow")
            from rag.retrieval.vector_store import entry
            await client.delete_collection("pib_chunks")
            await entry()
    else:
        from rag.retrieval.vector_store import entry
        print("Meow 2")
        await entry()
    await create_corpus()
    return

llm = ChatGroq(model="openai/gpt-oss-120b",temperature=0.3,api_key=os.getenv("GROQ_API"))
structured_llm = llm.with_structured_output(FinalVerdict)

third_umpire_prompt ="""
You get two verification JSONs for one claim: corpus (pre-vetted, may be stale) and live_web (fresh, less vetted).
If they agree, state the verdict with brief citations. If they disagree, favor whichever has the more recent most_recent_evidence_date,
and say so explicitly. Match confidence to the lower of the two. Surface any staleness_note prominently near the start.
Cite source type, publisher, and date. Plain prose only, no JSON, concise, neutral tone.
"""

REQUIRED_KEYS = {
    "verdict",
    "confidence",
    "reasoning",
    "cited_evidence",
    "conflicting_evidence",
    "most_recent_evidence_date",
    "staleness_note",
}

def _parse_verification(raw: Union[str, dict], label: str) -> dict:
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"{label} verifier returned invalid JSON: {e}\nraw: {raw!r}"
            ) from e
    else:
        parsed = raw
    missing = REQUIRED_KEYS - parsed.keys()
    if missing:
        raise ValueError(f"{label} verifier JSON missing keys: {missing}")
    return parsed
 

def merge_status(claim: str, corpus_result: Union[str, dict], web_result: Union[str, dict]) -> str:
    corpus_result = _parse_verification(corpus_result, "corpus")
    web_result = _parse_verification(web_result, "live_web")
    payload = {"claim": claim, "corpus": corpus_result, "live_web": web_result}
    return json.dumps(payload, separators=(",", ":"))

async def main():
    await create_vector_collection()
    data = None
    with open("rag/data/query.json",'r') as f: # File name needs to be changed
        data = json.load(f)
    print(type(data))
    if data:
        with open("rag/data/web_search.json",'r') as f:
            records = json.load(f)
        print(type(records[0]))
        # top5_raw = records[-5:]
        # top5 = [
        #     SimpleNamespace(
        #         chunk_id=str(i),
        #         document_id=r["doc_id"],
        #         content=r["content"]["clean_text"],
        #         title=r.get("title"),
        #         url=r["source"].get("source_url"),
        #         publisher=r["source"].get("organization") or r["source"].get("author_name"),
        #         author=r["source"].get("author_name"),
        #         published_at=r["dates"].get("published_at"),
        #         section_heading=None,
        #         source_type=r["source"].get("source_type"),
        #     )
        #     for i, r in enumerate(top5_raw)
        # ]
        # from rag.generation.llm import generate
        # from rag.generation.llm import GroqLLM
        # web_verification = await generate(query=data["query"],retrieved_chunks=top5,llm=GroqLLM(client=llm))
        # from rag.retrieval.vector_store import store
        # top5 = store.retrieve(query=data["query"])
        # corp_verification = generate(query=data["query"],retrieved_chunks=top5)
        # conv = [SystemMessage(content=third_umpire_prompt)
        #         ,merge_status(claim=data["query"],web_result=web_verification,corpus_result=corp_verification)]
        # final_verdict = structured_llm.ainvoke(conv)[-1].content
        print("Hello")
    else:
        print("No request made.")
    from rag.retrieval.vector_store import store
    if store is not None:
        await store.client.close()

asyncio.run(main())
import os
import json
import asyncio
from typing import Union
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq

load_dotenv()

collection_name="pib_chunks"


async def create_corpus():
    from rag.ingestion.corpus import iter_chunk_from_sqlite
    from pathlib import Path
    from rag.retrieval.vector_store import store
    for chunk in iter_chunk_from_sqlite(Path("rag/data/main_corpus.db")):
        store.ingest_chunks([chunk])

async def create_vector_collection():
    from qdrant_client import AsyncQdrantClient
    client = AsyncQdrantClient(url=os.getenv("CLUSTER_ENDPOINT"), api_key=os.getenv("CLUSTER_API"), check_compatibility=False)

    if client.collection_exists(collection_name=collection_name):
        collection_info = client.get_collection(collection_name=collection_name)
        points_count = collection_info.points_count
        if points_count >= 35000:
            return
        else:
            await client.delete_collection("pib_chunks")
            await entry()
    else:
        from rag.retrieval.vector_store import entry
        await entry()
    await create_corpus()
    return

llm = ChatGroq(name="openai/gpt-oss-120b",temperature=0.3,api_key=os.getenv("GROQ_API"))

third_umpire_prompt ="""
You get two verification JSONs for one claim: corpus (pre-vetted, may be stale) and live_web (fresh, less vetted).
If they agree, state the verdict with brief citations. If they disagree, favor whichever has the more recent most_recent_evidence_date,
and say so explicitly. Match confidence to the lower of the two. Surface any staleness_note prominently near the start.
Cite source type, publisher, and date. Plain prose only, no JSON, concise, neutral tone.
Reply with ONLY this JSON:
    {"verdict":"true|false|partially_true|unverifiable",
    "confidence":"high|medium|low",
    "reasoning":"1-2 sentences",
    "cited_evidence":[indices],
    "conflicting_evidence":true|false,
    "most_recent_evidence_date":"date or null",
    "staleness_note":"note or null"}
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
 

def merge_status(claim: str,corpus_result: Union[str, dict],web_result: Union[str, dict]) -> str:
    if isinstance(web_result,str):
        _parse_verification(web_result)
    if isinstance(corpus_result,str):
        _parse_verification(corpus_result)
    payload = {
        "claim": claim,
        "corpus": corpus_result,
        "live_web": web_result,
    }
    return json.dumps(payload, separators=(",", ":"))

async def main():
    await create_vector_collection()
    data = None
    with open("file.json",'r') as f: # File name needs to be changed
        data = f.read()
    if data:
        ## Arya's Code
        def arya_func(data):
            return {"content":"something"}
        records=arya_func()
        top5 = records[-5:]
        from rag.generation.llm import generate
        web_verification = generate(query=data["query"],retrieved_chunks=top5)
        from rag.retrieval.vector_store import store
        top5 = store.retrieve(query=data["query"])
        corp_verification = generate(query=data["query"],retrieved_chunks=top5)
        conv = [SystemMessage(content=third_umpire_prompt)
                ,merge_status(claim=data["query"],web_result=web_verification,corpus_result=corp_verification)]
        final_verdict = llm.ainvoke(conv)[-1].content
        print(final_verdict)
    else:
        print("No request made.")

asyncio.run(main())
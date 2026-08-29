import json
from pathlib import Path

from rag.models import Document


CORPUS_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_documents.json"


def test_sample_documents_match_the_document_contract() -> None:
    payload = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))

    documents = [Document.model_validate(item) for item in payload]

    assert len(documents) == 8
    assert {document.source_type.value for document in documents} == {"government_document"}
    assert all(str(document.url).startswith("https://www.pib.gov.in/") for document in documents)
    assert all(document.metadata and document.metadata["source_system"] == "PIB" for document in documents)

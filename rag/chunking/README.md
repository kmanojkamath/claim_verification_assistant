# RAG chunking

This component converts one validated `rag.models.Document` into a deterministic
`list[rag.models.Chunk]`. It is the boundary consumed by the retrieval layer:

```text
Document -> chunking -> list[Chunk]
```

`Document.content` is the canonical text for all character offsets. The chunker
does not read SQLite and does not perform HTML cleaning.

## Usage

```python
from rag.chunking import chunk_document
from rag.models import Document

document = Document.model_validate(payload)
chunks = chunk_document(document)
```

For reusable configuration or a shared tokenizer instance, use
`DocumentChunker(config=..., tokenizer=...)` and call `.chunk(document)`.

## Behavior

- Defaults are 700 target tokens, 900 maximum tokens, and 100 overlap tokens.
- Token measurements use the tokenizer for `BAAI/bge-m3` only; no embedding
  model is loaded.
- A document at or below the maximum is returned as exactly one chunk, even if
  it has multiple sections.
- Larger documents prefer section, paragraph, sentence, then token boundaries.
  Overlap comes from the end of the prior chunk, remains within the maximum,
  and stays section-local.
- `chunk_id` is deterministic: `<document_id>:<chunk_index>`.
- Each Chunk preserves document provenance: source IDs, content hash, title,
  URL, publisher, author, publication time, source type, language, metadata,
  and applicable section ID/heading.

The retrieval team receives valid `Chunk` models with `content`, exact
`char_start`/`char_end` offsets into `Document.content`, and the above
provenance. `Document.retrieved_at` remains a Document-level field and is not
part of the Chunk contract.

The SQLite loader in `rag/scripts/load_pib_sqlite.py` is development-only. The
production boundary is Team 1 final Document -> Document model -> chunking ->
list[Chunk].

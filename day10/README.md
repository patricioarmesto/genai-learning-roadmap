# Day 10 — Document chunking strategies for RAG

Indexes the same document corpus using four chunking strategies, then evaluates which strategy retrieves the most relevant chunk for each of 10 test questions.

## Strategies implemented

1. **Fixed-size** — split every N chars with M overlap
2. **Recursive** — try paragraph → sentence → word → char separators
3. **Semantic** — embed sentences, split where topic similarity drops
4. **Hierarchical** — small child chunks for retrieval, large parent for context

Each strategy gets its own Qdrant collection. The evaluator runs all 10 questions against all 4 strategies and scores Hit@1 and Hit@3.

## Requirements

```bash
docker run -d -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
uv add qdrant-client requests
ollama pull nomic-embed-text
ollama serve
```

## Run modes

- `uv run python main.py` — full eval (ingest + score)
- `uv run python main.py ingest` — ingest only (re-build all collections)
- `uv run python main.py search <q>` — search all strategies for one query
- `uv run python main.py inspect` — print chunks from each strategy
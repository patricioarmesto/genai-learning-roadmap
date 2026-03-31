# Day 11 — Building a Basic RAG Pipeline

A complete RAG (Retrieval Augmented Generation) Q&A system with query rewriting, semantic search, grounded generation, and LLM-as-judge evaluation.

## Features

- **Query Rewriting** — Expands conversational questions into keyword-dense queries for better retrieval
- **Qdrant Retrieval** — Semantic search over chunked document corpus
- **Context Assembly** — Ranks, deduplicates, and formats retrieved chunks
- **Grounded Generation** — System prompt that forces the model to cite only the provided context
- **Groundedness Checker** — LLM-as-judge scoring whether the answer is grounded in the context
- **Refusal Detection** — Model says "not in context" when it can't answer from the corpus

## Architecture

```
User Question
     │
     ▼
┌─────────────┐
│   Query     │  ← Rewrite conversational question to keyword query
│  Rewriting  │
└─────────────┘
     │
     ▼
┌─────────────┐
│  Retrieval  │  ← Embed query, search Qdrant, filter by score
│  (Qdrant)   │
└─────────────┘
     │
     ▼
┌─────────────┐
│   Context   │  ← Deduplicate, rank, truncate to max chars
│  Assembly   │
└─────────────┘
     │
     ▼
┌─────────────┐
│  Generation │  ← LLM generates answer using only provided context
│             │
└─────────────┘
     │
     ▼
┌─────────────┐
│ Groundedness│  ← LLM-as-judge evaluates if answer is supported
│   Check     │
└─────────────┘
     │
     ▼
   Response
```

## Requirements

**Services:**
```bash
# Qdrant vector database
docker run -d -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant

# Ollama (embedding + generation)
ollama serve
ollama pull nomic-embed-text
ollama pull llama3.2
```

**Python packages:**
```bash
uv add qdrant-client requests pydantic
```

**Data:**
```bash
uv run python day10_chunking.py ingest   # ← run this first to populate Qdrant
```

## Usage

**Interactive REPL:**
```bash
uv run python day11_rag.py
```

**Single question:**
```bash
uv run python day11_rag.py ask "What is Python's memory management?"
```

**Run evaluation set:**
```bash
uv run python day11_rag.py eval
```

**Demo questions:**
```bash
uv run python day11_rag.py demo
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `GEN_MODEL` | `llama3.2` | Generation model |
| `COLLECTION` | `chunks_recursive` | Qdrant collection |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `MIN_SCORE` | `0.45` | Minimum relevance threshold |
| `MAX_CONTEXT` | `2000` | Max chars for LLM context |

## Output Format

```python
{
  "question": "...",
  "rewritten_query": "...",      # Expanded keyword query
  "answer": "...",
  "groundedness": "grounded|partial|ungrounded|refused",
  "groundedness_score": 0.0-1.0,
  "sources": [...],
  "retrieved_chunks": 5,
  "latency_ms": {
    "rewrite_ms": 150,
    "retrieve_ms": 5,
    "assemble_ms": 1,
    "generate_ms": 2000,
    "groundedness_ms": 1500,
    "total_ms": 3656
  }
}
```

## Build On

- Day 8 — Embeddings
- Day 9 — Qdrant
- Day 10 — Chunking
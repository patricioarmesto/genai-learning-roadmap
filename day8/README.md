# Day 8 — Text Embeddings and Semantic Search

A semantic search engine built over a small corpus of documents to demonstrate the fundamental differences between exact-match (keyword) search and semantic search. It compares them side-by-side so the difference is immediately tangible.

## Key Concepts Demonstrated

- Calling Ollama's `/api/embeddings` endpoint
- Implementing **Cosine similarity** from scratch (no libraries needed)
- Embedding a corpus once and caching the vectors to disk (`embeddings_cache.json`)
- Querying: embedding a query and ranking all documents by similarity
- Exact-match (keyword) vs semantic comparison
- Understanding similarity thresholds and practical score implications
- A minimal Retrieval-Augmented Generation (RAG) implementation

## Prerequisites

1. **Ollama**: You need to have [Ollama](https://ollama.com/) installed and running locally.
2. **Models**: Pull the required embedding and generation models.
   
```bash
# Pull the embedding model used for generating vectors
ollama pull nomic-embed-text

# Pull the text generation model used for the RAG demo
ollama pull llama3.2
```

3. Ensure the Ollama server is running:

```bash
ollama serve
```

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management (as indicated by the `pyproject.toml` and `uv.lock` files), but you can also use standard `pip`.

### Using `uv` (Recommended)

```bash
# uv will automatically handle dependencies
uv run day8_embeddings.py
```

### Using `pip`

```bash
# Install the required dependency
pip install requests

# Run the script
python day8_embeddings.py
```

## Usage

The script `day8_embeddings.py` provides several execution modes to experiment with embeddings and search techniques. Note: if using `uv`, replace `python` with `uv run` in the commands below.

### Interactive Search REPL

Launch into an interactive search, where you can type queries and see semantic results.

```bash
python day8_embeddings.py
```

Once inside the REPL, you can use special commands:
- `text`: performs a semantic search
- `/kw text`: performs an exact keyword search
- `/both text`: performs a side-by-side comparison
- `/rag text`: retrieves context and generates an answer using the LLM
- `/stats`: displays index info
- `quit`: exits the REPL

### Command-Line Arguments

You can also run specific experiments directly from the terminal:

**1. Side-by-Side Comparison**
Run predefined comparison queries that showcase when semantic search shines vs. when keyword search wins.

```bash
python day8_embeddings.py compare
```

**2. Find Similar Documents**
Find documents most similar to any arbitrary text string.

```bash
python day8_embeddings.py similar "how do I store user authentication tokens"
```

**3. Minimal RAG Demo**
Retrieve relevant documents and generate an AI answer based *only* on the retrieved context.

```bash
python day8_embeddings.py rag "what is the difference between cache-aside and write-through?"
```

"""
Day 8 — Text embeddings and semantic search
=============================================
A semantic search engine over a small corpus of documents.
Shows keyword search vs semantic search side by side so the
difference is immediately tangible.

Key concepts demonstrated:
  • Calling Ollama's /api/embeddings endpoint
  • Cosine similarity from scratch (no libraries needed)
  • Embedding a corpus once and caching to disk
  • Querying: embed the query, rank all docs by similarity
  • Exact-match (keyword) vs semantic comparison
  • Similarity thresholds and what scores mean in practice

Requirements:
    uv add requests
    ollama pull nomic-embed-text
    ollama pull llama3.2      (for the RAG demo at the end)
    ollama serve

Run modes:
    python day8_embeddings.py                 # interactive search REPL
    python day8_embeddings.py compare         # keyword vs semantic side by side
    python day8_embeddings.py similar "text"  # find docs similar to any text
    python day8_embeddings.py rag "question"  # full RAG: retrieve + generate
"""

import json
import math
import sys
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "llama3.2"
CACHE_FILE = Path("embeddings_cache.json")
TOP_K = 5
SIM_THRESHOLD = 0.50  # scores below this are considered irrelevant

# ── Corpus ────────────────────────────────────────────────────────────────────
# Simulate a personal knowledge base of markdown notes.
# In week 2 day 9 this gets replaced by Qdrant.

CORPUS = [
    {
        "id": "py_lists",
        "title": "Python lists",
        "text": "Python lists are ordered, mutable sequences. Created with square brackets: my_list = [1, 2, 3]. Support indexing, slicing, and methods like append(), extend(), and sort(). Lists can hold mixed types. Time complexity: O(1) for index access, O(n) for search.",
    },
    {
        "id": "py_dicts",
        "title": "Python dictionaries",
        "text": "Dictionaries store key-value pairs. Keys must be hashable (strings, numbers, tuples). Created with curly braces: my_dict = {'key': 'value'}. Methods: keys(), values(), items(), get(). Python 3.7+ preserves insertion order. O(1) average for lookup and insert.",
    },
    {
        "id": "py_generators",
        "title": "Python generators",
        "text": "Generators are lazy iterators created with the yield keyword. They compute values on demand instead of storing them all in memory. Useful for large datasets. Generator expressions use parentheses: (x**2 for x in range(1000)). Cannot be reused after exhaustion.",
    },
    {
        "id": "py_async",
        "title": "Async programming in Python",
        "text": "Python's asyncio enables concurrent I/O-bound operations without threads. async def defines a coroutine. await pauses execution until the awaited coroutine completes. asyncio.gather() runs multiple coroutines concurrently. The event loop manages scheduling. Not suitable for CPU-bound tasks — use multiprocessing instead.",
    },
    {
        "id": "py_decorators",
        "title": "Python decorators",
        "text": "Decorators are functions that wrap other functions to add behaviour. Applied with @syntax above a function definition. Common uses: logging, timing, caching, authentication. functools.wraps preserves the wrapped function's metadata. Class decorators work similarly but wrap classes.",
    },
    {
        "id": "ml_supervised",
        "title": "Supervised learning",
        "text": "Supervised learning trains models on labelled data — each example has an input and a known output. Two main types: classification (predicting categories) and regression (predicting numbers). Common algorithms: linear regression, decision trees, SVMs, neural networks. Evaluated with accuracy, F1, RMSE depending on the task.",
    },
    {
        "id": "ml_embeddings",
        "title": "Word embeddings",
        "text": "Word embeddings represent words as dense vectors where similar words cluster together. Word2Vec and GloVe learn embeddings from co-occurrence statistics. Sentence embeddings (like SBERT) encode entire sentences. The key property: vector arithmetic captures semantic relationships. king - man + woman ≈ queen.",
    },
    {
        "id": "ml_transformers",
        "title": "Transformer architecture",
        "text": "Transformers use self-attention to weigh the importance of each token relative to others. Encoder-only models (BERT) are good for understanding. Decoder-only models (GPT) are good for generation. The attention formula: softmax(QK^T / sqrt(d_k)) * V. Positional encodings add sequence information.",
    },
    {
        "id": "ml_rag",
        "title": "Retrieval-augmented generation",
        "text": "RAG combines retrieval with generation. A query retrieves relevant documents from a corpus, which are injected into the LLM prompt as context. Reduces hallucination by grounding answers in retrieved facts. Key components: embedding model, vector store, retriever, and generator. Two-stage RAG adds re-ranking.",
    },
    {
        "id": "db_postgres",
        "title": "PostgreSQL basics",
        "text": "PostgreSQL is an open-source relational database. Supports ACID transactions, complex queries, and JSON. Key commands: SELECT, INSERT, UPDATE, DELETE. Indexes (B-tree by default) speed up queries. EXPLAIN ANALYZE shows query execution plans. Use connection pooling (pgBouncer) in production.",
    },
    {
        "id": "db_redis",
        "title": "Redis data structures",
        "text": "Redis is an in-memory key-value store. Supports strings, hashes, lists, sets, sorted sets, and streams. Used for caching, session storage, pub/sub, and rate limiting. Data can be persisted with RDB snapshots or AOF logs. TTL (time-to-live) automatically expires keys.",
    },
    {
        "id": "db_vectors",
        "title": "Vector databases",
        "text": "Vector databases store and query high-dimensional embeddings. Use approximate nearest-neighbour (ANN) algorithms like HNSW for fast similarity search. Popular options: Qdrant, Pinecone, Chroma, Weaviate, FAISS. Essential for semantic search, RAG pipelines, and recommendation systems. Support metadata filtering alongside vector search.",
    },
    {
        "id": "api_rest",
        "title": "REST API design",
        "text": "REST APIs use HTTP methods: GET (read), POST (create), PUT/PATCH (update), DELETE (remove). Resources are identified by URLs. Use nouns not verbs: /users not /getUsers. Status codes: 200 OK, 201 Created, 400 Bad Request, 401 Unauthorized, 404 Not Found, 500 Internal Server Error. Versioning: /api/v1/.",
    },
    {
        "id": "api_fastapi",
        "title": "FastAPI framework",
        "text": "FastAPI is a modern Python web framework. Auto-generates OpenAPI docs from type annotations. Uses Pydantic for request/response validation. Supports async request handlers natively. Dependency injection via Depends(). Background tasks with BackgroundTasks. Mount static files or other ASGI apps. Deploy with uvicorn or gunicorn+uvicorn workers.",
    },
    {
        "id": "devops_docker",
        "title": "Docker containers",
        "text": "Docker packages apps into containers — portable units with all dependencies. Dockerfile defines the image layer by layer. Multi-stage builds reduce final image size. docker-compose orchestrates multi-container apps. Volumes persist data beyond container lifecycle. Health checks ensure dependent services start in the right order. Use non-root users in production.",
    },
    {
        "id": "devops_k8s",
        "title": "Kubernetes basics",
        "text": "Kubernetes orchestrates containerised workloads. Key objects: Pod (smallest unit), Deployment (manages replicas), Service (stable network endpoint), ConfigMap and Secret (configuration). kubectl is the CLI. Rolling updates replace pods gradually. Horizontal Pod Autoscaler scales based on CPU/memory metrics.",
    },
    {
        "id": "security_jwt",
        "title": "JWT authentication",
        "text": "JSON Web Tokens have three parts: header (algorithm), payload (claims), signature. Signed with HMAC or RSA. The server validates the signature without storing session state — stateless auth. Access tokens should be short-lived (15 min). Refresh tokens are longer-lived and stored securely. Never store JWTs in localStorage — use httpOnly cookies.",
    },
    {
        "id": "security_sql_injection",
        "title": "SQL injection prevention",
        "text": "SQL injection occurs when user input is interpolated directly into SQL queries. Prevention: always use parameterised queries or prepared statements. ORM frameworks handle this automatically. Never construct queries with string concatenation. Use least-privilege database accounts. WAFs provide an extra layer but are not a substitute for parameterised queries.",
    },
    {
        "id": "testing_pytest",
        "title": "Testing with pytest",
        "text": "pytest is Python's most popular testing framework. Tests are functions prefixed with test_. Fixtures provide reusable setup via @pytest.fixture. Parametrize tests with @pytest.mark.parametrize. Mocking with pytest-mock or unittest.mock. Coverage with pytest-cov. Run specific tests: pytest tests/test_auth.py::test_login -v.",
    },
    {
        "id": "perf_caching",
        "title": "Caching strategies",
        "text": "Cache-aside: application checks cache first, fetches from DB on miss and populates cache. Write-through: write to cache and DB simultaneously. Write-behind: write to cache, async flush to DB. TTL-based expiration prevents stale data. Cache invalidation is one of the hardest problems. LRU eviction removes least recently used items when cache is full.",
    },
]


# ── Ollama helpers ────────────────────────────────────────────────────────────


def get_embedding(text: str) -> list[float]:
    """
    Call Ollama's embedding endpoint and return the vector.

    Important: this is a separate endpoint from /api/chat.
    The embedding model (nomic-embed-text) is a different model
    from the generation model (llama3.2) — it's purpose-built
    to produce good vector representations, not to generate text.
    """
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]
    except requests.exceptions.ConnectionError:
        print("[error] Cannot reach Ollama. Run: ollama serve")
        sys.exit(1)
    except KeyError:
        print(f"[error] No embedding in response. Is '{EMBED_MODEL}' pulled?")
        print(f"  Run: ollama pull {EMBED_MODEL}")
        sys.exit(1)


def generate(prompt: str, context: str = "") -> str:
    """Simple blocking generation call for the RAG demo."""
    full_prompt = f"{context}\n\n{prompt}" if context else prompt
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": GEN_MODEL,
                "prompt": full_prompt,
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "[error: Ollama not reachable]"


# ── Cosine similarity ─────────────────────────────────────────────────────────


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity between two vectors.

    Formula: cos(θ) = (A · B) / (|A| × |B|)

    Returns a float in [-1, 1]:
      1.0  = identical direction (very similar meaning)
      0.0  = orthogonal (unrelated)
     -1.0  = opposite direction (rare in text embeddings)

    We implement this from scratch so you can see exactly what's happening.
    In production you'd use numpy: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    """
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ── Corpus embedding + cache ──────────────────────────────────────────────────


def build_or_load_index() -> list[dict]:
    """
    Embed every document in the corpus and cache to disk.

    Embedding 20 documents takes ~10 seconds. The cache means
    subsequent runs are instant. In production you'd store these
    in Qdrant (day 9) instead of a JSON file.

    Each indexed document has all original fields plus:
      embedding: list[float]   the vector representation
    """
    # Load cache if it covers all current documents
    if CACHE_FILE.exists():
        cached = json.loads(CACHE_FILE.read_text())
        cached_ids = {d["id"] for d in cached}
        corpus_ids = {d["id"] for d in CORPUS}
        if cached_ids == corpus_ids:
            print(f"[index] Loaded {len(cached)} embeddings from cache")
            return cached

    print(f"[index] Embedding {len(CORPUS)} documents with {EMBED_MODEL}...")
    indexed = []
    for i, doc in enumerate(CORPUS):
        # Embed the title + text together for richer representation
        text_to_embed = f"{doc['title']}. {doc['text']}"
        embedding = get_embedding(text_to_embed)
        indexed.append({**doc, "embedding": embedding})
        print(f"  [{i + 1}/{len(CORPUS)}] {doc['title']}", end="\r")

    CACHE_FILE.write_text(json.dumps(indexed))
    print(f"\n[index] Saved to {CACHE_FILE}")
    return indexed


# ── Search functions ──────────────────────────────────────────────────────────


def semantic_search(query: str, index: list[dict], top_k: int = TOP_K) -> list[dict]:
    """
    Semantic search:
      1. Embed the query with the SAME model used for documents
      2. Compute cosine similarity between query vector and every doc vector
      3. Sort by similarity descending, return top_k

    The query and documents are compared in the same vector space,
    so similarity reflects meaning rather than word overlap.
    """
    query_vec = get_embedding(query)
    scored = []
    for doc in index:
        sim = cosine_similarity(query_vec, doc["embedding"])
        scored.append({**doc, "score": sim})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def keyword_search(query: str, corpus: list[dict], top_k: int = TOP_K) -> list[dict]:
    """
    Exact keyword search: count how many query words appear in each document.
    This is what a simple grep or basic search engine does.
    No understanding of meaning — purely lexical.
    """
    query_words = set(query.lower().split())
    # Remove common stop words for slightly better results
    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "and",
        "or",
        "but",
        "not",
        "with",
        "from",
        "by",
        "about",
        "how",
        "what",
        "when",
    }
    query_words -= stop_words

    scored = []
    for doc in corpus:
        doc_text = f"{doc['title']} {doc['text']}".lower()
        # Count word occurrences
        hits = sum(doc_text.count(w) for w in query_words)
        if hits > 0:
            scored.append({**doc, "score": hits})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ── Display helpers ───────────────────────────────────────────────────────────


def print_results(results: list[dict], label: str, show_score: bool = True) -> None:
    print(f"\n  {label}")
    print(f"  {'─' * 50}")
    if not results:
        print(f"  (no results)")
        return
    for i, r in enumerate(results, 1):
        score_str = f"  score={r['score']:.3f}" if show_score else ""
        above = r["score"] >= SIM_THRESHOLD if "score" in r else True
        marker = "●" if above else "○"
        print(f"  {marker} {i}. [{r['title']}]{score_str}")
        print(f"       {r['text'][:90]}...")


def print_comparison(query: str, semantic: list[dict], keyword: list[dict]) -> None:
    """Print semantic vs keyword results side by side."""
    print(f"\n{'═' * 60}")
    print(f'  Query: "{query}"')
    print(f"{'═' * 60}")
    print_results(semantic, f"Semantic search ({EMBED_MODEL})")
    print_results(keyword, "Keyword search (exact match)")


# ── RAG demo ──────────────────────────────────────────────────────────────────


def rag_answer(question: str, index: list[dict]) -> str:
    """
    Minimal RAG: retrieve top-3 semantically relevant docs, inject as context,
    generate an answer. This is the pattern you'll build properly in day 11.
    """
    results = semantic_search(question, index, top_k=3)
    relevant = [r for r in results if r["score"] >= SIM_THRESHOLD]

    if not relevant:
        return "No relevant documents found for that question."

    context = "\n\n".join(
        f"[{r['title']} — relevance {r['score']:.2f}]\n{r['text']}" for r in relevant
    )
    system_context = (
        f"Answer the question using only the provided context.\n\nContext:\n{context}"
    )
    return generate(question, system_context)


# ── Experiments ───────────────────────────────────────────────────────────────

COMPARISON_QUERIES = [
    # Semantic wins — query uses different words than the documents
    "how do I iterate without loading everything into memory",  # → generators
    "storing data that expires automatically",  # → Redis TTL
    "prevent bad guys from hijacking database queries",  # → SQL injection
    "make my API calls not block",  # → async
    "stateless user authentication with tokens",  # → JWT
    # Keyword wins — exact terminology in both query and document
    "pytest fixtures",
    "docker healthcheck",
    "PostgreSQL EXPLAIN ANALYZE",
]


def run_comparison(index: list[dict]) -> None:
    """
    Run all comparison queries and show keyword vs semantic side by side.
    This is the core experiment of day 8 — feel the difference.
    """
    print(f"\n{'═' * 60}")
    print(f"  Keyword vs Semantic Search Comparison")
    print(f"  {len(COMPARISON_QUERIES)} queries × 2 methods")
    print(f"{'═' * 60}")

    for query in COMPARISON_QUERIES:
        sem = semantic_search(query, index, top_k=3)
        kw = keyword_search(query, CORPUS, top_k=3)
        print_comparison(query, sem, kw)
        try:
            input("\n  Press Enter for next query...")
        except (KeyboardInterrupt, EOFError):
            break


def run_similar(text: str, index: list[dict]) -> None:
    """Find documents most similar to any arbitrary text."""
    print(f'\nFinding documents similar to: "{text}"')
    results = semantic_search(text, index, top_k=5)
    print_results(results, "Most similar documents")


def run_rag_demo(question: str, index: list[dict]) -> None:
    """Full mini-RAG: retrieve + generate."""
    print(f"\nQuestion: {question}")
    print("Retrieving relevant documents...")

    results = semantic_search(question, index, top_k=3)
    print_results(results, "Retrieved context")

    print("\nGenerating answer...")
    answer = rag_answer(question, index)
    print(f"\nAnswer:\n{answer}\n")


# ── REPL ──────────────────────────────────────────────────────────────────────


def repl(index: list[dict]) -> None:
    print(f"\n{'─' * 60}")
    print(f"  Day 8 — Semantic Search")
    print(f"  Corpus: {len(CORPUS)} documents | Model: {EMBED_MODEL}")
    print(f"  Threshold: {SIM_THRESHOLD} | Top-k: {TOP_K}")
    print(f"\n  Commands:")
    print(f"    <query>          semantic search")
    print(f"    /kw <query>      keyword search")
    print(f"    /both <query>    side-by-side comparison")
    print(f"    /rag <question>  retrieve + generate answer")
    print(f"    /stats           show index info")
    print(f"    quit             exit")
    print(f"{'─' * 60}\n")

    while True:
        try:
            raw = input("Search: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not raw or raw.lower() == "quit":
            break

        if raw.startswith("/kw "):
            query = raw[4:].strip()
            results = keyword_search(query, CORPUS)
            print_results(results, f'Keyword: "{query}"', show_score=True)

        elif raw.startswith("/both "):
            query = raw[6:].strip()
            sem = semantic_search(query, index, top_k=3)
            kw = keyword_search(query, CORPUS, top_k=3)
            print_comparison(query, sem, kw)

        elif raw.startswith("/rag "):
            question = raw[5:].strip()
            run_rag_demo(question, index)

        elif raw == "/stats":
            sample = index[0]["embedding"]
            print(f"\n  Documents : {len(index)}")
            print(f"  Model     : {EMBED_MODEL}")
            print(f"  Dimensions: {len(sample)}")
            print(f"  Cache     : {CACHE_FILE}")
            print(f"  Threshold : {SIM_THRESHOLD}\n")

        else:
            # Default: semantic search
            results = semantic_search(raw, index, top_k=TOP_K)
            print_results(results, f'Semantic: "{raw}"')
            # Flag below-threshold results
            below = [r for r in results if r["score"] < SIM_THRESHOLD]
            if below:
                print(
                    f"\n  ○ = below threshold ({SIM_THRESHOLD}) — likely not relevant"
                )

        print()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    # Build or load the embedding index (runs once, cached after)
    index = build_or_load_index()

    arg = sys.argv[1] if len(sys.argv) > 1 else None

    if arg == "compare":
        run_comparison(index)

    elif arg == "similar" and len(sys.argv) > 2:
        run_similar(" ".join(sys.argv[2:]), index)

    elif arg == "rag" and len(sys.argv) > 2:
        run_rag_demo(" ".join(sys.argv[2:]), index)

    else:
        repl(index)


if __name__ == "__main__":
    main()

"""
Day 9 — Vector databases with Qdrant
======================================
Ports the day-8 semantic search engine to Qdrant.
Same corpus, same embeddings — but stored with HNSW indexing,
metadata payload, and payload filtering.

Key concepts demonstrated:
  • Creating a Qdrant collection with vector config
  • Upserting points: id + vector + payload
  • Semantic search via client.search()
  • Payload filtering: filter by category, date range, tags
  • Scrolling (iterating all points without a query vector)
  • Latency benchmarking: JSON file vs Qdrant at different scales

Setup:
  docker run -d -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
  pip install qdrant-client requests
  ollama pull nomic-embed-text
  ollama serve

Run modes:
  python day9_qdrant.py                    # interactive search REPL
  python day9_qdrant.py ingest             # (re)load all documents
  python day9_qdrant.py search "query"     # single search
  python day9_qdrant.py filter             # demo payload filters
  python day9_qdrant.py benchmark          # latency: brute-force vs Qdrant
  python day9_qdrant.py stats              # collection info
"""

import math
import sys
import time
import uuid
from datetime import datetime

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

# ── Config ────────────────────────────────────────────────────────────────────

QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
COLLECTION = "knowledge"
VECTOR_DIM = 768  # nomic-embed-text output dimension
TOP_K = 5

# ── Qdrant client ─────────────────────────────────────────────────────────────

client = QdrantClient(url=QDRANT_URL)

# ── Corpus (same as day 8, now with rich metadata payloads) ───────────────────
# The payload can hold any JSON-serialisable fields.
# Qdrant indexes them for fast filtering — you can filter on any field
# without it affecting vector search performance.

CORPUS = [
    {
        "id": "py_lists",
        "title": "Python lists",
        "text": "Python lists are ordered, mutable sequences. Created with square brackets: my_list = [1, 2, 3]. Support indexing, slicing, and methods like append(), extend(), and sort(). Lists can hold mixed types. Time complexity: O(1) for index access, O(n) for search.",
        "category": "python",
        "tags": ["data-structures", "beginner"],
        "year": 2023,
    },
    {
        "id": "py_dicts",
        "title": "Python dictionaries",
        "text": "Dictionaries store key-value pairs. Keys must be hashable. Created with curly braces. Methods: keys(), values(), items(), get(). Python 3.7+ preserves insertion order. O(1) average for lookup and insert.",
        "category": "python",
        "tags": ["data-structures", "beginner"],
        "year": 2023,
    },
    {
        "id": "py_generators",
        "title": "Python generators",
        "text": "Generators are lazy iterators created with the yield keyword. They compute values on demand instead of storing them all in memory. Useful for large datasets. Generator expressions use parentheses. Cannot be reused after exhaustion.",
        "category": "python",
        "tags": ["advanced", "performance"],
        "year": 2023,
    },
    {
        "id": "py_async",
        "title": "Async programming in Python",
        "text": "Python's asyncio enables concurrent I/O-bound operations without threads. async def defines a coroutine. await pauses execution until the awaited coroutine completes. asyncio.gather() runs multiple coroutines concurrently. Not suitable for CPU-bound tasks.",
        "category": "python",
        "tags": ["advanced", "concurrency"],
        "year": 2024,
    },
    {
        "id": "py_decorators",
        "title": "Python decorators",
        "text": "Decorators are functions that wrap other functions to add behaviour. Applied with @syntax. Common uses: logging, timing, caching, authentication. functools.wraps preserves the wrapped function's metadata.",
        "category": "python",
        "tags": ["advanced", "patterns"],
        "year": 2023,
    },
    {
        "id": "ml_supervised",
        "title": "Supervised learning",
        "text": "Supervised learning trains models on labelled data. Two main types: classification and regression. Common algorithms: linear regression, decision trees, SVMs, neural networks. Evaluated with accuracy, F1, RMSE.",
        "category": "machine-learning",
        "tags": ["fundamentals", "beginner"],
        "year": 2023,
    },
    {
        "id": "ml_embeddings",
        "title": "Word and sentence embeddings",
        "text": "Embeddings represent text as dense vectors where similar texts cluster together. Word2Vec and GloVe learn from co-occurrence. Sentence embeddings (SBERT) encode entire sentences. Key property: vector arithmetic captures semantic relationships.",
        "category": "machine-learning",
        "tags": ["nlp", "advanced"],
        "year": 2024,
    },
    {
        "id": "ml_transformers",
        "title": "Transformer architecture",
        "text": "Transformers use self-attention to weigh the importance of each token. Encoder-only models (BERT) for understanding. Decoder-only (GPT) for generation. Attention formula: softmax(QK^T / sqrt(d_k)) * V.",
        "category": "machine-learning",
        "tags": ["nlp", "advanced", "architecture"],
        "year": 2024,
    },
    {
        "id": "ml_rag",
        "title": "Retrieval-augmented generation",
        "text": "RAG combines retrieval with generation. A query retrieves relevant documents which are injected into the LLM prompt as context. Reduces hallucination. Components: embedding model, vector store, retriever, generator.",
        "category": "machine-learning",
        "tags": ["llm", "advanced"],
        "year": 2024,
    },
    {
        "id": "db_postgres",
        "title": "PostgreSQL basics",
        "text": "PostgreSQL is an open-source relational database. Supports ACID transactions, complex queries, and JSON. Key commands: SELECT, INSERT, UPDATE, DELETE. EXPLAIN ANALYZE shows query execution plans.",
        "category": "databases",
        "tags": ["sql", "beginner"],
        "year": 2023,
    },
    {
        "id": "db_redis",
        "title": "Redis data structures",
        "text": "Redis is an in-memory key-value store. Supports strings, hashes, lists, sets, sorted sets, and streams. Used for caching, session storage, pub/sub, and rate limiting. TTL automatically expires keys.",
        "category": "databases",
        "tags": ["nosql", "caching", "beginner"],
        "year": 2023,
    },
    {
        "id": "db_vectors",
        "title": "Vector databases",
        "text": "Vector databases store and query high-dimensional embeddings. Use ANN algorithms like HNSW for fast similarity search. Popular options: Qdrant, Pinecone, Chroma, Weaviate, FAISS. Support metadata filtering alongside vector search.",
        "category": "databases",
        "tags": ["vectors", "advanced"],
        "year": 2024,
    },
    {
        "id": "api_rest",
        "title": "REST API design",
        "text": "REST APIs use HTTP methods: GET, POST, PUT/PATCH, DELETE. Resources are identified by URLs. Use nouns not verbs. Status codes: 200 OK, 201 Created, 400 Bad Request, 404 Not Found. Versioning: /api/v1/.",
        "category": "backend",
        "tags": ["api", "beginner"],
        "year": 2023,
    },
    {
        "id": "api_fastapi",
        "title": "FastAPI framework",
        "text": "FastAPI auto-generates OpenAPI docs from type annotations. Uses Pydantic for validation. Supports async handlers. Dependency injection via Depends(). Background tasks with BackgroundTasks. Deploy with uvicorn.",
        "category": "backend",
        "tags": ["api", "python", "intermediate"],
        "year": 2024,
    },
    {
        "id": "devops_docker",
        "title": "Docker containers",
        "text": "Docker packages apps into containers with all dependencies. Dockerfile defines the image. Multi-stage builds reduce image size. docker-compose orchestrates multi-container apps. Health checks ensure dependent services are ready.",
        "category": "devops",
        "tags": ["containers", "beginner"],
        "year": 2023,
    },
    {
        "id": "devops_k8s",
        "title": "Kubernetes basics",
        "text": "Kubernetes orchestrates containerised workloads. Key objects: Pod, Deployment, Service, ConfigMap, Secret. kubectl is the CLI. Rolling updates replace pods gradually. HPA scales based on CPU/memory metrics.",
        "category": "devops",
        "tags": ["containers", "advanced"],
        "year": 2024,
    },
    {
        "id": "security_jwt",
        "title": "JWT authentication",
        "text": "JSON Web Tokens have three parts: header, payload, signature. Signed with HMAC or RSA. Stateless auth — server validates signature without storing session. Access tokens short-lived (15 min). Never store JWTs in localStorage.",
        "category": "security",
        "tags": ["auth", "intermediate"],
        "year": 2023,
    },
    {
        "id": "security_sqli",
        "title": "SQL injection prevention",
        "text": "SQL injection occurs when user input is interpolated into SQL queries. Prevention: always use parameterised queries. ORM frameworks handle this automatically. Never construct queries with string concatenation. Use least-privilege DB accounts.",
        "category": "security",
        "tags": ["sql", "intermediate"],
        "year": 2023,
    },
    {
        "id": "testing_pytest",
        "title": "Testing with pytest",
        "text": "pytest is Python's most popular testing framework. Tests are functions prefixed with test_. Fixtures provide reusable setup. Parametrize with @pytest.mark.parametrize. Mocking with pytest-mock. Coverage with pytest-cov.",
        "category": "testing",
        "tags": ["python", "beginner"],
        "year": 2023,
    },
    {
        "id": "perf_caching",
        "title": "Caching strategies",
        "text": "Cache-aside: app checks cache first, fetches from DB on miss. Write-through: write to cache and DB simultaneously. Write-behind: write to cache, async flush to DB. TTL prevents stale data. LRU eviction removes least recently used items.",
        "category": "performance",
        "tags": ["caching", "intermediate"],
        "year": 2024,
    },
]


# ── Embedding helper ──────────────────────────────────────────────────────────


def embed(text: str) -> list[float]:
    """Call Ollama embedding endpoint. Same model as day 8."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]
    except requests.exceptions.ConnectionError:
        print("[error] Ollama not reachable. Run: ollama serve")
        sys.exit(1)


# ── Collection management ─────────────────────────────────────────────────────


def ensure_collection(recreate: bool = False) -> None:
    """
    Create the Qdrant collection if it doesn't exist.
    VectorParams specifies:
      size     = embedding dimension (must match your model — 768 for nomic-embed-text)
      distance = similarity metric (Cosine for text embeddings)
    """
    existing = [c.name for c in client.get_collections().collections]

    if recreate and COLLECTION in existing:
        client.delete_collection(COLLECTION)
        print(f"[qdrant] Deleted existing collection '{COLLECTION}'")
        existing.remove(COLLECTION)

    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=VECTOR_DIM,
                distance=Distance.COSINE,
            ),
        )
        print(
            f"[qdrant] Created collection '{COLLECTION}' (dim={VECTOR_DIM}, metric=cosine)"
        )
    else:
        print(f"[qdrant] Collection '{COLLECTION}' already exists")


def collection_count() -> int:
    return client.get_collection(COLLECTION).points_count or 0


# ── Ingestion ─────────────────────────────────────────────────────────────────


def ingest_corpus(force: bool = False) -> None:
    """
    Embed each document and upsert into Qdrant.

    Each Qdrant point has:
      id      : a UUID (Qdrant requires UUID or unsigned int)
      vector  : the embedding
      payload : all metadata (title, text, category, tags, year)

    We store the original doc_id in the payload so we can retrieve it.
    upsert() inserts new points and updates existing ones (idempotent).
    """
    ensure_collection(recreate=force)

    if not force and collection_count() >= len(CORPUS):
        print(f"[qdrant] Already have {collection_count()} points — skipping ingest")
        print(f"         Run with 'ingest' arg to force re-ingest")
        return

    print(f"[qdrant] Ingesting {len(CORPUS)} documents...")
    points = []

    for i, doc in enumerate(CORPUS):
        text_to_embed = f"{doc['title']}. {doc['text']}"
        vector = embed(text_to_embed)

        points.append(
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, doc["id"])),  # deterministic UUID
                vector=vector,
                payload={
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "text": doc["text"],
                    "category": doc["category"],
                    "tags": doc["tags"],
                    "year": doc["year"],
                },
            )
        )
        print(f"  [{i + 1}/{len(CORPUS)}] {doc['title']}", end="\r")

    client.upsert(collection_name=COLLECTION, points=points, wait=True)
    print(f"\n[qdrant] Upserted {len(points)} points")


# ── Search ────────────────────────────────────────────────────────────────────


def search(
    query: str,
    top_k: int = TOP_K,
    query_filter: Filter | None = None,
) -> list[dict]:
    """
    Semantic search with optional payload filter.

    query_filter is a Qdrant Filter object that restricts which points
    are eligible — e.g. only points where category == "python".
    The filter runs inside the HNSW index, not as a post-filter,
    so it's fast even with millions of points.
    """
    t0 = time.perf_counter()
    query_vec = embed(query)
    embed_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    results = client.query_points(
        collection_name=COLLECTION,
        query=query_vec,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )
    search_ms = (time.perf_counter() - t1) * 1000

    return [
        {
            "title": r.payload["title"],
            "text": r.payload["text"],
            "category": r.payload["category"],
            "tags": r.payload["tags"],
            "year": r.payload["year"],
            "doc_id": r.payload["doc_id"],
            "score": round(r.score, 4),
            "_embed_ms": round(embed_ms, 1),
            "_search_ms": round(search_ms, 2),
        }
        for r in results.points
    ]


# ── Filter helpers ────────────────────────────────────────────────────────────


def filter_by_category(category: str) -> Filter:
    """Only return points where payload.category == category."""
    return Filter(
        must=[FieldCondition(key="category", match=MatchValue(value=category))]
    )


def filter_by_tag(tag: str) -> Filter:
    """Only return points where payload.tags contains tag."""
    return Filter(must=[FieldCondition(key="tags", match=MatchValue(value=tag))])


def filter_by_year_range(min_year: int, max_year: int) -> Filter:
    """Only return points where min_year <= payload.year <= max_year."""
    return Filter(
        must=[FieldCondition(key="year", range=Range(gte=min_year, lte=max_year))]
    )


def filter_combined(category: str, min_year: int) -> Filter:
    """Category AND year — multiple must conditions are ANDed together."""
    return Filter(
        must=[
            FieldCondition(key="category", match=MatchValue(value=category)),
            FieldCondition(key="year", range=Range(gte=min_year)),
        ]
    )


# ── Display ───────────────────────────────────────────────────────────────────


def print_results(results: list[dict], label: str = "") -> None:
    if label:
        print(f"\n  {label}")
    print(f"  {'─' * 54}")
    if not results:
        print("  (no results)")
        return
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['title']}]  score={r['score']}  ({r['category']})")
        print(f"     {r['text'][:85]}...")
        print(f"     tags={r['tags']}  year={r['year']}")
    if results:
        r0 = results[0]
        print(f"\n  embed={r0['_embed_ms']}ms  search={r0['_search_ms']}ms")


# ── Filter demo ───────────────────────────────────────────────────────────────


def run_filter_demo() -> None:
    """
    Show how the same query returns different results with different filters.
    This is the most important Qdrant feature beyond basic search.
    """
    query = "how to handle data efficiently"

    print(f"\n{'═' * 58}")
    print(f'  Filter demo — query: "{query}"')
    print(f"{'═' * 58}")

    print("\n  No filter (all categories):")
    print_results(search(query, top_k=3))

    print("\n  Filter: category = python")
    print_results(search(query, top_k=3, query_filter=filter_by_category("python")))

    print("\n  Filter: category = databases")
    print_results(search(query, top_k=3, query_filter=filter_by_category("databases")))

    print("\n  Filter: tag = advanced")
    print_results(search(query, top_k=3, query_filter=filter_by_tag("advanced")))

    print("\n  Filter: year >= 2024")
    print_results(search(query, top_k=3, query_filter=filter_by_year_range(2024, 2099)))

    print("\n  Filter: category = machine-learning AND year >= 2024")
    print_results(
        search(query, top_k=3, query_filter=filter_combined("machine-learning", 2024))
    )


# ── Benchmark ─────────────────────────────────────────────────────────────────


def cosine_similarity_brute(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0


def run_benchmark() -> None:
    """
    Compare brute-force (day 8 approach) vs Qdrant HNSW.

    At 20 documents the difference is noise. The benchmark also
    synthesises a larger fake corpus to show how the gap widens.
    This is the key insight: Qdrant's value is at scale.
    """
    print(f"\n{'═' * 58}")
    print(f"  Benchmark: brute-force vs Qdrant")
    print(f"{'═' * 58}")

    query = "storing data in memory for fast access"
    query_vec = embed(query)

    # ── Brute force on actual corpus ──────────────────────────────────────
    # Scroll all points from Qdrant to get their vectors
    all_points = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=COLLECTION,
            limit=100,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )
        all_points.extend(batch)
        if offset is None:
            break

    corpus_size = len(all_points)

    # Brute force
    n_trials = 20
    t0 = time.perf_counter()
    for _ in range(n_trials):
        scored = [(cosine_similarity_brute(query_vec, p.vector), p) for p in all_points]
        scored.sort(key=lambda x: x[0], reverse=True)
        _ = scored[:TOP_K]
    brute_ms = (time.perf_counter() - t0) / n_trials * 1000

    # Qdrant HNSW (embedding already done)
    t1 = time.perf_counter()
    for _ in range(n_trials):
        client.query_points(
            COLLECTION, query=query_vec, limit=TOP_K, with_payload=False
        )
    qdrant_ms = (time.perf_counter() - t1) / n_trials * 1000

    print(f"\n  Corpus size    : {corpus_size} documents")
    print(f"  Brute-force    : {brute_ms:.2f}ms avg over {n_trials} runs")
    print(f"  Qdrant HNSW    : {qdrant_ms:.2f}ms avg over {n_trials} runs")
    speedup = brute_ms / qdrant_ms if qdrant_ms > 0 else 1
    print(f"  Speedup        : {speedup:.1f}×")

    print(f"\n  Projected at scale (HNSW is O(log n), brute-force is O(n)):")
    print(f"  {'Size':>12}  {'Brute-force':>14}  {'Qdrant (est)':>14}")
    print(f"  {'─' * 44}")
    base_bf = brute_ms / corpus_size
    base_q = qdrant_ms
    for n in [1_000, 10_000, 100_000, 1_000_000]:
        est_bf = base_bf * n
        # HNSW scales as O(log n) — roughly constant after index build
        est_q = (
            base_q * (math.log(n) / math.log(corpus_size))
            if n > corpus_size
            else base_q
        )
        print(f"  {n:>12,}  {est_bf:>12.0f}ms  {est_q:>12.1f}ms")


# ── Scroll demo ───────────────────────────────────────────────────────────────


def run_scroll_demo() -> None:
    """
    Scroll iterates all points in the collection without a query vector.
    Useful for: bulk export, offline re-processing, data audits.
    """
    print(f"\n  Scrolling all points in '{COLLECTION}':")
    offset = None
    count = 0
    while True:
        batch, offset = client.scroll(
            collection_name=COLLECTION,
            limit=5,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in batch:
            print(f"  • {p.payload['title']:35s}  [{p.payload['category']}]")
            count += 1
        if offset is None:
            break
    print(f"\n  Total: {count} points")


# ── REPL ──────────────────────────────────────────────────────────────────────


def repl() -> None:
    count = collection_count()
    print(f"\n{'─' * 58}")
    print(f"  Day 9 — Qdrant Semantic Search")
    print(f"  Collection: '{COLLECTION}'  |  Points: {count}")
    print(f"\n  Commands:")
    print(f"    <query>                     semantic search (all)")
    print(f"    /cat <cat> <query>          filter by category")
    print(f"    /tag <tag> <query>          filter by tag")
    print(f"    /year <min> <max> <query>   filter by year range")
    print(f"    /scroll                     list all documents")
    print(f"    /stats                      collection info")
    print(f"    quit")
    print(f"{'─' * 58}\n")

    while True:
        try:
            raw = input("Search: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not raw or raw.lower() == "quit":
            break

        parts = raw.split()

        if parts[0] == "/cat" and len(parts) >= 3:
            cat, query = parts[1], " ".join(parts[2:])
            results = search(query, query_filter=filter_by_category(cat))
            print_results(results, f"category={cat}")

        elif parts[0] == "/tag" and len(parts) >= 3:
            tag, query = parts[1], " ".join(parts[2:])
            results = search(query, query_filter=filter_by_tag(tag))
            print_results(results, f"tag={tag}")

        elif parts[0] == "/year" and len(parts) >= 4:
            min_y, max_y = int(parts[1]), int(parts[2])
            query = " ".join(parts[3:])
            results = search(query, query_filter=filter_by_year_range(min_y, max_y))
            print_results(results, f"year={min_y}–{max_y}")

        elif raw == "/scroll":
            run_scroll_demo()

        elif raw == "/stats":
            info = client.get_collection(COLLECTION)
            print(f"\n  Collection : {COLLECTION}")
            print(f"  Points     : {info.points_count}")
            print(f"  Vector dim : {info.config.params.vectors.size}")
            print(f"  Distance   : {info.config.params.vectors.distance}")
            print(f"  Status     : {info.status}\n")

        else:
            results = search(raw)
            print_results(results, f'"{raw}"')

        print()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    if arg == "ingest":
        ingest_corpus(force=True)

    elif arg == "search" and len(sys.argv) > 2:
        ingest_corpus()
        query = " ".join(sys.argv[2:])
        print_results(search(query), f'"{query}"')

    elif arg == "filter":
        ingest_corpus()
        run_filter_demo()

    elif arg == "benchmark":
        ingest_corpus()
        run_benchmark()

    elif arg == "scroll":
        ingest_corpus()
        run_scroll_demo()

    elif arg == "stats":
        ingest_corpus()
        info = client.get_collection(COLLECTION)
        print(f"Collection : {COLLECTION}")
        print(f"Points     : {info.points_count}")
        print(f"Dimensions : {info.config.params.vectors.size}")
        print(f"Distance   : {info.config.params.vectors.distance}")
        print(f"Status     : {info.status}")

    else:
        ingest_corpus()
        repl()


if __name__ == "__main__":
    main()

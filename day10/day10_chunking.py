"""
Day 10 — Document chunking strategies for RAG
===============================================
Indexes the same document corpus using four chunking strategies,
then evaluates which strategy retrieves the most relevant chunk
for each of 10 test questions.

Strategies implemented:
  1. Fixed-size     — split every N chars with M overlap
  2. Recursive      — try paragraph → sentence → word → char separators
  3. Semantic       — embed sentences, split where topic similarity drops
  4. Hierarchical   — small child chunks for retrieval, large parent for context

Each strategy gets its own Qdrant collection. The evaluator runs all
10 questions against all 4 strategies and scores Hit@1 and Hit@3.

Requirements:
  docker run -d -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
  uv add qdrant-client requests
  ollama pull nomic-embed-text
  ollama serve

Run modes:
  uv run python day10_chunking.py              # full eval (ingest + score)
  uv run python day10_chunking.py ingest       # ingest only (re-build all collections)
  uv run python day10_chunking.py search <q>   # search all strategies for one query
  uv run python day10_chunking.py inspect      # print chunks from each strategy
"""

import math
import re
import sys
import uuid
from dataclasses import dataclass
from textwrap import dedent

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# ── Config ────────────────────────────────────────────────────────────────────

QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
VECTOR_DIM = 768
TOP_K = 5

client = QdrantClient(url=QDRANT_URL)

# Collection names — one per strategy
COLLECTIONS = {
    "fixed": "chunks_fixed",
    "recursive": "chunks_recursive",
    "semantic": "chunks_semantic",
    "hierarchical": "chunks_hierarchical",
}

# ── Source documents ──────────────────────────────────────────────────────────
# Simulating a multi-section technical document.
# In practice this would be a PDF or markdown file you load from disk.

DOCUMENTS = [
    {
        "doc_id": "python_intro",
        "title": "Introduction to Python",
        "text": dedent("""
            Python is a high-level, general-purpose programming language. Its design philosophy
            emphasises code readability with the use of significant indentation. Python is
            dynamically typed and garbage-collected. It supports multiple programming paradigms,
            including structured, object-oriented, and functional programming.

            Python was created by Guido van Rossum and first released in 1991. The language's
            core philosophy is summarised in the Zen of Python, which includes aphorisms such as
            "Beautiful is better than ugly" and "Readability counts."

            Python consistently ranks as one of the most popular programming languages. It is
            widely used in web development, scientific computing, data science, artificial
            intelligence, and systems scripting.

            Variables in Python do not need to be declared with a specific type. The type is
            inferred at runtime. Python uses indentation to delimit code blocks, unlike many
            other languages that use braces. This enforced indentation makes Python code
            visually consistent across different codebases.

            Python's standard library is vast and covers areas including string processing,
            internet protocols, file I/O, and data serialisation. Third-party packages are
            available through the Python Package Index (PyPI) and can be installed with pip.

            Functions in Python are first-class objects, meaning they can be passed as
            arguments, returned from other functions, and assigned to variables. This enables
            functional programming patterns like map, filter, and reduce, as well as decorators.

            Python supports object-oriented programming through classes. Classes can inherit
            from one or more parent classes. Python uses duck typing — if an object has the
            required methods, it can be used regardless of its actual type.

            Error handling in Python uses try/except blocks. Exceptions are objects that can
            be caught, inspected, and re-raised. The finally block runs regardless of whether
            an exception occurred, making it useful for cleanup operations like closing files.

            List comprehensions provide a concise way to create lists. They are often more
            readable and faster than equivalent for loops. Dictionary and set comprehensions
            follow the same syntax. Generator expressions are similar but produce values lazily.

            Python's memory management uses reference counting combined with a cyclic garbage
            collector. Objects are automatically deallocated when their reference count drops
            to zero. The garbage collector handles circular references that reference counting
            alone cannot resolve.
        """).strip(),
    },
    {
        "doc_id": "python_data_structures",
        "title": "Python Data Structures",
        "text": dedent("""
            Python provides four built-in collection types that cover most use cases.
            Understanding when to use each is fundamental to writing efficient Python code.

            Lists are ordered, mutable sequences. They are created with square brackets and
            support indexing, slicing, and a rich set of methods. Appending to a list is O(1)
            amortised. Inserting at an arbitrary position is O(n) because elements must shift.
            Lists are backed by dynamic arrays and automatically resize when they reach capacity.

            Tuples are ordered but immutable sequences. Once created, their elements cannot be
            changed. Tuples are faster than lists for iteration and use less memory. They are
            commonly used for returning multiple values from functions and as dictionary keys
            when the data should not change.

            Dictionaries store key-value pairs in a hash table. Keys must be hashable — strings,
            numbers, and tuples of hashable types are all valid keys. Lookup, insertion, and
            deletion are O(1) on average. Python 3.7 and later guarantees insertion order is
            preserved. The get() method returns a default value instead of raising KeyError for
            missing keys.

            Sets are unordered collections of unique elements. They are backed by a hash table
            and support fast membership testing (O(1) average). Set operations include union,
            intersection, difference, and symmetric difference. Sets are useful for deduplication
            and for testing whether two collections have elements in common.

            The collections module provides specialised data structures beyond the built-ins.
            deque (double-ended queue) supports O(1) appends and pops from both ends. Counter
            counts hashable objects and supports arithmetic operations. OrderedDict remembers
            insertion order even on older Python versions. defaultdict provides a default factory
            for missing keys, avoiding explicit key initialisation.

            Choosing the right data structure has significant performance implications. Using a
            list for membership testing when a set would suffice is a common mistake — list
            membership is O(n), set membership is O(1). For sorted data with frequent lookups,
            the bisect module provides efficient binary search over sorted lists.
        """).strip(),
    },
    {
        "doc_id": "python_async",
        "title": "Asynchronous Python",
        "text": dedent("""
            Asynchronous programming allows a single thread to handle multiple I/O-bound tasks
            concurrently. Instead of blocking while waiting for a network response or file read,
            the program can switch to other tasks and return when the I/O completes.

            Python's asyncio library implements an event loop that manages coroutines. A coroutine
            is a function defined with async def. When a coroutine reaches an await expression,
            it suspends and yields control back to the event loop, which can then run another
            coroutine. This cooperative multitasking avoids the overhead of threads.

            The await keyword can only be used inside an async function. It can be applied to
            any awaitable object: coroutines, Tasks, or Futures. asyncio.sleep() is the async
            equivalent of time.sleep() — it suspends the coroutine without blocking the event loop.

            asyncio.gather() runs multiple coroutines concurrently and collects their results.
            It is the primary tool for parallelising independent I/O operations. If any coroutine
            raises an exception, gather() cancels the others by default unless return_exceptions
            is set to True.

            Tasks wrap coroutines and schedule them to run on the event loop. Creating a Task
            with asyncio.create_task() starts the coroutine immediately, whereas awaiting a
            coroutine directly runs it inline. Tasks are useful when you want to start work
            without immediately waiting for the result.

            Asynchronous context managers (async with) and asynchronous iterators (async for)
            extend the async model to resource management and streaming data. aiohttp is a
            popular library for making async HTTP requests. Databases can be accessed
            asynchronously using libraries like asyncpg for PostgreSQL.

            The key limitation of asyncio is that it only improves I/O-bound performance.
            CPU-bound tasks block the event loop and prevent other coroutines from running.
            For CPU-bound concurrency, use multiprocessing or ProcessPoolExecutor. For a mix,
            run CPU-bound work in an executor: await loop.run_in_executor(None, cpu_bound_fn).
        """).strip(),
    },
]


# ── Ollama helpers ────────────────────────────────────────────────────────────


def embed(text: str) -> list[float]:
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["embedding"]
    except requests.exceptions.ConnectionError:
        print("[error] Ollama not reachable. Run: ollama serve")
        sys.exit(1)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    ma = math.sqrt(sum(x * x for x in a))
    mb = math.sqrt(sum(x * x for x in b))
    return dot / (ma * mb) if ma and mb else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# CHUNKING STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Chunk:
    text: str
    doc_id: str
    title: str
    chunk_idx: int
    strategy: str
    parent_text: str = ""  # for hierarchical only


# ── Strategy 1: Fixed-size ────────────────────────────────────────────────────


def chunk_fixed(doc: dict, size: int = 400, overlap: int = 80) -> list[Chunk]:
    """
    Split text into fixed-size character windows with overlap.

    The overlap prevents a concept from being entirely cut off at a boundary.
    For example with size=400 and overlap=80, chunk 2 starts 320 chars into
    chunk 1 — so the last 80 chars of chunk 1 also appear at the start of chunk 2.

    This is the simplest strategy and the baseline all others are measured against.
    """
    text = doc["text"]
    chunks = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    text=chunk_text,
                    doc_id=doc["doc_id"],
                    title=doc["title"],
                    chunk_idx=idx,
                    strategy="fixed",
                )
            )
            idx += 1
        if end == len(text):
            break
        start += size - overlap

    return chunks


# ── Strategy 2: Recursive character splitting ─────────────────────────────────


def chunk_recursive(doc: dict, max_size: int = 400, overlap: int = 60) -> list[Chunk]:
    """
    Try separators in priority order: paragraph → sentence → word → character.
    This preserves natural language boundaries wherever possible.

    Algorithm:
      1. Try to split on \\n\\n (paragraph breaks)
      2. If any resulting piece is still > max_size, split it on \\n (newlines)
      3. If still too large, split on '. ' (sentence endings)
      4. If still too large, split on ' ' (word boundaries)
      5. Last resort: split on '' (character level, same as fixed-size)

    Then merge small splits back together up to max_size, with overlap.
    This is what LangChain's RecursiveCharacterTextSplitter does.
    """
    separators = ["\n\n", "\n", ". ", " ", ""]

    def split_with_sep(text: str, seps: list[str]) -> list[str]:
        if not seps:
            # Character-level fallback
            return [
                text[i : i + max_size] for i in range(0, len(text), max_size - overlap)
            ]
        sep = seps[0]
        pieces = text.split(sep) if sep else list(text)
        result = []
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if len(piece) <= max_size:
                result.append(piece)
            else:
                # Recursively split oversized pieces with the next separator
                result.extend(split_with_sep(piece, seps[1:]))
        return result

    raw_pieces = split_with_sep(doc["text"], separators)

    # Merge small pieces into chunks up to max_size, with overlap
    chunks = []
    current = ""
    prev_end = ""  # last `overlap` chars of previous chunk

    for piece in raw_pieces:
        candidate = (prev_end + " " + piece).strip() if prev_end else piece
        if len(current) + len(piece) + 1 <= max_size:
            current = (current + " " + piece).strip() if current else piece
        else:
            if current:
                chunks.append(current)
                # Keep last `overlap` chars as the start of the next chunk
                prev_end = current[-overlap:] if len(current) > overlap else current
            current = (prev_end + " " + piece).strip() if prev_end else piece

    if current:
        chunks.append(current)

    return [
        Chunk(
            text=c,
            doc_id=doc["doc_id"],
            title=doc["title"],
            chunk_idx=i,
            strategy="recursive",
        )
        for i, c in enumerate(chunks)
        if c.strip()
    ]


# ── Strategy 3: Semantic chunking ─────────────────────────────────────────────


def chunk_semantic(
    doc: dict, threshold_drop: float = 0.12, min_chunk_size: int = 150
) -> list[Chunk]:
    """
    Embed every sentence, then split where the cosine similarity between
    consecutive sentences drops significantly (a topic boundary).

    Algorithm:
      1. Split text into sentences on '. ', '! ', '? '
      2. Embed each sentence (this is the expensive step)
      3. For each adjacent pair, compute cosine similarity
      4. If similarity drops by more than `threshold_drop` from the rolling
         average, start a new chunk
      5. Merge chunks smaller than min_chunk_size with their neighbour

    The threshold_drop controls sensitivity: lower = more splits, higher = fewer.
    Typical values: 0.08–0.15 for technical documents.
    """
    # Split into sentences
    raw_sentences = re.split(r"(?<=[.!?])\s+", doc["text"])
    sentences = [s.strip() for s in raw_sentences if s.strip() and len(s.strip()) > 10]

    if len(sentences) <= 2:
        return chunk_recursive(doc)  # fallback for very short docs

    print(f"    [semantic] embedding {len(sentences)} sentences...", end="", flush=True)
    embeddings = [embed(s) for s in sentences]
    print(f" done")

    # Compute similarity between adjacent sentences
    sims = [
        cosine(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)
    ]

    # Rolling average similarity — used to detect relative drops
    window = 3
    avg_sim = sum(sims[:window]) / min(window, len(sims)) if sims else 0.7

    # Find split points where similarity drops significantly
    split_indices = set()
    for i, sim in enumerate(sims):
        avg_sim = 0.7 * avg_sim + 0.3 * sim  # exponential moving average
        if avg_sim - sim > threshold_drop:
            split_indices.add(i + 1)

    # Group sentences into chunks
    raw_chunks: list[list[str]] = []
    current_group: list[str] = []

    for i, sentence in enumerate(sentences):
        current_group.append(sentence)
        if i + 1 in split_indices:
            raw_chunks.append(current_group)
            current_group = []

    if current_group:
        raw_chunks.append(current_group)

    # Merge tiny chunks with their neighbour
    merged: list[str] = []
    for group in raw_chunks:
        text = " ".join(group)
        if len(text) < min_chunk_size and merged:
            merged[-1] += " " + text
        else:
            merged.append(text)

    return [
        Chunk(
            text=c,
            doc_id=doc["doc_id"],
            title=doc["title"],
            chunk_idx=i,
            strategy="semantic",
        )
        for i, c in enumerate(merged)
        if c.strip()
    ]


# ── Strategy 4: Hierarchical (parent-child) ───────────────────────────────────


def chunk_hierarchical(
    doc: dict, child_size: int = 150, parent_size: int = 600, overlap: int = 40
) -> list[Chunk]:
    """
    Two-level chunking:
      • Parent chunks (~600 chars) provide context to the LLM
      • Child chunks (~150 chars) are what gets embedded and retrieved

    Each child chunk stores its parent text in the payload.
    At query time: search by child embedding, return parent text to the LLM.

    This solves the precision/context tradeoff directly:
      - Small child embedding = precise semantic match
      - Large parent text = rich context for generation

    It's the pattern underlying production RAG systems (LlamaIndex's
    "sentence window" and "parent document retriever" are implementations
    of this idea).
    """
    chunks = []
    text = doc["text"]
    p_idx = 0

    # Generate parent windows
    p_start = 0
    while p_start < len(text):
        p_end = min(p_start + parent_size, len(text))
        parent_text = text[p_start:p_end].strip()

        # Generate child chunks within this parent
        c_start = p_start
        c_idx = 0
        while c_start < p_end:
            c_end = min(c_start + child_size, p_end)
            child_text = text[c_start:c_end].strip()
            if child_text:
                chunks.append(
                    Chunk(
                        text=child_text,
                        doc_id=doc["doc_id"],
                        title=doc["title"],
                        chunk_idx=p_idx * 100 + c_idx,
                        strategy="hierarchical",
                        parent_text=parent_text,
                    )
                )
                c_idx += 1
            if c_end == p_end:
                break
            c_start += child_size - overlap

        if p_end == len(text):
            break
        p_start += parent_size - overlap
        p_idx += 1

    return chunks


# ── Chunking dispatcher ───────────────────────────────────────────────────────


def chunk_document(doc: dict, strategy: str) -> list[Chunk]:
    if strategy == "fixed":
        return chunk_fixed(doc)
    elif strategy == "recursive":
        return chunk_recursive(doc)
    elif strategy == "semantic":
        return chunk_semantic(doc)
    elif strategy == "hierarchical":
        return chunk_hierarchical(doc)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ── Qdrant ingestion ──────────────────────────────────────────────────────────


def ingest_strategy(strategy: str, force: bool = False) -> None:
    collection = COLLECTIONS[strategy]
    existing = [c.name for c in client.get_collections().collections]

    if force and collection in existing:
        client.delete_collection(collection)
        existing.remove(collection)

    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )

    count = client.get_collection(collection).points_count or 0
    total_docs_chunks = sum(len(chunk_document(d, strategy)) for d in DOCUMENTS)
    if not force and count >= total_docs_chunks:
        print(f"  [{strategy}] already ingested ({count} points) — skipping")
        return

    print(f"  [{strategy}] chunking and ingesting...")
    all_chunks: list[Chunk] = []
    for doc in DOCUMENTS:
        chunks = chunk_document(doc, strategy)
        all_chunks.extend(chunks)
        print(f"    {doc['title']}: {len(chunks)} chunks")

    points = []
    for i, chunk in enumerate(all_chunks):
        # For hierarchical, embed the child text but store parent for context
        text_to_embed = chunk.text
        vector = embed(text_to_embed)
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk.text,
                    "parent_text": chunk.parent_text or chunk.text,
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "chunk_idx": chunk.chunk_idx,
                    "strategy": chunk.strategy,
                    "char_count": len(chunk.text),
                },
            )
        )

    client.upsert(collection_name=collection, points=points, wait=True)
    print(f"  [{strategy}] upserted {len(points)} points into '{collection}'")


def ingest_all(force: bool = False) -> None:
    print(f"\nIngesting {len(DOCUMENTS)} documents × {len(COLLECTIONS)} strategies...")
    for strategy in COLLECTIONS:
        ingest_strategy(strategy, force=force)
    print("Done.\n")


# ── Search across strategies ──────────────────────────────────────────────────


def search_strategy(query: str, strategy: str, top_k: int = TOP_K) -> list[dict]:
    collection = COLLECTIONS[strategy]
    q_vec = embed(query)
    results = client.query_points(
        collection_name=collection,
        query=q_vec,
        limit=top_k,
        with_payload=True,
    ).points
    return [
        {
            "score": round(r.score, 4),
            "text": r.payload.get("text", ""),
            # Return parent_text for hierarchical — that's what the LLM would see
            "context_text": r.payload.get("parent_text", ""),
            "doc_id": r.payload.get("doc_id", ""),
            "char_count": r.payload.get("char_count", 0),
            "chunk_idx": r.payload.get("chunk_idx", 0),
        }
        for r in results
        if r.payload is not None
    ]


# ── Evaluation harness ────────────────────────────────────────────────────────


@dataclass
class EvalQuestion:
    question: str
    expected_doc: str  # doc_id that should be retrieved
    expected_terms: list[str]  # key terms that should appear in the top chunk


EVAL_SET: list[EvalQuestion] = [
    EvalQuestion(
        "What year was Python first released?",
        "python_intro",
        ["1991", "van rossum", "guido"],
    ),
    EvalQuestion(
        "How does Python handle memory management?",
        "python_intro",
        ["reference counting", "garbage", "deallocated"],
    ),
    EvalQuestion(
        "What is the time complexity of list insertion?",
        "python_data_structures",
        ["o(n)", "shift", "arbitrary"],
    ),
    EvalQuestion(
        "How are dictionary keys stored internally?",
        "python_data_structures",
        ["hash table", "hashable", "o(1)"],
    ),
    EvalQuestion(
        "What is the difference between a list and a tuple?",
        "python_data_structures",
        ["mutable", "immutable", "tuple"],
    ),
    EvalQuestion(
        "How does asyncio avoid blocking the thread?",
        "python_async",
        ["event loop", "suspend", "cooperative", "await"],
    ),
    EvalQuestion(
        "How do I run multiple coroutines at the same time?",
        "python_async",
        ["gather", "concurrent", "task"],
    ),
    EvalQuestion(
        "When should I not use asyncio?",
        "python_async",
        ["cpu", "bound", "multiprocessing", "executor"],
    ),
    EvalQuestion(
        "How do list comprehensions work?",
        "python_intro",
        ["comprehension", "readable", "generator"],
    ),
    EvalQuestion(
        "What does the collections module provide?",
        "python_data_structures",
        ["deque", "counter", "defaultdict"],
    ),
]


def score_result(result: dict, question: EvalQuestion) -> tuple[bool, bool]:
    """
    Returns (doc_match, term_match).
    doc_match:  correct document was retrieved
    term_match: at least one expected term appears in the retrieved text
    """
    context = (result["text"] + " " + result["context_text"]).lower()
    doc_match = result["doc_id"] == question.expected_doc
    term_match = any(t.lower() in context for t in question.expected_terms)
    return doc_match, term_match


@dataclass
class StrategyScore:
    strategy: str
    hit1_doc: int = 0  # correct doc in position 1
    hit3_doc: int = 0  # correct doc in top 3
    hit1_term: int = 0  # expected term in position 1
    hit3_term: int = 0  # expected term in top 3
    total: int = 0

    @property
    def hit1_doc_pct(self):
        return self.hit1_doc / self.total * 100

    @property
    def hit3_doc_pct(self):
        return self.hit3_doc / self.total * 100

    @property
    def hit1_term_pct(self):
        return self.hit1_term / self.total * 100

    @property
    def hit3_term_pct(self):
        return self.hit3_term / self.total * 100


def run_eval() -> None:
    print(f"\n{'═' * 62}")
    print(f"  Chunking Strategy Evaluation")
    print(f"  {len(EVAL_SET)} questions × {len(COLLECTIONS)} strategies")
    print(f"{'═' * 62}\n")

    scores = {s: StrategyScore(strategy=s) for s in COLLECTIONS}

    for q in EVAL_SET:
        print(f"Q: {q.question}")
        for strategy in COLLECTIONS:
            results = search_strategy(q.question, strategy, top_k=3)
            scores[strategy].total += 1

            if results:
                d1, t1 = score_result(results[0], q)
                if d1:
                    scores[strategy].hit1_doc += 1
                if t1:
                    scores[strategy].hit1_term += 1

                d3 = any(score_result(r, q)[0] for r in results)
                t3 = any(score_result(r, q)[1] for r in results)
                if d3:
                    scores[strategy].hit3_doc += 1
                if t3:
                    scores[strategy].hit3_term += 1

                status = "●" if t1 else ("○" if t3 else "✗")
                top_preview = results[0]["text"][:60].replace("\n", " ")
                print(
                    f'  {status} [{strategy:12s}] score={results[0]["score"]:.3f}  "{top_preview}..."'
                )

        print()

    # Summary table
    print(f"\n{'─' * 62}")
    print(
        f"  {'Strategy':14s}  {'Hit@1 doc':>10}  {'Hit@3 doc':>10}  {'Hit@1 term':>11}  {'Hit@3 term':>11}"
    )
    print(f"{'─' * 62}")
    for s, sc in scores.items():
        bar = "█" * round(sc.hit1_term_pct / 10)
        print(
            f"  {s:14s}  {sc.hit1_doc_pct:>9.0f}%  {sc.hit3_doc_pct:>9.0f}%  "
            f"{sc.hit1_term_pct:>10.0f}%  {sc.hit3_term_pct:>10.0f}%  {bar}"
        )
    print(f"{'─' * 62}")

    best = max(scores.values(), key=lambda s: s.hit1_term_pct)
    print(
        f"\n  Best strategy: {best.strategy}  ({best.hit1_term_pct:.0f}% Hit@1 term match)\n"
    )


# ── Inspect chunks ────────────────────────────────────────────────────────────


def inspect_chunks() -> None:
    """Print the first 3 chunks from each strategy for the first document."""
    doc = DOCUMENTS[0]
    print(f"\nDocument: {doc['title']} ({len(doc['text'])} chars)\n")
    for strategy in COLLECTIONS:
        chunks = chunk_document(doc, strategy)
        print(f"{'─' * 58}")
        print(f"  {strategy.upper()}  —  {len(chunks)} chunks")
        print(f"{'─' * 58}")
        for i, chunk in enumerate(chunks[:3]):
            print(f"  Chunk {i + 1} ({len(chunk.text)} chars):")
            print(f"  {chunk.text[:200].replace(chr(10), ' ')}...")
            if chunk.parent_text and chunk.parent_text != chunk.text:
                print(f"  [parent: {len(chunk.parent_text)} chars]")
            print()


# ── Cross-strategy search display ────────────────────────────────────────────


def search_all(query: str) -> None:
    print(f"\n{'═' * 58}")
    print(f'  Query: "{query}"')
    print(f"{'═' * 58}")
    for strategy in COLLECTIONS:
        results = search_strategy(query, strategy, top_k=1)
        if results:
            r = results[0]
            print(f"\n  [{strategy}]  score={r['score']}  ({r['char_count']} chars)")
            print(f"  {r['text'][:160].replace(chr(10), ' ')}...")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    if arg == "ingest":
        ingest_all(force=True)

    elif arg == "search" and len(sys.argv) > 2:
        ingest_all()
        search_all(" ".join(sys.argv[2:]))

    elif arg == "inspect":
        inspect_chunks()

    else:
        ingest_all()
        run_eval()


if __name__ == "__main__":
    main()

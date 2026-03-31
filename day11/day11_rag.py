"""
Day 11 — Building a basic RAG pipeline
========================================
A complete RAG Q&A system with:
  • Query rewriting — expand the question for better recall
  • Qdrant retrieval — semantic search over the chunked corpus
  • Context assembly — rank, deduplicate, and format retrieved chunks
  • Grounded generation — system prompt that forces the model to cite context
  • Groundedness checker — LLM-as-judge scoring whether the answer is grounded
  • Refusal detection — the model says "not in context" when it can't answer

Builds on: day 8 (embeddings), day 9 (Qdrant), day 10 (chunking)

Requirements:
  docker run -d -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
  uv add qdrant-client requests pydantic
  ollama pull nomic-embed-text
  ollama pull llama3.2
  ollama serve
  uv run python day10_chunking.py ingest   ← run this first to populate Qdrant

Run modes:
  uv run python day11_rag.py                    # interactive Q&A REPL
  uv run python day11_rag.py ask "question"     # single question
  uv run python day11_rag.py eval               # run evaluation set
  uv run python day11_rag.py demo               # preset demo questions
"""

import json
import re
import sys
import time
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Literal

import requests
from pydantic import BaseModel, Field, field_validator
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# ── Config ────────────────────────────────────────────────────────────────────

QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "llama3.2"
COLLECTION = "chunks_recursive"  # use recursive strategy from day 10
TOP_K = 5
MIN_SCORE = 0.45  # chunks below this score are excluded
MAX_CONTEXT = 2000  # max chars of context to send to the LLM

client = QdrantClient(url=QDRANT_URL)


# ── Pydantic schemas ──────────────────────────────────────────────────────────


class RAGResponse(BaseModel):
    question: str
    rewritten_query: str
    answer: str
    groundedness: Literal["grounded", "partial", "ungrounded", "refused"]
    groundedness_score: float = Field(ge=0.0, le=1.0)
    sources: list[dict]
    retrieved_chunks: int
    latency_ms: dict


class GroundednessVerdict(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    label: Literal["grounded", "partial", "ungrounded", "refused"]
    reason: str

    @field_validator("score")
    @classmethod
    def round_score(cls, v: float) -> float:
        return round(v, 2)


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


def generate(
    prompt: str, system: str = "", temperature: float = 0.2, max_tokens: int = 600
) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": GEN_MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        return "[error: Ollama not reachable]"


def clean_json(raw: str) -> str:
    text = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start : end + 1]
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — QUERY REWRITING
# ══════════════════════════════════════════════════════════════════════════════


def rewrite_query(question: str) -> str:
    """
    Rewrite the user's question for better retrieval.

    User questions are conversational and often vague.
    Embedding a rewritten, information-dense version retrieves more
    relevant chunks because it better matches the language in the documents.

    Examples of what this fixes:
      "how does that memory thing work" → "Python memory management reference counting garbage collection"
      "is it fast?"                     → "Python list dictionary time complexity performance O(1) O(n)"
      "what's the diff between them?"   → needs conversation context to resolve "them"

    We use a very small, focused prompt — just keyword expansion,
    not a full transformation.
    """
    system = dedent("""
        You are a search query optimizer.
        Rewrite the question as a dense keyword query for semantic search.
        Output ONLY the rewritten query — no explanation, no preamble.
        Keep it under 20 words. Focus on technical terms and concepts.
    """).strip()

    rewritten = generate(
        f"Question: {question}\nRewritten search query:",
        system=system,
        temperature=0.1,
        max_tokens=60,
    )

    # Fallback to original if rewrite fails or is too similar
    rewritten = rewritten.strip().strip('"').strip("'")
    if not rewritten or len(rewritten) < 5:
        return question
    return rewritten


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class RetrievedChunk:
    text: str
    doc_id: str
    title: str
    score: float
    chunk_idx: int


def retrieve(
    query: str, top_k: int = TOP_K, doc_filter: str | None = None
) -> list[RetrievedChunk]:
    """
    Embed the query and search Qdrant.
    Optionally filter by doc_id for scoped Q&A.
    Excludes chunks below MIN_SCORE.
    """
    query_vec = embed(query)
    q_filter = None
    if doc_filter:
        q_filter = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_filter))]
        )

    results = client.query_points(
        collection_name=COLLECTION,
        query=query_vec,
        query_filter=q_filter,
        limit=top_k,
        with_payload=True,
    )

    return [
        RetrievedChunk(
            text=r.payload["text"],
            doc_id=r.payload["doc_id"],
            title=r.payload["title"],
            score=round(r.score, 4),
            chunk_idx=r.payload.get("chunk_idx", 0),
        )
        for r in results.points
        if r.score >= MIN_SCORE
    ]


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — CONTEXT ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════


def assemble_context(
    chunks: list[RetrievedChunk], max_chars: int = MAX_CONTEXT
) -> tuple[str, list[dict]]:
    """
    Format retrieved chunks into a context block for the LLM.

    Steps:
      1. Deduplicate — remove near-identical chunks (same doc_id + adjacent index)
      2. Rank — already sorted by score from Qdrant, keep that order
      3. Truncate — stop adding chunks once we hit max_chars
      4. Format — clear document markers so the model knows source boundaries

    Returns (context_string, source_list) where source_list is used
    to populate the RAGResponse.sources field.
    """
    # Deduplicate: skip a chunk if a very similar one from the same doc is already included
    seen: set[tuple[str, int]] = set()
    deduped: list[RetrievedChunk] = []
    for chunk in chunks:
        key = (chunk.doc_id, chunk.chunk_idx)
        # Also skip if adjacent chunk from same doc already included
        adjacent_included = (chunk.doc_id, chunk.chunk_idx - 1) in seen or (
            chunk.doc_id,
            chunk.chunk_idx + 1,
        ) in seen
        if key not in seen and not adjacent_included:
            deduped.append(chunk)
            seen.add(key)

    # Build context string respecting max_chars budget
    context_parts: list[str] = []
    sources: list[dict] = []
    total_chars = 0

    for i, chunk in enumerate(deduped, 1):
        entry = (
            f"[Document {i}: {chunk.title} | relevance: {chunk.score:.2f}]\n"
            f"{chunk.text}"
        )
        if total_chars + len(entry) > max_chars:
            break
        context_parts.append(entry)
        sources.append(
            {
                "title": chunk.title,
                "doc_id": chunk.doc_id,
                "score": chunk.score,
                "text": chunk.text[:100] + "...",
            }
        )
        total_chars += len(entry)

    return "\n\n".join(context_parts), sources


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — GROUNDED GENERATION
# ══════════════════════════════════════════════════════════════════════════════

GENERATION_SYSTEM = dedent("""
    You are a precise Q&A assistant. Answer questions using ONLY the provided context.

    Rules:
    - Base every claim on the context. Do not add information from outside it.
    - If the context does not contain the answer, say exactly:
      "The provided context does not contain information about [topic]."
    - Do not speculate or extrapolate beyond what is stated.
    - Keep answers concise — 2 to 4 sentences unless the question requires more.
    - Do not mention "the context" or "document N" in your answer unless asked.
      Write naturally as if you know this information.
""").strip()


def generate_answer(question: str, context: str) -> str:
    """Generate an answer grounded in the provided context."""
    if not context.strip():
        return "No relevant documents were found for this question."

    prompt = dedent(f"""
        Context:
        {context}

        Question: {question}

        Answer:
    """).strip()

    return generate(prompt, system=GENERATION_SYSTEM, temperature=0.1, max_tokens=400)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — GROUNDEDNESS CHECKING
# ══════════════════════════════════════════════════════════════════════════════

GROUNDEDNESS_SYSTEM = dedent("""
    You are a groundedness evaluator. You assess whether an answer is supported
    by the provided context.

    Output ONLY a JSON object with these fields:
    {
      "score": <float 0.0-1.0>,
      "label": <"grounded"|"partial"|"ungrounded"|"refused">,
      "reason": <one sentence explanation>
    }

    Scoring guide:
      1.0  = every claim in the answer is directly supported by the context
      0.7  = most claims are supported; minor additions from outside context
      0.4  = some claims supported; significant content from outside context
      0.0  = answer contradicts context or is entirely from outside it
      refused = model correctly said the context doesn't contain the answer
    """).strip()


def check_groundedness(question: str, answer: str, context: str) -> GroundednessVerdict:
    """
    LLM-as-judge: ask the model to evaluate whether the answer
    is grounded in the retrieved context.

    This is the day-2 structured output pattern applied to evaluation.
    Returns a GroundednessVerdict with score, label, and reason.
    """
    # Detect explicit refusals without an LLM call
    refusal_phrases = [
        "does not contain",
        "not in the context",
        "cannot find",
        "no information",
        "not provided",
        "not mentioned",
    ]
    if any(p in answer.lower() for p in refusal_phrases):
        return GroundednessVerdict(
            score=1.0,
            label="refused",
            reason="Model correctly identified missing information.",
        )

    prompt = dedent(f"""
        Context:
        {context[:1500]}

        Question: {question}

        Answer: {answer}

        Evaluate groundedness:
    """).strip()

    raw = generate(prompt, system=GROUNDEDNESS_SYSTEM, temperature=0.0, max_tokens=150)
    cleaned = clean_json(raw)

    for attempt in range(2):
        try:
            data = json.loads(cleaned)
            return GroundednessVerdict(**data)
        except Exception:
            if attempt == 0:
                # One retry with explicit format reminder
                raw = generate(
                    prompt + "\nReturn ONLY valid JSON, no other text.",
                    system=GROUNDEDNESS_SYSTEM,
                    temperature=0.0,
                    max_tokens=150,
                )
                cleaned = clean_json(raw)

    # Fallback heuristic: check word overlap between answer and context
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())
    overlap = len(answer_words & context_words) / max(len(answer_words), 1)
    score = min(overlap * 2.5, 1.0)
    label: Literal["grounded", "partial", "ungrounded", "refused"] = (
        "grounded" if score > 0.7 else "partial" if score > 0.4 else "ungrounded"
    )
    return GroundednessVerdict(
        score=round(score, 2),
        label=label,
        reason="Scored by word overlap (groundedness check failed).",
    )


# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


def rag(question: str, verbose: bool = True) -> RAGResponse:
    """
    Run the full RAG pipeline end to end.

    Timings are tracked per stage so you can see where latency lives.
    Typically: embedding (300ms) >> generation (2s) >> groundedness (2s) >> retrieval (5ms)
    """
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    # Stage 1: query rewriting
    t = time.perf_counter()
    rewritten = rewrite_query(question)
    timings["rewrite_ms"] = round((time.perf_counter() - t) * 1000)

    if verbose:
        print(f"\n  Query    : {question}")
        if rewritten != question:
            print(f"  Rewritten: {rewritten}")

    # Stage 2: retrieval
    t = time.perf_counter()
    chunks = retrieve(rewritten, top_k=TOP_K)
    timings["retrieve_ms"] = round((time.perf_counter() - t) * 1000)

    if verbose:
        print(f"  Retrieved: {len(chunks)} chunks (min score {MIN_SCORE})")
        for c in chunks:
            print(f"    [{c.score:.3f}] {c.title} — {c.text[:60]}...")

    # Stage 3: context assembly
    t = time.perf_counter()
    context, sources = assemble_context(chunks)
    timings["assemble_ms"] = round((time.perf_counter() - t) * 1000)

    # Stage 4: grounded generation
    t = time.perf_counter()
    answer = generate_answer(question, context)
    timings["generate_ms"] = round((time.perf_counter() - t) * 1000)

    if verbose:
        print(f"\n  Answer   : {answer}")

    # Stage 5: groundedness check
    t = time.perf_counter()
    verdict = check_groundedness(question, answer, context)
    timings["groundedness_ms"] = round((time.perf_counter() - t) * 1000)

    timings["total_ms"] = round((time.perf_counter() - t0) * 1000)

    if verbose:
        label_colour = {
            "grounded": "\033[32m",
            "partial": "\033[33m",
            "ungrounded": "\033[31m",
            "refused": "\033[36m",
        }.get(verdict.label, "")
        print(
            f"  Grounded : {label_colour}{verdict.label}\033[0m "
            f"(score={verdict.score:.2f}) — {verdict.reason}"
        )
        print(
            f"  Latency  : rewrite={timings['rewrite_ms']}ms  "
            f"retrieve={timings['retrieve_ms']}ms  "
            f"generate={timings['generate_ms']}ms  "
            f"groundedness={timings['groundedness_ms']}ms"
        )

    return RAGResponse(
        question=question,
        rewritten_query=rewritten,
        answer=answer,
        groundedness=verdict.label,
        groundedness_score=verdict.score,
        sources=sources,
        retrieved_chunks=len(chunks),
        latency_ms=timings,
    )


# ── Evaluation set ────────────────────────────────────────────────────────────


@dataclass
class EvalCase:
    question: str
    expected_terms: list[str]  # should appear in answer
    should_refuse: bool = False  # True if the question is unanswerable from corpus


EVAL_SET: list[EvalCase] = [
    # Answerable questions — should retrieve and answer correctly
    EvalCase("What year was Python first released?", ["1991"]),
    EvalCase(
        "What is the time complexity of appending to a Python list?",
        ["o(1)", "amortised", "constant"],
    ),
    EvalCase(
        "How does asyncio avoid blocking the event loop for CPU-bound tasks?",
        ["executor", "run_in_executor", "multiprocessing"],
    ),
    EvalCase(
        "What is the difference between a list and a tuple in Python?",
        ["mutable", "immutable"],
    ),
    EvalCase("How does asyncio.gather work?", ["concurrent", "coroutines", "results"]),
    EvalCase(
        "What data structure does Python use for dictionaries?",
        ["hash table", "hashable"],
    ),
    EvalCase(
        "What does the collections module provide?", ["deque", "counter", "defaultdict"]
    ),
    # Should-refuse questions — answer not in corpus
    EvalCase("What is the capital of Argentina?", [], should_refuse=True),
    EvalCase("How does Rust handle memory management?", [], should_refuse=True),
    EvalCase(
        "What is the recommended way to deploy a FastAPI app to AWS Lambda?",
        [],
        should_refuse=True,
    ),
]


def run_eval() -> None:
    print(f"\n{'═' * 62}")
    print(f"  RAG Pipeline Evaluation")
    print(
        f"  {len(EVAL_SET)} questions  ({sum(1 for e in EVAL_SET if not e.should_refuse)} answerable, "
        f"{sum(1 for e in EVAL_SET if e.should_refuse)} should-refuse)"
    )
    print(f"{'═' * 62}\n")

    results = []
    for case in EVAL_SET:
        response = rag(case.question, verbose=False)

        # Score
        answer_lower = response.answer.lower()
        term_hit = (
            any(t.lower() in answer_lower for t in case.expected_terms)
            if case.expected_terms
            else True
        )
        refused = response.groundedness == "refused"
        grounded = response.groundedness in ("grounded", "refused")

        if case.should_refuse:
            correct = refused
            status = "✓" if refused else "✗ should have refused"
        else:
            correct = term_hit and grounded
            status = (
                "✓"
                if correct
                else f"✗ (terms={'✓' if term_hit else '✗'} grounded={'✓' if grounded else '✗'})"
            )

        results.append(correct)
        print(
            f"  {status:30s} [{response.groundedness:10s} {response.groundedness_score:.2f}]  {case.question[:50]}"
        )

    accuracy = sum(results) / len(results) * 100
    print(f"\n{'─' * 62}")
    print(f"  Accuracy: {accuracy:.0f}%  ({sum(results)}/{len(results)} correct)")
    print(
        f"  Grounding: {sum(1 for r in [rag(c.question, verbose=False) for c in EVAL_SET[:3]] if r.groundedness in ('grounded', 'refused')) / 3 * 100:.0f}% grounded (sample)\n"
    )


# ── Demo questions ────────────────────────────────────────────────────────────

DEMO = [
    "How does Python's memory management work?",
    "What's the fastest way to check if an element is in a collection?",
    "When should I not use asyncio?",
    "What is the capital of France?",  # should refuse — not in corpus
]

# ── REPL ──────────────────────────────────────────────────────────────────────


def repl() -> None:
    print(f"\n{'─' * 60}")
    print(f"  Day 11 — RAG Q&A")
    print(f"  Collection: {COLLECTION}  |  Model: {GEN_MODEL}")
    print(f"  Min score: {MIN_SCORE}  |  Max context: {MAX_CONTEXT} chars")
    print(f"\n  Try questions about Python — lists, async, memory, data structures.")
    print(f"  Try an off-topic question to see the refusal behaviour.")
    print(f"  Type 'eval' to run the evaluation set, 'quit' to exit.")
    print(f"{'─' * 60}\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() == "quit":
            break
        if question.lower() == "eval":
            run_eval()
            continue

        rag(question, verbose=True)
        print()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    if arg == "ask" and len(sys.argv) > 2:
        question = " ".join(sys.argv[2:])
        rag(question, verbose=True)

    elif arg == "eval":
        run_eval()

    elif arg == "demo":
        for q in DEMO:
            rag(q, verbose=True)
            print()
            try:
                input("Press Enter for next question...")
            except (KeyboardInterrupt, EOFError):
                break

    else:
        repl()


if __name__ == "__main__":
    main()

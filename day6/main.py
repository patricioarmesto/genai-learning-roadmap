"""
Day 6 — Streaming responses from agents
=========================================
Two things in one file:
  1. A streaming ReAct agent for the terminal — tokens print as they arrive,
     thoughts in dim grey, tool calls in green, answers in full brightness.
  2. A FastAPI SSE endpoint that streams the same agent over HTTP,
     so any browser or curl client can consume it token by token.

Key concepts demonstrated:
  • Ollama streaming API (stream: true, NDJSON chunks)
  • Async generators as the streaming primitive
  • Detecting Action lines mid-stream without blocking display
  • ANSI colour codes for terminal differentiation
  • Server-Sent Events (SSE) for HTTP streaming
  • StreamingResponse in FastAPI

Run modes:
  python day6_streaming.py                    # streaming CLI agent (REPL)
  python day6_streaming.py demo               # preset questions, non-interactive
  python day6_streaming.py server             # start FastAPI SSE server
  curl -N http://localhost:8000/stream?q=...  # consume from another terminal

Requirements:
    pip install requests fastapi uvicorn
    ollama pull llama3.2
    ollama serve
"""

import asyncio
import json
import math
import re
import sys
import threading
import time
from datetime import datetime, timezone
from textwrap import dedent
from typing import AsyncGenerator, Generator

import requests

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"
TEMPERATURE = 0.2
MAX_TOKENS = 400
MAX_STEPS = 8

# ── ANSI colour helpers ───────────────────────────────────────────────────────
# These only render in a real terminal (not in logs or files).
# Each function wraps text in the escape sequence for that colour.


def dim(text: str) -> str:
    return f"\033[2m{text}\033[0m"


def green(text: str) -> str:
    return f"\033[32m{text}\033[0m"


def purple(text: str) -> str:
    return f"\033[35m{text}\033[0m"


def bold(text: str) -> str:
    return f"\033[1m{text}\033[0m"


def grey(text: str) -> str:
    return f"\033[90m{text}\033[0m"


def reset() -> str:
    return "\033[0m"


# ── Tools (same compact version from day 5) ───────────────────────────────────

KNOWLEDGE_BASE = {
    "einstein birth": "Albert Einstein was born on 14 March 1879.",
    "einstein death": "Albert Einstein died on 18 April 1955, aged 76.",
    "special relativity": "Einstein published special relativity in 1905.",
    "general relativity": "Einstein published general relativity in 1915.",
    "nobel einstein": "Einstein received the Nobel Prize in Physics in 1921.",
    "python created": "Python was created by Guido van Rossum, first released in 1991.",
    "speed of light": "The speed of light is 299,792,458 metres per second.",
    "distance earth moon": "The average Earth–Moon distance is 384,400 kilometres.",
    "mount everest": "Mount Everest is 8,848.86 metres tall.",
    "population argentina": "Argentina's population is approximately 46 million (2023).",
    "buenos aires founded": "Buenos Aires was founded on 11 June 1580.",
    "turing born": "Alan Turing was born on 23 June 1912.",
    "turing died": "Alan Turing died on 7 June 1954.",
    "turing paper": "Turing published 'On Computable Numbers' in 1936.",
}


def search(query: str) -> str:
    q = query.lower().strip()
    for key, val in KNOWLEDGE_BASE.items():
        if key in q or q in key:
            return val
    words = set(q.split())
    best_score, best_val = 0, None
    for key, val in KNOWLEDGE_BASE.items():
        s = len(words & set(key.split()))
        if s > best_score:
            best_score, best_val = s, val
    return best_val if best_val and best_score >= 1 else f"No result for '{query}'."


def calculate(expression: str) -> str:
    safe = {
        "__builtins__": {},
        "math": math,
        "sqrt": math.sqrt,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "round": round,
    }
    try:
        if any(b in expression for b in ["import", "exec", "__"]):
            return "Error: not allowed"
        return str(round(float(eval(expression, safe)), 6))  # noqa: S307
    except Exception as ex:
        return f"Error: {ex}"


def get_date(_: str = "") -> str:
    now = datetime.now(timezone.utc)
    return f"Today is {now.strftime('%A, %d %B %Y')} (UTC)."


TOOLS = {"search": search, "calculate": calculate, "get_date": get_date}


# ── System prompt (v3 from day 5) ─────────────────────────────────────────────

SYSTEM_PROMPT = dedent("""
    You are a precise research analyst. You MUST use tools to verify every fact.

    Available tools:
    - search(query)   : Look up factual information
    - calculate(expr) : Evaluate math. Python syntax: 2**10, 1905-1879
    - get_date()      : Get today's date
    - finish(answer)  : Return the final answer — ONLY call this after using search or calculate

    STRICT FORMAT — every single response must follow this exactly:
    Thought: <what you need to do next>
    Action: <tool name>
    Action Input: <argument>

    MANDATORY RULES:
    - You MUST call search or calculate at least once before calling finish
    - NEVER call finish as your first action — always search first
    - NEVER write an Observation line — the system adds it
    - NEVER answer a question from memory — always verify with search

    Example of CORRECT behaviour:
    Thought: I need to find the answer using search before I can respond.
    Action: search
    Action Input: Einstein birth year
    Observation: Albert Einstein was born on 14 March 1879.

    Thought: I have the data I need. Now I can answer.
    Action: finish
    Action Input: Einstein was born in 1879.
""").strip()


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — STREAMING GENERATOR
# Yields chunks as they arrive from Ollama.
# ══════════════════════════════════════════════════════════════════════════════


def stream_llm(messages: list[dict]) -> Generator[str, None, None]:
    """
    Generator that yields token strings from Ollama as they arrive.

    Ollama streaming sends newline-delimited JSON (NDJSON):
        {"message": {"content": "Th"}, "done": false}
        {"message": {"content": "ought"}, "done": false}
        {"message": {"content": ":"}, "done": false}
        ...
        {"message": {"content": ""}, "done": true}

    We yield each non-empty content string immediately.
    The caller accumulates these into a buffer while displaying them.
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": True,  # ← the key difference from previous days
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
            "stop": ["Observation:"],
        },
    }

    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    chunk = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token

                if chunk.get("done"):
                    break

    except requests.exceptions.ConnectionError:
        print("\n[error] Cannot reach Ollama. Run: ollama serve")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — STREAMING REACT LOOP (terminal)
# ══════════════════════════════════════════════════════════════════════════════


def _strip_think_tags(text: str) -> str:
    """
    Remove <think>...</think> blocks emitted by reasoning models
    (Qwen, DeepSeek-R1, etc.) before the actual ReAct output.
    Also strips any stray </think> closing tags.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"</think>", "", text)
    return text.strip()


def _parse_from_buffer(buf: str) -> tuple[str | None, str | None]:
    """Extract (action, action_input) from the accumulated buffer."""
    action = action_input = None
    m = re.search(r"Action:\s*(\w+)", buf, re.IGNORECASE)
    if m:
        action = m.group(1).strip().lower()
    m = re.search(
        r"Action Input:\s*(.+?)(?=\nObservation:|\nThought:|\Z)",
        buf,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        action_input = m.group(1).strip().strip('"').strip("'")
    return action, action_input


# Display state machine — tracks what kind of content we're currently printing
class DisplayState:
    THOUGHT = "thought"
    ACTION = "action"
    ACTION_INPUT = "action_input"
    ANSWER = "answer"
    OTHER = "other"


def streaming_react_agent(question: str) -> str:
    """
    ReAct agent with clean terminal display.

    Approach: buffer the full LLM response for each reasoning step
    (Thought + Action + Action Input), then render it in one clean
    block before executing the tool.  Only the *final answer* is
    streamed token-by-token — that is the part where streaming actually
    improves UX, because the user is waiting for prose, not scaffolding.

    Display rules:
      Thought     → dim grey   (internal reasoning, not primary)
      ▶ tool(arg) → green      (action event)
      ← result    → purple     (observation)
      Final answer→ bold white (streamed live)
    """
    scratchpad = ""
    step = 0
    tools_used = 0
    seen_calls: set[str] = set()  # tracks (action, action_input) to detect loops

    print(f"\n{grey('─' * 58)}")
    print(f"{bold(question)}")
    print(f"{grey('─' * 58)}\n")

    while step < MAX_STEPS:
        step += 1
        user_content = f"Question: {question}\n\n{scratchpad}".strip()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        # ── Accumulate the full response for this iteration ───────────────
        # We do NOT try to render mid-stream for reasoning steps.
        # The ReAct format (Thought / Action / Action Input) is only
        # complete once the stop sequence fires — partial buffers give
        # us half-lines and mis-ordered output.
        buffer = ""
        for token in stream_llm(messages):
            buffer += token

        # Strip <think> blocks before parsing — reasoning models emit these
        buffer = _strip_think_tags(buffer)

        # ── Parse ─────────────────────────────────────────────────────────
        action, action_input = _parse_from_buffer(buffer)

        # ── Extract and print thought ─────────────────────────────────────
        m = re.search(
            r"Thought:\s*(.+?)(?=\nAction:|\Z)", buffer, re.DOTALL | re.IGNORECASE
        )
        if m:
            thought_text = m.group(1).strip()
            print(dim(f"Thought: {thought_text}"))

        # ── finish → stream the answer live ──────────────────────────────
        if action == "finish":
            # Block premature finish — model must have used at least one tool
            if tools_used == 0:
                observation = (
                    "You must use search or calculate before calling finish. "
                    "Search for the information needed to answer the question first."
                )
                print(purple(f"  [blocked finish — no tools used yet]\n"))
            else:
                final = action_input or buffer
                print(f"\n{grey('─' * 58)}")
                for char in final:
                    print(bold(char), end="", flush=True)
                    time.sleep(0.012)
                print(reset() + "\n")
                return final

        # ── tool call → print label, run tool, print result ───────────────
        if not action or not action_input:
            observation = (
                "Parse error: your response must follow this format exactly:\n"
                "Thought: <reasoning>\nAction: <tool>\nAction Input: <argument>"
            )
            print(purple(f"  [parse error]\n"))
        elif action != "finish":
            # Duplicate detection — same (tool, input) seen before → break the loop
            call_key = f"{action}:{action_input.lower().strip()}"
            if call_key in seen_calls:
                observation = (
                    f"You already called {action}({action_input!r}) and got a result. "
                    "Do not repeat the same search. Use the information already retrieved, "
                    "search for something DIFFERENT, or call finish with your current answer."
                )
                print(purple(f"  [duplicate call blocked — forcing progress]\n"))
            else:
                seen_calls.add(call_key)
                tools_used += 1
                print(green(f"▶ {action}({action_input})"))
                fn = TOOLS.get(action)
                observation = (
                    fn(action_input)
                    if fn
                    else f"Unknown tool '{action}'. Available: {list(TOOLS)}"
                )
                print(purple(f"  ← {observation}\n"))
        else:
            pass

        new_chunk = buffer.rstrip() + f"\nObservation: {observation}\n\n"
        scratchpad = (scratchpad + new_chunk).strip() + "\n\n"

    # Max steps reached — pull the answer from the last thought if possible
    last_thought = ""
    for m in re.finditer(
        r"Thought:\s*(.+?)(?=\nAction:|\Z)", scratchpad, re.DOTALL | re.IGNORECASE
    ):
        last_thought = m.group(1).strip()
    fallback = last_thought or "[max steps reached without answer]"
    print(f"\n{grey('─' * 58)}")
    print(bold(fallback) + reset() + "\n")
    return fallback


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — FASTAPI SSE ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════


def build_sse_app():
    """
    Build and return the FastAPI app.
    Imported lazily so the file works without fastapi installed
    when only running the CLI.
    """
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse
    except ImportError:
        print("[error] FastAPI not installed. Run: pip install fastapi uvicorn")
        sys.exit(1)

    app = FastAPI(title="Day 6 — Streaming Agent", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    # SSE event types — the client uses these to route different content
    # to different UI elements (e.g. thoughts in a sidebar, answer in main).
    EVT_THOUGHT = "thought"
    EVT_TOOL_CALL = "tool_call"
    EVT_TOOL_RESULT = "tool_result"
    EVT_TOKEN = "token"  # final answer tokens
    EVT_DONE = "done"
    EVT_ERROR = "error"

    def sse_event(event_type: str, data: dict) -> str:
        """Format a Server-Sent Event string."""
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    def agent_sse_stream(question: str) -> Generator[str, None, None]:
        """
        Generator that yields SSE events for the entire agent run.

        Event flow per iteration:
          thought     → the model's reasoning text (streaming tokens)
          tool_call   → which tool is being called with what argument
          tool_result → what the tool returned
          [repeat]
          token       → final answer tokens (streaming)
          done        → signals the client the stream is complete
        """
        scratchpad = ""
        step = 0

        while step < MAX_STEPS:
            step += 1
            user_content = f"Question: {question}\n\n{scratchpad}".strip()
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]

            buffer = ""
            thought_sent = False
            thought_text = ""

            # Stream tokens from Ollama
            for token in stream_llm(messages):
                buffer += token

                # Detect thought section — emit tokens as they arrive
                if "Thought:" in buffer and not thought_sent:
                    # Extract just the thought content (after "Thought: ")
                    after = re.sub(
                        r".*?Thought:\s*", "", buffer, flags=re.DOTALL | re.IGNORECASE
                    )
                    # Only stream thought tokens (before Action:)
                    if "\nAction:" not in after:
                        # New thought token arrived
                        new_part = after[len(thought_text) :]
                        if new_part:
                            thought_text += new_part
                            yield sse_event(
                                EVT_THOUGHT, {"token": new_part, "step": step}
                            )
                    else:
                        thought_sent = True

            # Full response buffered — parse action
            action, action_input = _parse_from_buffer(buffer)

            # Extract complete thought for logging
            m = re.search(
                r"Thought:\s*(.+?)(?=\nAction:|\Z)", buffer, re.DOTALL | re.IGNORECASE
            )
            if m and not thought_sent:
                yield sse_event(
                    EVT_THOUGHT, {"token": m.group(1).strip(), "step": step}
                )

            if action == "finish":
                final_answer = action_input or buffer
                # Stream the final answer token by token
                for char in final_answer:
                    yield sse_event(EVT_TOKEN, {"token": char})
                yield sse_event(EVT_DONE, {"steps": step, "answer": final_answer})
                return

            if not action or not action_input:
                observation = "Parse error: use Thought/Action/Action Input format."
            else:
                # Emit tool call event before executing
                yield sse_event(
                    EVT_TOOL_CALL,
                    {
                        "tool": action,
                        "input": action_input,
                        "step": step,
                    },
                )

                fn = TOOLS.get(action)
                observation = fn(action_input) if fn else f"Unknown tool '{action}'"

                yield sse_event(
                    EVT_TOOL_RESULT,
                    {
                        "tool": action,
                        "result": observation,
                        "step": step,
                    },
                )

            new_chunk = buffer.rstrip() + f"\nObservation: {observation}\n\n"
            scratchpad = (scratchpad + new_chunk).strip() + "\n\n"

        yield sse_event(
            EVT_ERROR, {"message": "Max steps reached without a final answer."}
        )

    @app.get("/stream")
    def stream_endpoint(q: str = "What year was Python created?"):
        """
        Stream an agent run as Server-Sent Events.

        Usage:
          curl -N "http://localhost:8000/stream?q=How+old+was+Einstein+when+he+died"

        Each SSE event has a type (thought/tool_call/tool_result/token/done/error)
        and a JSON data payload. The client can use event type to route content
        to different parts of the UI.
        """
        return StreamingResponse(
            agent_sse_stream(q),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # disable nginx buffering
                "Access-Control-Allow-Origin": "*",
            },
        )

    @app.get("/health")
    def health():
        return {"status": "ok", "model": MODEL}

    return app


# ── Demo questions ────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "How old was Einstein when he published special relativity?",
    "How many years ago was Buenos Aires founded?",
    "If the Moon is 384400 km away and light travels at 299792 km/s, how many seconds to reach it?",
]


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    if arg == "server":
        try:
            import uvicorn
        except ImportError:
            print("[error] uvicorn not installed. Run: pip install uvicorn")
            sys.exit(1)

        app = build_sse_app()
        print(f"\nStreaming agent server starting...")
        print(f"Test with:")
        print(
            f'  curl -N "http://localhost:8000/stream?q=How+old+was+Turing+when+he+died"\n'
        )
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

    elif arg == "demo":
        for q in DEMO_QUESTIONS:
            streaming_react_agent(q)
            try:
                input(grey("\nPress Enter for next question..."))
            except (KeyboardInterrupt, EOFError):
                break

    else:
        # Interactive REPL
        print(f"\n{grey('─' * 58)}")
        print(f"  Day 6 — Streaming ReAct Agent")
        print(f"  Model: {MODEL}")
        print(f"  Run with 'server' arg to start the SSE endpoint")
        print(f"  Run with 'demo' to see preset questions")
        print(f"{grey('─' * 58)}\n")

        while True:
            try:
                question = input("Question: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye!")
                break
            if not question or question.lower() == "quit":
                break
            streaming_react_agent(question)


if __name__ == "__main__":
    main()

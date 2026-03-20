"""
Week 1 Capstone — Research CLI
================================
A CLI research assistant that:
  1. Takes a question as a CLI argument
  2. Runs a streaming ReAct agent (days 4-6) — thoughts/tools/answers in colour
  3. Extracts a structured report using Pydantic (day 2)
  4. Saves report.json to disk

Every concept from week 1 is present:
  Day 1 — LLM API, message format, temperature
  Day 2 — Structured output, Pydantic, retry-on-error
  Day 3 — Tool calling, tool registry
  Day 4 — ReAct loop, scratchpad, finish action
  Day 5 — Production system prompt, negative examples, few-shot
  Day 6 — Streaming display, ANSI colours, think-tag stripping

Usage:
  python week1_capstone.py "How old was Einstein when he published special relativity?"
  python week1_capstone.py "How far is the Moon in light-seconds?" --out moon.json
  python week1_capstone.py "When was Buenos Aires founded?" --no-stream

Requirements:
  uv add requests pydantic
  ollama pull llama3.2   (or mistral / llama3.1:8b for better compliance)
  ollama serve
"""

import argparse
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from textwrap import dedent
from typing import Literal

import requests
from pydantic import BaseModel, Field, ValidationError, field_validator

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:397b-cloud"
TEMPERATURE = 0.2
MAX_TOKENS = 400
MAX_STEPS = 8


# ── ANSI colours ──────────────────────────────────────────────────────────────


def dim(t):
    return f"\033[2m{t}\033[0m"


def green(t):
    return f"\033[32m{t}\033[0m"


def purple(t):
    return f"\033[35m{t}\033[0m"


def bold(t):
    return f"\033[1m{t}\033[0m"


def grey(t):
    return f"\033[90m{t}\033[0m"


def reset():
    return "\033[0m"


# ── Knowledge base ────────────────────────────────────────────────────────────

KNOWLEDGE_BASE = {
    "einstein birth": "Albert Einstein was born on 14 March 1879.",
    "einstein death": "Albert Einstein died on 18 April 1955, aged 76.",
    "special relativity": "Einstein published special relativity in 1905.",
    "general relativity": "Einstein published general relativity in 1915.",
    "nobel einstein": "Einstein received the Nobel Prize in Physics in 1921.",
    "python created": "Python was created by Guido van Rossum, first released in 1991.",
    "speed of light": "The speed of light is 299,792,458 metres per second (≈ 300,000 km/s).",
    "distance earth moon": "The average Earth–Moon distance is 384,400 kilometres.",
    "distance earth sun": "The average Earth–Sun distance is 149.6 million kilometres.",
    "mount everest": "Mount Everest is 8,848.86 metres tall.",
    "population argentina": "Argentina's population is approximately 46 million (2023).",
    "buenos aires founded": "Buenos Aires was founded on 11 June 1580 by Juan de Garay.",
    "turing born": "Alan Turing was born on 23 June 1912.",
    "turing died": "Alan Turing died on 7 June 1954.",
    "turing paper": "Turing published 'On Computable Numbers' in 1936.",
    "water boiling": "Water boils at 100°C at standard atmospheric pressure.",
    "sound speed": "The speed of sound in air at 20°C is approximately 343 metres per second.",
    "pi value": "Pi (π) is approximately 3.14159265358979.",
    "avogadro": "Avogadro's number is approximately 6.022 × 10²³ mol⁻¹.",
    "planck constant": "Planck's constant is approximately 6.626 × 10⁻³⁴ joule-seconds.",
}

# ── Tools ─────────────────────────────────────────────────────────────────────


def search(query: str) -> str:
    q = query.lower().strip()
    for key, val in KNOWLEDGE_BASE.items():
        if key in q or q in key:
            return val
    words = set(q.split())
    best_score, best_val = 0, None
    for key, val in KNOWLEDGE_BASE.items():
        score = len(words & set(key.split()))
        if score > best_score:
            best_score, best_val = score, val
    return best_val if best_val and best_score >= 1 else f"No result for '{query}'."


def calculate(expression: str) -> str:
    safe = {
        "__builtins__": {},
        "math": math,
        "sqrt": math.sqrt,
        "log": math.log,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "round": round,
    }
    try:
        if any(b in expression for b in ["import", "exec", "__"]):
            return "Error: not allowed"
        result = eval(expression, safe)  # noqa: S307
        return str(round(float(result), 8))
    except Exception as ex:
        return f"Error: {ex}"


def get_date(_: str = "") -> str:
    now = datetime.now(timezone.utc)
    return f"Today is {now.strftime('%A, %d %B %Y')} (UTC)."


TOOLS = {"search": search, "calculate": calculate, "get_date": get_date}

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = dedent("""
    You are a precise research analyst. You MUST use tools to verify every fact.

    Available tools:
    - search(query)   : Look up factual information in the knowledge base
    - calculate(expr) : Evaluate math using Python syntax: 2**10, 1905-1879, sqrt(144)
    - get_date()      : Get the current date
    - finish(answer)  : Return your final answer — ONLY after using search or calculate

    STRICT FORMAT — every response must follow exactly:
    Thought: <what you need to do and why>
    Action: <one tool name>
    Action Input: <argument — then STOP, write nothing else>

    RULES:
    - NEVER call finish as your first action — always search or calculate first
    - NEVER write "Observation:" — the system adds it
    - NEVER answer from memory — always verify with search
    - NEVER repeat the same search twice — if a search returns no result, try different wording once, then move on
    - Your first Thought must start with "PLAN —" and list what you need to find

    EXAMPLE:
    Thought: PLAN — I need: (1) Turing's birth year [search], (2) year of 1936 paper [given], (3) age [calculate].
    Action: search
    Action Input: Turing born
    Observation: Alan Turing was born on 23 June 1912.

    Thought: Born 1912, paper 1936. Will verify with calculate.
    Action: calculate
    Action Input: 1936 - 1912
    Observation: 24

    Thought: Confirmed 24. I have a complete verified answer.
    Action: finish
    Action Input: Alan Turing was 24 years old when he published his 1936 paper.

    BAD — never do these:
    Action: finish   ← as first action, before any search
    Observation: ... ← never write this yourself
    [repeating same search 3 times]
""").strip()

# ── Pydantic report schema (day 2) ────────────────────────────────────────────


class Source(BaseModel):
    tool: str = Field(description="Tool name: search, calculate, or get_date")
    query: str = Field(description="Argument passed to the tool")
    result: str = Field(description="What the tool returned")


class Report(BaseModel):
    question: str
    answer: str = Field(min_length=10)
    confidence: Literal["high", "medium", "low"]
    sources: list[Source]
    steps_taken: int = Field(ge=1)
    reasoning: str = Field(
        min_length=20,
        description="One paragraph synthesising how the answer was reached",
    )
    generated_at: str

    @field_validator("generated_at")
    @classmethod
    def must_be_iso(cls, v: str) -> str:
        # Accept any non-empty string — we generate it ourselves
        if not v:
            raise ValueError("generated_at must not be empty")
        return v


# ── LLM helpers ───────────────────────────────────────────────────────────────


def _call_llm(
    messages: list[dict],
    stream: bool = False,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
) -> str:
    """Single blocking or streaming call — streaming version accumulates full text."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "stop": ["Observation:"],
        },
    }
    try:
        if not stream:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        else:
            # Stream but return full accumulated text (for reasoning steps)
            full = ""
            with requests.post(
                OLLAMA_URL, json=payload, stream=True, timeout=120
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    full += chunk.get("message", {}).get("content", "")
                    if chunk.get("done"):
                        break
            return full.strip()
    except requests.exceptions.ConnectionError:
        print("\n[error] Cannot reach Ollama. Is it running? Run: ollama serve")
        sys.exit(1)


def _stream_answer_to_terminal(text: str) -> None:
    """Print the final answer character by character with a typewriter effect."""
    print(f"\n{grey('─' * 60)}")
    for char in text:
        print(bold(char), end="", flush=True)
        time.sleep(0.01)
    print(reset() + "\n")


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from reasoning models (Qwen, DeepSeek)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return re.sub(r"</think>", "", text).strip()


def _parse_action(buf: str) -> tuple[str | None, str | None]:
    """Extract (action, action_input) from a ReAct-formatted buffer."""
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


# ── ReAct agent loop ──────────────────────────────────────────────────────────


class AgentResult:
    """Collects everything that happened during the run for report generation."""

    def __init__(self, question: str):
        self.question = question
        self.answer = ""
        self.steps = 0
        self.sources: list[dict] = []
        self.scratchpad = ""
        self.had_errors = False
        self.hit_max_steps = False


def run_agent(question: str, use_stream: bool = True) -> AgentResult:
    """
    Full ReAct loop. Returns an AgentResult with everything needed
    to build the structured report.
    """
    result = AgentResult(question)
    seen_calls: set[str] = set()
    tools_used = 0

    print(f"\n{grey('═' * 60)}")
    print(f"  {bold(question)}")
    print(f"{grey('═' * 60)}\n")

    while result.steps < MAX_STEPS:
        result.steps += 1
        user_content = f"Question: {question}\n\n{result.scratchpad}".strip()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        # Get LLM response (streaming or blocking per flag)
        raw = _call_llm(messages, stream=use_stream)
        buffer = _strip_think_tags(raw)

        # Extract and display thought
        m = re.search(
            r"Thought:\s*(.+?)(?=\nAction:|\Z)", buffer, re.DOTALL | re.IGNORECASE
        )
        if m:
            print(dim(f"Thought: {m.group(1).strip()}"))

        action, action_input = _parse_action(buffer)

        # ── finish ────────────────────────────────────────────────────────
        if action == "finish":
            if tools_used == 0:
                observation = (
                    "You must use search or calculate before calling finish. "
                    "Search for the key facts first."
                )
                result.had_errors = True
                print(purple("  [blocked — no tools used yet]\n"))
            else:
                result.answer = action_input or buffer
                _stream_answer_to_terminal(result.answer)
                return result

        # ── parse error ───────────────────────────────────────────────────
        elif not action or not action_input:
            observation = (
                "Parse error: follow this format exactly — "
                "Thought: / Action: / Action Input:"
            )
            result.had_errors = True
            print(purple("  [parse error]\n"))

        # ── real tool call ────────────────────────────────────────────────
        else:
            call_key = f"{action}:{action_input.lower().strip()}"
            if call_key in seen_calls:
                observation = (
                    f"Already searched for that. Use the information already retrieved, "
                    f"try a DIFFERENT search term, or call finish with your current answer."
                )
                result.had_errors = True
                print(purple("  [duplicate — forcing progress]\n"))
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

                result.sources.append(
                    {
                        "tool": action,
                        "query": action_input,
                        "result": observation,
                    }
                )

        # Append to scratchpad
        new_chunk = buffer.rstrip() + f"\nObservation: {observation}\n\n"
        result.scratchpad = (result.scratchpad + new_chunk).strip() + "\n\n"

    # Max steps fallback
    result.hit_max_steps = True
    # Pull answer from last thought if available
    for m in re.finditer(
        r"Thought:\s*(.+?)(?=\nAction:|\Z)",
        result.scratchpad,
        re.DOTALL | re.IGNORECASE,
    ):
        result.answer = m.group(1).strip()
    if not result.answer:
        result.answer = "Could not determine a complete answer within the step limit."
    print(grey(f"\n[max steps reached — using last thought as answer]"))
    _stream_answer_to_terminal(result.answer)
    return result


# ── Report generation (day 2 pattern) ────────────────────────────────────────

REPORT_SYSTEM = dedent("""
    You are a report generator. Extract information from the research session
    and return ONLY a valid JSON object matching the schema exactly.
    No markdown, no code fences, no explanation — just the JSON object.

    Schema:
    {
      "question":     string — the original question,
      "answer":       string — the final answer (at least 10 chars),
      "confidence":   "high" | "medium" | "low",
      "sources":      array of {tool, query, result},
      "steps_taken":  integer,
      "reasoning":    string — one paragraph explaining how the answer was reached,
      "generated_at": string — ISO 8601 timestamp
    }

    Confidence rules:
      "high"   — every fact was verified by a tool call
      "medium" — some facts were inferred or one tool returned no result
      "low"    — agent hit max steps, had repeated parse errors, or answer is uncertain

    Use null for no value, false/true (not Python False/True), numbers not strings.
""").strip()


def _clean_json(raw: str) -> str:
    """Strip fences, extract first {...} block, fix Python booleans."""
    text = re.sub(r"```(?:json)?", "", raw).strip()
    text = re.sub(r"```", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start : end + 1]
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def generate_report(agent_result: AgentResult) -> Report:
    """
    Day 2 pattern: send the full research session to the LLM and ask it
    to extract a structured Report. Retry up to 3 times on validation failure.
    """
    session_summary = dedent(f"""
        Question: {agent_result.question}
        Final answer: {agent_result.answer}
        Steps taken: {agent_result.steps}
        Had errors: {agent_result.had_errors}
        Hit max steps: {agent_result.hit_max_steps}
        Current timestamp: {datetime.now(timezone.utc).isoformat()}

        Sources used:
        {json.dumps(agent_result.sources, indent=2)}

        Full scratchpad (agent's reasoning):
        {agent_result.scratchpad[:2000]}
    """).strip()

    messages = [
        {"role": "system", "content": REPORT_SYSTEM},
        {
            "role": "user",
            "content": f"Generate the report for this research session:\n\n{session_summary}",
        },
    ]

    last_error = None
    for attempt in range(1, 4):
        raw = _call_llm(messages, stream=False, temperature=0.1, max_tokens=800)
        cleaned = _clean_json(raw)

        try:
            data = json.loads(cleaned)
            # Ensure required fields the LLM might omit
            data.setdefault("question", agent_result.question)
            data.setdefault("steps_taken", agent_result.steps)
            data.setdefault("sources", agent_result.sources)
            data.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
            if not data.get("answer"):
                data["answer"] = agent_result.answer

            report = Report(**data)
            return report

        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            errors = str(e)[:300]
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": f"Validation failed: {errors}. Fix and return ONLY the corrected JSON.",
                }
            )

    # Final fallback — build the report manually from what we have
    print(grey(f"\n[report generation failed after 3 attempts: {last_error}]"))
    print(grey("[building report from agent data directly]"))
    return Report(
        question=agent_result.question,
        answer=agent_result.answer,
        confidence="low",
        sources=[Source(**s) for s in agent_result.sources],
        steps_taken=agent_result.steps,
        reasoning=(
            f"The agent ran {agent_result.steps} steps. "
            f"{'Errors occurred during execution. ' if agent_result.had_errors else ''}"
            f"Final answer was extracted from the research session."
        ),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ── Output ────────────────────────────────────────────────────────────────────


def save_and_print_report(report: Report, output_path: str) -> None:
    """Save JSON to disk and print a human-readable summary."""
    data = report.model_dump()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    confidence_colour = {"high": "\033[32m", "medium": "\033[33m", "low": "\033[31m"}
    cc = confidence_colour.get(report.confidence, "")

    print(f"{grey('─' * 60)}")
    print(f"  {bold('Report')}")
    print(f"{grey('─' * 60)}")
    print(f"  Question    : {report.question}")
    print(f"  Answer      : {report.answer}")
    print(f"  Confidence  : {cc}{report.confidence}\033[0m")
    print(f"  Steps taken : {report.steps_taken}")
    print(f"  Sources     : {len(report.sources)}")
    for s in report.sources:
        print(f"    • {s.tool}({s.query[:50]}) → {s.result[:60]}")
    print(f"  Reasoning   : {report.reasoning[:120]}...")
    print(f"  Saved to    : {output_path}")
    print(f"{grey('─' * 60)}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research CLI — ReAct agent with structured JSON report output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""
            Examples:
              python week1_capstone.py "How old was Einstein when he published special relativity?"
              python week1_capstone.py "How far is the Moon in light-seconds?" --out moon.json
              python week1_capstone.py "When was Buenos Aires founded?" --no-stream
              python week1_capstone.py --interactive
        """),
    )
    parser.add_argument("question", nargs="?", help="Question to research")
    parser.add_argument(
        "--out",
        default="report.json",
        help="Output path for the JSON report (default: report.json)",
    )
    parser.add_argument(
        "--no-stream", action="store_true", help="Disable streaming (blocking mode)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Ask questions interactively in a loop",
    )
    return parser.parse_args()


def run_one(question: str, output_path: str, stream: bool) -> None:
    """Research one question end to end."""
    # Step 1: ReAct agent
    agent_result = run_agent(question, use_stream=stream)

    # Step 2: structured report extraction (day 2 pattern)
    print(grey("\nExtracting structured report..."))
    report = generate_report(agent_result)

    # Step 3: save and display
    save_and_print_report(report, output_path)


def main() -> None:
    args = parse_args()
    stream = not args.no_stream

    if args.interactive:
        print(f"\n{bold('Research CLI')} — interactive mode")
        print(grey("Type a question, or 'quit' to exit.\n"))
        idx = 0
        while True:
            try:
                question = input("Question: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye!")
                break
            if not question or question.lower() == "quit":
                break
            idx += 1
            out = args.out if idx == 1 else args.out.replace(".json", f"_{idx}.json")
            run_one(question, out, stream)

    elif args.question:
        run_one(args.question, args.out, stream)

    else:
        # No question and no --interactive: run the demo set
        demo_questions = [
            (
                "How old was Einstein when he published special relativity?",
                "report_einstein.json",
            ),
            ("How many light-seconds away is the Moon?", "report_moon.json"),
            ("How many years ago was Buenos Aires founded?", "report_bsas.json"),
        ]
        for question, out in demo_questions:
            run_one(question, out, stream)
            try:
                input(grey("\nPress Enter for next question..."))
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()

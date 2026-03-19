"""
Day 5 — Prompt engineering for agents
=======================================
The day-4 ReAct agent rebuilt with three progressively better system prompts.
An eval harness scores each version on the same 10 questions so you can
measure what each prompt technique actually buys you.

Key concepts demonstrated:
  • Persona — shapes the model's prior on appropriate responses
  • Format contract — makes output structure non-negotiable
  • Few-shot examples — calibrates reasoning style and length
  • Chain-of-thought forcing — mandatory planning before first action
  • Negative examples — suppresses the most common failure modes
  • Measuring improvement — exact match + keyword scoring

Run modes:
  python day5_prompts.py              # compare all 3 prompts on eval set
  python day5_prompts.py v1           # run only prompt v1 interactively
  python day5_prompts.py v2           # run only prompt v2 interactively
  python day5_prompts.py v3           # run only prompt v3 (production quality)
  python day5_prompts.py diff         # print all 3 prompts side by side

Requirements:
    pip install requests
    ollama pull llama3.2
    ollama serve
"""

import math
import re
import sys
import time
from dataclasses import dataclass, field
from textwrap import dedent

import requests

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:397b-cloud"
MAX_STEPS = 8
MAX_TOKENS = 400


# ── Tools (same as day 4) ─────────────────────────────────────────────────────

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
    "water boiling": "Water boils at 100°C at standard pressure.",
    "pi value": "Pi (π) is approximately 3.14159265358979.",
}


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
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    return f"Today is {now.strftime('%A, %d %B %Y')} (UTC)."


TOOLS = {"search": search, "calculate": calculate, "get_date": get_date}


# ══════════════════════════════════════════════════════════════════════════════
# THE THREE SYSTEM PROMPTS
# ══════════════════════════════════════════════════════════════════════════════


def prompt_v1() -> str:
    """
    V1 — Bare minimum. Just the tool list and a vague format hint.
    What you'd write in 2 minutes without thinking about it.
    This is the baseline most tutorials stop at.
    """
    return dedent("""
        You are a helpful AI assistant with access to these tools:
        - search(query): search for information
        - calculate(expression): do math
        - get_date: get today's date
        - finish(answer): give the final answer

        Use the format:
        Thought: your thinking
        Action: tool name
        Action Input: the input
    """).strip()


def prompt_v2() -> str:
    """
    V2 — Adds persona + strict format contract + basic negative example.
    This is what most production agents look like after one debugging session.
    """
    return dedent("""
        You are a precise research agent. You only state facts verified by tools.
        You never guess numeric values — you always use calculate to confirm math.

        Available tools:
        - search(query)      : Look up factual information in the knowledge base
        - calculate(expr)    : Evaluate math using Python syntax: 2**10, 1905-1879
        - get_date(query)    : Get the current date
        - finish(answer)     : Return the final answer — use this when done

        STRICT FORMAT — your response must contain exactly these lines:
        Thought: <one sentence: what you need to do and why>
        Action: <one tool name — nothing else on this line>
        Action Input: <the argument — then STOP, do not write anything else>

        NEVER write "Observation:" — that line is added by the system.
        NEVER answer a factual question from memory — always search first.
        When you have a complete verified answer, use Action: finish.
    """).strip()


def prompt_v3() -> str:
    """
    V3 — Full production quality.
    Adds: forced planning thought, few-shot example, negative examples,
    self-correction instruction, and explicit uncertainty handling.

    This is what you'd ship after a week of prompt iteration.
    """
    example = dedent("""
        Question: How old was Turing when he published his 1936 paper?

        Thought: PLAN — I need: (1) Turing's birth year [search], (2) the year of his 1936 paper [given in question], (3) the age [calculate]. I'll search for his birth year first.
        Action: search
        Action Input: Turing born
        Observation: Alan Turing was born on 23 June 1912.

        Thought: Born 1912, paper 1936. I'll verify the subtraction with calculate.
        Action: calculate
        Action Input: 1936 - 1912
        Observation: 24

        Thought: Confirmed — Turing was 24. I have a complete, sourced answer.
        Action: finish
        Action Input: Alan Turing was 24 years old when he published his seminal 1936 paper.
    """).strip()

    bad_examples = dedent("""
        NEVER do these — they cause wrong answers:

        ✗ Writing your own Observation (fabrication):
          Action Input: world population
          Observation: 8 billion  ← NEVER write this line — you are not the system

        ✗ Answering from memory without searching:
          Thought: I know Einstein was born in 1879...
          Action: finish  ← WRONG — you must search before stating facts

        ✗ Repeating the same search:
          [searching "Einstein age" three times with minor wording changes]
          ← if search returns no result, try ONE different phrasing then move on

        ✗ Putting the answer in Thought instead of finish:
          Thought: The answer is 26 years old.
          Action: search  ← WRONG — if you have the answer, use finish
    """).strip()

    return dedent(f"""
        You are a meticulous research analyst. Your defining trait is accuracy:
        you verify every fact with tools before stating it, you use calculate
        for all arithmetic, and you explicitly flag when information is missing.

        Available tools:
        - search(query)      : Look up factual information. Try specific terms first.
        - calculate(expr)    : Evaluate math. Python syntax: 1955-1879, sqrt(144), 15*0.01
        - get_date(query)    : Get today's date when needed for calculations.
        - finish(answer)     : Output the final answer. Use complete sentences.

        STRICT FORMAT:
        Thought: <your reasoning — start with PLAN on the first thought>
        Action: <tool name only>
        Action Input: <argument string — then STOP>

        The word "Observation:" is written by the system after your Action Input.
        Never write it yourself.

        MANDATORY FIRST THOUGHT — always begin with:
        "Thought: PLAN — I need: (1) X [tool], (2) Y [tool], ..."
        This forces you to decompose the problem before acting.

        SELF-CORRECTION — if a search returns no result:
        1. Try one rephrasing
        2. If still no result, state what you could not verify in your finish answer

        EXAMPLE OF CORRECT BEHAVIOUR:
        {example}

        {bad_examples}
    """).strip()


PROMPTS = {"v1": prompt_v1, "v2": prompt_v2, "v3": prompt_v3}


# ── ReAct loop (same mechanics as day 4) ─────────────────────────────────────


def parse_react(text: str):
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()
    thought = action = action_input = None
    m = re.search(r"Thought:\s*(.+?)(?=\nAction:|\Z)", text, re.DOTALL | re.IGNORECASE)
    if m:
        thought = m.group(1).strip()
    m = re.search(r"Action:\s*(\w+)", text, re.IGNORECASE)
    if m:
        action = m.group(1).strip().lower()
    m = re.search(
        r"Action Input:\s*(.+?)(?=\nObservation:|\nThought:|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        action_input = m.group(1).strip().strip('"').strip("'")
    return thought, action, action_input


def run_agent(
    question: str,
    system_prompt: str,
    verbose: bool = True,
) -> dict:
    """
    Run the ReAct loop with a given system prompt.
    Returns a dict with answer, steps taken, and thoughts recorded.
    """
    scratchpad = ""
    steps = 0
    thoughts: list[str] = []
    start = time.time()

    while steps < MAX_STEPS:
        steps += 1
        user_content = f"Question: {question}\n\n{scratchpad}".strip()

        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": MAX_TOKENS,
                        "stop": ["Observation:"],
                    },
                },
                timeout=120,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            print("[error] Cannot reach Ollama. Run: ollama serve")
            sys.exit(1)

        raw = resp.json()["message"]["content"].strip()
        thought, action, action_input = parse_react(raw)

        if thought:
            thoughts.append(thought)

        if verbose:
            print(f"\n  Step {steps}")
            if thought:
                print(
                    f"  Thought:      {thought[:120]}{'...' if len(thought) > 120 else ''}"
                )
            if action:
                print(f"  Action:       {action}")
            if action_input:
                print(f"  Action Input: {action_input}")

        if action == "finish":
            answer = action_input or thought or raw
            if verbose:
                print(f"\n  Answer: {answer}")
            return {
                "answer": answer,
                "steps": steps,
                "thoughts": thoughts,
                "elapsed": round(time.time() - start, 1),
            }

        if not action or not action_input:
            observation = (
                "Parse error: use the exact format — Thought/Action/Action Input"
            )
        else:
            fn = TOOLS.get(action)
            observation = fn(action_input) if fn else f"Unknown tool '{action}'"

        if verbose:
            print(f"  Observation:  {observation}")

        new_chunk = raw.rstrip() + f"\nObservation: {observation}\n\n"
        scratchpad = (scratchpad + new_chunk).strip() + "\n\n"

    return {
        "answer": "[max steps reached]",
        "steps": steps,
        "thoughts": thoughts,
        "elapsed": round(time.time() - start, 1),
    }


# ── Eval harness ──────────────────────────────────────────────────────────────


@dataclass
class EvalCase:
    question: str
    expected_keywords: list[str]  # any of these in the answer = pass
    expected_exact: str | None = None  # exact substring match (optional)


EVAL_SET: list[EvalCase] = [
    # Single-step factual
    EvalCase("What year was Python first released?", ["1991"], "1991"),
    EvalCase(
        "What is the boiling point of water?", ["100", "100°c", "100 degrees"], "100"
    ),
    # Two-step: search + calculate
    EvalCase(
        "How old was Einstein when he published special relativity?", ["26"], "26"
    ),
    EvalCase(
        "How old was Alan Turing when he died?", ["41", "42"]
    ),  # 41 years old (born June 1912, died June 1954)
    # Three-step: two searches + calculate
    EvalCase(
        "How many years after special relativity did Einstein receive the Nobel Prize?",
        ["16"],
        "16",
    ),
    # Requires today's date (accept any reasonable current-year answer)
    EvalCase(
        "How many years ago was Buenos Aires founded?", ["444", "445", "443", "446"]
    ),  # ~1580 to ~2024-2025
    # Unit / multi-hop
    EvalCase(
        "If the Moon is 384400 km away and light travels at 299792 km/s, how many seconds does light take to reach the Moon?",
        ["1.28", "1.3", "1.27"],
    ),
    # Math only
    EvalCase(
        "What is the area of a circle with radius 7? Use pi = 3.14159.",
        ["153.9", "153.94", "154"],
    ),
    # Requires two searches
    EvalCase(
        "How many years passed between Turing's birth and his 1936 paper?", ["24"], "24"
    ),
    # Self-correction test: first search likely misses, needs rephrasing
    EvalCase("Who created Python and in what year?", ["van rossum", "1991"], "1991"),
]


def score_answer(answer: str, case: EvalCase) -> tuple[bool, str]:
    """Return (passed, reason)."""
    a = answer.lower()

    # Exact substring match
    if case.expected_exact and case.expected_exact.lower() in a:
        return True, f"exact match '{case.expected_exact}'"

    # Keyword match (any keyword)
    for kw in case.expected_keywords:
        if kw.lower() in a:
            return True, f"keyword '{kw}'"

    return False, f"none of {case.expected_keywords} found in answer"


@dataclass
class VersionResult:
    version: str
    scores: list[bool] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    times: list[float] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def avg_steps(self) -> float:
        return sum(self.steps) / len(self.steps) if self.steps else 0.0

    @property
    def avg_time(self) -> float:
        return sum(self.times) / len(self.times) if self.times else 0.0


def run_eval(versions: list[str] | None = None) -> None:
    """
    Run the eval set on each prompt version and print a comparison table.
    """
    if versions is None:
        versions = ["v1", "v2", "v3"]

    results: dict[str, VersionResult] = {v: VersionResult(version=v) for v in versions}

    print(f"\n{'═' * 65}")
    print(f"  Eval: {len(EVAL_SET)} questions × {len(versions)} prompt versions")
    print(f"  Model: {MODEL}")
    print(f"{'═' * 65}\n")

    for i, case in enumerate(EVAL_SET):
        print(f"Q{i + 1}: {case.question}")
        print(f"     Expected: {case.expected_keywords}")

        for v in versions:
            prompt_fn = PROMPTS[v]
            result = run_agent(case.question, prompt_fn(), verbose=False)
            passed, reason = score_answer(result["answer"], case)

            results[v].scores.append(passed)
            results[v].steps.append(result["steps"])
            results[v].times.append(result["elapsed"])

            status = "PASS" if passed else "FAIL"
            answer_preview = result["answer"][:60].replace("\n", " ")
            print(
                f"     [{v}] {status:4s} | {result['steps']} steps | {result['elapsed']}s | {answer_preview}..."
            )

        print()

    # Summary table
    print(f"\n{'─' * 65}")
    print(f"  {'Version':<10} {'Accuracy':>10} {'Avg steps':>12} {'Avg time':>10}")
    print(f"{'─' * 65}")
    for v, r in results.items():
        bar = "█" * round(r.accuracy * 20)
        print(
            f"  {v:<10} {r.accuracy:>9.0%}  {bar:<20}  {r.avg_steps:>6.1f} steps  {r.avg_time:>6.1f}s"
        )
    print(f"{'─' * 65}")

    best = max(results.values(), key=lambda r: r.accuracy)
    print(f"\n  Best: {best.version} with {best.accuracy:.0%} accuracy\n")


# ── Interactive mode ──────────────────────────────────────────────────────────


def repl(version: str) -> None:
    prompt_fn = PROMPTS.get(version)
    if not prompt_fn:
        print(f"Unknown version '{version}'. Choose: v1, v2, v3")
        sys.exit(1)

    print(f"\n{'─' * 60}")
    print(f"  Day 5 — Prompt engineering | {version.upper()}")
    print(f"  Model: {MODEL} | Type 'quit' to exit, 'prompt' to see the system prompt")
    print(f"{'─' * 60}\n")

    system_prompt = prompt_fn()

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
        if question.lower() == "prompt":
            print(f"\n{'─' * 60}")
            print(system_prompt)
            print(f"{'─' * 60}\n")
            continue

        result = run_agent(question, system_prompt, verbose=True)
        print(f"\n  Steps: {result['steps']} | Time: {result['elapsed']}s\n")


def print_diff() -> None:
    """Print all three prompts side by side for comparison."""
    for v, fn in PROMPTS.items():
        print(f"\n{'═' * 60}")
        print(f"  {v.upper()} — {fn.__doc__.strip().splitlines()[0]}")
        print(f"{'─' * 60}")
        print(fn())
    print(f"\n{'═' * 60}\n")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    if arg is None:
        run_eval()
    elif arg == "diff":
        print_diff()
    elif arg in PROMPTS:
        repl(arg)
    else:
        print(f"Usage: python day5_prompts.py [v1|v2|v3|diff]")
        print("  (no arg)  — run full eval on all 3 prompt versions")
        print("  v1/v2/v3  — interactive REPL with that prompt version")
        print("  diff      — print all 3 prompts for comparison")


if __name__ == "__main__":
    main()

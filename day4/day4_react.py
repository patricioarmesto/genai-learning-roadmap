"""
Day 4 — The ReAct agent loop
==============================
A ReAct agent that reasons through multi-step questions using
search (mocked), a calculator, and a date tool.

Key concepts demonstrated:
  • The Thought → Action → Observation cycle
  • Scratchpad as working memory
  • Parsing structured output from free text
  • Loop termination via the "finish" action
  • How the agent self-corrects when a tool returns an error

Architecture:
  - Style A: text-format ReAct (no native tools API)
  - System prompt defines the format; regex parses the output
  - Every iteration: append observation, call model, parse response

Requirements:
    uv add requests
    ollama pull llama3.2
    ollama serve
"""

import math
import re
import sys
from datetime import datetime, timezone
from textwrap import dedent

import requests

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:397b-cloud"
TEMPERATURE = 0.2
MAX_TOKENS = 512  # Per iteration — thoughts + action only, not full answer
MAX_STEPS = 10  # Hard limit on Thought/Action/Observation cycles


# ── Mock knowledge base ───────────────────────────────────────────────────────
# In day 8 this becomes a real vector DB. For now, a dict is enough
# to focus on the agent loop mechanics.

KNOWLEDGE_BASE = {
    "einstein birth": "Albert Einstein was born on 14 March 1879 in Ulm, Germany.",
    "einstein death": "Albert Einstein died on 18 April 1955 in Princeton, New Jersey.",
    "special relativity": "Einstein published the special theory of relativity in 1905 in the paper 'On the Electrodynamics of Moving Bodies'.",
    "general relativity": "Einstein published the general theory of relativity in 1915.",
    "nobel prize einstein": "Einstein received the Nobel Prize in Physics in 1921 for the photoelectric effect, not relativity.",
    "python created": "Python was created by Guido van Rossum. The first version (0.9.0) was released in February 1991.",
    "python latest": "Python 3.12 was released in October 2023. Python 3.13 was released in October 2024.",
    "speed of light": "The speed of light in a vacuum is approximately 299,792,458 metres per second (c).",
    "distance earth moon": "The average distance from Earth to the Moon is about 384,400 kilometres.",
    "distance earth sun": "The average distance from Earth to the Sun is about 149.6 million kilometres (1 AU).",
    "water boiling point": "Water boils at 100°C (212°F) at standard atmospheric pressure (1 atm).",
    "mount everest height": "Mount Everest is 8,848.86 metres (29,031.7 feet) above sea level.",
    "population world": "The world population reached 8 billion people in November 2022.",
    "population argentina": "Argentina's population is approximately 46 million people (2023 estimate).",
    "buenos aires founded": "Buenos Aires was formally founded on 11 June 1580 by Juan de Garay.",
    "turing born": "Alan Turing was born on 23 June 1912 in London.",
    "turing died": "Alan Turing died on 7 June 1954.",
    "turing machine": "Alan Turing described the Turing machine concept in his 1936 paper 'On Computable Numbers'.",
}


def search(query: str) -> str:
    """
    Keyword search over the mock knowledge base.
    In week 2, this becomes semantic search over Qdrant.
    """
    q = query.lower().strip()

    # Try progressively looser matches
    for key, value in KNOWLEDGE_BASE.items():
        if key in q or q in key:
            return value

    # Word overlap fallback
    q_words = set(q.split())
    best_score, best_value = 0, None
    for key, value in KNOWLEDGE_BASE.items():
        overlap = len(q_words & set(key.split()))
        if overlap > best_score:
            best_score, best_value = overlap, value

    if best_value and best_score >= 1:
        return best_value

    return f"No information found for '{query}'. Try rephrasing or search for a more specific term."


def calculate(expression: str) -> str:
    """Safe arithmetic evaluator. Returns result as a string."""
    safe_globals = {
        "__builtins__": {},
        "math": math,
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "sin": math.sin,
        "cos": math.cos,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "round": round,
    }
    try:
        if any(b in expression for b in ["import", "exec", "eval", "open", "__"]):
            return "Error: expression not allowed"
        result = eval(expression, safe_globals)  # noqa: S307
        return str(round(float(result), 6))
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as e:
        return f"Error: could not evaluate '{expression}' — {e}"


def get_date(query: str = "") -> str:
    """Return current UTC date/time information."""
    now = datetime.now(timezone.utc)
    return (
        f"Current UTC date: {now.strftime('%Y-%m-%d')} "
        f"({now.strftime('%A, %d %B %Y')}). "
        f"Current time: {now.strftime('%H:%M')} UTC."
    )


TOOLS = {
    "search": search,
    "calculate": calculate,
    "get_date": get_date,
}


# ── System prompt ─────────────────────────────────────────────────────────────


def build_system_prompt() -> str:
    tool_list = "\n".join(
        [
            "- search(query)    : Search the knowledge base for factual information",
            "- calculate(expr)  : Evaluate a math expression. Use Python syntax: 2**10, sqrt(144)",
            "- get_date(query)  : Get the current date and time",
            "- finish(answer)   : Return the final answer to the user and stop",
        ]
    )

    example = dedent("""
        Question: How old was Alan Turing when he died?

        Thought: I need Turing's birth year and death year, then subtract.
        Action: search
        Action Input: Turing born
        Observation: Alan Turing was born on 23 June 1912 in London.

        Thought: Born in 1912. Now I need his death year.
        Action: search
        Action Input: Turing died
        Observation: Alan Turing died on 7 June 1954.

        Thought: He died in 1954, born in 1912. Age = 1954 - 1912 = 41. Let me verify.
        Action: calculate
        Action Input: 1954 - 1912
        Observation: 42

        Thought: The calculator says 42. He was born in June and died in June, so he had just turned 41 — but the calculator gives 42 because that's the year difference. He was 41 years old.
        Action: finish
        Action Input: Alan Turing was 41 years old when he died on 7 June 1954.
    """).strip()

    return dedent(f"""
        You are a reasoning agent. You solve questions step by step using tools.

        Available tools:
        {tool_list}

        STRICT FORMAT RULES — follow exactly, no deviations:
        1. Always start with "Thought:" and write your reasoning
        2. Follow with "Action:" on a new line (one tool name only)
        3. Follow with "Action Input:" on a new line (the argument string)
        4. Stop there — wait for the Observation before continuing
        5. Never invent Observations — only real tool results count
        6. Use "finish" only when you have a complete, verified answer

        {example}

        Now solve the following question using the same format.
    """).strip()


# ── Output parser ─────────────────────────────────────────────────────────────


def parse_react_output(text: str) -> tuple[str | None, str | None, str | None]:
    """
    Extract (thought, action, action_input) from a ReAct-formatted response.

    The model should output:
        Thought: <reasoning text>
        Action: <tool name>
        Action Input: <argument>

    Returns (thought, action, action_input).
    Any field may be None if parsing fails.

    This parser is intentionally lenient — small models often add
    extra whitespace, capitalization variations, or light formatting.
    """
    # Normalise line endings and strip code fences
    text = text.strip()
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    thought = None
    action = None
    action_input = None

    # Thought — everything between "Thought:" and the next "Action:"
    m = re.search(r"Thought:\s*(.+?)(?=\nAction:|\Z)", text, re.DOTALL | re.IGNORECASE)
    if m:
        thought = m.group(1).strip()

    # Action — the tool name on its own line
    m = re.search(r"Action:\s*(\w+)", text, re.IGNORECASE)
    if m:
        action = m.group(1).strip().lower()

    # Action Input — everything after "Action Input:" to end of relevant section
    m = re.search(
        r"Action Input:\s*(.+?)(?=\nObservation:|\nThought:|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        action_input = m.group(1).strip().strip('"').strip("'")

    return thought, action, action_input


# ── The ReAct loop ────────────────────────────────────────────────────────────


def run_react_agent(question: str, verbose: bool = True) -> str:
    """
    Run the full ReAct loop for a question.

    The scratchpad grows with each iteration:
        Thought → Action → Action Input → Observation → Thought → ...

    The model receives the full scratchpad on every call so it can
    read its own prior reasoning and observations before deciding
    what to do next.

    Returns the final answer string.
    """
    scratchpad = ""  # Grows with every Thought/Action/Observation
    step = 0

    if verbose:
        print(f"\n{'═' * 60}")
        print(f"  Question: {question}")
        print(f"{'═' * 60}")

    while step < MAX_STEPS:
        step += 1

        # ── Build the prompt for this iteration ───────────────────────────────
        # The user message contains the question + full scratchpad so far.
        # The model reads the entire history and appends the next T/A/AI block.
        user_content = f"Question: {question}\n\n{scratchpad}".strip()

        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": user_content},
        ]

        # ── Call the model ────────────────────────────────────────────────────
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": TEMPERATURE,
                        "num_predict": MAX_TOKENS,
                        "stop": [
                            "Observation:"
                        ],  # Halt before the model invents results
                    },
                },
                timeout=120,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            print("[error] Cannot reach Ollama. Run: ollama serve")
            sys.exit(1)

        raw_output = resp.json()["message"]["content"].strip()

        # ── Parse the output ──────────────────────────────────────────────────
        thought, action, action_input = parse_react_output(raw_output)

        if verbose:
            print(f"\n  Step {step}")
            if thought:
                print(f"  Thought:      {thought}")
            if action:
                print(f"  Action:       {action}")
            if action_input:
                print(f"  Action Input: {action_input}")

        # ── Handle finish ─────────────────────────────────────────────────────
        if action == "finish":
            final_answer = action_input or thought or raw_output
            if verbose:
                print(f"\n{'─' * 60}")
                print(f"  Final answer: {final_answer}")
                print(f"  Steps taken:  {step}")
                print(f"{'═' * 60}")
            return final_answer

        # ── Handle missing action ─────────────────────────────────────────────
        if not action or not action_input:
            # Model didn't follow the format — inject a nudge as the observation
            observation = (
                "Error: I could not parse an Action and Action Input from your response. "
                "Please follow the exact format:\n"
                "Thought: your reasoning\nAction: tool_name\nAction Input: argument"
            )
            if verbose:
                print(f"  [parse fail] nudging model back to format")
        else:
            # ── Execute the tool ──────────────────────────────────────────────
            tool_fn = TOOLS.get(action)
            if tool_fn:
                observation = tool_fn(action_input)
            else:
                available = ", ".join(TOOLS.keys())
                observation = (
                    f"Error: unknown tool '{action}'. Available tools: {available}"
                )

        if verbose:
            print(f"  Observation:  {observation}")

        # ── Append to scratchpad ──────────────────────────────────────────────
        # Build the new chunk to add. We include the full thought + action +
        # action input from the model's output, then add the real observation.
        new_chunk = raw_output.rstrip()
        if not new_chunk.endswith("\n"):
            new_chunk += "\n"
        new_chunk += f"Observation: {observation}\n\n"

        scratchpad = (scratchpad + new_chunk).strip() + "\n\n"

    # ── Max steps reached ─────────────────────────────────────────────────────
    # Ask for a direct answer with everything collected so far
    if verbose:
        print(f"\n  [max steps reached] requesting direct answer")

    messages = [
        {
            "role": "system",
            "content": "Summarise what you found and give a concise final answer.",
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nResearch so far:\n{scratchpad}",
        },
    ]
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": TEMPERATURE, "num_predict": 300},
        },
        timeout=120,
    )
    return resp.json()["message"]["content"].strip()


# ── Demo questions ────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    # Single-step (sanity check)
    "What is the speed of light?",
    # Two-step: search + calculate
    "How old was Einstein when he died?",
    # Three-step: two searches + calculate
    "How many years passed between Einstein publishing special relativity and receiving the Nobel Prize?",
    # Requires today's date + known fact
    "How many years ago was Buenos Aires founded?",
    # Multi-step with unit conversion
    "If light travels at 299792 km/s, how many seconds does it take to travel from Earth to the Moon?",
    # Adversarial: tests self-correction when first search is vague
    "What is the height of the tallest mountain in metres divided by the number of people in Argentina (in millions)?",
]


def run_demo() -> None:
    for i, q in enumerate(DEMO_QUESTIONS):
        run_react_agent(q, verbose=True)
        if i < len(DEMO_QUESTIONS) - 1:
            try:
                input("\nPress Enter for next question...")
            except (KeyboardInterrupt, EOFError):
                break


def repl() -> None:
    print(f"\n{'─' * 60}")
    print(f"  Day 4 — ReAct Agent")
    print(f"  Model: {MODEL}  |  Max steps: {MAX_STEPS}")
    print(f"  Type 'demo' to run example questions, 'quit' to exit")
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
        if question.lower() == "demo":
            run_demo()
            continue

        run_react_agent(question, verbose=True)
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_demo()
    else:
        repl()

"""
Day 1 — LLM API basics
=======================
A CLI chatbot that persists conversation history to a JSON file.
Each run loads prior messages, so the model "remembers" past turns.

Key concepts demonstrated:
  • The message format (role + content)
  • How to fake memory by replaying history
  • Temperature and its effect on output
  • Max tokens as a hard output cap
  • Stop sequences to control generation

Requirements:
    uv add requests
    ollama pull llama3.2
    ollama serve
"""

import json
import os
import sys
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:397b-cloud"
HISTORY_FILE = Path("chat_history.json")

# Temperature experiments — try each and notice how responses change:
#   0.0  → deterministic, robotic, best for structured output
#   0.2  → focused, reliable  ← good default for agents
#   0.7  → natural, slightly varied  ← good default for conversation
#   1.2  → creative, occasionally surprising
#   2.0  → chaotic, often incoherent
TEMPERATURE = 0.7

# Max tokens caps the response length. The model stops HERE, mid-sentence
# if needed. Try setting this to 20 and see what happens.
MAX_TOKENS = 500

# Stop sequences: the model stops generating as soon as it produces this string.
# Useful in agents to halt after a structured section. Empty list = no early stop.
STOP_SEQUENCES: list[str] = []

# The system prompt is sent as the first message on every call.
# The USER never sees it, but the model always reads it.
# This is where you set the model's persona, rules, and output format.
SYSTEM_PROMPT = """You are a helpful programming tutor. You explain concepts clearly,
use concrete examples, and ask follow-up questions to check understanding.
Keep responses concise — 3 paragraphs max unless the user asks for more detail."""


# ── History helpers ───────────────────────────────────────────────────────────


def load_history() -> list[dict]:
    """
    Load conversation history from disk.

    The history is a list of message dicts:
        [
            {"role": "user",      "content": "What is recursion?"},
            {"role": "assistant", "content": "Recursion is ..."},
            ...
        ]

    This is the SAME format the API expects, which is why replaying it
    gives the model its "memory" — it literally re-reads past turns.
    """
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text())
    return []


def save_history(history: list[dict]) -> None:
    """Persist history to disk so it survives between runs."""
    HISTORY_FILE.write_text(json.dumps(history, indent=2, ensure_ascii=False))


def trim_history(history: list[dict], max_turns: int = 20) -> list[dict]:
    """
    Keep only the last N turns to avoid overflowing the context window.

    In production you'd use a smarter strategy (e.g. summarise old turns),
    but trimming is the simplest approach and works well for most chatbots.

    Each "turn" = 1 user message + 1 assistant message = 2 items.
    """
    max_messages = max_turns * 2
    if len(history) > max_messages:
        # Always keep an even number so we don't split a turn
        history = history[-max_messages:]
    return history


# ── LLM call ─────────────────────────────────────────────────────────────────


def chat(history: list[dict]) -> str:
    """
    Send the full conversation history to Ollama and get the next response.

    Notice what we send:
      1. A system message (always first — sets the model's behaviour)
      2. The full history (so the model has "memory")

    The model sees all of this as one big context and predicts the next token.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,  # Ollama's name for max_tokens
            "stop": STOP_SEQUENCES,
        },
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    except requests.exceptions.ConnectionError:
        print("\n[error] Cannot reach Ollama. Is it running?")
        print("  Run: ollama serve")
        sys.exit(1)

    except requests.exceptions.HTTPError as e:
        print(f"\n[error] Ollama returned an error: {e}")
        sys.exit(1)


# ── Token counting ────────────────────────────────────────────────────────────


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate: 1 token ≈ 4 characters (English).
    Real tokenisers are more precise but this is good enough for monitoring.
    """
    return len(text) // 4


def history_token_estimate(history: list[dict]) -> int:
    total = estimate_tokens(SYSTEM_PROMPT)
    for msg in history:
        total += estimate_tokens(msg["content"]) + 4  # +4 for role overhead
    return total


# ── Main loop ─────────────────────────────────────────────────────────────────


def main() -> None:
    history = load_history()

    print(f"\n{'─' * 50}")
    print(f"  Day 1 Chatbot  |  model: {MODEL}  |  temp: {TEMPERATURE}")
    if history:
        print(f"  Loaded {len(history) // 2} previous turn(s) from {HISTORY_FILE}")
    print(f"  Commands: 'quit' to exit, 'clear' to reset history, 'stats' for info")
    print(f"{'─' * 50}\n")

    while True:
        # ── Get user input ────────────────────────────────────────────────────
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue

        # ── Handle commands ───────────────────────────────────────────────────
        if user_input.lower() == "quit":
            print("Bye!")
            break

        if user_input.lower() == "clear":
            history = []
            save_history(history)
            print("[history cleared]\n")
            continue

        if user_input.lower() == "stats":
            tok = history_token_estimate(history)
            print(f"\n  Turns in memory : {len(history) // 2}")
            print(f"  Est. context    : ~{tok:,} tokens")
            print(f"  History file    : {HISTORY_FILE.absolute()}")
            print(f"  Temperature     : {TEMPERATURE}")
            print(f"  Max tokens      : {MAX_TOKENS}\n")
            continue

        # ── Add user message to history ───────────────────────────────────────
        history.append({"role": "user", "content": user_input})

        # Trim before sending to avoid overflowing the context window
        history = trim_history(history)

        # ── Call the LLM ──────────────────────────────────────────────────────
        print("\nAssistant: ", end="", flush=True)
        response = chat(history)
        print(response)

        # ── Add assistant response to history and save ────────────────────────
        # This is the critical step: we store what the model said so that
        # next turn (or next run of the script) it can "remember" it.
        history.append({"role": "assistant", "content": response})
        save_history(history)

        # Show a quick token estimate so you can feel the context growing
        tok = history_token_estimate(history)
        print(f"\n  [~{tok:,} tokens in context]\n")


# ── Experiments entrypoint ────────────────────────────────────────────────────


def run_experiments() -> None:
    """
    Non-interactive mode: run the same prompt at different temperatures
    and print all outputs side by side.

    Run with:  python day1_chatbot.py experiment
    """
    prompt = "Explain what a Python list is in one sentence."
    temperatures = [0.0, 0.5, 1.0, 1.5]

    print(f"\nPrompt: {prompt!r}\n")
    print("=" * 60)

    for temp in temperatures:
        history = [{"role": "user", "content": prompt}]
        payload = {
            "model": MODEL,
            "messages": [{"role": "system", "content": "Be concise."}] + history,
            "stream": False,
            "options": {"temperature": temp, "num_predict": 80},
        }
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
            resp.raise_for_status()
            output = resp.json()["message"]["content"].strip()
        except Exception as e:
            output = f"[error: {e}]"

        print(f"\nTemperature {temp}:")
        print(f"  {output}")

    print("\n" + "=" * 60)
    print("\nObservation: at temp=0.0 the output is near-identical every run.")
    print("At temp=1.5+ it becomes varied and sometimes odd.")
    print("For agents, stay in the 0.1–0.3 range for reliable structured output.\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "experiment":
        run_experiments()
    else:
        main()

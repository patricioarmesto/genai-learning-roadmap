"""
Day 3 — Tool calling and function use
======================================
A weather + calculator assistant that demonstrates the full tool-call cycle.

Key concepts demonstrated:
  • Tool schema definition (name, description, parameters)
  • The tool-call / tool-result message cycle
  • Parallel tool calls (model calls multiple tools in one step)
  • The agent loop: run until no more tool calls
  • How bad tool descriptions lead to wrong calls

Requirements:
    uv add requests
    ollama pull llama3.2   (must be >= 0.3 for tools API)
    ollama serve
"""

import json
import math
import sys
from datetime import datetime

import requests

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:397b-cloud"
TEMPERATURE = 0.2  # Low = reliable tool argument formatting
MAX_TOKENS = 1024

# ── Tool implementations ──────────────────────────────────────────────────────
# These are the REAL functions your code runs.
# The model never sees this code — only the schema descriptions below.

# Simulated weather data (no API key needed for the exercise)
WEATHER_DB = {
    "buenos aires": {
        "temp": 22,
        "condition": "partly cloudy",
        "humidity": 65,
        "wind_kph": 15,
    },
    "london": {"temp": 9, "condition": "overcast", "humidity": 82, "wind_kph": 22},
    "tokyo": {"temp": 14, "condition": "clear", "humidity": 55, "wind_kph": 8},
    "new york": {"temp": 18, "condition": "sunny", "humidity": 48, "wind_kph": 12},
    "sydney": {"temp": 26, "condition": "sunny", "humidity": 60, "wind_kph": 18},
    "paris": {"temp": 11, "condition": "light rain", "humidity": 78, "wind_kph": 20},
    "cairo": {"temp": 30, "condition": "clear", "humidity": 25, "wind_kph": 10},
}


def get_weather(city: str, unit: str = "celsius") -> dict:
    """
    Look up weather for a city.
    Returns a dict the model will read as context.
    """
    key = city.lower().strip()
    data = WEATHER_DB.get(key)

    if not data:
        return {
            "error": f"No weather data for '{city}'. Available: {', '.join(WEATHER_DB)}"
        }

    temp = data["temp"]
    if unit == "fahrenheit":
        temp = round(temp * 9 / 5 + 32, 1)

    return {
        "city": city.title(),
        "temp": temp,
        "unit": unit,
        "condition": data["condition"],
        "humidity": data["humidity"],
        "wind_kph": data["wind_kph"],
        "retrieved_at": datetime.now().strftime("%H:%M"),
    }


def calculate(expression: str) -> dict:
    """
    Safely evaluate a mathematical expression.
    Supports: +, -, *, /, **, sqrt, log, sin, cos, pi, e
    """
    # Whitelist approach: only allow safe math operations
    safe_globals = {
        "__builtins__": {},
        "math": math,
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "round": round,
    }
    try:
        # Block any attempt to access builtins through the expression
        if any(bad in expression for bad in ["import", "exec", "eval", "open", "__"]):
            return {"error": "Expression not allowed"}
        result = eval(expression, safe_globals)  # noqa: S307
        return {"expression": expression, "result": round(float(result), 6)}
    except ZeroDivisionError:
        return {"error": "Division by zero"}
    except Exception as e:
        return {"error": f"Could not evaluate '{expression}': {e}"}


def get_current_time(timezone: str = "UTC") -> dict:
    """Return the current date and time (always UTC in this demo)."""
    now = datetime.utcnow()
    return {
        "timezone": timezone,
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M"),
        "day_of_week": now.strftime("%A"),
        "note": "Time is always UTC in this demo regardless of timezone arg",
    }


# ── Tool registry ─────────────────────────────────────────────────────────────
# Maps tool names to their Python implementations.
# This is what your loop calls when the model requests a tool.

TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "calculate": calculate,
    "get_current_time": get_current_time,
}


# ── Tool schemas ──────────────────────────────────────────────────────────────
# These are what the MODEL sees. Write descriptions as if explaining to a
# smart colleague who doesn't know your codebase.
#
# Quality of descriptions directly determines quality of tool calls.
# Run the BAD_SCHEMAS experiment below to see the difference.

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get the current weather conditions for a city. "
                "Use this whenever the user asks about weather, temperature, "
                "climate, or whether to bring an umbrella. "
                "Returns temperature, condition, humidity, and wind speed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. 'London', 'Buenos Aires', 'New York'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit. Default is celsius.",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate a mathematical expression and return the result. "
                "Use this for any arithmetic, percentages, unit conversions, "
                "or numeric calculations. Do NOT calculate in your head — "
                "always use this tool for numbers to ensure accuracy. "
                "Supports: +, -, *, /, ** (power), sqrt(), log(), sin(), cos(), pi, e"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "A Python math expression as a string. "
                            "Examples: '15 * 340', 'sqrt(144)', '(100 - 32) * 5/9', "
                            "'2 ** 10', 'pi * 5 ** 2'"
                        ),
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "Get the current date and time. Use this when the user asks "
                "what time or date it is, what day of the week it is, or needs "
                "to know the current date for any calculation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Timezone name e.g. 'UTC', 'America/New_York'. Default: UTC",
                    },
                },
                "required": [],
            },
        },
    },
]


# ── Tool executor ─────────────────────────────────────────────────────────────


def execute_tool(tool_name: str, arguments: dict) -> str:
    """
    Look up and run the real function behind a tool call.
    Returns a JSON string — this is what gets appended as the tool result.
    """
    fn = TOOL_FUNCTIONS.get(tool_name)
    if not fn:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    try:
        result = fn(**arguments)
        return json.dumps(result)
    except TypeError as e:
        return json.dumps({"error": f"Wrong arguments for {tool_name}: {e}"})
    except Exception as e:
        return json.dumps({"error": f"Tool {tool_name} failed: {e}"})


# ── Core agent loop ───────────────────────────────────────────────────────────


def run_tool_loop(
    messages: list[dict],
    verbose: bool = True,
) -> tuple[str, list[dict]]:
    """
    The tool-calling loop:
      1. Call the model with current messages + tool schemas
      2. If response has tool_calls → execute each, append results, go to 1
      3. If response has content (no tool_calls) → we're done, return answer

    Returns (final_answer, updated_messages).

    This loop IS the agent. Day 4 wraps this in a ReAct wrapper.
    Day 5 adds memory. Week 3 runs multiple of these in parallel.
    Everything builds on this loop.
    """
    MAX_ITERATIONS = 10  # safety cap against infinite tool loops
    iteration = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1

        # ── Call the model ────────────────────────────────────────────────────
        payload = {
            "model": MODEL,
            "messages": messages,
            "tools": TOOL_SCHEMAS,
            "stream": False,
            "options": {
                "temperature": TEMPERATURE,
                "num_predict": MAX_TOKENS,
            },
        }

        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
            resp.raise_for_status()
            response_data = resp.json()
        except requests.exceptions.ConnectionError:
            print("[error] Cannot reach Ollama. Run: ollama serve")
            sys.exit(1)

        message = response_data.get("message", {})
        tool_calls = message.get("tool_calls", [])

        # ── No tool calls → model is done, return the text answer ─────────────
        if not tool_calls:
            final_answer = message.get("content", "").strip()
            messages.append({"role": "assistant", "content": final_answer})
            return final_answer, messages

        # ── Tool calls requested → execute each and append results ────────────
        # Append the assistant's tool-call request to history first
        messages.append(
            {
                "role": "assistant",
                "content": message.get("content", ""),
                "tool_calls": tool_calls,
            }
        )

        if verbose:
            print(
                f"\n  [iter {iteration}] Model requested {len(tool_calls)} tool call(s):"
            )

        for call in tool_calls:
            fn_info = call.get("function", {})
            tool_name = fn_info.get("name", "")
            arguments = fn_info.get("arguments", {})

            # Ollama sometimes returns arguments as a JSON string instead of dict
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            if verbose:
                print(f"    → {tool_name}({json.dumps(arguments)})")

            # Run the real function
            result_json = execute_tool(tool_name, arguments)

            if verbose:
                print(
                    f"      ← {result_json[:100]}{'...' if len(result_json) > 100 else ''}"
                )

            # Append tool result — role must be "tool"
            # tool_call_id links this result to its request
            messages.append(
                {
                    "role": "tool",
                    "content": result_json,
                    "tool_call_id": call.get("id", f"call_{iteration}"),
                }
            )

    # Safety: if we hit the iteration limit, ask for a direct answer
    messages.append(
        {
            "role": "user",
            "content": "Please summarise what you found and give a final answer.",
        }
    )
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": TEMPERATURE},
        },
        timeout=120,
    )
    return resp.json()["message"]["content"].strip(), messages


# ── Conversation wrapper ──────────────────────────────────────────────────────


def ask(question: str, verbose: bool = True) -> str:
    """Single-turn question with tool support."""
    print(f"\nQuestion: {question}")
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to weather data, "
                "a calculator, and the current time. "
                "Always use the calculate tool for any math — never compute in your head. "
                "Use tools in parallel when multiple are needed."
            ),
        },
        {"role": "user", "content": question},
    ]
    answer, _ = run_tool_loop(messages, verbose=verbose)
    print(f"\nAnswer: {answer}")
    return answer


# ── Interactive REPL ──────────────────────────────────────────────────────────


def repl() -> None:
    """
    Multi-turn chat that keeps context across tool calls.
    Type 'debug' to print the full message history.
    """
    print(f"\n{'─' * 55}")
    print(f"  Day 3 — Tool-calling assistant")
    print(f"  Model: {MODEL}  |  Tools: {', '.join(TOOL_FUNCTIONS)}")
    print(f"  Try: 'weather in Tokyo', '15% of 847', 'what day is it?'")
    print(f"  Type 'debug' to see the message history, 'quit' to exit")
    print(f"{'─' * 55}\n")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to weather data, "
                "a calculator, and the current time. "
                "Always use the calculate tool for any math — never compute in your head. "
                "Use tools in parallel when multiple are needed."
            ),
        }
    ]

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "debug":
            print(f"\n── Message history ({len(messages)} messages) ──")
            for i, m in enumerate(messages):
                role = m["role"].upper()
                body = m.get("content", "")
                calls = m.get("tool_calls", [])
                if calls:
                    body = f"[tool_calls: {[c['function']['name'] for c in calls]}]"
                print(f"  [{i}] {role}: {str(body)[:100]}")
            print()
            continue

        messages.append({"role": "user", "content": user_input})
        print()
        answer, messages = run_tool_loop(messages, verbose=True)
        print(f"\nAssistant: {answer}\n")


# ── Experiments ───────────────────────────────────────────────────────────────


def run_parallel_experiment() -> None:
    """
    Show parallel tool calls in action.
    The model should request both get_weather and calculate simultaneously.

    Run with: python day3_tools.py parallel
    """
    print("\n── Parallel tool call experiment ──────────────────────")
    ask(
        "What's the weather in London and Paris? "
        "Also, what's the area of a circle with radius 7?",
        verbose=True,
    )


def run_schema_quality_experiment() -> None:
    """
    Compare good vs bad tool descriptions.
    Shows how description quality directly affects tool selection.

    Run with: python day3_tools.py schema
    """
    print("\n── Schema quality experiment ───────────────────────────")
    print("Testing with GOOD descriptions (current schemas):")
    print("─" * 50)

    question = "How hot is it in Cairo right now? And what is 847 divided by 7?"

    # Good schemas (current TOOL_SCHEMAS)
    messages_good = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question},
    ]
    answer_good, msgs_good = run_tool_loop(messages_good, verbose=True)
    tool_calls_good = sum(1 for m in msgs_good if m.get("role") == "tool")
    print(f"Result: {answer_good}")
    print(f"Tool calls made: {tool_calls_good}")

    # Now try with deliberately vague descriptions
    BAD_SCHEMAS = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "weather",  # ← too vague
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "city"},
                    },
                    "required": ["city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "math",  # ← too vague, no instruction to always use it
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "expression"},
                    },
                    "required": ["expression"],
                },
            },
        },
    ]

    print("\nTesting with BAD descriptions (vague, no examples):")
    print("─" * 50)
    messages_bad = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question},
    ]

    payload = {
        "model": MODEL,
        "messages": messages_bad,
        "tools": BAD_SCHEMAS,
        "stream": False,
        "options": {"temperature": TEMPERATURE},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    bad_message = resp.json().get("message", {})
    bad_calls = bad_message.get("tool_calls", [])

    if bad_calls:
        print(
            f"Tool calls made: {len(bad_calls)} (model still called tools despite bad descriptions)"
        )
    else:
        print("Tool calls made: 0 — model answered from memory instead of using tools!")
        print(f"Answer (possibly wrong): {bad_message.get('content', '')[:200]}")

    print("\nKey insight: 'Do NOT calculate in your head — always use this tool'")
    print("in the description is what forces the model to use calculate reliably.")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "parallel":
            run_parallel_experiment()
        elif cmd == "schema":
            run_schema_quality_experiment()
        elif cmd == "demo":
            # Run a few preset questions without interactive input
            questions = [
                "What's the weather like in Tokyo?",
                "What is 15% of 847, and what's the square root of 256?",
                "Compare the weather in London and Buenos Aires. Which is warmer?",
                "If I fly from New York to Paris (5570 km) at 900 km/h, how long is the flight?",
                "What day of the week is it, and what's 2 to the power of 16?",
            ]
            for q in questions:
                ask(q, verbose=True)
                print()
                input("Press Enter for next question...")
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python day3_tools.py [parallel|schema|demo]")
    else:
        repl()


if __name__ == "__main__":
    main()

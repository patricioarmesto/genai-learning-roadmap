# Day 3 — Tool calling and function use

A weather + calculator assistant that demonstrates the full tool-call cycle. The agent loop calls tools as needed until it has enough information to answer the user.

## Concepts

| Concept | What it does |
|---------|--------------|
| **Tool schema** | JSON definition describing what a function does and its parameters |
| **Tool-call cycle** | Model requests a tool → your code executes it → result is fed back |
| **Agent loop** | Repeatedly call the model until it returns a text answer (no more tool calls) |
| **Parallel calls** | Model can request multiple tools in a single response |
| **Schema quality** | Vague descriptions lead to wrong or missed tool calls |

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running (model must support tools API)

```bash
uv add requests
ollama pull llama3.2   # or minimax-m2:cloud
ollama serve
```

## Usage

**Interactive REPL:**

```bash
uv run python day3_tools.py
```

Ask questions like:
- "What's the weather in Tokyo?"
- "Calculate 15% of 847"
- "Compare weather in London vs Paris"

Commands: `quit` to exit

**Run preset demo questions:**

```bash
uv run python day3_tools.py demo
```

**Parallel tool call experiment:**

```bash
uv run python day3_tools.py parallel
```
Tests whether the model can call multiple tools in one response.

**Schema quality experiment:**

```bash
uv run python day3_tools.py schema
```
Compares good vs bad tool descriptions to show why wording matters.

## Available tools

| Tool | Description |
|------|-------------|
| `get_weather(city, unit)` | Get weather for a city (celsius/fahrenheit) |
| `calculate(expression)` | Evaluate math expressions (`+`, `-`, `*`, `/`, `**`, `sqrt`, `log`, etc.) |
| `get_current_time(timezone)` | Get current date/time (UTC in this demo) |

## Key insight

The `calculate` tool description includes: *"Do NOT calculate in your head — always use this tool"*. This explicit instruction dramatically improves reliability. Without it, the model may try to do math from memory and get it wrong.

## Project layout

- `day3_tools.py` — Tool implementations, schemas, agent loop, REPL, and experiments
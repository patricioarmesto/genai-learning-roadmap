# ReAct Research Agent

A CLI tool that uses a **ReAct** (Reasoning + Acting) agent loop to research questions, make tool calls, and generate structured JSON reports.

## Features

- **ReAct agent loop** — Iterative reasoning with tool use (search, calculate)
- **Streaming output** — Watch the agent think in real-time
- **Structured reports** — Outputs JSON with answer, confidence, sources, and reasoning
- **Interactive mode** — Ask multiple questions in a loop
- **Duplicate detection** — Prevents redundant tool calls

## Tools

| Tool | Description |
|------|-------------|
| `search` | Web search for facts |
| `calculate` | Evaluates mathematical expressions |
| `finish` | Signal end of research with final answer |

## Usage

```bash
# Single question
python main.py "How old was Einstein when he published special relativity?"

# With output file
python main.py "How far is the Moon in light-seconds?" --out report_moon.json

# Non-streaming (faster)
python main.py "When was Buenos Aires founded?" --no-stream

# Interactive mode
python main.py --interactive
```

## Output

Generates a JSON report:

```json
{
  "question": "...",
  "answer": "...",
  "confidence": "high|medium|low",
  "sources": [{"tool": "search", "query": "...", "result": "..."}],
  "steps_taken": 4,
  "reasoning": "...",
  "generated_at": "..."
}
```

## Setup

```bash
uv sync
# Configure ANTHROPIC_API_KEY in your environment
```

Requires Python 3.12+ and an Anthropic API key for the LLM.

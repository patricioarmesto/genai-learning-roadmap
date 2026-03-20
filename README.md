# Agentic AI Roadmap

A hands-on roadmap for building agentic AI systems. Each folder (`day1/`, `day2/`, etc.) is a standalone project that focuses on a specific concept, building toward intelligent, autonomous agents.

## Projects overview

| Day | Project | Description |
|-----|---------|-------------|
| — | Capstone | ReAct research CLI with structured JSON report output |
| 1 | CLI chatbot | Chat with memory that persists conversation history |
| 2 | Invoice parser | Extract structured data from raw email text |
| 3 | Tool-calling agent | Weather + calculator assistant that calls tools in real time |
| 4 | ReAct agent | Reasoning + Acting with scratchpad for multi-step problems |
| 5 | Streaming ReAct | Real-time token streaming with visual state display |
| 6 | SSE streaming agent | FastAPI endpoint with Server-Sent Events for web clients |
| 8 | Semantic search | Semantic search engine demonstrating exact-match vs semantic search |

## Concepts covered

| Day | Concept | Key ideas |
|-----|---------|-----------|
| 1 | LLM API basics | Message format, memory, temperature, system prompts |
| 2 | Structured output | JSON schemas, Pydantic validation, retry-on-error |
| 3 | Tool calling | Function definitions, tool-result cycle, agent loop |
| 4 | ReAct prompting | Thought/Action/Action Input loop, reasoning scratchpad |
| 5 | Streaming UX | Token-by-token display, visual state machine, live rendering |
| 6 | SSE streaming | Server-Sent Events, FastAPI, real-time web integration |
| 8 | Text embeddings | Cosine similarity, vector search, minimal RAG |

## Global setup

All days use [Ollama](https://ollama.com/) as the LLM backend.

```bash
# Install Ollama (macOS)
brew install ollama

# Pull models
ollama pull llama3.2          # Day 1, 3, 4, 5, 8
ollama pull minimax-m2:cloud  # Day 2 (better JSON output)
ollama pull qwen2.5:14b       # Day 4, 5, 6 (for reasoning/reasoning models)
ollama pull nomic-embed-text  # Day 8 (for generating vectors)

# Start server (keep running)
ollama serve
```

---

## Day 1 — CLI Chatbot

A chat interface with persistent conversation history.

### What it demonstrates
- **Message format** — `role` + `content` (user, assistant, system)
- **Fake memory** — replaying history so the model sees past turns
- **Temperature** — effect on determinism vs creativity
- **Max tokens** — hard cap on response length
- **System prompt** — invisible instructions that shape behaviour

### Usage

```bash
cd day1
uv sync
python day1_chatbot.py
```

Commands: `quit` | `clear` | `stats`

Temperature experiment:
```bash
python day1_chatbot.py experiment
```

### Temperature guide

| Value | Behavior |
|-------|----------|
| 0.0 | Deterministic, best for structured output |
| 0.2 | Focused, good default for agents |
| 0.7 | Natural, good default for chat |
| 1.2 | Creative, sometimes surprising |
| 2.0 | Chaotic, often incoherent |

### How memory works

History is stored in `chat_history.json`. On each request, the full history is sent to the API. The model has no real memory — it just re-reads past turns. `trim_history()` keeps only the last 20 turns to avoid overflowing the context window.

---

## Day 2 — Invoice Parser

Extracts structured data from raw email text.

### What it demonstrates
Three-layer approach to reliable structured output:

| Layer | What it does |
|-------|--------------|
| 1 | Prompt design that enforces a JSON schema (task + example output) |
| 2 | Output cleaning: strip code fences, fix booleans, remove trailing commas |
| 3 | Pydantic validation with typed fields and constraints |

Plus: **Retry-on-error** — on parse/validation failure, feed the error back to the model so it can self-correct.

### Usage

```bash
cd day2
uv add requests pydantic
python day2_invoice.py
```

Breakage experiment (shows retry/cleaning pipeline):
```bash
python day2_invoice.py breakage
```

### Output schema

```python
{
    vendor: str
    amount_due: float  # positive
    currency: str      # 3-letter code
    due_date: str
    invoice_number: str
    line_items: list[{description, quantity, unit_price, total}]
    is_overdue: bool
    notes: str | None
}
```

---

## Day 3 — Tool Calling Agent

An agent that calls tools in real time to answer questions.

### What it demonstrates
- **Tool schema** — JSON definition describing function parameters
- **Tool-call cycle** — model requests tool → code executes → result fed back
- **Agent loop** — call model repeatedly until it returns text (no more tool calls)
- **Parallel calls** — model can request multiple tools in one response
- **Schema quality** — vague descriptions lead to wrong or missed tool calls

### Usage

```bash
cd day3
uv sync
python day3_tools.py
```

Example questions:
- "What's the weather in Tokyo?"
- "Calculate 15% of 847"
- "Compare weather in London vs Paris"

Other modes:
```bash
python day3_tools.py demo     # preset demo questions
python day3_tools.py parallel # test parallel tool calls
python day3_tools.py schema   # compare good vs bad tool descriptions
```

### Available tools

| Tool | Description |
|------|-------------|
| `get_weather(city, unit)` | Get weather for a city |
| `calculate(expression)` | Evaluate math expressions |
| `get_current_time(timezone)` | Get current date/time |

### Key insight

Include explicit instructions like *"Do NOT calculate in your head — always use this tool"*. This dramatically improves reliability.

---

## Day 4 — ReAct Agent

**ReAct** (Reasoning + Acting) is a prompting technique where an LLM interleaves reasoning traces with tool actions.

### What it demonstrates
- **Thought/Action/Action Input** — explicit reasoning before tool calls
- **Scratchpad** — accumulated history of thoughts, actions, and observations
- **Self-correction** — agent can adjust based on tool feedback
- **Multi-step problems** — breaks complex questions into manageable steps

### The ReAct loop

1. **Thought** — reasoning about what to do next
2. **Action** — which tool to call
3. **Action Input** — the argument for that tool
4. **Observation** — the result returned by the tool

### This implementation

The agent has three tools:
- **search** — keyword search over a mock knowledge base
- **calculate** — safe math expression evaluator
- **get_date** — returns current UTC date/time

The loop terminates when the model outputs `finish` action with the final answer.

### Usage

```bash
cd day4
uv sync
python day4_react.py        # Start interactive REPL
python day4_react.py demo   # Run demo questions
```

---

## Day 5 — Streaming ReAct

Builds on Day 4 with real-time token streaming and visual display states.

### What it demonstrates
- **Streaming tokens** — display model output as it arrives
- **Visual state machine** — color-coded display (thoughts, actions, answers)
- **Think tag stripping** — handles `<think>` blocks from reasoning models
- **Duplicate call detection** — prevents infinite loops
- **Blocked finish** — forces agent to use tools before finishing

### Display states

| State | Color | Description |
|-------|-------|-------------|
| Thought | dim grey | Internal reasoning |
| ▶ tool(arg) | green | Action event |
| ← result | purple | Observation |
| Final answer | bold white | Streamed live |

### Usage

```bash
cd day5
uv sync
python day5_prompts.py        # Interactive REPL
python day5_prompts.py demo   # Run demo questions
python day5_prompts.py chat   # Chat mode with history
```

---

## Day 6 — SSE Streaming Agent

FastAPI server with Server-Sent Events for real-time web streaming.

### What it demonstrates
- **Server-Sent Events (SSE)** — server push to web clients
- **FastAPI integration** — async streaming endpoints
- **Event types** — separate streams for thoughts, tool calls, results, and final answer
- **Web client** — HTML/JS interface for real-time agent visualization

### SSE event types

| Event | Data |
|-------|------|
| `thought` | Streaming reasoning tokens |
| `tool_call` | Which tool is being called |
| `tool_result` | What the tool returned |
| `token` | Final answer tokens (streamed) |
| `done` | Stream complete with final answer |
| `error` | Error message if something fails |

### Usage

```bash
cd day6
uv sync
python day6_streaming.py server  # Start FastAPI server on port 8000
```

Then open `client.html` in a browser or test with curl:
```bash
curl -N "http://localhost:8000/stream?q=How+old+was+Turing+when+he+died"
```

Other modes:
```bash
python day6_streaming.py        # Interactive REPL
python day6_streaming.py demo   # Run demo questions in terminal
```

---

## Day 8 — Text Embeddings and Semantic Search

A semantic search engine built over a small corpus of documents to demonstrate the fundamental differences between exact-match and semantic search.

### What it demonstrates
- **Embeddings** — converting text to vector representations
- **Cosine similarity** — computing distance between vectors
- **Semantic vs exact match** — comparing search techniques side-by-side
- **Minimal RAG** — retrieving context to answer queries

### Usage

```bash
cd day8
uv sync
python day8_embeddings.py
```

Other modes:
```bash
python day8_embeddings.py compare
python day8_embeddings.py similar "your query"
python day8_embeddings.py rag "your query"
```

---

## File layout

```
.
├── README.md  # this file
├── week1_capstone/
│   ├── week1_capstone.py
│   ├── report_*.json  # sample output reports
│   ├── pyproject.toml
│   └── README.md
├── day1/
│   ├── day1_chatbot.py
│   ├── chat_history.json  # created on first run
│   ├── pyproject.toml
│   └── README.md
├── day2/
│   ├── day2_invoice.py
│   ├── pyproject.toml
│   └── README.md
├── day3/
│   ├── day3_tools.py
│   ├── pyproject.toml
│   └── README.md
├── day4/
│   ├── day4_react.py
│   ├── pyproject.toml
│   └── README.md
├── day5/
│   ├── day5_prompts.py
│   ├── pyproject.toml
│   └── README.md
├── day6/
│   ├── day6_streaming.py
│   ├── client.html
│   ├── pyproject.toml
│   └── README.md
├── day8/
│   ├── day8_embeddings.py
│   ├── embeddings_cache.json
│   ├── pyproject.toml
│   └── README.md
```

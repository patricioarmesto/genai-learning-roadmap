# Day 1 — LLM API basics

A CLI chatbot that talks to [Ollama](https://ollama.com/) and persists conversation history to a JSON file. Each run loads prior messages, so the model "remembers" past turns.

Part of the **Agentic AI Roadmap** — this day focuses on core LLM API concepts.

## What it demonstrates

- **Message format** — `role` + `content` (user, assistant, system)
- **Fake memory** — replaying history so the model sees past turns
- **Temperature** — effect on determinism vs creativity
- **Max tokens** — hard cap on response length
- **Stop sequences** — early halt on a given string
- **System prompt** — invisible instructions that shape behaviour

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com/) installed and running
- Model: `llama3.2` (or change `MODEL` in `day1_chatbot.py`)

## Setup

1. **Install dependencies** (uv):

   ```bash
   uv sync
   ```

   Or with pip:

   ```bash
   pip install requests
   ```

2. **Install and run Ollama**:

   ```bash
   ollama pull llama3.2
   ollama serve
   ```

   Keep `ollama serve` running in a separate terminal.

## Usage

**Interactive chat:**

```bash
python day1_chatbot.py
```

- Type your message and press Enter.
- Commands:
  - `quit` — exit
  - `clear` — reset conversation history
  - `stats` — show turns in memory, estimated context size, and settings

**Temperature experiment** (non-interactive):

```bash
python day1_chatbot.py experiment
```

Runs the same prompt at temperatures 0.0, 0.5, 1.0, and 1.5 so you can compare output style.

## Configuration

Edit the constants at the top of `day1_chatbot.py`:

| Constant          | Default | Purpose                                      |
|-------------------|---------|----------------------------------------------|
| `OLLAMA_URL`      | `http://localhost:11434/api/chat` | Ollama API endpoint   |
| `MODEL`           | `llama3.2` | Model name                            |
| `TEMPERATURE`     | `0.7`   | 0 = deterministic, higher = more creative   |
| `MAX_TOKENS`      | `500`   | Max response length                         |
| `STOP_SEQUENCES`  | `[]`    | Stop generation when these strings appear   |
| `SYSTEM_PROMPT`   | (tutor) | Invisible instructions for the model        |
| `HISTORY_FILE`    | `chat_history.json` | Where history is saved              |

**Temperature guide (from the code):**

- `0.0` — deterministic, best for structured output
- `0.2` — focused, good default for agents
- `0.7` — natural, good default for chat
- `1.2` — creative, sometimes surprising
- `2.0` — chaotic, often incoherent

## How “memory” works

History is stored in `chat_history.json` as a list of `{"role": "user"|"assistant", "content": "..."}`. On each request, the full history (plus a system message) is sent to the API. The model has no real memory — it just re-reads the last N turns. `trim_history()` keeps only the last 20 turns to avoid overflowing the context window.

## File layout

```
day1/
├── day1_chatbot.py           # Chatbot + experiment entrypoint
├── chat_history.json # Persisted conversation (created on first run)
├── pyproject.toml    # Project and dependencies (uv/pip)
└── README.md         # This file
```

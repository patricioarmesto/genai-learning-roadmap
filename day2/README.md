# Day 2 — Structured output from LLMs

An **invoice parser** that extracts structured data from raw email text using a local LLM (Ollama). Demonstrates a three-layer approach to reliable structured output plus retry-on-error.

## Concepts

| Layer | What it does |
|-------|----------------|
| **Layer 1** | Prompt design that enforces a JSON schema (task + example output, no markdown) |
| **Layer 2** | Output cleaning: strip code fences, fix `True`/`False`/`None`, remove trailing commas |
| **Layer 3** | Pydantic validation with typed fields and constraints |

Additional behavior:

- **Retry-on-error**: on parse or validation failure, the model’s broken output and the error are fed back into the conversation so it can self-correct (up to `MAX_RETRIES`).
- **Temperature 0.1** for more deterministic JSON.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running

```bash
uv add requests pydantic
ollama pull minimax-m2:cloud
ollama serve
```

## Usage

Run the default demo (processes three sample invoices):

```bash
uv run python day2_invoice.py
```

Run the **breakage experiment** (deliberately triggers bad JSON to show the retry/cleaning pipeline):

```bash
uv run python day2_invoice.py breakage
```

## Output schema

The parser produces an `Invoice` with:

- `vendor`, `amount_due`, `currency`, `due_date`, `invoice_number`
- `line_items`: list of `{description, quantity, unit_price, total}`
- `is_overdue`, `notes`

All validated and typed via Pydantic (e.g. positive amounts, 3-letter currency codes, optional fields).

## Project layout

- `day2_invoice.py` — system prompt, cleaning, LLM call, Pydantic models, sample invoices, and CLI entrypoint.

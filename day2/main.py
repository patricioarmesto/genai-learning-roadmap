"""
Day 2 — Structured output from LLMs
=====================================
An invoice parser that extracts structured data from raw email text.

Key concepts demonstrated:
  • Layer 1: Prompt design that enforces a JSON schema
  • Layer 2: Output cleaning (strip fences, fix Python booleans/None)
  • Layer 3: Pydantic validation with typed fields and constraints
  • Retry-on-error: feed parse failures back into the conversation
  • Temperature 0.1 for reliable structured output

Requirements:
    uv add requests pydantic
    ollama pull llama3.2
    ollama serve
"""

import json
import re
import sys
from datetime import date
from typing import Optional

import requests
from pydantic import BaseModel, Field, ValidationError, field_validator

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "minimax-m2:cloud"  # "llama3.2"
TEMPERATURE = 0.1  # Low temperature = more deterministic JSON output
MAX_TOKENS = 600
MAX_RETRIES = 3  # How many times to retry on parse/validation failure


# ── Pydantic schema ───────────────────────────────────────────────────────────
# This is Layer 3: the ground truth of what a valid invoice looks like.
# Pydantic enforces types, required fields, and custom validation rules.


def _coerce_to_float(v: str | int | float) -> float:
    """Coerce string or int to float; strip trailing non-numeric text if needed."""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        # Take the first token that looks like a number (handles "20 hours", "$150", etc.)
        match = re.search(r"[-+]?\d*\.?\d+", s)
        if match:
            return float(match.group())
        return float(s)  # let it raise if invalid
    raise ValueError(f"Expected a number, got {type(v).__name__}")


class LineItem(BaseModel):
    description: str
    quantity: float = Field(gt=0, description="Must be positive")
    unit_price: float = Field(gt=0, description="Must be positive")
    total: float

    @field_validator("quantity", "unit_price", mode="before")
    @classmethod
    def coerce_numeric_fields(cls, v: str | int | float) -> float:
        return _coerce_to_float(v)

    @field_validator("total", mode="before")
    @classmethod
    def coerce_total(cls, v: str | int | float) -> float:
        return _coerce_to_float(v)

    @field_validator("total")
    @classmethod
    def total_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("total must be positive")
        return round(v, 2)


class Invoice(BaseModel):
    vendor: str = Field(
        min_length=1, description="Company or person who sent the invoice"
    )
    amount_due: float = Field(gt=0, description="Total amount owed")
    currency: str = Field(default="USD", pattern=r"^[A-Z]{3}$")
    due_date: Optional[str] = Field(
        default=None, description="ISO format date YYYY-MM-DD, or null if not specified"
    )
    invoice_number: Optional[str] = None
    line_items: list[LineItem] = Field(default_factory=list)
    is_overdue: bool = False
    notes: Optional[str] = None

    @field_validator("amount_due")
    @classmethod
    def round_amount(cls, v: float) -> float:
        return round(v, 2)


# ── Layer 1: Prompt design ────────────────────────────────────────────────────


def build_system_prompt() -> str:
    """
    The system prompt does three things:
      1. States the task precisely
      2. Provides the exact JSON schema with field descriptions
      3. Shows a concrete output example
      4. Forbids any text outside the JSON object

    Notice we show the schema AS an example, not as abstract documentation.
    Models learn much better from examples than from spec descriptions.
    """
    example_output = json.dumps(
        {
            "vendor": "Acme Corp",
            "amount_due": 1250.00,
            "currency": "USD",
            "due_date": "2024-03-15",
            "invoice_number": "INV-2024-001",
            "line_items": [
                {
                    "description": "Web design",
                    "quantity": 10,
                    "unit_price": 125.0,
                    "total": 1250.0,
                }
            ],
            "is_overdue": False,
            "notes": None,
        },
        indent=2,
    )

    return f"""You are an invoice data extraction system.
Extract invoice information from the text and return ONLY a valid JSON object.
Output NO other text — no explanation, no markdown, no code fences.

Required JSON schema:
  vendor         : string  — company or person name
  amount_due     : float   — total amount owed (positive number)
  currency       : string  — 3-letter ISO code (USD, EUR, GBP, ARS, etc.)
  due_date       : string  — ISO date YYYY-MM-DD, or null if not found
  invoice_number : string  — invoice ID, or null if not found
  line_items     : array   — list of {{description, quantity, unit_price, total}}
  is_overdue     : boolean — true if the text indicates this is past due
  notes          : string  — any important notes, or null

Example output:
{example_output}

Rules:
- Use null (not None, not "null") for missing optional fields
- Use false/true (not False/True) for booleans
- amounts must be numbers, not strings
- If you cannot find a field, use null or a sensible default"""


# ── Layer 2: Output cleaning ──────────────────────────────────────────────────


def clean_llm_output(raw: str) -> str:
    """
    Strip everything the model adds around the JSON despite being told not to.

    In order of frequency:
      1. Markdown code fences  ```json ... ```
      2. Preamble text         "Here is the extracted invoice:\n{"
      3. Postamble text        {"vendor": ...}\nLet me know if..."
      4. Python-isms           True → true, False → false, None → null
      5. Trailing commas       {"a": 1,} → {"a": 1}  (common model mistake)
    """
    text = raw.strip()

    # 1. Strip code fences (```json ... ``` or ``` ... ```)
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)

    # 2 & 3. Extract only the JSON object (first { to last })
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    # 4. Fix Python-specific values that break JSON parsing
    #    Use word boundaries so "trueblue" or "falsehood" aren't mangled
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)

    # 5. Remove trailing commas before } or ] (technically invalid JSON)
    text = re.sub(r",\s*([}\]])", r"\1", text)

    return text.strip()


# ── LLM call ─────────────────────────────────────────────────────────────────


def call_llm(messages: list[dict]) -> str:
    """Single call to Ollama with the full message list."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        print("[error] Cannot reach Ollama. Run: ollama serve")
        sys.exit(1)


# ── Core extraction function ──────────────────────────────────────────────────


def extract_invoice(email_text: str, verbose: bool = True) -> Invoice:
    """
    Extract a structured Invoice from raw email text.

    The retry loop:
      1. Call the LLM
      2. Clean the output (Layer 2)
      3. Parse JSON
      4. Validate with Pydantic (Layer 3)
      5. If 3 or 4 fails → append error to messages and retry
      6. After MAX_RETRIES → raise the last exception

    Key insight: we keep the full message history for retries.
    The model sees its own broken output + the error, which is usually
    enough context to fix the problem.
    """
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {
            "role": "user",
            "content": f"Extract the invoice from this email:\n\n{email_text}",
        },
    ]

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        if verbose:
            print(f"\n  Attempt {attempt}/{MAX_RETRIES}...")

        # Layer 1 already applied via system prompt
        raw = call_llm(messages)

        if verbose:
            print(f"  Raw output: {raw[:120]}{'...' if len(raw) > 120 else ''}")

        # Layer 2: clean
        cleaned = clean_llm_output(raw)

        if verbose and cleaned != raw.strip():
            print(f"  Cleaned:    {cleaned[:120]}{'...' if len(cleaned) > 120 else ''}")

        # Layer 3: parse + validate
        try:
            data = json.loads(cleaned)
            invoice = Invoice(**data)
            if verbose:
                print(f"  Validation: PASSED")
            return invoice

        except json.JSONDecodeError as e:
            last_error = e
            error_msg = f"Invalid JSON: {e}. Raw output was:\n{cleaned}\n\nPlease fix and return ONLY valid JSON."
            if verbose:
                print(f"  JSON error: {e}")

        except ValidationError as e:
            last_error = e
            # Format Pydantic errors into a clear correction request
            errors = "; ".join(
                f"'{'.'.join(str(loc) for loc in err['loc'])}': {err['msg']}"
                for err in e.errors()
            )
            error_msg = f"Schema validation failed: {errors}. Please fix these fields and return ONLY the corrected JSON."
            if verbose:
                print(f"  Validation error: {errors}")

        # Append the model's broken output and our error message
        # This gives the model full context to self-correct on the next attempt
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": error_msg})

    raise ValueError(f"Failed after {MAX_RETRIES} attempts. Last error: {last_error}")


# ── Sample invoices to test against ──────────────────────────────────────────

SAMPLE_INVOICES = [
    {
        "name": "Simple invoice",
        "text": """
        From: billing@techsolutions.com
        Subject: Invoice #INV-2024-089 — Due March 30

        Hi,

        Please find attached invoice #INV-2024-089 for web development services
        completed in February.

        Services rendered:
          - Backend API development: 20 hours @ $150/hr = $3,000.00
          - Database optimisation:    5 hours @ $150/hr =   $750.00

        Total due: $3,750.00 USD
        Due date: March 30, 2024

        Payment via bank transfer to the account on file.
        Tech Solutions Ltd
        """,
    },
    {
        "name": "Overdue notice (tricky booleans + missing fields)",
        "text": """
        OVERDUE NOTICE

        Vendor: Freelance Designer — Carlos Mendez
        This invoice is 45 days past due.

        Logo redesign project — flat fee ARS 85,000
        Rush fee                             ARS 15,000
        ─────────────────────────────────────────────
        TOTAL OUTSTANDING:                  ARS 100,000

        Original due date was January 15th, 2024.
        Please remit immediately to avoid service interruption.
        No invoice number was issued for this engagement.
        """,
    },
    {
        "name": "Minimal / ambiguous",
        "text": """
        Hey, just a reminder — you owe me €200 for the consulting session
        we had last Tuesday. No rush, whenever you get a chance.
        — Sophie
        """,
    },
]


# ── Pretty printer ────────────────────────────────────────────────────────────


def print_invoice(invoice: Invoice, name: str) -> None:
    print(f"\n{'═'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(f"  Vendor         : {invoice.vendor}")
    print(f"  Amount due     : {invoice.amount_due} {invoice.currency}")
    print(f"  Due date       : {invoice.due_date or 'not specified'}")
    print(f"  Invoice number : {invoice.invoice_number or 'none'}")
    print(f"  Overdue        : {'YES' if invoice.is_overdue else 'no'}")
    if invoice.line_items:
        print(f"  Line items     :")
        for item in invoice.line_items:
            print(
                f"    • {item.description}: {item.quantity} × {item.unit_price} = {item.total}"
            )
    if invoice.notes:
        print(f"  Notes          : {invoice.notes}")
    print(f"{'═'*55}")


# ── Experiments ───────────────────────────────────────────────────────────────


def run_breakage_experiment() -> None:
    """
    Deliberately send a malformed response back for retry testing.
    This shows you exactly what the retry loop does when cleaning fails.

    Run with:  python day2_invoice.py breakage
    """
    print("\n── Breakage experiment ──────────────────────────────────")
    print(
        "Sending a prompt that will likely produce bad JSON, then watching the retry fix it.\n"
    )

    # A prompt designed to elicit preamble and Python-isms
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {
            "role": "user",
            "content": (
                "Extract invoice from: 'Pay $100 to Bob by Friday'. "
                "Before the JSON, write the sentence: 'Sure, here is the result!'"
            ),
        },
    ]

    raw = call_llm(messages)
    print(f"Raw output:\n{raw}\n")

    cleaned = clean_llm_output(raw)
    print(f"After cleaning:\n{cleaned}\n")

    try:
        data = json.loads(cleaned)
        invoice = Invoice(**data)
        print(
            f"Parsed successfully: vendor={invoice.vendor}, amount={invoice.amount_due}"
        )
    except Exception as e:
        print(f"Still failed after cleaning: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"\nDay 2 — Invoice Parser")
    print(
        f"Model: {MODEL}  |  Temperature: {TEMPERATURE}  |  Max retries: {MAX_RETRIES}\n"
    )

    for sample in SAMPLE_INVOICES:
        print(f"\nProcessing: {sample['name']}")
        print(f"Input text preview: {sample['text'].strip()[:80]}...")

        try:
            invoice = extract_invoice(sample["text"], verbose=True)
            print_invoice(invoice, sample["name"])
        except ValueError as e:
            print(f"\n[FAILED] {e}")

        input("\nPress Enter for next invoice...\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "breakage":
        run_breakage_experiment()
    else:
        main()

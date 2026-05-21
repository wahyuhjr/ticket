"""
Central configuration for the ticket triage pipeline.
All tunable parameters live here.
"""
import os

# ── LLM Provider ──────────────────────────────────────────────────────────────
LLM_PROVIDER   = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL      = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_API_KEY    = os.getenv("OPENAI_API_KEY", "")
LLM_TEMPERATURE = 0.0           # deterministic outputs
LLM_MAX_TOKENS  = 512

# ── Routing ───────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))

# ── Paths ─────────────────────────────────────────────────────────────────────
DEFAULT_TICKETS_PATH = "data/tickets.json"
DEFAULT_SCHEMA_PATH  = "data/label_schema.json"
DEFAULT_OUTPUT_DIR   = "output"

# ── Retry ─────────────────────────────────────────────────────────────────────
MAX_PARSE_RETRIES = 1           # one retry with stricter prompt on parse failure

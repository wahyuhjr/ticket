"""
Stage: INPUTS_LOADED -> TEXT_PREPROCESSED
Applies deterministic text normalization to each ticket.
"""
import re
from typing import Any


def clean_text(text: str) -> str:
    """
    Deterministic preprocessing:
    1. Strip leading/trailing whitespace
    2. Collapse internal whitespace (tabs, multiple spaces, newlines) to single space
    3. Normalize repeated punctuation (e.g. '!!!' -> '!')
    4. Lowercase
    """
    # Strip
    text = text.strip()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # Normalize repeated punctuation (keep one)
    text = re.sub(r"([!?,.]){2,}", r"\1", text)
    # Lowercase
    text = text.lower()
    return text


def preprocess_ticket(ticket: dict[str, Any]) -> dict[str, Any]:
    """Return a preprocessed record for a single ticket."""
    original = ticket["customer_message"]
    cleaned  = clean_text(original)
    words    = cleaned.split()
    return {
        "ticket_id":     ticket["ticket_id"],
        "original_text": original,
        "cleaned_text":  cleaned,
        "char_count":    len(cleaned),
        "word_count":    len(words),
    }


def preprocess_all(tickets: list[dict]) -> list[dict]:
    """Preprocess all tickets and return list of preprocessed records."""
    return [preprocess_ticket(t) for t in tickets]

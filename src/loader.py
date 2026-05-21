"""
Stage: INIT -> INPUTS_LOADED
Loads tickets.json and label_schema.json from disk and validates structure.
"""
import json
from pathlib import Path


def load_json(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p.resolve()}")
    with open(p, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {p}: {e}") from e


def load_tickets(path):
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"tickets.json must be a JSON array, got {type(data).__name__}")
    required_keys = {"ticket_id", "customer_message"}
    for i, ticket in enumerate(data):
        missing = required_keys - set(ticket.keys())
        if missing:
            raise ValueError(f"Ticket at index {i} missing keys: {missing}")
    return data


def load_schema(path):
    data = load_json(path)
    if "categories" not in data or "urgency_levels" not in data:
        raise ValueError("label_schema.json must contain 'categories' and 'urgency_levels'")
    if not isinstance(data["categories"], list) or not isinstance(data["urgency_levels"], list):
        raise ValueError("'categories' and 'urgency_levels' must be arrays")
    return data
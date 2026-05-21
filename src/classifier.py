"""
Stage: MODEL_PROMPTED -> STRUCTURED_OUTPUT_PARSED
Calls LLM for classification and parses/validates structured JSON output.
"""
import json
import re
from typing import Any

from config import CONFIDENCE_THRESHOLD


REQUIRED_FIELDS = {"category", "urgency", "confidence", "reasoning_summary", "needs_human_review"}


def _extract_json(text: str) -> str:
    """
    Robustness: extract the first JSON object from potentially noisy LLM output.
    Handles markdown code fences and extra surrounding text.
    """
    # Remove markdown fences
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    # Find first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def parse_classification(raw: str, schema: dict) -> tuple[dict | None, str | None]:
    """
    Parse and validate a classification JSON response.
    Returns (parsed_dict, error_message).
    error_message is None on success.
    """
    try:
        extracted = _extract_json(raw)
        data = json.loads(extracted)
    except (json.JSONDecodeError, ValueError) as e:
        return None, f"JSON parse error: {e} | raw={raw[:200]}"

    # Check required fields
    missing = REQUIRED_FIELDS - set(data.keys())
    if missing:
        return None, f"Missing fields: {missing}"

    # Validate category
    if data["category"] not in schema["categories"]:
        return None, f"Invalid category '{data['category']}' not in schema"

    # Validate urgency
    if data["urgency"] not in schema["urgency_levels"]:
        return None, f"Invalid urgency '{data['urgency']}' not in schema"

    # Validate confidence
    try:
        conf = float(data["confidence"])
        if not (0.0 <= conf <= 1.0):
            raise ValueError
        data["confidence"] = conf
    except (ValueError, TypeError):
        return None, f"Invalid confidence value: {data.get('confidence')}"

    # Coerce needs_human_review to bool
    data["needs_human_review"] = bool(data.get("needs_human_review", False))

    return data, None

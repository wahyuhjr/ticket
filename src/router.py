"""
Stage: CONFIDENCE_CHECKED -> ROUTED
Deterministic routing — does NOT rely solely on model's needs_human_review flag.
"""
from config import CONFIDENCE_THRESHOLD


def route_ticket(ticket_id: str, parsed: dict | None, parse_error: str | None) -> dict:
    """
    Routing rules (deterministic):
    1. If parse failed → human_review (error reason)
    2. If confidence < CONFIDENCE_THRESHOLD → human_review
    3. Otherwise → auto_triage
    """
    if parsed is None or parse_error:
        return {
            "ticket_id":      ticket_id,
            "route":          "human_review",
            "confidence":     0.0,
            "routing_reason": f"parse_failure: {parse_error}",
        }

    confidence = parsed["confidence"]
    if confidence < CONFIDENCE_THRESHOLD:
        reason = f"confidence {confidence:.2f} below threshold {CONFIDENCE_THRESHOLD}"
        route  = "human_review"
    else:
        reason = f"confidence {confidence:.2f} meets threshold {CONFIDENCE_THRESHOLD}"
        route  = "auto_triage"

    return {
        "ticket_id":      ticket_id,
        "route":          route,
        "confidence":     confidence,
        "routing_reason": reason,
    }
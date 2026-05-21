"""Lightweight tests for core logic."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessor import clean_text
from src.classifier   import parse_classification
from src.router       import route_ticket
from src.metrics      import compute_metrics


SCHEMA = {
    "categories":    ["billing", "login_access", "technical_issue", "other"],
    "urgency_levels": ["low", "medium", "high"],
}


# ── Preprocessor ─────────────────────────────────────────────────────────────
def test_clean_text_lowercases():
    assert clean_text("Hello WORLD") == "hello world"

def test_clean_text_collapses_whitespace():
    assert clean_text("hello   world") == "hello world"

def test_clean_text_strips():
    assert clean_text("  hi  ") == "hi"

def test_clean_text_repeated_punctuation():
    assert clean_text("help!!!") == "help!"

def test_clean_text_deterministic():
    assert clean_text("Test!!") == clean_text("Test!!")


# ── Parser ────────────────────────────────────────────────────────────────────
def test_parse_valid_classification():
    raw = '{"category":"billing","urgency":"high","confidence":0.9,"reasoning_summary":"x","needs_human_review":false}'
    parsed, error = parse_classification(raw, SCHEMA)
    assert error is None
    assert parsed["category"] == "billing"
    assert parsed["confidence"] == 0.9

def test_parse_invalid_category_rejected():
    raw = '{"category":"FAKE","urgency":"high","confidence":0.9,"reasoning_summary":"x","needs_human_review":false}'
    parsed, error = parse_classification(raw, SCHEMA)
    assert parsed is None
    assert "Invalid category" in error

def test_parse_malformed_json():
    raw = "not json at all"
    parsed, error = parse_classification(raw, SCHEMA)
    assert parsed is None
    assert error is not None

def test_parse_extracts_json_from_noisy_text():
    raw = 'Sure! Here you go: {"category":"billing","urgency":"low","confidence":0.7,"reasoning_summary":"x","needs_human_review":false} Hope that helps!'
    parsed, error = parse_classification(raw, SCHEMA)
    assert parsed is not None
    assert error is None


# ── Router ────────────────────────────────────────────────────────────────────
def test_route_above_threshold():
    parsed = {"confidence": 0.9, "category": "billing", "urgency": "high",
              "reasoning_summary": "x", "needs_human_review": False}
    result = route_ticket("T1", parsed, None)
    assert result["route"] == "auto_triage"

def test_route_below_threshold():
    parsed = {"confidence": 0.4, "category": "billing", "urgency": "high",
              "reasoning_summary": "x", "needs_human_review": False}
    result = route_ticket("T1", parsed, None)
    assert result["route"] == "human_review"

def test_route_exactly_at_threshold():
    # Exactly at threshold → auto_triage (>= threshold)
    parsed = {"confidence": 0.65, "category": "billing", "urgency": "high",
              "reasoning_summary": "x", "needs_human_review": False}
    result = route_ticket("T1", parsed, None)
    assert result["route"] == "auto_triage"

def test_route_parse_failure():
    result = route_ticket("T1", None, "JSON parse error")
    assert result["route"] == "human_review"
    assert "parse_failure" in result["routing_reason"]


# ── Metrics ───────────────────────────────────────────────────────────────────
def test_metrics_perfect_accuracy():
    tickets = [
        {"ticket_id": "T1", "customer_message": "", "expected_category": "billing",  "expected_urgency": "high"},
        {"ticket_id": "T2", "customer_message": "", "expected_category": "login_access", "expected_urgency": "low"},
    ]
    results = [
        {"ticket_id": "T1", "predicted_category": "billing",      "predicted_urgency": "high", "confidence": 0.9, "route": "auto_triage"},
        {"ticket_id": "T2", "predicted_category": "login_access", "predicted_urgency": "low",  "confidence": 0.8, "route": "auto_triage"},
    ]
    report, _, _ = compute_metrics(results, tickets)
    assert report["category_accuracy"] == 1.0
    assert report["urgency_accuracy"]  == 1.0

def test_metrics_partial_accuracy():
    tickets = [
        {"ticket_id": "T1", "customer_message": "", "expected_category": "billing", "expected_urgency": "high"},
        {"ticket_id": "T2", "customer_message": "", "expected_category": "billing", "expected_urgency": "high"},
    ]
    results = [
        {"ticket_id": "T1", "predicted_category": "billing", "predicted_urgency": "high",   "confidence": 0.9, "route": "auto_triage"},
        {"ticket_id": "T2", "predicted_category": "other",   "predicted_urgency": "medium", "confidence": 0.5, "route": "human_review"},
    ]
    report, _, _ = compute_metrics(results, tickets)
    assert report["category_accuracy"] == 0.5
    assert report["human_review_count"] == 1


if __name__ == "__main__":
    # Simple runner (no pytest needed)
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
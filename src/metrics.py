"""
Stage: EVALUATION_COMPUTED
Computes accuracy metrics using expected labels from tickets.json.
"""
from collections import defaultdict


def compute_metrics(triage_results: list[dict], tickets: list[dict]) -> tuple[dict, list[dict]]:
    """
    Returns (evaluation_report, prediction_comparison).
    """
    ticket_map = {t["ticket_id"]: t for t in tickets}
    total = len(triage_results)
    cat_correct = 0
    urg_correct = 0
    human_review_count = 0
    parse_failures = 0
    comparisons = []

    cat_confusion = defaultdict(lambda: defaultdict(int))

    for result in triage_results:
        tid = result["ticket_id"]
        original = ticket_map.get(tid, {})
        expected_cat = original.get("expected_category")
        expected_urg = original.get("expected_urgency")
        predicted_cat = result.get("predicted_category")
        predicted_urg = result.get("predicted_urgency")

        if result["route"] == "human_review":
            human_review_count += 1
        if result.get("predicted_category") is None:
            parse_failures += 1

        cat_match = (expected_cat == predicted_cat) if expected_cat else None
        urg_match = (expected_urg == predicted_urg) if expected_urg else None

        if cat_match:
            cat_correct += 1
        if urg_match:
            urg_correct += 1

        if expected_cat and predicted_cat:
            cat_confusion[expected_cat][predicted_cat] += 1

        comparisons.append({
            "ticket_id":          tid,
            "expected_category":  expected_cat,
            "predicted_category": predicted_cat,
            "category_match":     cat_match,
            "expected_urgency":   expected_urg,
            "predicted_urgency":  predicted_urg,
            "urgency_match":      urg_match,
            "route":              result["route"],
            "confidence":         result.get("confidence", 0.0),
        })

    with_labels = sum(1 for t in tickets if t.get("expected_category"))
    cat_accuracy = round(cat_correct / with_labels, 4) if with_labels else None
    urg_accuracy = round(urg_correct / with_labels, 4) if with_labels else None

    report = {
        "total_tickets":        total,
        "category_accuracy":    cat_accuracy,
        "urgency_accuracy":     urg_accuracy,
        "human_review_count":   human_review_count,
        "auto_triage_count":    total - human_review_count,
        "parse_failures":       parse_failures,
        "human_review_pct":     round(human_review_count / total * 100, 1) if total else 0,
    }

    confusion_summary = []
    for expected, preds in cat_confusion.items():
        for predicted, count in preds.items():
            confusion_summary.append({
                "expected":  expected,
                "predicted": predicted,
                "count":     count,
                "correct":   expected == predicted,
            })

    return report, comparisons, confusion_summary
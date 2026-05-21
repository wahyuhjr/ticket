"""
Validation command — checks all required artifacts and data integrity.
Run: python validate.py
"""
import json
import sys
from pathlib import Path

OUTPUT_DIR = Path("output")
REQUIRED_FILES = [
    "preprocessed_tickets.json",
    "routing_decisions.json",
    "triage_results.json",
    "prediction_comparison.json",
    "evaluation_report.json",
    "llm_calls.jsonl",
]

errors = []
warnings = []


def check(condition: bool, msg: str, is_error: bool = True):
    if not condition:
        (errors if is_error else warnings).append(msg)


# 1. Required files exist
print("Checking required artifacts...")
for fname in REQUIRED_FILES:
    fpath = OUTPUT_DIR / fname
    check(fpath.exists(), f"MISSING: {fpath}")

# 2. JSON files are valid + load data
data = {}
for fname in REQUIRED_FILES:
    fpath = OUTPUT_DIR / fname
    if not fpath.exists():
        continue
    try:
        if fname.endswith(".jsonl"):
            records = [json.loads(l) for l in fpath.read_text().strip().splitlines() if l.strip()]
            data[fname] = records
        else:
            data[fname] = json.loads(fpath.read_text())
    except Exception as e:
        errors.append(f"INVALID JSON in {fname}: {e}")

# 3. Load schema for label validation
schema_path = Path("data/label_schema.json")
schema = json.loads(schema_path.read_text()) if schema_path.exists() else None
allowed_cats  = set(schema["categories"])      if schema else set()
allowed_urgs  = set(schema["urgency_levels"])  if schema else set()

# 4. All tickets have routing decision
print("Checking routing completeness...")
routing = {r["ticket_id"]: r for r in data.get("routing_decisions.json", [])}
triage  = {r["ticket_id"]: r for r in data.get("triage_results.json",   [])}
for tid in triage:
    check(tid in routing, f"Ticket {tid} has no routing decision")

# 5. auto_triage tickets have customer_reply
print("Checking reply completeness...")
for tid, result in triage.items():
    if result["route"] == "auto_triage":
        check(result.get("customer_reply") not in (None, ""),
              f"Ticket {tid} is auto_triage but has no customer_reply")
    if result["route"] == "human_review":
        check(result.get("internal_note") not in (None, ""),
              f"Ticket {tid} is human_review but has no internal_note")

# 6. Predicted labels belong to schema
print("Checking label schema compliance...")
if schema:
    for tid, result in triage.items():
        cat = result.get("predicted_category")
        urg = result.get("predicted_urgency")
        if cat:
            check(cat in allowed_cats, f"Ticket {tid} has invalid category '{cat}'")
        if urg:
            check(urg in allowed_urgs, f"Ticket {tid} has invalid urgency '{urg}'")

# 7. Evaluation metrics present
print("Checking evaluation report...")
eval_report = data.get("evaluation_report.json", {})
for key in ["total_tickets", "category_accuracy", "urgency_accuracy", "human_review_count"]:
    check(key in eval_report, f"evaluation_report.json missing key: '{key}'")

# ── Results ────────────────────────────────────────────────────────────────────
print("\n" + "="*50)
if errors:
    print(f"VALIDATION FAILED — {len(errors)} error(s):")
    for e in errors:
        print(f"  ✗ {e}")
    sys.exit(1)
else:
    print(f"VALIDATION PASSED ✓")
    print(f"  {len(triage)} tickets validated, {len(routing)} routing decisions checked")
    if warnings:
        for w in warnings:
            print(f"  ⚠ {w}")
    sys.exit(0)
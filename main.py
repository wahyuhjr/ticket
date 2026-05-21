"""
Entry point — runs the full triage pipeline.
Usage: python main.py --tickets data/tickets.json --schema data/label_schema.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.loader       import load_tickets, load_schema
from src.preprocessor import preprocess_all
from src.classifier   import parse_classification
from src.router       import route_ticket
from src.metrics      import compute_metrics
from src.llm_client   import get_llm_client, make_log_entry
import config


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved {path}")


def save_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  ✓ Saved {path}")


def run_pipeline(tickets_path: str, schema_path: str, output_dir: str):
    out = Path(output_dir)
    llm = get_llm_client()
    llm_logs = []

    # ── STAGE: INPUTS_LOADED ──────────────────────────────────────────────────
    print("\n[1/7] Loading inputs...")
    tickets = load_tickets(tickets_path)
    schema  = load_schema(schema_path)
    print(f"  Loaded {len(tickets)} tickets, schema has "
          f"{len(schema['categories'])} categories")

    # ── STAGE: TEXT_PREPROCESSED ──────────────────────────────────────────────
    print("\n[2/7] Preprocessing tickets...")
    preprocessed = preprocess_all(tickets)
    save_json(preprocessed, out / "preprocessed_tickets.json")
    preprocessed_map = {p["ticket_id"]: p for p in preprocessed}

    # ── STAGE: MODEL_PROMPTED + STRUCTURED_OUTPUT_PARSED ─────────────────────
    print("\n[3/7] Classifying tickets (LLM calls)...")
    raw_outputs   = {}
    parsed_map    = {}
    error_map     = {}

    for ticket in tickets:
        tid   = ticket["ticket_id"]
        clean = preprocessed_map[tid]["cleaned_text"]

        # First attempt
        raw, model = llm.classify(tid, clean, schema)
        raw_outputs[tid] = raw
        parsed, error = parse_classification(raw, schema)

        # ── STAGE: SHOULD ATTEMPT #8 — one retry on parse failure ────────────
        if parsed is None:
            print(f"  ⚠ Parse failed for {tid}, retrying with strict prompt...")
            if hasattr(llm, "classify_strict"):
                raw2, model = llm.classify_strict(tid, clean, schema)
                raw_outputs[tid] = raw2
                parsed, error = parse_classification(raw2, schema)
                if parsed is None:
                    print(f"  ✗ Retry also failed for {tid}: {error}")
            else:
                # mock client — keep original error
                pass

        parsed_map[tid] = parsed
        error_map[tid]  = error

        # Log the LLM call
        llm_logs.append(make_log_entry(
            stage="classification", ticket_id=tid,
            provider=config.LLM_PROVIDER, model=model,
            prompt_text=clean,
            output_artifact=str(out / "triage_results.json"),
        ))

    # Save raw model outputs
    save_json(raw_outputs, out / "raw_model_outputs.json")

    # ── STAGE: CONFIDENCE_CHECKED + ROUTED ────────────────────────────────────
    print("\n[4/7] Routing tickets...")
    routing_decisions = []
    for ticket in tickets:
        tid     = ticket["ticket_id"]
        parsed  = parsed_map.get(tid)
        error   = error_map.get(tid)
        decision = route_ticket(tid, parsed, error)
        routing_decisions.append(decision)

    save_json(routing_decisions, out / "routing_decisions.json")
    routing_map = {r["ticket_id"]: r for r in routing_decisions}

    # ── STAGE: RESPONSE_GENERATED ────────────────────────────────────────────
    print("\n[5/7] Generating replies and internal notes...")
    triage_results = []
    for ticket in tickets:
        tid      = ticket["ticket_id"]
        parsed   = parsed_map.get(tid)
        decision = routing_map[tid]
        route    = decision["route"]
        is_hr    = (route == "human_review")

        cat = parsed["category"]  if parsed else "other"
        urg = parsed["urgency"]   if parsed else "low"
        clean = preprocessed_map[tid]["cleaned_text"]

        reply_text, model = llm.generate_reply(
            ticket_id=tid, category=cat, urgency=urg,
            cleaned_text=clean, is_human_review=is_hr
        )

        llm_logs.append(make_log_entry(
            stage="reply_generation", ticket_id=tid,
            provider=config.LLM_PROVIDER, model=model,
            prompt_text=clean,
            output_artifact=str(out / "triage_results.json"),
        ))

        triage_results.append({
            "ticket_id":          tid,
            "predicted_category": cat,
            "predicted_urgency":  urg,
            "confidence":         decision["confidence"],
            "route":              route,
            "customer_reply":     reply_text if not is_hr else None,
            "internal_note":      reply_text if is_hr     else None,
        })

    save_json(triage_results, out / "triage_results.json")

    # ── STAGE: RESULTS_SAVED ─────────────────────────────────────────────────
    print("\n[6/7] Saving LLM call log...")
    save_jsonl(llm_logs, out / "llm_calls.jsonl")

    # ── STAGE: EVALUATION_COMPUTED ───────────────────────────────────────────
    print("\n[7/7] Computing evaluation metrics...")
    eval_report, comparisons, confusion = compute_metrics(triage_results, tickets)
    save_json(eval_report,   out / "evaluation_report.json")
    save_json(comparisons,   out / "prediction_comparison.json")
    save_json(confusion,     out / "confusion_summary.json")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("PIPELINE COMPLETE — Summary")
    print("="*50)
    print(f"  Total tickets:       {eval_report['total_tickets']}")
    print(f"  Category accuracy:   {eval_report['category_accuracy']:.0%}")
    print(f"  Urgency accuracy:    {eval_report['urgency_accuracy']:.0%}")
    print(f"  Auto-triaged:        {eval_report['auto_triage_count']}")
    print(f"  Human review:        {eval_report['human_review_count']} "
          f"({eval_report['human_review_pct']}%)")
    print(f"  Parse failures:      {eval_report['parse_failures']}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="AI Ticket Triage Pipeline")
    parser.add_argument("--tickets", default=config.DEFAULT_TICKETS_PATH,
                        help="Path to tickets.json")
    parser.add_argument("--schema",  default=config.DEFAULT_SCHEMA_PATH,
                        help="Path to label_schema.json")
    parser.add_argument("--output",  default=config.DEFAULT_OUTPUT_DIR,
                        help="Output directory (default: output/)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Confidence threshold override (default from config)")
    args = parser.parse_args()

    if args.threshold is not None:
        config.CONFIDENCE_THRESHOLD = args.threshold

    run_pipeline(args.tickets, args.schema, args.output)


if __name__ == "__main__":
    main()
# AI Ticket Triage Pipeline

A replayable, deterministic pipeline that classifies customer support tickets,
generates suggested replies, and routes low-confidence cases to human review.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI API key (optional — uses mock client if not set)
export OPENAI_API_KEY=sk-...

# 3. Run the pipeline
python3 main.py

# With custom paths
python3 main.py --tickets data/tickets.json --schema data/label_schema.json --output output/

# Override confidence threshold
python3 main.py --threshold 0.75

# 4. Validate outputs
python3 validate.py

# 5. Run tests
python3 tests/test_core.py
# or with pytest:
pytest tests/
```

## Pipeline Stages

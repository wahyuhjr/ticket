"""
LLM client wrapper. Falls back to deterministic mock when no API key is set.
"""
import hashlib
import json
import os
from datetime import datetime, timezone

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class MockLLMClient:
    """Deterministic mock — used when OPENAI_API_KEY is not set."""

    _CLASSIFICATIONS = {
        "t1": {"category": "login_access",    "urgency": "high",   "confidence": 0.95,
               "reasoning_summary": "User reports persistent login failure after password reset.",
               "needs_human_review": False},
        "t2": {"category": "billing",         "urgency": "high",   "confidence": 0.92,
               "reasoning_summary": "Withdrawal not reflected after two days.",
               "needs_human_review": False},
        "t3": {"category": "technical_issue", "urgency": "medium", "confidence": 0.88,
               "reasoning_summary": "Performance and UI freeze on mobile platform.",
               "needs_human_review": False},
        "t4": {"category": "verification",    "urgency": "medium", "confidence": 0.85,
               "reasoning_summary": "Document requirement query after address change.",
               "needs_human_review": False},
        "t5": {"category": "account_closure", "urgency": "medium", "confidence": 0.97,
               "reasoning_summary": "Explicit account closure request.",
               "needs_human_review": False},
        "t6": {"category": "feature_request", "urgency": "low",    "confidence": 0.91,
               "reasoning_summary": "Customer suggesting a UI enhancement (dark mode).",
               "needs_human_review": False},
    }

    _REPLIES = {
        "t1": "Thank you for reaching out. We're sorry to hear you're having trouble logging in after your password reset. Our team will investigate this issue promptly. Please try clearing your browser cache or using a different browser in the meantime.",
        "t2": "Thank you for contacting us regarding your withdrawal. We understand this is urgent and will investigate the delay with our payments team immediately. You can expect an update within 24 hours.",
        "t3": "Thank you for reporting these performance issues on our mobile site. Our technical team has been notified and will investigate the slowness and chart freeze behavior. We appreciate your patience.",
        "t4": "Thank you for your question about account verification. To verify your account with a recent address change, you will typically need a government-issued ID and a recent document showing your new address. Please check our help center for the full list of accepted documents.",
        "t5": "We have received your request to permanently close your account. Our account management team will process this and reach out to confirm the closure. Please allow 1-2 business days for this process.",
        "t6": "Thank you for the suggestion! We have logged your request for dark mode in our dashboard. Our product team reviews feature requests regularly and we will keep you updated on any developments.",
    }

    def classify(self, ticket_id, cleaned_text, schema):
        key = ticket_id.lower()
        result = self._CLASSIFICATIONS.get(key, {
            "category": "other", "urgency": "low", "confidence": 0.50,
            "reasoning_summary": "Could not confidently classify this ticket.",
            "needs_human_review": True,
        })
        return json.dumps(result), "mock-model"

    def generate_reply(self, ticket_id, category, urgency, cleaned_text, is_human_review):
        key = ticket_id.lower()
        if is_human_review:
            note = (
                f"INTERNAL NOTE: Ticket {ticket_id} routed to human review due to low "
                f"classification confidence. Category hint: {category}, urgency hint: {urgency}. "
                f"Please review the original message and respond manually."
            )
            return note, "mock-model"
        reply = self._REPLIES.get(key,
            "Thank you for contacting us. A support agent will review your request and respond shortly.")
        return reply, "mock-model"


class OpenAILLMClient:
    """Real OpenAI client."""

    def __init__(self):
        self._client = OpenAI(api_key=config.LLM_API_KEY)
        self._model  = config.LLM_MODEL

    def _chat(self, messages):
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )
        return resp.choices[0].message.content.strip(), self._model

    def classify(self, ticket_id, cleaned_text, schema):
        categories = schema["categories"]
        urgencies  = schema["urgency_levels"]
        system_msg = (
            "You are a customer support ticket classifier. "
            "Respond with valid JSON only — no markdown, no explanation outside the JSON."
        )
        user_msg = (
            f'Classify the following customer support ticket.\n\n'
            f'Ticket text: "{cleaned_text}"\n\n'
            f'Allowed categories: {json.dumps(categories)}\n'
            f'Allowed urgency levels: {json.dumps(urgencies)}\n\n'
            f'Return a JSON object with exactly these fields:\n'
            f'{{"category": "<one of allowed categories>", '
            f'"urgency": "<one of allowed urgency levels>", '
            f'"confidence": <float 0.0-1.0>, '
            f'"reasoning_summary": "<1-2 sentence explanation>", '
            f'"needs_human_review": <true if confidence < 0.65 else false>}}'
        )
        return self._chat([
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ])

    def classify_strict(self, ticket_id, cleaned_text, schema):
        """Stricter retry prompt — called when first parse fails."""
        categories = schema["categories"]
        urgencies  = schema["urgency_levels"]
        user_msg = (
            f'Ticket: "{cleaned_text}"\n\n'
            f'You MUST respond with ONLY a raw JSON object. No markdown, no extra text.\n'
            f'Required fields: category (one of {categories}), '
            f'urgency (one of {urgencies}), confidence (0.0-1.0), '
            f'reasoning_summary (string), needs_human_review (boolean).\n'
            f'Example: {{"category":"billing","urgency":"high","confidence":0.9,'
            f'"reasoning_summary":"...","needs_human_review":false}}'
        )
        return self._chat([{"role": "user", "content": user_msg}])

    def generate_reply(self, ticket_id, category, urgency, cleaned_text, is_human_review):
        if is_human_review:
            prompt = (
                f"Write a brief INTERNAL escalation note (2-3 sentences) for a support agent.\n"
                f"Ticket ID: {ticket_id}\nCategory: {category}\nUrgency: {urgency}\n"
                f"Customer message: \"{cleaned_text}\"\n"
                f"Explain why this was escalated and what the agent should focus on."
            )
        else:
            prompt = (
                f"Write a professional customer support reply (2-4 sentences).\n"
                f"Category: {category}, Urgency: {urgency}\n"
                f"Customer message: \"{cleaned_text}\"\n\n"
                f"Rules: acknowledge the issue, do NOT make up account-specific facts, "
                f"do NOT promise specific actions not implied by the ticket, be empathetic.\n"
                f"Return only the reply text."
            )
        return self._chat([{"role": "user", "content": prompt}])


def get_llm_client():
    """Returns real OpenAI client if API key present, mock otherwise."""
    if _OPENAI_AVAILABLE and config.LLM_API_KEY:
        return OpenAILLMClient()
    return MockLLMClient()


def hash_prompt(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_log_entry(stage, ticket_id, provider, model, prompt_text, output_artifact):
    return {
        "stage":           stage,
        "ticket_id":       ticket_id,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "provider":        provider,
        "model":           model,
        "prompt_hash":     hash_prompt(prompt_text),
        "output_artifact": output_artifact,
    }
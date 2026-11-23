"""Safety & Guardrails Module for KhareedaarAI.

Enforces:
1. Rejection of off-topic requests (coding, medical diagnosis, politics, homework).
2. Defense against prompt injections & jailbreaks (system prompt extraction, price override, DAN).
3. Human confirmation before risky financial actions (placing orders, spending money).
"""
import re
from typing import Tuple

# Common prompt injection patterns
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
    r"forget\s+(all\s+)?(your\s+)?(instructions|rules|system\s+prompt)",
    r"you\s+are\s+now\s+(in\s+developer\s+mode|dan|an\s+unfiltered|an\s+ai\s+without\s+rules)",
    r"(reveal|print|show|output|leak)\s+(your\s+)?(system\s+prompt|developer\s+instructions|system\s+instructions)",
    r"set\s+(the\s+)?(price|total|cost)\s+to\s+(0|zero|\$0|free)",
    r"grant\s+(me\s+)?(admin|root|seller)\s+(access|privileges)",
    r"(bypass|override)\s+(security|safety|rules|filters)",
    r"show\s+(all|other)\s+(users'|customers'|client)\s+(data|orders|credit\s+cards|information)"
]

# Off-topic request patterns
OFF_TOPIC_PATTERNS = [
    r"\b(write|generate|debug|fix|create|show)\b.*\b(python|java|javascript|c\+\+|sql|code|script|html|css|algorithm|function|program|app)\b",
    r"\b(solve|calculate)\b.*\b(calculus|equation|derivative|integral|algebra|homework|math)\b",
    r"\b(differential\s+equation|calculus|homework\s+assignment|write\s+(an|my)\s+essay)\b",
    r"\b(diagnose|cure|prescribe|medication|medicine|pills|treatment|doctor|medical|symptom|blood\s+pressure|diabetes|cancer)\b",
    r"\b(who\s+should\s+i\s+vote\s+for|presidential\s+election|political\s+party|democrat|republican)\b",
    r"\b(what\s+is\s+the\s+meaning\s+of|define|definition\s+of|explain\s+the\s+concept\s+of)\b",
    r"\b(translate|write\s+a\s+poem|tell\s+a\s+joke|write\s+a\s+story|summarize\s+this\s+article)\b"
]

SAFE_STORE_KEYWORDS = [
    "coffee", "bean", "roast", "espresso", "latte", "pour-over", "dripper", "kettle",
    "grinder", "mug", "tumbler", "tee", "shirt", "tote", "bag", "order", "cart",
    "buy", "price", "catalog", "product", "discount", "coupon", "shipping", "size",
    "recommend", "stock", "track", "status", "store", "shop", "preference", "hello", "hi", "hey",
    "return", "refund", "hours", "rustic", "item", "purchase", "cost"
]


def check_prompt_injection(message: str) -> Tuple[bool, str]:
    """Check if message contains a prompt injection or jailbreak attempt."""
    msg = message.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, msg):
            return True, "I cannot fulfill this request. I am programmed to assist only with shopping, products, and orders at The Rustic Stop."
    return False, ""


def check_off_topic(message: str) -> Tuple[bool, str]:
    """Check if message is completely off-topic from storefront activities."""
    msg = message.lower()
    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, msg):
            return True, (
                "I am KhareedaarAI, the dedicated shopping and sales assistant for The Rustic Stop. "
                "That request is outside of my capabilities, as I only assist with our products, recommendations, "
                "orders, and store services. Please let me know if you would like help browsing our catalog!"
            )

    # If message has non-trivial length and lacks any store-related keywords or greetings
    words = set(re.findall(r"\w+", msg))
    if len(words) >= 6 and not any(kw in msg for kw in SAFE_STORE_KEYWORDS):
        return True, (
            "I apologize, but this falls outside my scope of expertise. As the sales assistant for The Rustic Stop, "
            "I can assist you with product recommendations, order tracking, and store policies. "
            "How can I help you with our store today?"
        )

    return False, ""


def check_risky_action_confirmation(message: str, is_order_intent: bool, already_confirmed: bool) -> Tuple[bool, str]:
    """Ensure financial transactions require explicit customer confirmation before execution."""
    if not is_order_intent:
        return False, ""

    msg = message.lower()
    confirm_words = ["yes", "confirm", "proceed", "place order", "i confirm", "go ahead", "sounds good", "do it"]

    if already_confirmed or any(w in msg for w in confirm_words):
        return False, ""

    return True, "Please confirm: Would you like me to proceed with placing this order and charging your account?"

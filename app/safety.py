"""Safety & Guardrails Module for ShopBot.

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
    r"\b(write|generate|debug|fix)\s+(a\s+)?(python|java|javascript|c\+\+|sql|code|script|html|css)\b",
    r"\b(solve|calculate)\b.*\b(calculus|equation|derivative|integral|algebra|homework|math)\b",
    r"\b(differential\s+equation|calculus|homework\s+assignment|write\s+(an|my)\s+essay)\b",
    r"\b(diagnose|cure|prescribe|medication|medicine|pills|treatment|doctor|medical|symptom|blood\s+pressure|diabetes|cancer)\b",
    r"\b(who\s+should\s+i\s+vote\s+for|presidential\s+election|political\s+party|democrat|republican)\b"
]

SAFE_STORE_KEYWORDS = [
    "coffee", "bean", "roast", "espresso", "latte", "pour-over", "dripper", "kettle",
    "grinder", "mug", "tumbler", "tee", "shirt", "tote", "bag", "order", "cart",
    "buy", "price", "catalog", "product", "discount", "coupon", "shipping", "size",
    "recommend", "stock", "track", "status", "store", "shop", "preference", "hello", "hi"
]


def check_prompt_injection(message: str) -> Tuple[bool, str]:
    """Check if message contains a prompt injection or jailbreak attempt."""
    msg = message.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, msg):
            return True, "I cannot fulfill this request. I am programmed to assist only with shopping, products, and orders at Artisan & Oak."
    return False, ""


def check_off_topic(message: str) -> Tuple[bool, str]:
    """Check if message is completely off-topic from storefront activities."""
    msg = message.lower()
    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, msg):
            return True, "I am ShopBot, your sales assistant for Artisan & Oak. I can only assist with product recommendations, catalog browsing, orders, and store policies."

    # If message is long and lacks any store-related keywords, flag as potentially off-topic
    words = set(re.findall(r"\w+", msg))
    if len(words) > 12 and not any(kw in msg for kw in SAFE_STORE_KEYWORDS):
        # Allow pleasantries or general questions if short, but reject completely unrelated long queries
        return True, "I specialize in our lifestyle and coffee catalog. How can I help you find products or manage your orders today?"

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

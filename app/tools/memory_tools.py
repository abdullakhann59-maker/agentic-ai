"""Long-term memory stored in the vector DB."""
from __future__ import annotations

import re

from langchain_core.tools import tool

from app.knowledge import get_kb

SECRET = re.compile(r"(password|passcode|otp|cvv|pin)\b|\b\d{12,19}\b", re.I)


@tool
def memory_save(fact: str) -> str:
    """Remember a fact for future conversations (only when the user asks you to remember something)."""
    if SECRET.search(fact):
        return "Refused: passwords, OTPs, PINs and card numbers are never stored in memory."
    get_kb().memory_save(fact.strip())
    return f"Saved to memory: {fact.strip()}"


@tool
def memory_search(query: str) -> str:
    """Search facts the user asked you to remember earlier."""
    hits = get_kb().memory_search(query)
    hits = [h for h in hits if h.score > 0.3]
    return "\n".join(f"- {h.text} (match {h.score:.2f})" for h in hits) or "Nothing relevant in memory."

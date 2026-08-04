import re
from typing import List, Dict, Any, Sequence
from config.settings import *
from datetime import datetime, timezone

def normalize_query(query: str) -> str:
    """Applies strict high-performance regex sequence normalization patterns."""
    text = query.lower()
    for pattern in STRUCTURAL_PHRASES:
        text = pattern.sub(" ", text)
    for pattern in QUERY_FILLERS:
        text = pattern.sub(" ", text)
    
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def detect_search_domain(query: str) -> str:
    """Word-safe search intent classifier executing via exact token boundaries."""
    # Tokenize input clean query strictly around alphanumeric boundaries
    tokens = set(TOKEN_REGEX.findall(query.lower()))
    
    if tokens & ALLERGY_KEYWORDS:
        return "clinical"
    if tokens & NUTRITION_KEYWORDS:
        return "nutrition"
    return "clinical"

def build_metadata(source, url, condition, section, trust_score, risk, category) -> Dict[str, Any]:
    """Uniform structural telemetry metadata dictionary factory pipeline."""
    return {
        "source": source,
        "condition": condition,
        "section_header": section,
        "category": category,
        "risk_level": risk,
        "url": url,
        "trust_score": trust_score,
        "content_type": "web_scraped",
        "cached_from_web": True,
        "retrieved_at": datetime.now(timezone.utc).isoformat()
    }


def is_generic_vision_query(query: str, generic_phrases: Sequence[str]) -> bool:
    """
    Evaluates whether a user query is a generic/vague vision request,
    including command-style prompts like 'analyze this image'.
    """
    if not query:
        return True

    # 1. Clean and normalize whitespace/punctuation
    cleaned = re.sub(r"[^\w\s]", " ", query.lower()).strip()
    cleaned = " ".join(cleaned.split())

    if not cleaned:
        return True

    # 2. Exact match against settings list
    normalized_phrases = {
        re.sub(r"[^\w\s]", " ", p.lower()).strip() 
        for p in generic_phrases
    }
    if cleaned in normalized_phrases:
        return True

    # 3. Action / Command-style generic prefixes (e.g., "analyze the image", "explain what to do")
    generic_action_prefixes = (
        "analyze",
        "examine",
        "evaluate",
        "assess",
        "check",
        "look at",
        "take a look at",
        "explain what to do",
        "tell me about this",
        "what to do for this",
        "what should i do for this",
        "what do you see in this",
    )

    # Generic visual target words
    generic_visual_nouns = (
        "image", "picture", "photo", "pic", "asset", "file", 
        "rash", "rashes", "skin", "spot", "spots", "condition"
    )

    # If the prompt starts with an action verb (e.g. "analyze the image...")
    if any(cleaned.startswith(prefix) for prefix in generic_action_prefixes):
        # And contains a generic noun or is short (<= 12 words)
        words = cleaned.split()
        if len(words) <= 12 or any(noun in cleaned for noun in generic_visual_nouns):
            return True

    return False
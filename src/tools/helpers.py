import re
from typing import List, Dict, Any, Sequence
from config.settings import *
from datetime import datetime, timezone

# Updated normalize_query in src/tools/helpers.py

def normalize_query(query: str) -> str:
    """Applies strict high-performance regex sequence normalization patterns."""
    if not query:
        return ""

    text = query.lower()

    # Safely apply structural phrase cleanups
    for pattern in STRUCTURAL_PHRASES:
        if isinstance(pattern, re.Pattern):
            text = pattern.sub(" ", text)
        else:
            text = re.sub(r"\b" + re.escape(str(pattern)) + r"\b", " ", text)

    # Safely apply filler cleanups
    for pattern in QUERY_FILLERS:
        if isinstance(pattern, re.Pattern):
            text = pattern.sub(" ", text)
        else:
            text = re.sub(r"\b" + re.escape(str(pattern)) + r"\b", " ", text)

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

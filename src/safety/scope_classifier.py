"""
Semantic scope classifier using BGE embeddings + cosine prototype matching.

The classifier uses the same embedding backend and SCOPE_PROTOTYPES
validated in notebooks/04_scope_classifier.ipynb.

Architecture:
    User Query
        ↓
    BGE Scope Classifier
        ↓
    Margin Gate
        ├── confident → use BGE prediction
        └── uncertain → resolve using existing Dialogue Manager context

This module intentionally does NOT use an LLM or a separate Context Resolver
agent. Ambiguous queries are resolved using state already owned by the
Dialogue Manager, primarily clinical_subject.

Thresholds should be re-tuned in the notebook before changing them here.
"""

import logging
import re
import time
from typing import Any, Dict, Optional

import numpy as np

from src.models.embeddings import initialize_embeddings, normalize_vectors


logger = logging.getLogger("mediqwen_scope_classifier")

CONFIDENCE_THRESHOLD = 0.50
MARGIN_THRESHOLD = 0.05

# Scope prototypes
SCOPE_PROTOTYPES = {
    "MEDICAL": [
        "What are the treatment options for this condition?",
        "What could be causing this symptom?",
        "Why does my skin itch?",
        "Should I see a doctor?",
        "What are the warning signs?",
        "How can I manage this condition?",
        "What medications are commonly used?",
        "Is this symptom serious?",
        "What can I do to relieve these symptoms?",
        "What are the possible causes?",
        "How is this diagnosed?",
        "What are the complications of this illness?",
        "When should I seek medical help?",
        "What are the side effects of this medication?",
        "How can I prevent this condition from worsening?",
        "How can i cure this condition?",
        "What are the home remedies for this symptom?",
    ],
    "NUTRITION": [
        "What is a healthy diet?",
        "What foods are good for heart health?",
        "What should I eat to get more protein?",
        "What foods should I avoid?",
        "Can you suggest a healthy meal?",
        "How can I improve my diet?",
        "What nutrients do I need?",
        "What are healthy foods for breakfast?",
        "How much protein should I eat?",
        "What is a balanced diet?",
        "How do I read a nutrition label?",
        "How much protein is in one egg?",
        "What are the benefits of eating vegetables?",
        "What are the best sources of vitamin C?",
        "Importance of hydration and drinking water.",
        "What are the important sources of zinc?"
    ],
    "CASUAL": [
        "How are you?",
        "Good morning.",
        "Let's chat.",
        "What can you do?",
        "How is your day?",
        "What's something fun to talk about?",
        "Can we have a conversation?",
        "Hello",
        "Hy there",
        "Nice to meet you!",
        "Thank you very much",
        "Okay sounds good",
        "Tell me about yourself.",
        "What are your capabilities?",
        "Who are you?",
        "What can you help me with?",
    ],
    "OUT_OF_SCOPE": [
        "Write Python code for a game.",
        "Explain quantum computing.",
        "Help me configure a Linux server.",
        "Write a marketing strategy.",
        "How do I build a website?",
        "Help me debug this JavaScript code.",
        "Explain how a car engine works.",
        "Write a Python web scraper.",
        "Help me with my programming assignment.",
        "Explain computer networking.",
        "What is the stock market trend today?",
        "Who won the football game?",
        "Tell me a joke.",
        "Tell me a story.",
        "Who won the election?",
        "What is the weather like today?",
    ],
}


# ---------------------------------------------------------------------------
# Deterministic fallback markers.
#
# These are used ONLY after the BGE margin gate marks a query UNCERTAIN.
# They are not a replacement for the semantic classifier.
#
# Word boundaries are important:
#   "eat" must not match "treatment".
# ---------------------------------------------------------------------------

# Regex pattern for anaphoric pronouns and conversational context indicators
ANAPHORIC_PATTERNS = re.compile(
    r"\b("
    r"this issue|this problem|the rash|the lesion|"
    r"its|"
    r"it|this|that|them|these|those"
    r")\b",
    re.IGNORECASE,
)

_TECHNICAL_MARKERS = (
    "python",
    "code",
    "script",
    "javascript",
    "docker",
    "website",
    "scraper",
    "sql",
    "html",
    "algorithm",
)

_NUTRITION_MARKERS = (
    "eat",
    "food",
    "diet",
    "meal",
    "nutrient",
    "calorie",
    "vitamin",
    "protein",
    "carb",
    "fat",
    "sugar",
    "fiber",
    "muscle building",
    "weight loss",
    "weight gain",
    "healthy eating",
)


def _contains_marker(query: str, markers: tuple[str, ...]) -> bool:
    """
    Return True when a marker appears as a complete word.

    This prevents false matches such as:
        "eat" matching "treatment"
        "sql" matching unrelated substrings
    """
    return any(
        re.search(rf"\b{re.escape(marker)}\b", query)
        for marker in markers
    )


class BGEScopeClassifier:
    """
    BGE-based semantic scope classifier.

    Prototype embeddings are generated once during initialization.
    Each query is embedded once and compared against all scope prototypes
    using cosine similarity.
    """

    def __init__(self):
        logger.info("Initializing BGE scope classifier...")

        self.hf_embeddings = initialize_embeddings()
        self.prototype_embeddings: Dict[str, np.ndarray] = {}

        for scope, examples in SCOPE_PROTOTYPES.items():
            raw_vectors = self.hf_embeddings.embed_documents(examples)

            self.prototype_embeddings[scope] = normalize_vectors(
                np.asarray(raw_vectors)
            )

            logger.info(
                "scope=%-12s prototypes=%d",
                scope,
                len(examples),
            )

        logger.info("BGE scope classifier ready.")

    def classify_scope(self, query: str) -> Dict[str, Any]:
        """
        Classify a query using maximum cosine similarity against each
        scope's prototype examples.

        Returns:
            best_scope
            best_score
            second_scope
            second_score
            margin
            decision
            latency_ms
        """

        start_time = time.perf_counter()

        query = query.strip()

        if not query:
            return {
                "query": query,
                "best_scope": "CASUAL",
                "best_score": 0.0,
                "second_scope": None,
                "second_score": 0.0,
                "margin": 0.0,
                "scores": {},
                "decision": "UNCERTAIN",
                "latency_ms": 0.0,
            }

        raw_query_vector = self.hf_embeddings.embed_query(query)

        query_vector = normalize_vectors(
            np.asarray([raw_query_vector])
        )[0]

        scores: Dict[str, float] = {}

        for scope, prototype_matrix in self.prototype_embeddings.items():
            similarities = prototype_matrix @ query_vector
            scores[scope] = float(np.max(similarities))

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        best_scope, best_score = ranked[0]
        second_scope, second_score = ranked[1]

        margin = best_score - second_score

        is_confident = (
            best_score >= CONFIDENCE_THRESHOLD
            and margin >= MARGIN_THRESHOLD
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return {
            "query": query,
            "best_scope": best_scope,
            "best_score": round(best_score, 4),
            "second_scope": second_scope,
            "second_score": round(second_score, 4),
            "margin": round(margin, 4),
            "scores": {
                scope: round(score, 4)
                for scope, score in scores.items()
            },
            "decision": (
                "CONFIDENT"
                if is_confident
                else "UNCERTAIN"
            ),
            "latency_ms": round(elapsed_ms, 2),
        }


def resolve_scope_with_context(
    query: str,
    clinical_subject: Optional[str],
) -> str:
    """
    Resolve an UNCERTAIN BGE classification using existing dialogue state.

    This is intentionally deterministic and lightweight. It is not a
    separate agent or LLM call.

    Priority:
        1. Explicit technical intent → OUT_OF_SCOPE
        2. Explicit nutrition intent → NUTRITION
        3. Existing clinical subject → MEDICAL
        4. Otherwise → CASUAL

    Explicit current-turn intent takes priority over stale clinical context.
    """

    q = query.lower().strip()

    # Current-turn technical intent overrides clinical context.
    if _contains_marker(q, _TECHNICAL_MARKERS):
        return "OUT_OF_SCOPE"

    # Current-turn nutrition intent is distinct from medical follow-up.
    if _contains_marker(q, _NUTRITION_MARKERS):
        return "NUTRITION"

    # Existing clinical context resolves otherwise ambiguous follow-ups.
    if clinical_subject:
        return "MEDICAL"

    # No useful context: retain the conservative conversational default.
    return "CASUAL"


def classify_scope_with_gate(
    classifier: BGEScopeClassifier,
    query: str,
    clinical_subject: Optional[str],
) -> Dict[str, Any]:
    """
    Main scope-classification entry point for the Dialogue Manager.
    """
    bge_result = classifier.classify_scope(query)

    best_scope = bge_result["best_scope"]
    best_score = bge_result["best_score"]
    margin = bge_result["margin"]

    q_lower = query.lower().strip()

    # 1. Technical / Out-of-Scope override (Hard safety check)
    if _contains_marker(q_lower, _TECHNICAL_MARKERS):
        return {
            **bge_result,
            "scope": "OUT_OF_SCOPE",
            "scope_source": "TECHNICAL_OVERRIDE",
            "decision": "HIGH_CONFIDENCE",
        }

    # 2. Anaphora check + Active dialogue context
    has_anaphora = bool(ANAPHORIC_PATTERNS.search(q_lower))

    if has_anaphora and clinical_subject:
        return {
            **bge_result,
            "scope": "MEDICAL",
            "scope_source": "ANAPHORA_CONTEXT_OVERRIDE",
            "decision": "CONTEXT_RESOLVED",
        }

    # 3. Standard BGE confidence + margin threshold gate
    is_confident = (
        best_score >= CONFIDENCE_THRESHOLD
        and margin >= MARGIN_THRESHOLD
    )

    if is_confident:
        return {
            **bge_result,
            "scope": best_scope,
            "scope_source": "BGE",
            "decision": "HIGH_CONFIDENCE",
        }

    # 4. Low-confidence / Low-margin fallback via resolve_scope_with_context
    resolved_scope = resolve_scope_with_context(
        query=q_lower,
        clinical_subject=clinical_subject,
    )

    scope_source = (
        "DIALOG_CONTEXT"
        if clinical_subject and resolved_scope == "MEDICAL"
        else "DETERMINISTIC_FALLBACK"
    )

    return {
        **bge_result,
        "scope": resolved_scope,
        "scope_source": scope_source,
        "decision": "FALLBACK_ROUTED",
    }
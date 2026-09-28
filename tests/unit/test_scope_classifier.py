# tests/unit/test_scope_classifier.py

import pytest
from src.safety.scope_classifier import (
    BGEScopeClassifier,
    classify_scope_with_gate,
)


@pytest.fixture(scope="module")
def scope_classifier():
    return BGEScopeClassifier()


# ============================================================================
# 1. Parameterized Matrix Test
# ============================================================================
@pytest.mark.parametrize(
    "query, expected_scope, clinical_subject",
    [
        # Clear standalone semantic classifications
        ("What are the treatment options for psoriasis?", "MEDICAL", None),
        ("What should I eat for a healthy heart?", "NUTRITION", None),
        ("Hello, how are you?", "CASUAL", None),
        ("Explain quantum mechanics.", "OUT_OF_SCOPE", None),
        # Clinical anaphora + existing context overrides
        ("What are the possible treatments for this?", "MEDICAL", "skin lesion"),
        ("What do you recommend for it?", "MEDICAL", "skin lesion"),
        ("How can I make it better?", "MEDICAL", "skin lesion"),
        ("What should I do?", "MEDICAL", "skin lesion"),
        # Explicit technical intent overriding clinical context
        ("Write Python code to analyze hives.", "OUT_OF_SCOPE", None),
        ("Generate HTML for this diagnosis.", "OUT_OF_SCOPE", "skin lesion"),
    ],
)
def test_scope_classification_matrix(
    scope_classifier, query, expected_scope, clinical_subject
):
    result = classify_scope_with_gate(
        scope_classifier,
        query,
        clinical_subject=clinical_subject,
    )
    assert result["scope"] == expected_scope


# ============================================================================
# 2. Mechanism-Specific Unit Tests
# ============================================================================
def test_clear_medical_query_uses_bge(scope_classifier):
    result = classify_scope_with_gate(
        scope_classifier,
        "What are the treatment options for psoriasis?",
        clinical_subject=None,
    )

    assert result["scope"] == "MEDICAL"
    assert result["scope_source"] == "BGE"
    assert result["decision"] == "HIGH_CONFIDENCE"


def test_anaphoric_followup_uses_clinical_context(scope_classifier):
    result = classify_scope_with_gate(
        scope_classifier,
        "What do you recommend for it?",
        clinical_subject="skin lesion",
    )

    assert result["scope"] == "MEDICAL"
    assert result["scope_source"] == "ANAPHORA_CONTEXT_OVERRIDE"


def test_technical_query_overrides_clinical_context(scope_classifier):
    result = classify_scope_with_gate(
        scope_classifier,
        "Generate HTML for this diagnosis.",
        clinical_subject="skin lesion",
    )

    assert result["scope"] == "OUT_OF_SCOPE"
    assert result["scope_source"] == "TECHNICAL_OVERRIDE"


# ============================================================================
# 3. Explicit Regression Lock Tests
# ============================================================================
def test_treatment_followup_after_image_analysis(scope_classifier):
    """
    Prevents regression where valid clinical follow-up after multimodal vision analysis
    gets misrouted or blocked.
    """
    result = classify_scope_with_gate(
        scope_classifier,
        "What are the possible treatments for this?",
        clinical_subject="skin lesion",
    )

    assert result["scope"] == "MEDICAL"
    assert result["decision"] in ["HIGH_CONFIDENCE", "CONTEXT_RESOLVED"]
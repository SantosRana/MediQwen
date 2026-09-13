# tests/unit/test_risk_classifier.py
"""
Unit Test Suite for MediQwen Risk Classifier Engine.
Validates domain vs. severity separation, active self-report detection,
multi-turn worsening escalations, and emergency signature overrides.
"""

import pytest
from src.safety.risk_classifier import RiskClassifier


@pytest.fixture
def classifier() -> RiskClassifier:
    """Provides a fresh RiskClassifier instance for unit testing."""
    return RiskClassifier()


# ============================================================================
# 🩺 DOMAIN vs. SEVERITY SEPARATION
# ============================================================================

def test_informational_medical_query_low_risk(classifier: RiskClassifier):
    """
    Verifies that informational medical questions ('What is a rash?')
    remain at LOW risk rather than being over-triaged.
    """
    query = "What is a localized skin rash?"
    result = classifier.classify_query(query)
    
    assert result["risk_level"] == "LOW"
    assert "general_medicine" in result["clinical_categories"]


def test_active_self_report_medium_risk(classifier: RiskClassifier):
    """
    Verifies that active patient self-reports ('I have a rash')
    trigger MEDIUM risk classification.
    """
    query = "I have a localized skin rash on my forearm."
    result = classifier.classify_query(query)

    assert result["risk_level"] == "MEDIUM"
    assert "active_symptoms" in result["clinical_categories"]


# ============================================================================
# 📈 WORSENING ESCALATIONS & MULTI-TURN HISTORY
# ============================================================================

def test_single_turn_worsening_modifier_high_risk(classifier: RiskClassifier):
    """
    Verifies that active self-reports with worsening modifiers ('spreading rapidly')
    escalate risk to HIGH.
    """
    query = "I have a severe skin rash that is spreading rapidly and worsening."
    result = classifier.classify_query(query)
    
    assert result["risk_level"] == "HIGH"
    assert "worsening_escalation" in result["clinical_categories"]


def test_multi_turn_history_worsening_escalation(classifier: RiskClassifier):
    """
    Verifies that cumulative worsening indicators across multi-turn history
    escalate the assigned risk level from MEDIUM to HIGH via classify_with_history().
    """
    current_query = "I have a localized skin rash on my arm."

    history = [
        {"role": "user", "content": "I noticed a rash yesterday and it is getting worse."},
        {"role": "assistant", "content": "Please describe any additional changes or symptoms."},
        {"role": "user", "content": "It seems to be spreading rapidly across my forearm."}
    ]

    result = classifier.classify_with_history(current_query, history)

    assert result["risk_level"] == "HIGH"
    assert result.get("history_escalated") is True


# ============================================================================
# 🚨 EMERGENCY SIGNATURES & CASUAL CHAT
# ============================================================================

def test_emergency_signature_override(classifier: RiskClassifier):
    """
    Verifies that critical emergency signatures (e.g., stroke or crushing chest pain)
    immediately trigger EMERGENCY risk assignment.
    """
    emergency_queries = [
        "I have sudden crushing chest pain radiating to my neck!",
        "My dad suddenly has trouble speaking and his face is drooping",
        "Severe trouble breathing and turning blue.",
    ]
    
    for query in emergency_queries:
        result = classifier.classify_query(query)
        assert result["risk_level"] == "EMERGENCY"
        assert result["requires_doctor"] is True
        assert result.get("guardrail_emergency") is True


def test_casual_chat_low_risk(classifier: RiskClassifier):
    """
    Verifies that non-clinical conversational queries assign LOW risk.
    """
    chat_queries = [
        "Hello! How can you assist me today?",
        "Thank you for the information.",
        "What are your core capabilities?",
    ]
    
    for query in chat_queries:
        result = classifier.classify_query(query)
        assert result["risk_level"] == "LOW"
        assert result["trigger_count"] == 0
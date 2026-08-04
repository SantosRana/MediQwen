# tests/unit/test_risk_classifier.py
import pytest
from src.safety.risk_classifier import classify_triage_tier  # Replace with your actual path if different

def test_risk_classifier_flags_emergency_chest_pain():
    """Asserts that severe acute cardiac presentation symptoms force an EMERGENCY classification."""
    mock_state = {"user_query": "I have sudden crushing chest pain radiating down my left arm and jaw!"}
    result = classify_triage_tier(mock_state)
    
    assert result.get("risk_level") == "EMERGENCY"
    assert result.get("matched_triggers_count", 0) >= 1

def test_risk_classifier_handles_low_risk_queries():
    """Asserts that standard superficial inquiries settle correctly into a LOW risk level tier."""
    mock_state = {"user_query": "How do I clear a mild dry cough at home?"}
    result = classify_triage_tier(mock_state)
    
    assert result.get("risk_level") == "LOW"
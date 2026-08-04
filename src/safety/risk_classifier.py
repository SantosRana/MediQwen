# src/safety/risk_classifier.py
"""
Data-Driven Risk Classification Engine for MediGemma.
Accumulates all matching clinical rules to preserve compound symptoms, extracts the
highest severity risk tier, and leverages pre-compiled guardrail proximity checks.
"""

import re
import logging
from enum import Enum
from typing import Dict, List, Tuple, Any, Iterable, Set
import config.settings as settings
from src.safety.guardrails import Guardrails

logger = logging.getLogger("risk_classifier")

class RiskLevel(Enum):
    EMERGENCY = "emergency"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def severity_rank(self) -> int:
        """Assigns an integer rank to compare enum severities programmatically."""
        mapping = {RiskLevel.EMERGENCY: 4, RiskLevel.HIGH: 3, RiskLevel.MEDIUM: 2, RiskLevel.LOW: 1}
        return mapping[self]

class ClinicalRule:
    """Represents a structured clinical assessment and triage routing rule."""
    def __init__(self, pattern: str, risk_level: RiskLevel, score: int, category: str, action: str):
        self.raw_pattern = pattern
        self.risk_level = risk_level
        self.score = score
        self.category = category
        self.action = action
        self.compiled_regex = re.compile(pattern, re.IGNORECASE)

class RiskClassifier:
    """
    Accumulative Multi-Layered Triage Engine.
    Gathers all matched clinical factors to assess complex symptom profiles.
    """

    def __init__(self, enable_checks: bool = None):
        self.enable_checks = enable_checks if enable_checks is not None else settings.ENABLE_SAFETY_CHECKS
        # Instantiate Guardrails to borrow the system's foundational pre-compiled safety patterns
        self.guardrails = Guardrails()
        self._initialize_rules_engine()
        logger.info(f"⚡ Accumulative Rules Engine Active. Safety Enforcement: {self.enable_checks}")

    def _initialize_rules_engine(self):
        """Declarative local clinical taxonomy registry."""
        self.rules_registry: List[ClinicalRule] = [
            # --- HIGH RISK CARDIO/RESPIRATORY INDICATORS (Score: 7-8) ---
            ClinicalRule(r"\bsevere\b.*\bpain\b", RiskLevel.HIGH, 7, "general", "urgent_medical_consultation"),
            ClinicalRule(r"\bhigh\b.*\bfever\b|\bcoughing\b.*\bblood\b", RiskLevel.HIGH, 7, "infectious", "urgent_medical_consultation"),
            ClinicalRule(r"\bblood\b.*\b(stool|vomit)\b", RiskLevel.HIGH, 7, "gastrointestinal", "urgent_medical_consultation"),
            ClinicalRule(r"\b(sudden\s+weakness|confusion|disorientation)\b", RiskLevel.HIGH, 7, "neurological", "urgent_medical_consultation"),
            ClinicalRule(r"\b(seizure|convulsion|head\s+injury|concussion)\b", RiskLevel.HIGH, 7, "neurological", "urgent_medical_consultation"),
            
            # --- MEDIUM RISK TIER (Score: 4) ---
            ClinicalRule(r"\b(fever|cough|vomiting|diarrhoea|rash)\b", RiskLevel.MEDIUM, 4, "general", "routine_medical_advice"),
            ClinicalRule(r"\babdominal\b.*\bpain\b|\bstomach\b.*\bache\b", RiskLevel.MEDIUM, 4, "gastrointestinal", "routine_medical_advice"),
            ClinicalRule(r"\b(dizziness|lightheaded|earache|sore\s+throat)\b", RiskLevel.MEDIUM, 4, "general", "routine_medical_advice"),
            ClinicalRule(r"\b(hypertension|blood\s+pressure|asthma|infection)\b", RiskLevel.MEDIUM, 4, "chronic", "routine_medical_advice")
        ]

        self.WORSENING_INDICATORS = [
            "worse", "worsening", "getting worse", "increasing", 
            "not improving", "cannot tolerate", "spread", "spreading", "intense"
        ]

    def classify_query(self, text: str) -> Dict[str, Any]:
        """
        Processes inputs across multiple safety validation layers.
        Accumulates all matching indicators to ensure compound clinical scenarios are preserved.
        """
        if not self.enable_checks:
            return {
                "risk_level": RiskLevel.LOW.value, "score": 0, "triggers": [],
                "action_required": "none", "requires_doctor": False, "classification_mode": "disabled"
            }
            
        text_clean = re.sub(r"\s+", " ", text.lower()).strip()
        
        # ─── LAYER 1: FOUNDATIONAL GUARDRAILS PROXIMITY SCANS ───
        # Restores foundational precedence for the global emergency arrays
        matched_guardrail_patterns = [
            pattern.pattern for pattern in self.guardrails.compiled_emergencies
            if pattern.search(text_clean)
        ]
        
        # ─── LAYER 2: LOCAL RULES ACCUMULATION MATRIX ───
        # Accumulates all matches rather than short-circuiting on the first hit
        matched_rules: List[ClinicalRule] = [
            rule for rule in self.rules_registry if rule.compiled_regex.search(text_clean)
        ]

        # ─── LAYER 3: MULTI-VARIABLE CONTEXT EVALUATION ───
        if matched_guardrail_patterns or matched_rules:
            # Determine the baseline values from guardrails layer if it fires
            highest_risk = RiskLevel.EMERGENCY if matched_guardrail_patterns else RiskLevel.LOW
            highest_score = 10 if matched_guardrail_patterns else 0
            primary_action = "immediate_emergency_services" if matched_guardrail_patterns else "none"
            
            # Map out and audit every matched element
            all_triggers = list(matched_guardrail_patterns)
            categories = {"emergency_guardrail"} if matched_guardrail_patterns else set()

            # Process all matched rules to extract the highest severity peak
            for rule in matched_rules:
                all_triggers.append(rule.raw_pattern)
                categories.add(rule.category)
                
                # Check severity ranks to find the highest risk level
                if rule.risk_level.severity_rank > highest_risk.severity_rank:
                    highest_risk = rule.risk_level
                    highest_score = rule.score
                    primary_action = rule.action

            logger.warning(
                f"🎯 Triage Evaluation Complete. Matched Triggers Count: {len(all_triggers)} | "
                f"Assigned Risk Peak: {highest_risk.value.upper()}"
            )

            return {
                "risk_level": highest_risk.value,
                "score": max(highest_score, 1 if highest_risk == RiskLevel.LOW else highest_score),
                "triggers": all_triggers,
                "action_required": primary_action,
                "requires_doctor": highest_score >= 4 or highest_risk == RiskLevel.EMERGENCY,
                "classification_mode": "accumulative_rules_engine",
                "clinical_categories": list(categories),
                "trigger_count": len(all_triggers)
            }
                
        # Baseline low risk fallback
        return {
            "risk_level": RiskLevel.LOW.value,
            "score": 1,
            "triggers": [],
            "action_required": "general_information",
            "requires_doctor": False,
            "classification_mode": "default",
            "clinical_categories": ["general_medicine"],
            "trigger_count": 0
        }

    def classify_with_history(self, current_query: str, history: Any) -> Dict[str, Any]:
        """
        Dynamically calculates history log risk tracking variables.
        Protects against string duplication using sets cardinality verification.
        """
        base_classification = self.classify_query(current_query)
        
        if not self.enable_checks or not history:
            return base_classification
            
        history_text_pieces = []
        if isinstance(history, str):
            history_text_pieces.append(history)
        elif isinstance(history, Iterable):
            for msg in history:
                if hasattr(msg, "content"):
                    history_text_pieces.append(str(msg.content))
                elif isinstance(msg, dict) and "content" in msg:
                    history_text_pieces.append(str(msg["content"]))
                elif isinstance(msg, str):
                    history_text_pieces.append(msg)
                    
        history_combined = " ".join(history_text_pieces).lower()
        
        if base_classification["risk_level"] in [RiskLevel.MEDIUM.value, RiskLevel.HIGH.value]:
            detected_worsening_terms = [
                indicator for indicator in self.WORSENING_INDICATORS
                if re.search(rf"\b{re.escape(indicator)}\b", history_combined)
            ]
            
            unique_indicators = set(detected_worsening_terms)
            
            if len(unique_indicators) >= 2:
                current_level = base_classification["risk_level"]
                
                if current_level == RiskLevel.MEDIUM.value:
                    escalated_level = RiskLevel.HIGH.value
                    score_bump = 7
                    action = "urgent_medical_consultation"
                else:
                    escalated_level = RiskLevel.EMERGENCY.value
                    score_bump = 10
                    action = "immediate_emergency_services"
                    
                result = base_classification.copy()
                result.update({
                    "risk_level": escalated_level,
                    "score": score_bump,
                    "action_required": action,
                    "requires_doctor": True,
                    "history_escalated": True,
                    "escalation_triggers": list(unique_indicators)
                })
                
                logger.warning(
                    f"⚠️ CRITICAL STATE ESCALATION: Upgraded from {current_level} to {escalated_level}. "
                    f"Unique cardinality indicators found: {unique_indicators}"
                )
                return result
                
        return base_classification
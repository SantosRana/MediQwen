# src/safety/risk_classifier.py
"""
Data-Driven Risk Classification Engine for MediQwen.
Separates medical domain relevance from clinical risk severity using 
accumulative rules, active self-reporting context, and worsening indicator escalations.
"""

import re
import logging
from enum import Enum
from typing import Dict, List, Tuple, Any, Iterable, Set
import config.settings as settings
from src.safety.guardrails import Guardrails

logger = logging.getLogger("risk_classifier")


class RiskLevel(Enum):
    EMERGENCY = "EMERGENCY"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

    @property
    def severity_rank(self) -> int:
        """Assigns an integer rank to compare enum severities programmatically."""
        mapping = {
            RiskLevel.EMERGENCY: 4,
            RiskLevel.HIGH: 3,
            RiskLevel.MEDIUM: 2,
            RiskLevel.LOW: 1
        }
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
    Distinguishes informational medical queries (LOW) from active, severe,
    or worsening clinical scenarios (MEDIUM / HIGH / EMERGENCY).
    """

    def __init__(self, enable_checks: bool = None):
        self.enable_checks = enable_checks if enable_checks is not None else settings.ENABLE_SAFETY_CHECKS
        self.guardrails = Guardrails()
        self._initialize_rules_engine()
        logger.info(f"⚡ Accumulative Rules Engine Active. Safety Enforcement: {self.enable_checks}")

    def _initialize_rules_engine(self):
        """Declarative clinical taxonomy registry."""
        self.rules_registry: List[ClinicalRule] = [
            # --- HIGH RISK CARDIO/RESPIRATORY/NEURO INDICATORS (Score: 7-8) ---
            ClinicalRule(r"\bsevere\b.*\bpain\b", RiskLevel.HIGH, 7, "general", "urgent_medical_consultation"),
            ClinicalRule(r"\bhigh\b.*\bfever\b|\bcoughing\b.*\bblood\b", RiskLevel.HIGH, 7, "infectious", "urgent_medical_consultation"),
            ClinicalRule(r"\bblood\b.*\b(stool|vomit)\b", RiskLevel.HIGH, 7, "gastrointestinal", "urgent_medical_consultation"),
            ClinicalRule(r"\b(sudden\s+weakness|confusion|disorientation)\b", RiskLevel.HIGH, 7, "neurological", "urgent_medical_consultation"),
            ClinicalRule(r"\b(seizure|convulsion|head\s+injury|concussion)\b", RiskLevel.HIGH, 7, "neurological", "urgent_medical_consultation"),

            # --- MEDIUM RISK TIER: ACTIVE ACUTE / SELF-REPORTED SYMPTOMS (Score: 4) ---
            ClinicalRule(r"\b(i\s+have|my|suffering|experiencing|dealing\s+with)\b.*\b(fever|cough|vomiting|diarrhoea|rash|pain|lesion)\b", RiskLevel.MEDIUM, 4, "active_symptoms", "routine_medical_advice"),
            ClinicalRule(r"\babdominal\b.*\bpain\b|\bstomach\b.*\bache\b", RiskLevel.MEDIUM, 4, "gastrointestinal", "routine_medical_advice"),
            ClinicalRule(r"\b(hypertension|blood\s+pressure|asthma|active\s+infection)\b", RiskLevel.MEDIUM, 4, "chronic", "routine_medical_advice"),
            
            # --- LOW RISK TIER: INFORMATIONAL MEDICINE & BASE SYMPTOM INQUIRIES (Score: 1-2) ---
            ClinicalRule(r"\b(fever|cough|vomiting|diarrhoea|rash|dizziness|lightheaded|earache|sore\s+throat)\b", RiskLevel.LOW, 2, "general_medicine", "general_information"),
        ]

        self.WORSENING_INDICATORS = [
            "worse", "worsening", "getting worse", "increasing", 
            "not improving", "cannot tolerate", "spread", "spreading", "rapidly", "intense", "severe"
        ]

    def classify_query(self, text: str) -> Dict[str, Any]:
        """
        Processes inputs across multiple safety validation layers.
        Evaluates emergency guardrails, local clinical rules, and active worsening modifiers.
        """
        if not self.enable_checks:
            return {
                "risk_level": RiskLevel.LOW.value, "score": 0, "triggers": [],
                "action_required": "none", "requires_doctor": False, "classification_mode": "disabled"
            }
            
        text_clean = re.sub(r"\s+", " ", text.lower()).strip()
        
        # ─── LAYER 1: FOUNDATIONAL GUARDRAILS PROXIMITY SCANS ───
        emergency_eval = self.guardrails.detect_emergency(text_clean)
        matched_guardrail_patterns = []
        if emergency_eval.get("is_emergency"):
            matched_guardrail_patterns = [emergency_eval.get("matched_pattern")]
        
        # ─── LAYER 2: LOCAL RULES ACCUMULATION MATRIX ───
        matched_rules: List[ClinicalRule] = [
            rule for rule in self.rules_registry if rule.compiled_regex.search(text_clean)
        ]

        # ─── LAYER 3: SINGLE-TURN WORSENING / SEVERITY ESCALATION SCAN ───
        detected_worsening = [
            indicator for indicator in self.WORSENING_INDICATORS
            if re.search(rf"\b{re.escape(indicator)}\b", text_clean)
        ]

        if matched_guardrail_patterns or matched_rules or detected_worsening:
            highest_risk = RiskLevel.EMERGENCY if matched_guardrail_patterns else RiskLevel.LOW
            highest_score = 10 if matched_guardrail_patterns else 0
            primary_action = "immediate_emergency_services" if matched_guardrail_patterns else "none"
            
            all_triggers = list(matched_guardrail_patterns)
            categories = {"emergency_guardrail"} if matched_guardrail_patterns else set()

            for rule in matched_rules:
                all_triggers.append(rule.raw_pattern)
                categories.add(rule.category)
                if rule.risk_level.severity_rank > highest_risk.severity_rank:
                    highest_risk = rule.risk_level
                    highest_score = rule.score
                    primary_action = rule.action

            # Single-turn escalation: Elevate LOW -> MEDIUM or MEDIUM -> HIGH on worsening terms
            if detected_worsening and highest_risk != RiskLevel.EMERGENCY:
                all_triggers.extend([f"worsening_indicator:{w}" for w in detected_worsening])
                categories.add("worsening_escalation")
                
                if highest_risk == RiskLevel.LOW:
                    highest_risk = RiskLevel.MEDIUM
                    highest_score = max(highest_score, 4)
                    primary_action = "routine_medical_advice"
                elif highest_risk == RiskLevel.MEDIUM:
                    highest_risk = RiskLevel.HIGH
                    highest_score = max(highest_score, 7)
                    primary_action = "urgent_medical_consultation"

            logger.warning(
                f"🎯 Triage Evaluation Complete. Matched Triggers Count: {len(all_triggers)} | "
                f"Assigned Risk Peak: {highest_risk.value}"
            )

            return {
                "risk_level": highest_risk.value,
                "score": max(highest_score, 1 if highest_risk == RiskLevel.LOW else highest_score),
                "triggers": all_triggers,
                "action_required": primary_action,
                "requires_doctor": highest_score >= 4 or highest_risk == RiskLevel.EMERGENCY,
                "classification_mode": "accumulative_rules_engine",
                "clinical_categories": list(categories),
                "trigger_count": len(all_triggers),
                "is_emergency": highest_risk == RiskLevel.EMERGENCY,
                "guardrail_emergency": bool(matched_guardrail_patterns)
            }
                
        # Baseline LOW risk fallback for casual chat / non-symptom queries
        return {
            "risk_level": RiskLevel.LOW.value,
            "score": 1,
            "triggers": [],
            "action_required": "general_information",
            "requires_doctor": False,
            "classification_mode": "default",
            "clinical_categories": ["general_medicine"],
            "trigger_count": 0,
            "is_emergency": False,
            "guardrail_emergency": False
        }

    def classify_with_history(self, current_query: str, history: Any) -> Dict[str, Any]:
        """
        Dynamically calculates multi-turn history risk tracking variables.
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
                    "escalation_triggers": list(unique_indicators),
                    "is_emergency": escalated_level == RiskLevel.EMERGENCY.value
                })
                
                logger.warning(
                    f"⚠️ CRITICAL STATE ESCALATION: Upgraded from {current_level} to {escalated_level}. "
                    f"Unique cardinality indicators found: {unique_indicators}"
                )
                return result
                
        return base_classification
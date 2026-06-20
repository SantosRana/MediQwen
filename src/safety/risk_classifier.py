"""
Risk Classification Layer
Classifies medical queries by urgency and risk level.
Critical for ensuring user safety.
"""

import re
import time
from typing import Dict, List, Tuple
from enum import Enum
import logging
import config.settings as settings

# Configure logger
logger = logging.getLogger("risk_classifier")

class RiskLevel(Enum):
    EMERGENCY = "emergency"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class RiskClassifier:
    """
    Multi-layered risk classification system:
    1. Keyword-based emergency detection
    2. Symptom severity scoring
    3. Context-aware risk assessment
    """

    def __init__(self, enable_checks: bool = None):
        # Use config setting if not provided
        self.enable_checks = enable_checks if enable_checks is not None else settings.ENABLE_SAFETY_CHECKS
        
        # Load keywords from config or use defaults
        self._load_keywords()
        
        logger.info(f"RiskClassifier initialized. Safety checks enabled: {self.enable_checks}")
    
    def _load_keywords(self):
        """Load risk keywords from config or use built-in lists."""
        # Emergency keywords - require immediate professional care
        self.EMERGENCY_KEYWORDS = [
            "chest pain", "heart attack", "can't breathe", "not breathing", 
            "unconscious", "unresponsive", "severe bleeding", "bleeding heavily",
            "stroke", "seizure", "anaphylaxis", "allergic reaction", "swelling throat",
            "suicide", "kill myself", "want to die", "overdose", "poisoning",
            "severe burn", "third degree burn", "electrocution", "drowning",
            "gunshot", "stab wound", "head injury", "spinal injury",
            "baby not breathing", "infant unresponsive", "pregnant bleeding",
            "severe dehydration", "shock", "cyanosis", "blue lips"
        ]
        
        # High-risk keywords - need prompt medical attention
        self.HIGH_RISK_KEYWORDS = [
            "high fever", "fever 104", "fever 40", "persistent vomiting", 
            "blood in stool", "blood in urine", "blood in vomit", "black stool",
            "severe headache", "worst headache", "stiff neck", "confusion",
            "chest tightness", "irregular heartbeat", "fainting", "passed out",
            "severe abdominal pain", "rigid abdomen", "jaundice", "severe back pain",
            "vision loss", "sudden blindness", "slurred speech", "weakness one side",
            "severe allergic reaction", "difficulty swallowing", "dehydration",
            "diabetic emergency", "ketones", "fruity breath", "rapid breathing",
            "pregnant pain", "pregnant bleeding", "decreased fetal movement"
        ]
        
        # Medium-risk keywords - should see doctor soon
        self.MEDIUM_RISK_KEYWORDS = [
            "fever", "cough", "sore throat", "ear pain", "sinus pain",
            "rash", "itching", "joint pain", "muscle pain", "back pain",
            "stomach pain", "nausea", "diarrhea", "constipation", "heartburn",
            "headache", "dizziness", "fatigue", "insomnia", "anxiety",
            "depression", "weight loss", "weight gain", "swelling", "numbness",
            "tingling", "urinary problems", "menstrual irregularities"
        ]
        
        logger.info(f"Loaded {len(self.EMERGENCY_KEYWORDS)} emergency keywords")
        logger.info(f"Loaded {len(self.HIGH_RISK_KEYWORDS)} high-risk keywords")
        logger.info(f"Loaded {len(self.MEDIUM_RISK_KEYWORDS)} medium-risk keywords")
    
    def classify(self, text: str, image_description: str = "") -> Dict:
        """
        Classify the risk level of a medical query.
        
        Args:
            text: The user's query text
            image_description: Description of any accompanying image (for multimodal)
            
        Returns:
            Dict with risk_level, score, triggers, and recommendation
        """
        
        start_time = time.time()
        
        # Check if safety checks are enabled
        if not self.enable_checks:
            elapsed = time.time() - start_time
            logger.warning(f"Safety checks disabled. Classification returned LOW. Time: {elapsed:.3f}s")
            return {
                "risk_level": RiskLevel.LOW.value,
                "score": 1,
                "triggers": [],
                "recommendation": "Safety checks disabled by user.",
                "requires_doctor": False,
                "is_emergency": False,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        
        # Combine text and image description for analysis
        combined_text = f"{text} {image_description}".lower()
        
        # Check emergency keywords first (highest priority)
        emergency_triggers = self._find_triggers(combined_text, self.EMERGENCY_KEYWORDS)
        if emergency_triggers:
            elapsed = time.time() - start_time
            logger.warning(
                f"EMERGENCY DETECTED! Triggers: {emergency_triggers}. "
                f"Classification time: {elapsed:.3f}s"
            )
            return {
                "risk_level": RiskLevel.EMERGENCY.value,
                "score": settings.RISK_LEVELS[RiskLevel.EMERGENCY.value],
                "triggers": emergency_triggers,
                "recommendation": "🚨 EMERGENCY: Call emergency services (911/112) immediately. "
                                "Do not wait.",
                "requires_doctor": True,
                "is_emergency": True,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        
        # Check high-risk keywords
        high_triggers = self._find_triggers(combined_text, self.HIGH_RISK_KEYWORDS)
        if high_triggers:
            elapsed = time.time() - start_time
            logger.warning(
                f"HIGH RISK DETECTED! Triggers: {high_triggers}. "
                f"Classification time: {elapsed:.3f}s"
            )
            return {
                "risk_level": RiskLevel.HIGH.value,
                "score": settings.RISK_LEVELS[RiskLevel.HIGH.value],
                "triggers": high_triggers,
                "recommendation": "⚠️ HIGH RISK: Seek medical attention promptly "
                                "(within hours). Do not delay.",
                "requires_doctor": True,
                "is_emergency": False,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        
        # Check medium-risk keywords
        medium_triggers = self._find_triggers(combined_text, self.MEDIUM_RISK_KEYWORDS)
        if medium_triggers:
            elapsed = time.time() - start_time
            logger.info(f"MEDIUM RISK DETECTED! Triggers: {medium_triggers}. "
                        f"Classification time: {elapsed:.3f}s")
            return {
                "risk_level": RiskLevel.MEDIUM.value,
                "score": settings.RISK_LEVELS[RiskLevel.MEDIUM.value],
                "triggers": medium_triggers,
                "recommendation": "ℹ️ MODERATE: Monitor symptoms. "
                                "See a doctor if symptoms persist >2-3 days or worsen.",
                "requires_doctor": False,
                "is_emergency": False,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        
        # Low risk - no specific triggers
        elapsed = time.time() - start_time
        logger.info(f"LOW RISK! No specific triggers found. Classification time: {elapsed:.3f}s")
        return {
            "risk_level": RiskLevel.LOW.value,
            "score": settings.RISK_LEVELS[RiskLevel.LOW.value],
            "triggers": [],
            "recommendation": "✅ LOW RISK: General health information provided. "
                            "Consult doctor if concerned.",
            "requires_doctor": False,
            "is_emergency": False,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def classify_with_context(
        self, 
        text: str, 
        conversation_history: List[str] = None,
        image_description: str = ""
    ) -> Dict:
        """
        Enhanced classification using conversation context.
        Escalates risk if symptoms are worsening or persisting.
        
        Args:
            text: Current query
            conversation_history: List of previous turns
            image_description: Image description
            
        Returns:
            Dict with enhanced classification
        """
        # Base classification
        base_classification = self.classify(text, image_description)
        
        # Skip escalation if already high risk or above
        if base_classification["risk_level"] in [RiskLevel.EMERGENCY.value, 
                                                  RiskLevel.HIGH.value]:
            return base_classification
        
        # Check for worsening symptoms in conversation
        if conversation_history and len(conversation_history) >= 2:
            worsening_indicators = [
                "worse", "getting worse", "increasing", "more severe", 
                "not better", "persisting", "still", "continuing", 
                "still hurting", "pain won't go away", "worsening"
            ]
            
            # Use recent history
            recent_history = " ".join(conversation_history[-3:]).lower()
            
            if any(ind in recent_history for ind in worsening_indicators):
                current_level = base_classification["risk_level"]
                escalation_map = {
                    RiskLevel.LOW.value: RiskLevel.MEDIUM.value,
                    RiskLevel.MEDIUM.value: RiskLevel.HIGH.value,
                    RiskLevel.HIGH.value: RiskLevel.HIGH.value  # cap at HIGH
                }
                
                escalated_level = escalation_map.get(current_level, current_level)
                score_map = {
                    RiskLevel.LOW.value: 2,
                    RiskLevel.MEDIUM.value: 3,
                    RiskLevel.HIGH.value: 4
                }
                
                # Update classification
                result = base_classification.copy()
                result["risk_level"] = escalated_level
                result["score"] = score_map.get(escalated_level, 0)
                result["triggers"].append("worsening_symptoms")
                result["recommendation"] = (
                    "⚠️ Worsening symptoms detected. Seek medical attention promptly."
                    if escalated_level == RiskLevel.HIGH.value
                    else "⚠️ Symptoms persisting/worsening. Recommend seeing a doctor."
                )
                result["requires_doctor"] = escalated_level != RiskLevel.LOW.value
                
                logger.warning(
                    f"SYMPTOM ESCALATION: Escalated from {current_level} to {escalated_level} "
                    f"due to worsening indicators in conversation history."
                )
                return result
        
        return base_classification
    
    def _find_triggers(self, text: str, keywords: List[str]) -> List[str]:
        """Find which keywords are triggered in the text."""
        triggers = []
        for keyword in keywords:
            if keyword in text:
                triggers.append(keyword)
        return triggers
    
    def get_risk_level(self) -> RiskLevel:
        """Get current risk level setting."""
        return RiskLevel(settings.MAX_RISK_LEVEL if settings.MAX_RISK_LEVEL else "low")
    
    def get_keywords(self, level: RiskLevel) -> List[str]:
        """Get keywords for a specific risk level."""
        keyword_map = {
            RiskLevel.EMERGENCY.value: self.EMERGENCY_KEYWORDS,
            RiskLevel.HIGH.value: self.HIGH_RISK_KEYWORDS,
            RiskLevel.MEDIUM.value: self.MEDIUM_RISK_KEYWORDS
        }
        return keyword_map.get(level, [])
    
    def set_enable_checks(self, enable: bool):
        """Enable/disable safety checks."""
        self.enable_checks = enable
        logger.info(f"Safety checks set to: {enable}")
"""
Safety guardrails for medical AI responses.
Ensures responses meet safety standards.
"""

import re
from typing import Dict, Tuple, Optional
import logging
from enum import Enum
import config.settings as settings

# Configure logger
logger = logging.getLogger("guardrails")

class GuardrailType(Enum):
    INPUT_VALIDATION = "input_validation"
    OUTPUT_FILTERING = "output_filtering"
    DIAGNOSTIC_SOFTENING = "diagnostic_softening"
    DISCLAIMER_ADDITION = "disclaimer_addition"

class Guardrails:
    """
    Safety guardrails for medical AI responses.
    Ensures responses meet safety standards.
    """
    
    # Topics to refuse or redirect
    RESTRICTED_TOPICS = [
        r"how to (make|create|synthesize|manufacture) (drugs|meth|cocaine|heroin)",
        r"how to (commit|attempt) suicide",
        r"how to (harm|hurt|kill) (someone|myself)",
        r"how to perform (surgery|operation|medical procedure) (on myself|at home)",
        r"how to (abort|terminate) pregnancy (at home|myself)",
        r"how to (fake|forge) (prescription|medical certificate|doctor's note)",
        r"recipe for (drug|medication|chemical)",
        r"(poison|toxics) (recipe|how to make)"
    ]
    
    # Required disclaimers
    DISCLAIMER = (
        "\n\n---\n⚕️ **Medical Disclaimer**: This information is for educational purposes only "
        "and is not a substitute for professional medical advice, diagnosis, or treatment. "
        "Always seek the advice of your physician or other qualified health provider with any "
        "questions you may have regarding a medical condition.\n"
    )
    
    EMERGENCY_DISCLAIMER = (
        "\n\n🚨 **URGENT**: If you are experiencing a medical emergency, call your local "
        "emergency number (911/112/999) immediately. Do not rely on this AI for emergency care. "
        "This chatbot is not a substitute for professional emergency care.\n"
    )
    
    GENERAL_ADVICE_DISCLAIMER = (
        "\n\n---\nℹ️ **Important**: The following information is general health guidance. "
        "Individual cases vary. Always consult with a healthcare professional for "
        "personalized advice.\n"
    )

    def __init__(self, enable_checks: bool = None):
        # Use config setting if not provided
        self.enable_checks = enable_checks if enable_checks is not None else settings.ENABLE_SAFETY_CHECKS
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        # Load additional safety settings
        self.ENABLE_DIAGNOSTIC_SOFTENING = getattr(settings, 'ENABLE_DIAGNOSTIC_SOFTENING', True)
        
        logger.info(f"Guardrails initialized. Safety checks enabled: {self.enable_checks}")
        logger.info(f"Compiled {len(self._compiled_patterns)} restricted topic patterns")
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        self._compiled_patterns = []
        for pattern in self.RESTRICTED_TOPICS:
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
                self._compiled_patterns.append(compiled)
                logger.debug(f"Compiled pattern: {pattern}")
            except re.error as e:
                logger.error(f"Failed to compile pattern {pattern}: {e}")
    
    def check_input(self, text: str) -> Tuple[bool, str]:
        """
        Check if input violates safety policies.
        
        Args:
            text: User input text
            
        Returns:
            Tuple of (is_safe, message_if_unsafe)
        """
        if not self.enable_checks:
            return True, ""
        
        for pattern in self._compiled_patterns:
            if pattern.search(text):
                # Determine the type of restriction
                pattern_str = pattern.pattern
                
                if "suicide" in pattern_str or "self-harm" in pattern_str:
                    return False, (
                        "I'm concerned about you. If you're feeling suicidal or "
                        "having thoughts of self-harm, please contact a crisis helpline immediately. "
                        "In the US: 988 Suicide & Crisis Lifeline. "
                        "In other countries, please search for local crisis services. "
                        "I can provide resources, but I cannot continue this conversation."
                    )
                
                elif "drug" in pattern_str or "meth" in pattern_str:
                    return False, (
                        "I cannot provide information on creating or manufacturing "
                        "drugs or illegal substances. This is important for safety and legal reasons. "
                        "If you have a substance use problem, please seek help from a "
                        "professional.\n"
                        "Resources:\n- SAMHSA National Helpline: 1-800-662-HELP (4357)\n"
                        "- National Institute on Drug Abuse: https://www.drugabuse.gov"
                    )
                
                else:
                    return False, (
                        f"I cannot provide information on this topic. "
                        f"If you have questions about health or safety, please ask in a different way.") 
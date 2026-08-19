# src/safety/guardrails.py
"""
Production-hardened, optimized safety guardrails for medical AI responses.
Implements pre-compiled regex architectures, modular taxonomy sets,
and multi-tier input/output policy enforcement.
"""

import re
import logging
from enum import Enum
from typing import Dict, Tuple, Optional, Any, Set, List

logger = logging.getLogger("guardrails")


class GuardrailType(Enum):
    INPUT_VALIDATION = "input_validation"
    OUTPUT_FILTERING = "output_filtering"
    DIAGNOSTIC_SOFTENING = "diagnostic_softening"
    DISCLAIMER_ADDITION = "disclaimer_addition"


class Guardrails:
    """
    Safety guardrails enforcement agent for MediQwen.
    Validates health-only focus scopes, intercepts prompt injections,
    and restricts software execution vulnerabilities.
    """
    
    # -----------------------------------------------------------------------
    # 1. EXPANDED MODULAR TAXONOMY CATEGORIES
    # -----------------------------------------------------------------------
    BODY_PARTS: Set[str] = {
        "heart", "stomach", "skin", "brain", "chest", "arm", "neck", "jaw", "back", 
        "liver", "kidney", "lung", "head", "eye", "throat", "bone", "abdomen", "belly", 
        "shoulder", "elbow", "wrist", "hand", "finger", "thumb", "hip", "pelvis", "groin", 
        "knee", "ankle", "foot", "toe", "ear", "nose", "mouth", "tongue", "teeth", "gums", 
        "cheek", "scalp", "spine", "breast", "joint", "muscle", "nerve", "vein", "artery",
        "tonsil", "sinus", "bladder", "colon", "bowel", "rectum", "rib", "skull"
    }

    SYMPTOMS: Set[str] = {
        "pain", "symptom", "ache", "cough", "fever", "rash", "bleed", "bleeding", "swelling", 
        "hurt", "injury", "cramp", "nausea", "vomiting", "dizzy", "fatigue", "tired", 
        "weakness", "numbness", "tingling", "diarrhea", "constipation", "chills", "sweating", 
        "dizziness", "fainting", "passed out", "palpitations", "shortness of breath", "wheezing", 
        "sneezing", "runny nose", "congestion", "itching", "burning", "loss of appetite", 
        "weight loss", "weight gain", "insomnia", "confusion", "memory loss", "sore", "stiff",
        "spasm", "discharge", "lump", "bump", "blister", "bruise", "redness", "inflammation", 
        "acne", "pimples", "breakout", "blackheads", "ulcer", "lesion", "tightness", "pressure", 
        "crushing", "numb", "weak", "exhaustion", "light headed"
    }

    MEDICATIONS: Set[str] = {
        "medicine", "pill", "drug", "dose", "treatment", "cure", "vaccine", "antibiotic", 
        "aspirin", "paracetamol", "ibuprofen", "insulin", "acetaminophen", "advil", "tylenol",
        "motrin", "prescription", "meds", "ointment", "cream", "drops", "syrup"
    }

    CLINICAL_CONCEPTS: Set[str] = {
        "doctor", "hospital", "disease", "condition", "illness", "sick", "clinical", "virus", 
        "infection", "angina", "attack", "hives", "ecg", "ekg", "mri", "ct scan", "xray",
        "ultrasound", "blood test", "lab", "clinic", "nurse", "physician", "pediatrician"
    }

    MENTAL_HEALTH: Set[str] = {
        "mental", "stress", "anxiety", "depression", "ptsd", "autism", "adhd", "bipolar", 
        "trauma", "therapy", "psychology", "panic", "insomnia", "hallucination", "delusion"
    }

    MEDICAL_EDUCATION_TERMS: Set[str] = {
        "hypertension", "diabetes", "asthma", "cancer", "stroke", "immune system", "vaccination",
        "vaccinated", "allergy", "allergies", "cholesterol", "blood pressure", "glucose",
        "arthritis", "migraine", "influenza", "flu", "pneumonia", "bronchitis"
    }

    MEDICAL_HEALTH_KEYWORDS: Set[str] = (
        BODY_PARTS | SYMPTOMS | MEDICATIONS | CLINICAL_CONCEPTS | MENTAL_HEALTH | MEDICAL_EDUCATION_TERMS
    )

    # -----------------------------------------------------------------------
    # 2. EMERGENCY PATTERNS
    # -----------------------------------------------------------------------
    EMERGENCY_PATTERNS: List[str] = [
        r"\b(chest|heart)\b.*\b(pain|hurt|ache|crushing|pressure|tight)\b",
        r"\b(pain|hurt|ache|crushing|pressure|tight)\b.*\b(chest|heart)\b",
        r"\b(can't|cannot|struggling|difficulty|trouble|unable)\b.*\b(breathe|breathing|breath)\b",
        r"\b(breathe|breathing|breath)\b.*\b(difficult|trouble|hard|short|loss)\b",
        r"\b(breathless|suffocating|choking|gasping)\b",
        r"\bface\b.*\b(droop|drooping|weak)\b",
        r"\b(weak|weakness|numb|numbness)\b.*\b(one\s+side|left\s+side)\b",
        r"\b(heavy|uncontrolled|severe)\b.*\b(bleed|bleeding|hemorrhage)\b",
        r"\b(loss\s+of\s+consciousness|unconscious|fainted|fainting|passed\s+out)\b",
        r"\b(heart\s+attack|stroke|seizure|anaphylaxis|shock)\b"
    ]

    # -----------------------------------------------------------------------
    # 3. POLICY INFRACTIONS & DANGEROUS TOPICS
    # -----------------------------------------------------------------------
    RESTRICTED_TOPICS: Set[str] = {
        r"how to (make|create|synthesize|manufacture) (drugs|meth|cocaine|heroin)",
        r"how to (commit|attempt) suicide",
        r"how to (harm|hurt|kill) (someone|myself)",
        r"how to perform (surgery|operation|medical procedure) (on myself|at home)",
        r"how to (abort|terminate) pregnancy (at home|myself)",
        r"how to (fake|forge) (prescription|medical certificate|doctor's note)",
        r"recipe for (drug|medication|chemical)",
        r"(poison|toxics) (recipe|how to make)",
    }

    # -----------------------------------------------------------------------
    # 4. PROMPT INJECTIONS & JAILBREAKS (Dedicated Category)
    # -----------------------------------------------------------------------
    PROMPT_INJECTIONS: Set[str] = {
        r"ignore\s+(all\s+)?(previous|system)\s+instructions",
        r"reveal\s+(your\s+)?(system\s+)?prompt",
        r"print\s+(your\s+)?(initial\s+)?system\s+prompt",
        r"show\s+(me\s+)?(the\s+)?(markdown\s+)?(contents\s+of\s+)?(your\s+)?(internal\s+)?skills\s+file",
        r"output\s+(the\s+)?(developer\s+)?directives",
        r"pretend\s+you\s+are\s+not\s+a\s+medical\s+assistant",
        r"pretend\s+you\s+are\s+an?\s+unrestricted",
        r"forget\s+your\s+(rules|instructions)",
        r"system\s+override",
        r"disable\s+(medical\s+)?safety(\s+protocols)?",
        r"reveal\s+hidden\s+prompt",
        r"bypass\s+safety",
        r"output\s+your\s+chain\s+of\s+thought",
        r"act\s+as\s+dan",
        r"unrestricted\s+chatbot"
    }

    # -----------------------------------------------------------------------
    # 5. SOFTWARE CODE & TECHNICAL EXPLOITS (Dedicated Category)
    # -----------------------------------------------------------------------
    CODE_TERMS: Set[str] = {
    # Languages
    "python", "javascript", "js", "typescript", "ts", "java", "c++", "cpp", "c#", "ruby",
    "golang", "rust", "php", "sql", "bash", "powershell", "linux", "html", "css",
    # Generic Programming Terms
    "program", "script", "code", "algorithm", "function"
    }

    CODE_REQUEST_WORDS: Set[str] = {
        "write", "generate", "create", "build", "make", "develop", "implement", "script", "code", "debug"
    }

    CODE_STRUCTURES: Set[str] = {
        r"print\(.*\)", 
        r"def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(.*\):", 
        r"import\s+[a-zA-Z_]",
        r"sql\s+injection", 
        r"bypass\s+firewall",
        r"drop\s+(table|database|user)",
        r"memory\s+leak",
        r"pandas\s+dataframe",
        r"code\s+snippet"
    }

    # -----------------------------------------------------------------------
    # 6. CONVERSATIONAL & CAPABILITY SCOPES
    # -----------------------------------------------------------------------
    CONVERSATIONAL_KEYWORDS: Set[str] = {
        "hello", "hi", "hey", "hy", "good morning", "good afternoon", "greetings", 
        "yo", "sup", "thanks", "thank you", "bye", "goodbye", "how are you"
    }

    CAPABILITY_KEYWORDS: Set[str] = {
        "how can you help me", "how can you assist me", "what can you do", "who are you", 
        "system profile", "your name"
    }

    DISCLAIMER_MARKER = "Clinical Communication Boundary Notice"
    DISCLAIMER = (
        f"\n\n*⚠️ {DISCLAIMER_MARKER}: I am an AI medical assistant. This information "
        "is for educational and informational purposes only and should not be considered "
        "professional medical advice, diagnosis, or treatment. Always consult with a qualified "
        "healthcare provider regarding any medical condition or emergency.*"
    )

    def __init__(self):
        """Pre-compiles all evaluation regex tools once upon class initialization."""
        self.compiled_emergencies = [re.compile(p, re.IGNORECASE) for p in self.EMERGENCY_PATTERNS]
        self.compiled_restricted = [re.compile(p, re.IGNORECASE) for p in self.RESTRICTED_TOPICS]
        self.compiled_prompt_injections = [re.compile(p, re.IGNORECASE) for p in self.PROMPT_INJECTIONS]
        self.compiled_code_structures = [re.compile(p, re.IGNORECASE) for p in self.CODE_STRUCTURES]
        logger.info("⚡ MediQwen Pre-compiled Guardrails Safety Engine Online.")

    def validate_input(self, text: str) -> Tuple[bool, str]:
        """Evaluates input safety using high-performance pre-compiled lookups."""
        text_clean = re.sub(r"\s+", " ", text).strip()
        text_lower = text_clean.lower()
        
        # Step 1: Emergency Interceptor Gate
        if any(pattern.search(text_lower) for pattern in self.compiled_emergencies):
            logger.info("🚨 Emergency signature validated. Routing to clinical priority queue.")
            return True, text_clean

        # Step 2: Prompt Injections & System Prompt Extractions
        for pattern in self.compiled_prompt_injections:
            if pattern.search(text_lower):
                logger.warning(f"🛑 Intercepted prompt injection attempt: {pattern.pattern}")
                return False, "I cannot fulfill this request. I am a specialized AI medical assistant and restricted from providing or revealing system directives."

        # Step 3: Restricted & Dangerous Topics
        for pattern in self.compiled_restricted:
            if pattern.search(text_lower):
                logger.warning(f"🛑 Intercepted restricted topic: {pattern.pattern}")
                if "suicide" in pattern.pattern or "harm" in pattern.pattern:
                    return False, (
                        "If you are in immediate danger or believe you may act on thoughts of self-harm, "
                        "please contact your local emergency services (like 999 or 911) or proceed straight "
                        "to your nearest medical center or crisis support hotline immediately. Help is available 24/7."
                    )
                return False, "I cannot fulfill this request. I am only permitted to assist with safe, standard medical inquiries."

        # Step 4: Technical & Functional Code Requests
        has_code_target = any(re.search(rf"\b{re.escape(term)}\b", text_lower) for term in self.CODE_TERMS)
        has_action_verb = any(re.search(rf"\b{re.escape(verb)}\b", text_lower) for verb in self.CODE_REQUEST_WORDS)
        has_code_syntax = any(pattern.search(text_lower) for pattern in self.compiled_code_structures)

        if (has_code_target and has_action_verb) or has_code_syntax:
            logger.warning("🛑 Intercepted non-medical technical code/program request.")
            return False, "I am a specialized AI medical assistant. I cannot write, debug, or execute programming code or software scripts."
        
        # Step 5: Whitelisted Conversational & System Competency Scopes
        has_isolated_greeting = any(re.search(rf"\b{re.escape(word)}\b", text_lower) for word in self.CONVERSATIONAL_KEYWORDS)
        has_capability_inquiry = any(phrase in text_lower for phrase in self.CAPABILITY_KEYWORDS)
        
        if has_isolated_greeting or has_capability_inquiry:
            logger.info("👋 Conversational framework approved. Passing down to routing layers.")
            return True, text_clean
        
        return True, text_clean

    def apply_output_guardrails(self, response_text: str, risk_metadata: Dict[str, Any]) -> str:
        """Appends mandatory institutional disclaimers using robust signature detection bounds."""
        if not response_text:
            return "An unexpected error occurred while compiling your medical response update."

        processed_text = response_text
        diagnostic_patterns = [
            (r"\byou\s+definitely\s+have\b", "your symptoms could be indicative of"),
            (r"\byou\s+have\b(?!\s+asked)", "your symptoms could be indicative of"),
            (r"\byou\s+are\s+suffering\s+from\b", "your clinical profile might suggest")
        ]
        for pattern, replacement in diagnostic_patterns:
            processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)

        if risk_metadata.get("risk_level") == "emergency":
            emergency_notice = (
                "🚨 **URGENT NOTICE:** Your query triggers critical emergency indicators. "
                "Please stop reading immediately and seek immediate medical attention from your local emergency services (Call 999 or 911).\n\n"
            )
            processed_text = emergency_notice + processed_text

        if self.DISCLAIMER_MARKER not in processed_text:
            processed_text += self.DISCLAIMER
            
        return processed_text

    # Alias for test compatibility
    format_output = apply_output_guardrails
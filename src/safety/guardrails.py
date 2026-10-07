# src/safety/guardrails.py
"""
Simplified, production-oriented safety guardrails for MediQwen.
Responsibilities:
1. Block prompt injection and system manipulation attempts.
2. Detect restricted or dangerous requests.
3. Detect self-harm / crisis signals.
4. Detect emergency medical signals.
5. Validate MediQwen's medical scope.
6. Apply conservative output safety checks.

Guardrails protect the pipeline.
LangGraph nodes remain responsible for routing, retrieval,
risk classification, and agent orchestration.
"""

import logging
import re
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger("guardrails")


class Guardrails:
    """
    Safety boundary for MediQwen.
    This class intentionally does NOT manage:
        - LangGraph routing
        - RAG retrieval
        - Web search
        - Dialogue state
        - Full risk classification

    It only validates safety, detects urgent signals,
    and determines whether input is within MediQwen's scope.
    """

    # =======================================================================
    # 1. MEDICAL TAXONOMY
    # =======================================================================

    BODY_PARTS: Set[str] = {
        "heart", "stomach", "skin", "brain", "chest", "arm",
        "neck", "jaw", "back", "liver", "kidney", "lung",
        "head", "eye", "throat", "bone", "abdomen", "belly",
        "shoulder", "elbow", "wrist", "hand", "finger", "thumb",
        "hip", "pelvis", "groin", "knee", "ankle", "foot",
        "toe", "ear", "nose", "mouth", "tongue", "teeth",
        "gums", "cheek", "scalp", "spine", "breast", "joint",
        "muscle", "nerve", "vein", "artery", "tonsil", "sinus",
        "bladder", "colon", "bowel", "rectum", "rib", "skull"
    }

    SYMPTOMS: Set[str] = {
        "pain", "symptom", "ache", "cough", "fever", "rash",
        "bleed", "bleeding", "swelling", "hurt", "injury",
        "cramp", "nausea", "vomiting", "dizzy", "dizziness",
        "fatigue", "tired", "weakness", "numbness", "tingling",
        "diarrhea", "constipation", "chills", "sweating",
        "fainting", "passed out", "palpitations",
        "shortness of breath", "wheezing", "sneezing",
        "runny nose", "congestion", "itching", "burning",
        "loss of appetite", "weight loss", "weight gain",
        "insomnia", "confusion", "memory loss", "sore", "stiff",
        "spasm", "discharge", "lump", "bump", "blister",
        "bruise", "redness", "inflammation", "acne",
        "pimples", "breakout", "blackheads", "ulcer",
        "lesion", "tightness", "pressure", "crushing",
        "numb", "weak", "exhaustion", "light headed", "cold", "colds", "flu", "infection", "allergy", "allergic reaction",
        "turning blue", "blue", "red", "yellow", "pale", "sweaty", "clammy", "chest tightness", "headache",
        "migraine", "blurred vision", "double vision", "vision loss", "hearing loss", "ringing in ears"
    }

    MEDICATIONS: Set[str] = {
        "medicine", "pill", "drug", "dose", "treatment",
        "cure", "vaccine", "antibiotic", "aspirin",
        "paracetamol", "ibuprofen", "insulin",
        "acetaminophen", "advil", "tylenol", "motrin",
        "prescription", "meds", "ointment", "cream",
        "drops", "syrup"
    }

    CLINICAL_CONCEPTS: Set[str] = {
        "doctor", "hospital", "disease", "condition",
        "illness", "sick", "clinical", "virus", "infection",
        "angina", "hives", "ecg", "ekg", "mri", "ct scan",
        "xray", "ultrasound", "blood test", "lab",
        "clinic", "nurse", "physician", "pediatrician"
    }

    MENTAL_HEALTH: Set[str] = {
        "mental", "stress", "anxiety", "depression",
        "ptsd", "autism", "adhd", "bipolar", "trauma",
        "therapy", "psychology", "panic", "insomnia",
        "hallucination", "delusion"
    }

    MEDICAL_EDUCATION_TERMS: Set[str] = {
        "hypertension", "diabetes", "asthma", "cancer",
        "stroke", "immune system", "vaccination",
        "vaccinated", "allergy", "allergies", "cholesterol",
        "blood pressure", "glucose", "arthritis", "migraine",
        "influenza", "flu", "pneumonia", "bronchitis"
    }

    # Preserved for backwards compatibility
    MEDICAL_HEALTH_KEYWORDS: Set[str] = (
        BODY_PARTS
        | SYMPTOMS
        | MEDICATIONS
        | CLINICAL_CONCEPTS
        | MENTAL_HEALTH
        | MEDICAL_EDUCATION_TERMS
    )

    NUTRITION_KEYWORDS: Set[str] = {
        "diet", "nutrition", "food", "foods", "meal",
        "calories", "protein", "carbohydrate", "carbs",
        "fat", "fiber", "vitamin", "minerals",
        "healthy eating", "weight management",
        "heart healthy", "nutrition plan"
    }

    CONVERSATIONAL_KEYWORDS: Set[str] = {
        "hello", "hi", "hey", "hy", "good morning",
        "good afternoon", "good evening", "greetings",
        "yo", "sup", "thanks", "thank you", "bye",
        "goodbye", "how are you"
    }

    CAPABILITY_KEYWORDS: Set[str] = {
        "how can you help me",
        "how can you assist me",
        "what can you do",
        "who are you",
        "your name",
        "capabilities",
        "capability",
        "assistant",
        "ai"
    }

    VAGUE_VISUAL_QUERIES: Set[str] = {
        "what is this",
        "what's this",
        "what could this be",
        "what is that",
        "what's that",
        "identify this",
        "identify it",
        "can you identify this",
        "is this normal",
        "does this look normal",
        "what happened",
        "what happened here",
        "help",
        "help me",
        "thoughts",
        "any thoughts",
        "can you look at this",
        "can you check this",
        "what do you think"
    }

    # =======================================================================
    # 2. SECURITY BOUNDARY
    # =====================================================================

    PROMPT_INJECTIONS: List[str] = [
    # -----------------------------------------------------------------------
    # 1. Direct instruction override
    # -----------------------------------------------------------------------
    r"\bignore\s+(?:all\s+)?(?:the\s+)?"
    r"(?:previous|prior|above|earlier|system|developer)\s+"
    r"(?:instructions?|rules?|directives?|constraints?|guidelines?)\b",

    r"\bdisregard\s+(?:all\s+)?(?:the\s+)?"
    r"(?:previous|prior|above|earlier|system|developer)\s+"
    r"(?:instructions?|rules?|directives?|constraints?|guidelines?)\b",

    r"\bforget\s+(?:all\s+)?(?:the\s+)?"
    r"(?:previous|prior|above|earlier)\s+"
    r"(?:instructions?|rules?|directives?|constraints?)\b",

    r"\bdo\s+not\s+follow\s+(?:the\s+)?"
    r"(?:previous|prior|system|developer)\s+"
    r"(?:instructions?|rules?|guidelines?)\b",

    r"\bstop\s+following\s+(?:your\s+)?"
    r"(?:system|developer|safety)\s+"
    r"(?:instructions?|rules?|policies?|guidelines?)\b",

    # -----------------------------------------------------------------------
    # 2. Explicit system/developer prompt extraction
    # -----------------------------------------------------------------------
    r"\b(?:reveal|show|display|print|output|provide|give)\s+"
    r"(?:me\s+)?(?:your\s+)?"
    r"(?:system|developer|hidden|internal)\s+"
    r"(?:prompt|instructions?|directives?|rules?|message)\b",

    r"\b(?:reveal|show|display|print|output|provide|give)\s+"
    r"(?:me\s+)?(?:the\s+)?"
    r"(?:full\s+|complete\s+|exact\s+)?"
    r"(?:system|developer)\s+prompt\b",

    r"\bwhat\s+(?:is|are)\s+(?:your\s+)?"
    r"(?:system|developer|hidden|internal)\s+"
    r"(?:prompt|instructions?|rules?|directives?)\b",

    r"\b(?:repeat|quote|recite|echo)\s+"
    r"(?:your\s+)?(?:system|developer|hidden|internal)\s+"
    r"(?:prompt|instructions?|message)\b",

    # -----------------------------------------------------------------------
    # 3. Hidden context / internal data extraction
    # -----------------------------------------------------------------------
    r"\b(?:reveal|show|print|output|dump|expose)\s+"
    r"(?:your\s+)?(?:hidden|internal|private|secret)\s+"
    r"(?:context|instructions?|message|configuration|config|rules?)\b",

    r"\b(?:show|reveal|output|dump)\s+"
    r"(?:everything|all\s+text|all\s+content)\s+"
    r"(?:before|above|behind)\s+"
    r"(?:my\s+)?(?:message|request|query)\b",

    r"\b(?:what|tell\s+me)\s+(?:came|was)\s+"
    r"(?:before|above)\s+(?:my\s+)?"
    r"(?:message|question|request)\b",

    # -----------------------------------------------------------------------
    # 4. Instruction hierarchy manipulation
    # -----------------------------------------------------------------------
    r"\b(?:system|developer|assistant|user)\s+message\s+"
    r"(?:override|replacement|replacement\s+instructions?)\b",

    r"\b(?:treat|consider|interpret)\s+(?:my\s+)?"
    r"(?:next|following)\s+(?:message|text|instruction)\s+"
    r"(?:as|like)\s+(?:a\s+)?"
    r"(?:system|developer)\s+(?:message|instruction)\b",

    r"\b(?:this|the\s+following)\s+"
    r"(?:is|should\s+be)\s+(?:your\s+)?"
    r"(?:new|replacement|updated)\s+"
    r"(?:system|developer)\s+(?:prompt|instructions?|rules?)\b",

    r"\b(?:pretend|assume|act\s+as\s+if)\s+"
    r"(?:the\s+)?(?:system|developer)\s+"
    r"(?:said|instructed|commanded)\b",

    # -----------------------------------------------------------------------
    # 5. Role / identity manipulation
    # -----------------------------------------------------------------------
    r"\b(?:you\s+are\s+now|from\s+now\s+on\s+you\s+are|"
    r"your\s+new\s+role\s+is)\b",

    r"\b(?:act|behave|respond|operate)\s+as\s+"
    r"(?:an?\s+)?(?:unrestricted|uncensored|unfiltered|"
    r"jailbroken|evil|malicious)\s+(?:ai|assistant|chatbot|model)\b",

    r"\b(?:pretend|roleplay)\s+(?:that\s+you\s+are|as)\s+"
    r"(?:an?\s+)?(?:unrestricted|uncensored|unfiltered|"
    r"jailbroken)\s+(?:ai|assistant|chatbot|model)\b",

    r"\b(?:you\s+are|act\s+as)\s+(?:DAN|dan)\b",

    r"\b(?:DAN|dan)\s+(?:mode|protocol|prompt)\b",

    # -----------------------------------------------------------------------
    # 6. Safety-policy bypass
    # -----------------------------------------------------------------------
    r"\b(?:bypass|circumvent|disable|remove|ignore|evade)\s+"
    r"(?:the\s+)?"
    r"(?:safety|security|content|moderation|guardrail|guardrails)\s+"
    r"(?:rules?|filters?|protocols?|restrictions?|controls?)?\b",

    r"\b(?:turn|switch)\s+off\s+"
    r"(?:the\s+)?(?:safety|security|moderation|guardrail|guardrails)\b",

    r"\b(?:disable|deactivate|remove)\s+"
    r"(?:your\s+)?(?:safety|security|content)\s+"
    r"(?:filters?|restrictions?|rules?|protocols?)\b",

    r"\b(?:ignore|bypass)\s+"
    r"(?:medical|clinical)\s+"
    r"(?:safety|security)\s+"
    r"(?:rules?|protocols?|restrictions?|guidelines?)\b",

    # -----------------------------------------------------------------------
    # 7. Jailbreak terminology
    # -----------------------------------------------------------------------
    r"\b(?:jailbreak|jailbroken|uncensored|unfiltered|unrestricted)\s+"
    r"(?:mode|model|assistant|chatbot|response)\b",

    r"\b(?:enable|activate|enter|switch\s+to)\s+"
    r"(?:jailbreak|developer|debug|unrestricted|uncensored|"
    r"unfiltered)\s+mode\b",

    r"\b(?:no|without)\s+(?:safety|content|moderation|"
    r"guardrail|guardrails)\s+(?:restrictions?|filters?|rules?)\b",

    # -----------------------------------------------------------------------
    # 8. Chain-of-thought / hidden reasoning extraction
    # -----------------------------------------------------------------------
    r"\b(?:reveal|show|print|output|provide|give)\s+"
    r"(?:your\s+)?"
    r"(?:chain\s+of\s+thought|chain-of-thought|"
    r"hidden\s+reasoning|private\s+reasoning|internal\s+reasoning)\b",

    r"\b(?:show|reveal|print|output)\s+"
    r"(?:all\s+)?(?:your\s+)?"
    r"(?:reasoning|thought\s+process|thinking\s+process)\b",

    r"\b(?:explain|describe)\s+"
    r"(?:your\s+)?(?:private|hidden|internal)\s+"
    r"(?:reasoning|thought\s+process)\b",

    # -----------------------------------------------------------------------
    # 9. Prompt reconstruction / extraction techniques
    # -----------------------------------------------------------------------
    r"\b(?:reconstruct|recover|extract|retrieve)\s+"
    r"(?:your\s+)?(?:system|developer|hidden|internal)\s+"
    r"(?:prompt|instructions?|message|context)\b",

    r"\b(?:leak|leakage|exfiltrate|extract)\s+"
    r"(?:the\s+)?(?:system|developer|hidden|internal)\s+"
    r"(?:prompt|instructions?|context|message)\b",

    r"\b(?:what\s+was|what\s+were)\s+"
    r"(?:your\s+)?(?:original|initial|hidden|system|developer)\s+"
    r"(?:prompt|instructions?|rules?)\b",

    # -----------------------------------------------------------------------
    # 10. Explicit instruction to expose policy/rules
    # -----------------------------------------------------------------------
    r"\b(?:list|enumerate|print|show|reveal)\s+"
    r"(?:all\s+)?(?:your\s+)?"
    r"(?:safety|security|moderation|guardrail|guardrails)\s+"
    r"(?:rules?|policies?|restrictions?|criteria|conditions)\b",

    r"\b(?:tell|show|explain)\s+me\s+"
    r"(?:exactly\s+)?how\s+your\s+"
    r"(?:safety|moderation|guardrail|guardrails)\s+"
    r"(?:works|rules?|filters?)\b",

    # -----------------------------------------------------------------------
    # 11. Common instruction-injection phrasing
    # -----------------------------------------------------------------------
    r"\b(?:new|updated|replacement)\s+instructions?\s*:",
    r"\b(?:system|developer)\s+prompt\s*:",
    r"\b(?:system|developer)\s+message\s*:",
    r"\b(?:assistant|model)\s+instructions?\s*:",
]

    # =======================================================================
    # 3. SELF-HARM / CRISIS DETECTION
    # =======================================================================

    SELF_HARM_PATTERNS: List[str] = [
        r"\bhow\s+(do|can)\s+i\s+(commit|attempt)\s+suicide\b",
        r"\bhow\s+to\s+(commit|attempt)\s+suicide\b",
        r"\bi\s+want\s+to\s+die\b",
        r"\bi\s+(don't|do\s+not)\s+want\s+to\s+live\b",
        r"\bi\s+want\s+to\s+kill\s+myself\b",
        r"\bi\s+am\s+going\s+to\s+kill\s+myself\b",
        r"\bi\s+(plan|planning)\s+to\s+kill\s+myself\b",
        r"\bi\s+want\s+to\s+hurt\s+myself\b",
        r"\bi\s+am\s+going\s+to\s+hurt\s+myself\b",
        r"\bi\s+(plan|planning)\s+to\s+hurt\s+myself\b",
        r"\bkill\s+myself\b"
    ]

    # =======================================================================
    # 4. RESTRICTED / DANGEROUS REQUESTS
    # =======================================================================

    RESTRICTED_TOPICS: List[str] = [
        r"how\s+to\s+(make|create|synthesize|manufacture)\s+(drugs|meth|cocaine|heroin)",
        r"how\s+to\s+(harm|hurt|kill)\s+(someone|another\s+person)",
        r"how\s+to\s+perform\s+(surgery|operation|medical\s+procedure)\s+(on\s+myself|at\s+home)",
        r"how\s+to\s+(fake|forge)\s+(prescription|medical\s+certificate|doctor'?s\s+note)",
        r"recipe\s+for\s+(drug|medication|poison)",
        r"(poison|toxic)\s+(recipe|how\s+to\s+make)"
    ]

    # =======================================================================
    # 5. EMERGENCY DETECTION
    # =======================================================================

    EMERGENCY_RULES: Dict[str, List[str]] = {
        "cardiac": [
            r"\b(chest|heart)\b.*\b(pain|hurt|ache|crushing|pressure|tightness|tight)\b",
            r"\b(pain|hurt|ache|crushing|pressure|tightness|tight)\b.*\b(chest|heart)\b",
            r"\bchest\s+pain\b.*\b(arm|jaw|neck|back)\b",
            r"\b(crushing|severe)\s+chest\s+pain\b"
        ],
        "respiratory": [
            r"\b(can't|cannot|unable|struggling|difficulty|trouble)\b.*\b(breathe|breathing|breath)\b",
            r"\bshortness\s+of\s+breath\b.*\b(severe|sudden|worse|worsening)\b",
            r"\b(severe|sudden)\b.*\bshortness\s+of\s+breath\b",
            r"\b(breathless|suffocating|choking|gasping|turning\s+blue)\b"
        ],
        "neurological": [
            r"\bface\b.*\b(droop|drooping)\b",
            r"\b(droop|drooping)\b.*\bface\b",
            r"\b(one\s+side|one-sided)\b.*\b(weak|weakness|numb|numbness)\b",
            r"\b(weak|weakness|numb|numbness)\b.*\b(one\s+side|one-sided)\b",
            r"\b(sudden|suddenly)\b.*\b(weakness|numbness|confusion|vision\s+loss)\b",
            r"\b(weakness|numbness|confusion|vision\s+loss)\b.*\b(sudden|suddenly)\b",
            r"\b(trouble|difficulty)\s+(speaking|talking)\b.*\b(sudden|suddenly)\b",
            r"\b(sudden|suddenly)\b.*\btrouble\s+(speaking|talking)\b",
            r"\b(slurred|slurring)\s+speech\b",
            r"\bsuddenly\s+(cannot|can't)\s+(speak|talk|move)\b"
        ],
        "severe_bleeding": [
            r"\b(heavy|uncontrolled|severe|profuse)\b.*\b(bleed|bleeding)\b",
            r"\b(bleed|bleeding)\b.*\b(heavy|uncontrolled|severe|profuse)\b",
            r"\bbleeding\b.*\b(won't|will\s+not|cannot|can't)\s+stop\b"
        ],
        "loss_of_consciousness": [
            r"\b(unconscious|loss\s+of\s+consciousness)\b",
            r"\b(passed\s+out|collapsed)\b",
            r"\bfainted\b.*\b(chest|breath|breathing|pain)\b"
        ],
        "anaphylaxis": [
            r"\b(having|experiencing)\b.*\b(anaphylaxis|anaphylactic)\b",
            r"\b(anaphylaxis|anaphylactic)\b.*\b(happening|now|currently)\b",
            r"\b(swelling)\b.*\b(throat|tongue|lips)\b.*\b(breathe|breathing|breath)\b",
            r"\b(throat|tongue|lips)\b.*\b(swelling)\b.*\b(breathe|breathing|breath)\b",
            r"\b(allergic\s+reaction)\b.*\b(breathe|breathing|breath|swelling)\b"
        ],
        "seizure": [
            r"\b(having|currently\s+having)\s+(a\s+)?seizure\b",
            r"\bseizure\b.*\b(not\s+stopping|won't\s+stop|cannot\s+stop)\b"
        ]
    }

    # =======================================================================
    # 6. TECHNICAL / CODE INTERCEPT PATTERNS
    # =======================================================================

    CODE_PATTERNS: List[str] = [
    # 1. Explicit fenced code blocks
    r"```[\s\S]*?```",

    # 2. Programming declarations / syntax (Self-contained flags & loops)
    r"(?m)^\s*(?:def|class|function|interface|struct|enum)\s+\w+",
    r"(?m)^\s*(?:import|from|using|include)\s+[A-Za-z_][\w.]*",
    r"\b(?:public|private|protected)\s+(?:class|static|void|int|string)\b",
    r"\b(?:const|let|var)\s+\w+\s*=",
    r"\b(?:if|for|while|switch)\s*\([^)]*\)\s*\{",  # C-style loops
    r"\bfor\s+\w+\s+in\s+(?:range|enumerate|zip)\s*\([^)]*\)\s*:",  # Python for loops
    r"\b(?:for|while|if|elif)\s+.*?\s*:\s*(?:print|return|break|continue|pass|raise)\b", # Inline Python blocks

    # 3. Explicit requests to write / generate / create code
    r"\b(?:write|generate|create|build|implement|develop)\s+(?:a\s+)?(?:script|program|function|class|code|application|app)\b",
    r"\b(?:write|generate|create|build|implement)\s+(?:a\s+)?(?:python|javascript|typescript|java|c\+\+|cpp|c#|csharp|sql|bash|shell|rust|golang|go|php|ruby|kotlin|swift)\b",

    # 4. Programming language + code context
    r"\b(?:python|javascript|typescript|java|c\+\+|cpp|c#|csharp|sql|bash|shell|rust|golang|go|php|ruby|kotlin|swift)\s+(?:code|script|program|function|class|application)\b",
    r"\b(?:code|script|program|function|class)\s+(?:in|using|with)\s+(?:python|javascript|typescript|java|c\+\+|cpp|c#|csharp|sql|bash|shell|rust|golang|go|php|ruby|kotlin|swift)\b",

    # 5. Debugging / programming error requests
    r"\b(?:debug|fix|refactor|optimize|compile|run)\s+(?:this|the|my)?\s*(?:code|script|program|function|class)\b",
    r"\b(?:debug|fix|troubleshoot)\s+(?:this\s+)?(?:python|javascript|typescript|java|c\+\+|cpp|c#|sql|bash|rust|golang|go|php|ruby|kotlin|swift)\b",

    # 6. Common programming / compiler errors
    r"\bsegmentation\s+fault\b",
    r"\bsegfault\b",
    r"\b(?:null|nil|nullptr)\s+pointer\b",
    r"\bpointer\s+(?:error|exception|issue|bug)\b",
    r"\bmemory\s+leak\b",
    r"\b(?:stack|heap)\s+(?:overflow|corruption)\b",
    r"\bcompiler\s+(?:error|warning)\b",
    r"\b(?:syntax|runtime|type|compile[ \t]*time)\s+error\b",
    r"\b(?:traceback|stack\s+trace)\b",
    r"\bundefined\s+(?:reference|variable|symbol)\b",
    r"\b(?:dependency|package)\s+(?:error|conflict)\b",

    # 7. Programming-specific terminology with strong technical context
    r"\b(?:API|SDK)\s+(?:endpoint|request|response|integration|authentication)\b",
    r"\b(?:REST|GraphQL)\s+(?:API|endpoint|query)\b",
    r"\b(?:database|DB)\s+(?:query|schema|migration|connection)\b",
    r"\b(?:regex|regular\s+expression)\s+(?:code|pattern|bug|syntax)\b",
    r"\b(?:Git|GitHub|GitLab)\s+(?:command|repository|branch|merge|commit)\b",

    # 8. SQL / database programming
    r"\b(?:SELECT|INSERT|UPDATE|DELETE)\s+.+\s+(?:FROM|INTO|SET|WHERE)\b",
    r"\bsql\s+injection\b",
    r"\b(?:write|generate|create)\s+(?:a\s+)?sql\s+(?:query|statement)\b",

    # 9. Explicit cybersecurity / system-bypass requests
    r"\b(?:bypass|disable|circumvent)\s+(?:a\s+)?(?:firewall|authentication|authorization|security|access\s+control)\b",
    r"\b(?:exploit|hack|penetrate)\s+(?:a\s+)?(?:system|server|network|website|application|database)\b",
    r"\b(?:reverse\s+shell|remote\s+shell|privilege\s+escalation)\b",

    # 10. Strong programming context
    r"\b(?:IDE|compiler|interpreter|runtime|package\s+manager|virtual\s+environment|dependency|repository|commit|branch)\b"
]
    
    # =======================================================================
    # 7. OUTPUT SAFETY
    # =======================================================================

    DISCLAIMER_MARKER = "Clinical Communication Boundary Notice"

    DISCLAIMER = (
        "\n\n"
        f"*⚠️ {DISCLAIMER_MARKER}: MediQwen provides educational and "
        "informational support and does not replace professional medical "
        "advice, diagnosis, or treatment. Consult a qualified healthcare "
        "professional for medical concerns.*"
    )

    EMERGENCY_NOTICE = (
        "🚨 **URGENT:** Your symptoms may indicate a medical emergency. "
        "Please seek immediate medical care or contact your local emergency "
        "service. Do not rely on this chat as a substitute for emergency care.\n\n"
    )

    DIAGNOSTIC_CERTAINTY_PATTERNS: List[Tuple[str, str]] = [
        (
            r"\byou\s+definitely\s+have\b",
            "your symptoms may be consistent with"
        ),
        (
            r"\byou\s+certainly\s+have\b",
            "your symptoms may be consistent with"
        ),
        (
            r"\bthis\s+is\s+definitely\b",
            "this may be"
        ),
        (
            r"\bwithout\s+a\s+doubt\b",
            "based on the available information"
        )
    ]

    # =======================================================================
    # 8. PRE-COMPILED REGEX
    # =======================================================================

    _COMPILED_PROMPT_INJECTIONS = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in PROMPT_INJECTIONS
    ]

    _COMPILED_SELF_HARM = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in SELF_HARM_PATTERNS
    ]

    _COMPILED_RESTRICTED = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in RESTRICTED_TOPICS
    ]

    _COMPILED_CODE = [
        re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        for pattern in CODE_PATTERNS
    ]

    _COMPILED_EMERGENCY_RULES = {
        category: [
            re.compile(pattern, re.IGNORECASE)
            for pattern in patterns
        ]
        for category, patterns in EMERGENCY_RULES.items()
    }

    _RE_HEALTH_KEYWORDS = re.compile(
        r"\b("
        + "|".join(
            re.escape(word)
            for word in sorted(
                MEDICAL_HEALTH_KEYWORDS,
                key=len,
                reverse=True
            )
        )
        + r")\b",
        re.IGNORECASE
    )

    _RE_NUTRITION_KEYWORDS = re.compile(
        r"\b("
        + "|".join(
            re.escape(word)
            for word in sorted(
                NUTRITION_KEYWORDS,
                key=len,
                reverse=True
            )
        )
        + r")\b",
        re.IGNORECASE
    )

    _RE_CONVERSATIONAL = re.compile(
        r"\b("
        + "|".join(
            re.escape(word)
            for word in sorted(
                CONVERSATIONAL_KEYWORDS,
                key=len,
                reverse=True
            )
        )
        + r")\b",
        re.IGNORECASE
    )

    _COMPILED_DIAGNOSTIC_CERTAINTY = [
        (
            re.compile(pattern, re.IGNORECASE),
            replacement
        )
        for pattern, replacement in DIAGNOSTIC_CERTAINTY_PATTERNS
    ]

    # =======================================================================
    # INITIALIZATION
    # =======================================================================

    def __init__(self):
        logger.info("⚡ MediQwen Guardrails Safety Engine Online.")

    # =======================================================================
    # NORMALIZATION
    # =======================================================================

    @staticmethod
    def normalize_input(text: str) -> str:
        """
        Normalize whitespace for single-line semantic matching.
        """
        if not isinstance(text, str):
            return ""

        return re.sub(r"\s+", " ", text).strip()

    # =======================================================================
    # SECURITY DETECTION
    # =======================================================================

    def detect_prompt_injection(self, text: str) -> Dict[str, Any]:
        """
        Detect explicit prompt injection or system manipulation attempts.
        """
        for pattern in self._COMPILED_PROMPT_INJECTIONS:
            if pattern.search(text):
                return {
                    "detected": True,
                    "matched_pattern": pattern.pattern
                }

        return {
            "detected": False,
            "matched_pattern": None
        }

    # =======================================================================
    # SELF-HARM / CRISIS DETECTION
    # =======================================================================

    def detect_self_harm(self, text: str) -> Dict[str, Any]:
        """
        Detect explicit self-harm or suicide crisis signals.
        """
        for pattern in self._COMPILED_SELF_HARM:
            if pattern.search(text):
                return {
                    "detected": True,
                    "matched_pattern": pattern.pattern
                }

        return {
            "detected": False,
            "matched_pattern": None
        }

    @staticmethod
    def self_harm_response() -> str:
        """
        Safe response for detected self-harm or suicide signals.
        """
        return (
            "I'm sorry that you're going through this. "
            "If you may be in immediate danger or think you might hurt "
            "yourself, please contact your local emergency service or go "
            "to the nearest emergency department now. "
            "If possible, stay with someone you trust and tell them what "
            "you're experiencing."
        )

    # =======================================================================
    # RESTRICTED CONTENT
    # =======================================================================

    def detect_restricted_content(self, text: str) -> Dict[str, Any]:
        """
        Detect dangerous requests outside MediQwen's permitted scope.
        """
        for pattern in self._COMPILED_RESTRICTED:
            if pattern.search(text):
                return {
                    "detected": True,
                    "matched_pattern": pattern.pattern
                }

        return {
            "detected": False,
            "matched_pattern": None
        }

    # =======================================================================
    # EMERGENCY DETECTION
    # =======================================================================

    def detect_emergency(self, text: str) -> Dict[str, Any]:
        """
        Detect possible emergency medical signals.
        """
        normalized = self.normalize_input(text).lower()

        for category, patterns in self._COMPILED_EMERGENCY_RULES.items():
            for pattern in patterns:
                match = pattern.search(normalized)
                if match:
                    return {
                        "is_emergency": True,
                        "category": category,
                        "matched_pattern": pattern.pattern,
                        "matched_text": match.group(0),
                        "risk_level": "EMERGENCY"
                    }

        return {
            "is_emergency": False,
            "category": None,
            "matched_pattern": None,
            "matched_text": None,
            "risk_level": None
        }

    def is_emergency_query(self, text: str) -> bool:
        """
        Backwards-compatible emergency helper.
        """
        return self.detect_emergency(text)["is_emergency"]

    # =======================================================================
    # TECHNICAL / CODE DETECTION
    # =======================================================================

    def is_code_request(self, text: str) -> bool:
        """
        Detect programming or technical requests using raw text.
        Preserves original newlines for multiline pattern anchors.
        """
        return any(
            pattern.search(text)
            for pattern in self._COMPILED_CODE
        )

    # =======================================================================
    # SCOPE DETECTION
    # =======================================================================

    def classify_scope(
        self,
        text: str,
        has_image: bool = False
    ) -> str:
        """
        Lightweight scope classification.
        """
        normalized = self.normalize_input(text).lower()

        if has_image:
            if (
                normalized in self.VAGUE_VISUAL_QUERIES
                or len(normalized.split()) <= 6
            ):
                return "MULTIMODAL"

        if self._RE_HEALTH_KEYWORDS.search(normalized):
            return "MEDICAL"

        if self._RE_NUTRITION_KEYWORDS.search(normalized):
            return "NUTRITION"

        if self._RE_CONVERSATIONAL.search(normalized):
            return "CASUAL"

        if any(phrase in normalized for phrase in self.CAPABILITY_KEYWORDS):
            return "CASUAL"

        return "OUT_OF_SCOPE"

    # =======================================================================
    # MAIN INPUT VALIDATION
    # =======================================================================

    def validate_input(
        self,
        text: str,
        has_image: bool = False
    ) -> Tuple[bool, str]:
        """
        Validate user input using normalized and raw representations.
        Returns (is_safe, processed_text_or_response).
        """
        text_raw = text
        text_clean = self.normalize_input(text)

        if not text_clean:
            return False, "Please provide a medical or health-related question."

        # 1. Prompt Injection
        injection = self.detect_prompt_injection(text_clean)
        if injection["detected"]:
            logger.warning(
                "🛑 Prompt injection intercepted: %s",
                injection["matched_pattern"]
            )
            return False, (
                "I cannot fulfill requests to override instructions, "
                "bypass safety measures, or reveal internal system details."
            )

        # 2. Self-Harm / Crisis
        crisis = self.detect_self_harm(text_clean)
        if crisis["detected"]:
            logger.warning("🚨 Self-harm / crisis signal detected.")
            return False, self.self_harm_response()

        # 3. Restricted Content
        restricted = self.detect_restricted_content(text_clean)
        if restricted["detected"]:
            logger.warning(
                "🛑 Restricted request intercepted: %s",
                restricted["matched_pattern"]
            )
            return False, (
                "I cannot provide instructions that could facilitate "
                "harmful, dangerous, or unsafe activities."
            )

        # 4. Emergency Detection (Non-blocking)
        emergency = self.detect_emergency(text_clean)
        if emergency["is_emergency"]:
            logger.warning(
                "🚨 Emergency signal detected | category=%s | match=%s",
                emergency["category"],
                emergency["matched_text"]
            )

        # 5. Code / Technical Requests (Evaluated against raw multiline text)
        if self.is_code_request(text_raw):
            logger.info("🛑 Technical/code request blocked.")
            return False, (
                "I am a specialized medical AI assistant designed specifically for health and clinical information, so I cannot "
                "provide programming or software development assistance. If you have a health-related question, I’d be happy to help."
            )

    
        return True, text_clean

    # =======================================================================
    # OUTPUT GUARDRAILS
    # =======================================================================

    def apply_output_guardrails(
        self,
        response_text: str,
        risk_metadata: Dict[str, Any] = None
    ) -> str:
        """
        Applies deterministic safety controls to model outputs.
        
        Responsibilities:
            - Softens diagnostic certainty claims.
            - Prepends deterministic, region-neutral emergency guidance.
            - Appends educational disclaimers exclusively for non-emergency medical responses.
        """
        if not response_text:
            return "I was unable to generate a response. Please try again."

        if risk_metadata is None:
            risk_metadata = {}

        processed_text = response_text.strip()

        # 1. Soften diagnostic certainty phrasing
        for pattern, replacement in self._COMPILED_DIAGNOSTIC_CERTAINTY:
            processed_text = pattern.sub(replacement, processed_text)

        risk_level = str(risk_metadata.get("risk_level", "")).upper()
        is_emergency = (
            risk_level == "EMERGENCY" 
            or risk_metadata.get("is_emergency", False)
            or risk_metadata.get("guardrail_emergency", False)
        )

        # 2. EMERGENCY PATH: Prepend deterministic banner and return (no duplicate disclaimers)
        if is_emergency:
            emergency_banner = (
                "🚨 **Emergency Guidance**\n\n"
                "Your symptoms may indicate a medical emergency. "
                "Please contact your local emergency services immediately "
                "or go to the nearest emergency department. "
                "Do not delay seeking urgent medical care.\n\n"
                "───────────────────────────────────────\n\n"
            )
            if "Emergency Guidance" not in processed_text:
                processed_text = emergency_banner + processed_text
            
            return processed_text

        # 3. STANDARD CLINICAL PATH: Append medical disclaimer
        is_medical = risk_metadata.get("is_medical", False) or (risk_level in ["MEDIUM", "HIGH"])
        if is_medical and self.DISCLAIMER_MARKER not in processed_text:
            processed_text += self.DISCLAIMER

        return processed_text


# Backwards compatibility alias
format_output = Guardrails.apply_output_guardrails
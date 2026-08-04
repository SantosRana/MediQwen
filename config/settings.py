from pathlib import Path
import re

# Model Configuration
GEMMA_MODEL_ID="ollama/mediqwen:latest"  # Ollama model identifier for MediQwen

# Device
DEVICE="auto"  # auto, cuda, cpu, mps
# ChromaDB
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_db")
CHROMA_COLLECTION_NAME="medical_kb"
# Embeddings
EMBEDDING_MODEL="BAAI/bge-large-en-v1.5"
# Safety
ENABLE_SAFETY_CHECKS=True
MAX_RISK_LEVEL="high"  # emergency, high, medium, low
# Audio (optional)
ENABLE_AUDIO=False
WHISPER_MODEL="base"
# LangGraph
MAX_ITERATIONS=5
MAX_TOOL_CALLS=10


# Risk Levels
RISK_LEVELS = {
    "emergency": 4,
    "high": 3,
    "medium": 2,
    "low": 1
}

# Medical domains for RAG filtering
MEDICAL_DOMAINS = [
    "general_medicine",
    "first_aid",
    "pediatrics",
    "dermatology",
    "cardiology",
    "respiratory",
    "gastroenterology",
    "neurology",
    "mental",
    "cardiovascular",
    "pharmacology",
    "global_health",
    "diabetes"
]

# Structured output schema
OUTPUT_SCHEMA = {
    "possible_conditions": [],
    "risk_level": "",
    "confidence": "",
    "advice": "",
    "doctor_recommendation": False,
    "urgency_note": "",
    "disclaimer": "",
    "sources": []
}

TRUSTED_WEB_SOURCES = {
    "who.int": {"name": "WHO", "score": 10},
    "nhs.uk": {"name": "NHS", "score": 10},
    "mayoclinic.org": {"name": "Mayo Clinic", "score": 9},
    "hsph.harvard.edu": {"name": "Harvard Nutrition", "score": 9},
    "healthline.com": {"name": "Healthline", "score": 8},
    "webmd.com": {"name": "WebMD", "score": 8},
    "eatright.org": {"name": "Academy of Nutrition", "score": 8}
    
}

QUERY_FILLERS = [
    re.compile(r"\bwhat is\b", re.IGNORECASE),
    re.compile(r"\bwhat are\b", re.IGNORECASE),
    re.compile(r"\btell me about\b", re.IGNORECASE),
    re.compile(r"\bplease explain\b", re.IGNORECASE),
    re.compile(r"\bsymptoms of\b", re.IGNORECASE),
    re.compile(r"\bcauses of\b", re.IGNORECASE),
    re.compile(r"\btreatment for\b", re.IGNORECASE),
    re.compile(r"\bdiagnosis of\b", re.IGNORECASE),
    re.compile(r"\bcan you provide me the\b", re.IGNORECASE),
    re.compile(r"\bgive me an\b", re.IGNORECASE),
    re.compile(r"\bgive me a\b", re.IGNORECASE),
    re.compile(r"\bi want a\b", re.IGNORECASE)
]

STRUCTURAL_PHRASES = [
    re.compile(r"\bok(?:ay)?\b", re.IGNORECASE),
    re.compile(r"\blet'?s move on\b", re.IGNORECASE),
    re.compile(r"\blet'?s change the subject\b", re.IGNORECASE),
    re.compile(r"\bby the way\b", re.IGNORECASE),
    re.compile(r"\bcan you provide\b", re.IGNORECASE),
    re.compile(r"\bi want\b", re.IGNORECASE),
    re.compile(r"\bhow do use\b", re.IGNORECASE)
]

NUTRITION_KEYWORDS = {
    "nutrition", "protein", "vitamin", "mineral", "diet", "calories", "food", "fat",
    "carbohydrate", "fiber", "healthy eating", "meal", "nutrient"
}

ALLERGY_KEYWORDS = {
    "allergy", "allergen", "rash", "swelling", "anaphylaxis", "reaction", "hives"
}

# Pre-compile clean word tokens extraction rule
TOKEN_REGEX = re.compile(r"\b\w+\b")

GENERIC_IMAGE_PHRASES = [
    # Identification
    "what is this",
    "what's this",
    "what could this be",
    "what might this be",
    "what does this look like",
    "what am i looking at",
    "what do you think this is",
    "identify this",
    "identify it",
    "can you identify this",
    "can you identify it",
    "can you tell what this is",

    # Opinion
    "what do you think",
    "thoughts",
    "any thoughts",
    "your thoughts",
    "opinions",
    "any idea",
    "ideas",

    # Normality
    "is this normal",
    "does this look normal",
    "is this okay",
    "does this look okay",
    "is this bad",
    "does this look bad",
    "should i be worried",

    # Medical uncertainty
    "what could be causing this",
    "what could cause this",
    "any clue what this is",
    "any idea what this is",
    "what do you make of this",
    "what is causing this",
    "what is causing it",
    "what is the reason for this",
    

    # Very short prompts
    "help",
    "please help",
    "look at this",
    "check this",
    "check this out",
    "can you check this",
    "can you look at this",
    "take a look",
    "have a look",

    # Single-word prompts
    "help?",
    "why",
    "why?",
    "advice",
    "thoughts?",
    "opinion",
    "question",

    # Common follow-ups
    "is this concerning",
    "is this serious",
    "is this something to worry about",
    "should i see a doctor",
    "can you explain this",
]
from pathlib import Path
import re

# Model Configuration
MODEL_ID="ollama/mediqwen:latest"  # Ollama model identifier for MediQwen

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
    # General nutrition
    "nutrition",
    "nutritional",
    "diet",
    "dietary",
    "food",
    "foods",
    "eating",
    "meal",
    "meals",

    # Healthy eating
    "healthy diet",
    "healthy eating",
    "balanced diet",
    "heart-healthy diet",
    "heart healthy diet",
    "healthy foods",
    "nutritious",
    "nutrient",

    # Dietary components
    "protein",
    "fiber",
    "fibre",
    "carbohydrates",
    "carbs",
    "fat",
    "fats",
    "sodium",
    "salt",
    "sugar",
    "calories",
    "cholesterol",

    # Dietary patterns
    "mediterranean diet",
    "low-sodium diet",
    "low salt diet",
    "plant-based diet",
    "vegetarian diet",
    "vegan diet",
}

ALLERGY_KEYWORDS = {
    "allergy", "allergen", "rash", "swelling", "anaphylaxis", "reaction", "hives"
}

# Pre-compile clean word tokens extraction rule
TOKEN_REGEX = re.compile(r"\b\w+\b")

# Intents that trigger vector retrieval on follow-ups (Skill 1: Deferred Retrieval)
RETRIEVAL_PHRASES = (
    # Direct imperative / request phrases
    "give me",
    "give me a",
    "provide",
    "show me",
    "recommend",
    "diet for",
    "plan for",
    "guidelines for",
    
    # Treatment / management
    "treatment",
    "what should i do",
    "how should i treat",
    "how is this treated",
    "what is the treatment",
    "treatment options",
    "recommended treatment",
    "management options",
    "medication options",
    "at home relief",
    "how to cure",

    # Causes / prevention
    "what causes this",
    "what are the causes",
    "what are the triggers",
    "how can i prevent this",

    # Evidence / guidelines
    "guidelines",
    "clinical guidelines",
    "evidence-based",
    "evidence based",
    "evidence-based guidelines",
    "evidence based guidelines",
    "recommendations",
    "clinical recommendations",
    "management guidelines",

    # Symptoms / warning signs
    "symptoms",
    "warning signs",
    "what are the warning signs",
    "when should i seek medical help",
    "when should i see a doctor",

    # Diagnosis / complications
    "diagnosis",
    "complications",
    "what are the complications",
    "side effects",
    "risks",
)

AFFIRMATION_EXPRESSIONS = {
    "yes", "yeah", "yep", "sure", "ok", "okay", "please", 
    "tell me more", "go ahead", "yes please", "sure thing", "i would"
}

# Regex subject extractor with dynamic term fallback
MEDICAL_SUBJECT_PATTERNS = [
    r"\b(hives|urticaria)\b",
    r"\b(eczema|dermatitis)\b",
    r"\b(rash|lesion|welts?)\b",
    r"\b(blister|burn)\b",
    r"\b(ringworm|fungal|tinea)\b",
    r"\b(psoriasis|acne|rosacea)\b",
    r"\b(shingles|herpes|chickenpox)\b"
]
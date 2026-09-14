from pathlib import Path
import re

# Model Configuration
MODEL_ID = "mediqwen:latest"  # Direct Ollama /api/chat model naming

# Device & Storage
DEVICE = "auto"  # auto, cuda, cpu, mps
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_db")
CHROMA_COLLECTION_NAME = "medical_kb"

# Embeddings
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# Safety Controls
ENABLE_SAFETY_CHECKS = True

# FSM & Intent Keywords (Used in run_dialogue_manager)
AFFIRMATION_EXPRESSIONS = {
    "yes",
    "yeah",
    "yep",
    "sure",
    "ok",
    "okay",
    "please",
    "tell me more",
    "go ahead",
    "yes please",
    "sure thing",
    "please do",
    "why not",
    "absolutely",
    "go on",
    "sounds good",
    "carry on",
    "please continue",
    "continue",
    "i'm listening",
    
}

# Intents that route toward clinical information requests and trigger deferred RAG
CLINICAL_INFORMATION_INTENTS = (
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

# Trusted Web Domain Scoring for Fallback Search
TRUSTED_WEB_SOURCES = {
    "who.int": {"name": "WHO", "score": 10},
    "nhs.uk": {"name": "NHS", "score": 10},
    "mayoclinic.org": {"name": "Mayo Clinic", "score": 9},
    "hsph.harvard.edu": {"name": "Harvard Nutrition", "score": 9},
    "healthline.com": {"name": "Healthline", "score": 8},
    "webmd.com": {"name": "WebMD", "score": 8},
    "eatright.org": {"name": "Academy of Nutrition", "score": 8},
}

# Regex subject extractor with dynamic term fallback for multimodal turns
MEDICAL_SUBJECT_PATTERNS = [
    # Dermatology
    r"\b(hives|urticaria)\b",
    r"\b(eczema|dermatitis)\b",
    r"\b(rash|lesion|welts?)\b",
    r"\b(blister|burn)\b",
    r"\b(ringworm|fungal infection|tinea)\b",
    r"\b(psoriasis|acne|rosacea)\b",
    r"\b(shingles|herpes|chickenpox)\b",

    # Cardiovascular
    r"\b(angina|heart disease|hypertension)\b",

    # Neurological
    r"\b(migraine|headache|stroke)\b",

    # Autoimmune / inflammatory
    r"\b(lupus|arthritis)\b",

    # Respiratory
    r"\b(asthma|pneumonia|bronchitis)\b",
]
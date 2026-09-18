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
    # --- Dermatology ---
    r"\b(hives|urticaria)\b",
    r"\b(eczema|dermatitis)\b",
    r"\b(rash|lesion|welts?)\b",
    r"\b(blister|burn)\b",
    r"\b(ringworm|fungal infection|tinea)\b",
    r"\b(psoriasis|acne|rosacea)\b",
    r"\b(shingles|herpes|chickenpox)\b",
    r"\b(hair loss|alopecia)\b",
    r"\b(dandruff)\b",
    r"\b(wart|warts)\b",

    # --- Cardiovascular ---
    r"\b(angina|heart disease|hypertension)\b",
    r"\b(chest pain|chest tightness|palpitations)\b",
    r"\b(arrhythmia|irregular heartbeat)\b",
    r"\b(heart attack|myocardial infarction)\b",
    r"\b(high cholesterol)\b",

    # --- Neurological ---
    r"\b(migraine|headache|stroke)\b",
    r"\b(numbness|dizziness|fainting)\b",
    r"\b(seizure|epilepsy)\b",
    r"\b(tremor)\b",
    r"\b(vertigo)\b",
    r"\b(memory loss|confusion)\b",

    # --- Respiratory ---
    r"\b(asthma|pneumonia|bronchitis)\b",
    r"\b(cough|wheezing|shortness of breath)\b",
    r"\b(common cold|flu|influenza)\b",
    r"\b(sore throat)\b",

    # --- Gastrointestinal ---
    r"\b(stomach pain|abdominal pain|diarrhea|constipation)\b",
    r"\b(nausea|vomiting|heartburn|acid reflux|gerd)\b",
    r"\b(bloating)\b",
    r"\b(stomach ulcer|peptic ulcer)\b",
    r"\b(hemorrhoids)\b",

    # --- Musculoskeletal ---
    r"\b(arthritis)\b",
    r"\b(joint pain|back pain|muscle strain)\b",
    r"\b(sprain|fracture)\b",
    r"\b(tendinitis)\b",

    # --- ENT ---
    r"\b(ear infection|otitis)\b",
    r"\b(tinnitus|ringing in ears)\b",
    r"\b(hearing loss)\b",
    r"\b(sinusitis|sinus infection)\b",

    # --- Endocrine / Metabolic ---
    r"\b(diabetes)\b",
    r"\b(hypothyroidism|hyperthyroidism|thyroid disorder)\b",
    r"\b(obesity)\b",

    # --- Mental Health (the condition itself, not the act of seeking care) ---
    r"\b(anxiety|panic attack)\b",
    r"\b(depression)\b",
    r"\b(insomnia)\b",
    r"\b(ptsd|post-traumatic stress)\b",
    r"\b(bipolar disorder)\b",

    # --- Infectious Disease ---
    r"\b(covid|coronavirus)\b",
    r"\b(strep throat)\b",

    # --- Ophthalmologic ---
    r"\b(conjunctivitis|pink eye)\b",
    r"\b(blurred vision|double vision|vision loss)\b",
    r"\b(cataract)\b",

    # --- Urological / Renal ---
    r"\b(kidney stones?)\b",
    r"\b(urinary tract infection|uti|bladder infection)\b",

    # --- Women's Health ---
    r"\b(menstrual cramps)\b",
    r"\b(yeast infection)\b",

    # --- Allergy / Immune ---
    r"\b(hay fever|seasonal allergies)\b",
    r"\b(food allergy)\b",
    r"\b(anaphylaxis)\b",

    # --- Dental / Oral ---
    r"\b(toothache)\b",
    r"\b(gum disease|gingivitis)\b",
    r"\b(cavity|cavities)\b",

    # --- Hematological ---
    r"\b(anemia)\b",
    r"\b(blood clot)\b",

    # --- General (symptom presentations broad enough to still be a subject) ---
    r"\b(fever|infection|allergic reaction|inflammation)\b",
    r"\b(fatigue|weight loss|weight gain)\b",
]
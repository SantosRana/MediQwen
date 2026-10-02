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

# Regex Search Tokens & Filler Normalizers
TOKEN_REGEX = re.compile(r"\b\w+\b")

STRUCTURAL_PHRASES = [
    re.compile(r"\bwhat (are|is) (the|its)?\b", re.IGNORECASE),
    re.compile(r"\bhow (can|do) i (treat|manage|prevent)\b", re.IGNORECASE),
    re.compile(r"\btell me about\b", re.IGNORECASE),
]

QUERY_FILLERS = [
    re.compile(r"\bplease\b", re.IGNORECASE),
    re.compile(r"\bcan you\b", re.IGNORECASE),
    re.compile(r"\bcould you\b", re.IGNORECASE),
]

# Intent Keywords for Web Search Scoping
ALLERGY_KEYWORDS = {
    "hives", "urticaria", "rash", "allergy", "allergic", "swelling",
    "itching", "eczema", "dermatitis", "anaphylaxis", "histamine"
}

NUTRITION_KEYWORDS = {
    "diet", "food", "nutrition", "eat", "eating", "vitamin", "meal",
    "calories", "protein", "carbs", "magnesium", "supplement", "nutrients"
}

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
    # --- Infectious / Zoonotic / Viral (Fixes Rabies & Common Pathogens) ---
    r"\b(rabies|hydrophobia)\b",
    r"\b(tetanus|lockjaw)\b",
    r"\b(sepsis|septicemia)\b",
    r"\b(tuberculosis|tb)\b",
    r"\b(malaria|dengue|zika|cholera|typhoid)\b",
    r"\b(measles|mumps|rubella|chickenpox|shingles|smallpox|mpox)\b",
    r"\b(covid(?:-?19)?|coronavirus|sars)\b",
    r"\b(common colds?|flus?|influenza)\b",
    r"\b(strep(?:tococcal)?\s*(?:throat)?)\b",
    r"\b(mono|mononucleosis)\b",

    # --- Dermatology ---
    r"\b(hives|urticaria)\b",
    r"\b(eczema|dermatitis)\b",
    r"\b(rash(?:es)?|lesion(?:s)?|welts?)\b",
    r"\b(blister(?:s)?|burn(?:s)?)\b",
    r"\b(ringworm|fungal infection|tinea)\b",
    r"\b(psoriasis|acne|rosacea)\b",
    r"\b(hair loss|alopecia)\b",
    r"\b(dandruff)\b",
    r"\b(warts?)\b",

    # --- Cardiovascular ---
    r"\b(angina|heart disease|hypertension|high blood pressure)\b",
    r"\b(chest pain|chest tightness|palpitations?)\b",
    r"\b(arrhythmia|irregular heartbeat)\b",
    r"\b(heart attack|myocardial infarction)\b",
    r"\b(high cholesterol)\b",

    # --- Neurological ---
    r"\b(migraines?|headaches?|strokes?)\b",
    r"\b(numbness|dizziness|fainting|syncope)\b",
    r"\b(seizures?|epilepsy)\b",
    r"\b(tremors?)\b",
    r"\b(vertigo)\b",
    r"\b(memory loss|dementia|alzheimer'?s|confusion)\b",

    # --- Respiratory ---
    r"\b(asthma|pneumonia|bronchitis)\b",
    r"\b(coughs?|coughing|wheezing|shortness of breath|dyspnea)\b",
    r"\b(sore throat)\b",
    r"\b(copd|emphysema)\b",

    # --- Gastrointestinal ---
    r"\b(stomach pain|abdominal pain|diarrhea|constipation)\b",
    r"\b(nausea|vomiting|heartburn|acid reflux|gerd)\b",
    r"\b(bloating)\b",
    r"\b(stomach ulcers?|peptic ulcers?)\b",
    r"\b(hemorrhoids?)\b",
    r"\b(ibs|irritable bowel syndrome|crohn'?s|ulcerative colitis)\b",

    # --- Musculoskeletal ---
    r"\b(arthritis|gout)\b",
    r"\b(joint pain|back pain|muscle strain|neck pain)\b",
    r"\b(sprains?|fractures?|broken bone(?:s)?)\b",
    r"\b(tendinitis|bursitis)\b",
    r"\b(osteoporosis)\b",

    # --- ENT ---
    r"\b(ear infections?|otitis(?:\s+media)?)\b",
    r"\b(tinnitus|ringing in ears?)\b",
    r"\b(hearing loss)\b",
    r"\b(sinusitis|sinus infections?)\b",

    # --- Endocrine / Metabolic / Autoimmune ---
    r"\b(diabetes|diabetic)\b",
    r"\b(hypothyroidism|hyperthyroidism|thyroid disorders?)\b",
    r"\b(obesity)\b",
    r"\b(lupus|rheumatoid arthritis)\b",

    # --- Mental Health ---
    r"\b(anxiety|panic attacks?)\b",
    r"\b(depression|depressive)\b",
    r"\b(insomnia|sleep apnea)\b",
    r"\b(ptsd|post-traumatic stress)\b",
    r"\b(bipolar disorder)\b",

    # --- Ophthalmologic ---
    r"\b(conjunctivitis|pink\s*eye)\b",
    r"\b(blurred vision|double vision|vision loss)\b",
    r"\b(cataracts?|glaucoma)\b",

    # --- Urological / Renal ---
    r"\b(kidney stones?|renal calculi)\b",
    r"\b(urinary tract infections?|uti|bladder infections?)\b",

    # --- Women's Health ---
    r"\b(menstrual cramps|dysmenorrhea)\b",
    r"\b(yeast infections?|vaginosis)\b",

    # --- Allergy / Immune ---
    r"\b(hay fever|seasonal allergies?)\b",
    r"\b(food allerg(?:y|ies))\b",
    r"\b(anaphylaxis|anaphylactic shock)\b",

    # --- Dental / Oral ---
    r"\b(toothaches?|dental pain)\b",
    r"\b(gum disease|gingivitis|periodontitis)\b",
    r"\b(cavit(?:y|ies))\b",

    # --- Hematological & Oncology ---
    r"\b(anemia)\b",
    r"\b(blood clots?|deep vein thrombosis|dvt)\b",
    r"\b(cancers?|tumors?|leukemia|lymphoma)\b",

    # --- Broad Presentation Fallbacks ---
    r"\b(fevers?|infections?|allergic reactions?|inflammation)\b",
    r"\b(fatigue|weight loss|weight gain)\b",
]

# Subject-Bearing Intents (Filtered subset explicitly tied to noun phrases)
SUBJECT_EXTRACTION_INTENTS = (
    "treatment",
    "treatment options",
    "recommended treatment",
    "management options",
    "medication options",
    "how to cure",
    "symptoms",
    "warning signs",
    "diagnosis",
    "complications",
    "side effects",
    "risks",
    "guidelines",
    "clinical guidelines",
    "recommendations",
    "clinical recommendations",
    "management guidelines",
)

# Compile dynamic regex patterns using ONLY subject-bearing intents
_subject_intent_block = "|".join(re.escape(intent) for intent in SUBJECT_EXTRACTION_INTENTS)

DYNAMIC_SUBJECT_PATTERNS = [
    re.compile(
        rf"\b(?:{_subject_intent_block})\s+"
        r"(?:of|for|about|on)\s+"
        r"([^?.!,]+?)(?:\s+\b(?:and|but|or)\b|[?.!,]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bhow\s+is\s+([^?.!,]+?)\s+"
        r"(?:treated|managed|diagnosed|prevented)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat\s+is\s+([^?.!,]+)",
        re.IGNORECASE,
    ),
]
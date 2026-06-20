# Model Configuration
GEMMA_MODEL_ID="ollama/gemma4:e2b"
# Quantization (4-bit for laptop) - Note: Ollama handles this via the model file, but we keep it for conceptual clarity.
LOAD_IN_4BIT=True
LOAD_IN_8BIT=False
# Device
DEVICE="auto"  # auto, cuda, cpu, mps
# ChromaDB
CHROMA_PERSIST_DIR="./chroma_db"
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
    "mental_health",
    "nutrition",
    "women_health",
    "men_health",
    "elderly_care",
    "infectious_disease",
    "chronic_disease"
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
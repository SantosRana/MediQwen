
import os
import logging

logger = logging.getLogger("mediqwen_prompts")

def get_system_behavioral_spec() -> str:
    """Reads and returns the master skills specification from config/skills.md."""
    skills_path = os.path.join("config", "skills.md")
    
    if not os.path.exists(skills_path):
        logger.warning(f"⚠️ Prompt Warning: File '{skills_path}' not found. Initializing generic fallback.")
        return "You are MediQwen, a safe, evidence-grounded clinical medical assistant."
        
    try:
        with open(skills_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"❌ Critical failure loading system prompt configuration: {e}")
        return "You are MediQwen, a safe, evidence-grounded clinical medical assistant."

# Pre-cache baseline system spec at import time
SYSTEM_BEHAVIORAL_SPEC = get_system_behavioral_spec()
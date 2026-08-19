
from pathlib import Path
import os
import logging

logger = logging.getLogger("mediqwen_prompts")

def get_system_behavioral_spec() -> str:
    """Reads and returns the master skills specification from config/skills.md."""
    PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Adjust parents level if file is in config/ (use parents[1])
    skills_path = PROJECT_ROOT / "config" / "skills.md"
    
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
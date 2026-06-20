# src/utils/helpers.py

import os
import json
from pathlib import Path

def load_config():
    """
    Load configuration settings from 'config/settings.py'.
    """
    # Import settings directly
    from config.settings import (
        GEMMA_MODEL_ID,
        CHROMA_PERSIST_DIR,
        CHROMA_COLLECTION_NAME,
        EMBEDDING_MODEL,
        ENABLE_SAFETY_CHECKS,
        MAX_RISK_LEVEL,
        RISK_LEVELS,
        MEDICAL_DOMAINS,
        OUTPUT_SCHEMA
    )
    
    config = {
        "model_id": GEMMA_MODEL_ID,
        "chroma_dir": CHROMA_PERSIST_DIR,
        "collection_name": CHROMA_COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "safety_enabled": ENABLE_SAFETY_CHECKS,
        "max_risk_level": MAX_RISK_LEVEL,
        "risk_levels": RISK_LEVELS,
        "medical_domains": MEDICAL_DOMAINS,
        "output_schema": OUTPUT_SCHEMA
    }
    return config

def initialize_directories():
    """
    Ensure all necessary directories exist.
    """
    directories = [
        "src", "src/models", "src/preprocessing", "src/rag", "src/safety", "src/agents", "src/utils",
        "config", "data", "data/knowledge_base", "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
def log_info(message):
    """
    Print information with a timestamp.
    """
    import time
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[INFO] {timestamp} - {message}")
    
def log_error(message):
    """
    Print error messages with a timestamp.
    """
    import time
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[ERROR] {timestamp} - {message}")

if __name__ == "__main__":
    # Test helpers
    print("Testing helpers...")
    load_config()
    initialize_directories()
    log_info("Directories initialized successfully.")
    log_error("This was a test error.")
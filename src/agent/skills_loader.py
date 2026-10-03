# src/agent/skills_loader.py
"""
Dynamic Skill Loader for MediQwen Runtime System Prompts.
Loads and caches focused Markdown skill specifications to inject into Ollama payloads.
"""

import logging
from functools import lru_cache
from pathlib import Path
from config.settings import BASE_DIR

logger = logging.getLogger("skills_loader")

SKILLS_DIR = BASE_DIR / "config" / "skills"


@lru_cache(maxsize=8)
def load_skill(skill_name: str) -> str:
    """
    Load and cache a MediQwen runtime skill specification.

    Args:
        skill_name: Filename without the .md extension.

    Returns:
        Skill content, or an empty string if unavailable.
    """
    file_path = SKILLS_DIR / f"{skill_name}.md"

    try:
        content = file_path.read_text(encoding="utf-8").strip()

        logger.debug(
            "📖 Loaded skill '%s' (%d chars)",
            skill_name,
            len(content),
        )

        return content

    except FileNotFoundError:
        logger.warning(
            "⚠️ Skill file not found: %s",
            file_path,
        )
        return ""

    except Exception as err:
        logger.exception(
            "❌ Failed to load skill '%s': %s",
            skill_name,
            err,
        )
        return ""
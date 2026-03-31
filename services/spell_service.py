from __future__ import annotations

from functools import lru_cache
from typing import Dict

from models.spell_checker import SpellChecker


@lru_cache(maxsize=1)
def _checker() -> SpellChecker:
    return SpellChecker()


def correct_spelling(text: str, suggest_only: bool = True) -> Dict[str, object]:
    clean = text.strip()
    if not clean:
        raise ValueError("text must not be empty")

    corrected, suggestions = _checker().correct(clean, suggest_only=suggest_only)
    return {
        "original_text": clean,
        "corrected_text": corrected,
        "suggestions": suggestions,
    }

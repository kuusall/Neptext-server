from __future__ import annotations

from functools import lru_cache
from typing import Dict

from config.env import get_settings
from models.word_perdictor import WordPredictor


@lru_cache(maxsize=1)
def _predictor() -> WordPredictor:
    return WordPredictor()


def predict_next_word(text: str, top_k: int | None = None) -> Dict[str, object]:
    clean = text.strip()
    if not clean:
        raise ValueError("text must not be empty")

    effective_top_k = top_k if top_k is not None else get_settings().default_top_k
    predictions = _predictor().predict(clean, top_k=effective_top_k)
    return {
        "context": clean,
        "predictions": predictions,
    }

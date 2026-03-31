from __future__ import annotations

from functools import lru_cache
from typing import Dict

from models.sentiment_analyzer import SentimentAnalyzer


@lru_cache(maxsize=1)
def _analyzer() -> SentimentAnalyzer:
    return SentimentAnalyzer()


def analyze_sentiment(text: str) -> Dict[str, object]:
    clean = text.strip()
    if not clean:
        raise ValueError("text must not be empty")
    return _analyzer().predict(clean)

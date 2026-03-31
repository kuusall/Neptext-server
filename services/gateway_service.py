from __future__ import annotations

from typing import Dict

from services.sentiment_service import analyze_sentiment
from services.spell_service import correct_spelling
from services.word_prediction_service import predict_next_word


class APIGatewayService:
    def analyze_sentiment(self, text: str) -> Dict[str, object]:
        return analyze_sentiment(text)

    def correct_spelling(self, text: str, suggest_only: bool = True) -> Dict[str, object]:
        return correct_spelling(text, suggest_only=suggest_only)

    def predict_next_word(self, text: str, top_k: int = 5) -> Dict[str, object]:
        return predict_next_word(text, top_k=top_k)

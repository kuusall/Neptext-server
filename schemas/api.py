from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    message: str


class ErrorResponse(BaseModel):
    detail: str


class SentimentRequest(TextRequest):
    pass


class SentimentResponse(BaseModel):
    sentiment: Literal["negative", "semi_negative", "neutral", "semi_positive", "positive"]
    label_id: int = Field(..., ge=0, le=4)
    confidence: float = Field(..., ge=0.0, le=1.0)
    model: str


class SpellCorrectRequest(TextRequest):
    suggest_only: bool = True


class SpellSuggestion(BaseModel):
    index: int = Field(..., ge=0)
    from_word: str = Field(..., alias="from")
    suggestion: str = Field(..., alias="suggest")
    edit_distance: int = Field(..., ge=0)

    model_config = {
        "populate_by_name": True,
    }


class SpellCorrectResponse(BaseModel):
    original_text: str
    corrected_text: str
    suggestions: List[SpellSuggestion]


class WordPredictRequest(TextRequest):
    top_k: int = Field(default=5, ge=1, le=20)


class PredictionItem(BaseModel):
    word: str
    probability: float = Field(..., ge=0.0, le=1.0)


class WordPredictResponse(BaseModel):
    context: str
    predictions: List[PredictionItem]

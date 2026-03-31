from __future__ import annotations

import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

from config.env import get_settings
from models.sentiment_preprocessing import (
    SENTIMENT_LABELS,
    SENTIMENT_TO_ID,
    augment_text_for_model,
    build_dense_feature_matrix,
    load_emoji_resources,
    sentiment_strength,
    split_long_text,
    text_signal_features,
    tokenize_words,
)

try:
    from transformers import pipeline
except Exception:
    pipeline = None


@dataclass
class SentimentResult:
    label: str
    label_id: int
    confidence: float
    model: str


class SentimentAnalyzer:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._classifier = None
        self._load_error: Optional[str] = None
        self._model_name = self._settings.sentiment_model_name

        self._artifact = None
        self._word_vectorizer = None
        self._char_vectorizer = None
        self._dense_scaler = None
        self._emoji_polarity: Dict[str, float] = {}
        self._loaded_artifact_name = ""

        dicts_dir = Path(self._settings.model_dicts_dir)
        self._artifact_candidates = self._resolve_artifact_candidates(dicts_dir)
        emoji_dict_path = os.getenv("EMOJI_DICT_PATH", str(dicts_dir / "Emoji_Dict.p"))
        self._resources = load_emoji_resources(emoji_dict_path)

        self._try_load_trained_model()
        self._try_load_transformers_model()

    def _resolve_artifact_candidates(self, dicts_dir: Path) -> list[Path]:
        explicit_path = os.getenv("SENTIMENT_ARTIFACT_PATH", "").strip()
        if explicit_path:
            return [Path(explicit_path)]

        # Prefer pickle artifact, then fallback to joblib for backwards compatibility.
        return [
            dicts_dir / "sentiment_5class_model.pkl",
            dicts_dir / "sentiment_5class_model.joblib",
        ]

    def _load_artifact_bundle(self, artifact_path: Path) -> Dict[str, object]:
        if artifact_path.suffix.lower() in {".pkl", ".pickle"}:
            with artifact_path.open("rb") as file_obj:
                bundle = pickle.load(file_obj)
        else:
            bundle = joblib.load(artifact_path)

        if not isinstance(bundle, dict):
            raise ValueError("artifact payload is not a dict bundle")
        return bundle

    def _try_load_trained_model(self) -> None:
        last_error: Optional[str] = None
        for artifact_path in self._artifact_candidates:
            if not artifact_path.exists():
                continue

            try:
                bundle = self._load_artifact_bundle(artifact_path)
                self._word_vectorizer = bundle["word_vectorizer"]
                self._char_vectorizer = bundle["char_vectorizer"]
                self._dense_scaler = bundle["dense_scaler"]
                self._artifact = bundle["classifier"]
                self._emoji_polarity = {
                    str(key): float(value)
                    for key, value in dict(bundle.get("emoji_polarity", {})).items()
                }
                self._loaded_artifact_name = artifact_path.name
                self._load_error = None
                return
            except Exception as exc:
                last_error = f"{artifact_path.name}: {exc}"

        self._artifact = None
        if last_error:
            self._load_error = f"failed to load trained model: {last_error}"

    def _try_load_transformers_model(self) -> None:
        if self._artifact is not None:
            return

        if pipeline is None:
            if self._load_error is None:
                self._load_error = "transformers is not available"
            return

        try:
            self._classifier = pipeline(
                "sentiment-analysis",
                model=self._model_name,
                tokenizer=self._model_name,
            )
        except Exception as exc:
            self._classifier = None
            if self._load_error is None:
                self._load_error = str(exc)

    def _map_transformer_label(self, raw_label: str) -> str:
        normalized = raw_label.lower().strip()

        if "neg" in normalized or normalized == "label_0":
            return "semi_negative"
        if "neu" in normalized or normalized == "label_1":
            return "neutral"
        if "pos" in normalized or normalized == "label_2":
            return "semi_positive"

        return "neutral"

    def _predict_with_trained_model(self, text: str) -> SentimentResult:
        chunks = split_long_text(text, max_chars=360)
        if not chunks:
            chunks = [text]

        probabilities: list[np.ndarray] = []
        weights: list[float] = []

        for chunk in chunks:
            augmented = augment_text_for_model(chunk, self._resources, self._emoji_polarity)
            signal = text_signal_features(chunk, self._resources, self._emoji_polarity)
            normalized = str(signal["normalized"])

            x_word = self._word_vectorizer.transform([augmented])
            x_char = self._char_vectorizer.transform([normalized])
            x_dense = build_dense_feature_matrix([chunk], self._resources, self._emoji_polarity)
            x_dense_scaled = csr_matrix(self._dense_scaler.transform(x_dense))
            x_input = hstack([x_word, x_char, x_dense_scaled], format="csr")

            probabilities.append(self._artifact.predict_proba(x_input)[0])
            weights.append(max(1.0, float(len(tokenize_words(chunk)))))

        aggregated = np.average(
            np.vstack(probabilities),
            axis=0,
            weights=np.asarray(weights, dtype=np.float32),
        )
        label_id = int(np.argmax(aggregated))
        label = SENTIMENT_LABELS[label_id]
        confidence = float(aggregated[label_id])

        return SentimentResult(
            label=label,
            label_id=label_id,
            confidence=round(confidence, 4),
            model=self._loaded_artifact_name or "trained-model",
        )

    def _predict_with_transformers(self, text: str) -> SentimentResult:
        chunks = split_long_text(text, max_chars=420)
        if not chunks:
            chunks = [text]

        votes: Dict[str, float] = {label: 0.0 for label in SENTIMENT_LABELS}
        for chunk in chunks:
            output = self._classifier(chunk[:512], truncation=True)[0]
            mapped = self._map_transformer_label(str(output.get("label", "neutral")))
            raw_score = float(output.get("score", 0.5))

            if mapped == "semi_negative":
                votes["semi_negative"] += raw_score
                if raw_score > 0.82:
                    votes["negative"] += raw_score - 0.75
            elif mapped == "semi_positive":
                votes["semi_positive"] += raw_score
                if raw_score > 0.82:
                    votes["positive"] += raw_score - 0.75
            else:
                votes["neutral"] += raw_score

        total_vote = sum(votes.values()) or 1.0
        label = max(votes, key=votes.get)
        confidence = votes[label] / total_vote

        return SentimentResult(
            label=label,
            label_id=SENTIMENT_TO_ID[label],
            confidence=round(confidence, 4),
            model=self._model_name,
        )

    def _heuristic_predict(self, text: str) -> SentimentResult:
        signal = text_signal_features(text, self._resources, self._emoji_polarity)
        score = sentiment_strength(signal)

        if score <= -3.0:
            label = "negative"
        elif score < -1.0:
            label = "semi_negative"
        elif score >= 3.0:
            label = "positive"
        elif score > 1.0:
            label = "semi_positive"
        else:
            label = "neutral"

        raw_conf = 0.52 + (1 - math.exp(-abs(score))) * 0.43
        confidence = min(0.98, max(0.51, raw_conf))

        return SentimentResult(
            label=label,
            label_id=SENTIMENT_TO_ID[label],
            confidence=round(confidence, 4),
            model="heuristic-fallback",
        )

    def predict(self, text: str) -> Dict[str, object]:
        clean = text.strip()
        if not clean:
            raise ValueError("text must not be empty")

        if self._artifact is not None:
            result = self._predict_with_trained_model(clean)
            return {
                "sentiment": result.label,
                "label_id": result.label_id,
                "confidence": result.confidence,
                "model": result.model,
            }

        if self._classifier is None:
            result = self._heuristic_predict(clean)
            return {
                "sentiment": result.label,
                "label_id": result.label_id,
                "confidence": result.confidence,
                "model": result.model,
            }

        try:
            result = self._predict_with_transformers(clean)
            return {
                "sentiment": result.label,
                "label_id": result.label_id,
                "confidence": result.confidence,
                "model": result.model,
            }
        except Exception:
            result = self._heuristic_predict(clean)
            return {
                "sentiment": result.label,
                "label_id": result.label_id,
                "confidence": result.confidence,
                "model": result.model,
            }

from __future__ import annotations

import csv
import pickle
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


SENTIMENT_LABELS: tuple[str, ...] = (
    "negative",
    "semi_negative",
    "neutral",
    "semi_positive",
    "positive",
)
SENTIMENT_TO_ID: Dict[str, int] = {label: idx for idx, label in enumerate(SENTIMENT_LABELS)}


WORD_RE = re.compile(r"[\u0900-\u097Fa-zA-Z0-9_]+", re.UNICODE)
SPLIT_RE = re.compile(r"(?<=[.!?।])\s+|\n+")


# Balanced mixed-language hints frequently seen in Nepali social comments.
POSITIVE_HINTS = {
    "good",
    "great",
    "awesome",
    "best",
    "love",
    "excellent",
    "ramro",
    "mitho",
    "happy",
    "wow",
    "thanks",
    "thankyou",
    "sahi",
    "nice",
    "धन्यवाद",
    "राम्रो",
    "सहि",
    "जिन्दावाद",
    "सलाम",
}

NEGATIVE_HINTS = {
    "bad",
    "worse",
    "worst",
    "hate",
    "awful",
    "terrible",
    "bekar",
    "naramro",
    "angry",
    "annoying",
    "fake",
    "chor",
    "thukk",
    "भ्रष्टाचार",
    "भ्रष्टाचारी",
    "चोर",
    "थुक्क",
    "मुर्दावाद",
    "धिक्कार",
}

STRONG_POSITIVE_HINTS = {
    "excellent",
    "best",
    "legend",
    "जिन्दावाद",
    "सलाम",
    "great",
}

STRONG_NEGATIVE_HINTS = {
    "hate",
    "worst",
    "awful",
    "thukk",
    "chor",
    "चोर",
    "थुक्क",
    "मुर्दावाद",
}


@dataclass
class EmojiResources:
    alias_by_emoji: Dict[str, str]
    emoji_pattern: re.Pattern[str] | None


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("\u200d", "")
    return re.sub(r"\s+", " ", normalized).strip().lower()


def tokenize_words(text: str) -> List[str]:
    return WORD_RE.findall(normalize_text(text))


def _normalize_alias(alias: str) -> str:
    cleaned = alias.strip().strip(":").lower()
    cleaned = re.sub(r"[^a-z0-9_]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "emoji"


def _expand_emoji_forms(value: str) -> List[str]:
    value = (value or "").strip()
    if not value:
        return []

    forms = {value}
    forms.add(value.replace("\ufe0f", ""))
    if " " in value:
        forms.add(value.replace(" ", ""))
        forms.add(value.replace(" ", "").replace("\ufe0f", ""))

    return [item for item in forms if item]


def load_emoji_resources(emoji_dict_path: str | Path) -> EmojiResources:
    path = Path(emoji_dict_path)
    if not path.exists():
        return EmojiResources(alias_by_emoji={}, emoji_pattern=None)

    with path.open("rb") as file_obj:
        raw_dict = pickle.load(file_obj)

    if not isinstance(raw_dict, dict):
        return EmojiResources(alias_by_emoji={}, emoji_pattern=None)

    alias_by_emoji: Dict[str, str] = {}
    for alias, emoji_value in raw_dict.items():
        normalized_alias = _normalize_alias(str(alias))
        for form in _expand_emoji_forms(str(emoji_value)):
            alias_by_emoji.setdefault(form, normalized_alias)

    if not alias_by_emoji:
        return EmojiResources(alias_by_emoji={}, emoji_pattern=None)

    pattern = re.compile(
        "|".join(sorted((re.escape(item) for item in alias_by_emoji.keys()), key=len, reverse=True))
    )
    return EmojiResources(alias_by_emoji=alias_by_emoji, emoji_pattern=pattern)


def extract_emojis(text: str, resources: EmojiResources) -> List[str]:
    if not text or resources.emoji_pattern is None:
        return []
    return resources.emoji_pattern.findall(text)


def read_labeled_dataset(csv_path: str | Path) -> List[Tuple[str, int]]:
    path = Path(csv_path)
    rows: List[Tuple[str, int]] = []

    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            text = (row.get("text") or "").strip()
            label_raw = str(row.get("label", "")).strip()
            if not text:
                continue

            try:
                label = int(label_raw)
            except ValueError:
                continue

            if label in (0, 1, 2):
                rows.append((text, label))

    return rows


def build_emoji_polarity(
    rows: Sequence[Tuple[str, int]],
    resources: EmojiResources,
    min_support: int = 3,
) -> Dict[str, float]:
    counts: Dict[str, list[int]] = {}

    for text, label in rows:
        for emoji in set(extract_emojis(text, resources)):
            bucket = counts.setdefault(emoji, [0, 0, 0])
            bucket[label] += 1

    polarity: Dict[str, float] = {}
    for emoji, bucket in counts.items():
        neg_count, pos_count, neu_count = bucket
        support = neg_count + pos_count + neu_count
        if support < min_support:
            continue

        score = (pos_count - neg_count) / float(support)
        if abs(score) < 0.05:
            continue
        polarity[emoji] = float(round(score, 4))

    return polarity


def text_signal_features(
    text: str,
    resources: EmojiResources,
    emoji_polarity: Dict[str, float],
) -> Dict[str, float | int | list[str]]:
    normalized = normalize_text(text)
    tokens = WORD_RE.findall(normalized)

    pos_hint = sum(1 for token in tokens if token in POSITIVE_HINTS)
    neg_hint = sum(1 for token in tokens if token in NEGATIVE_HINTS)
    strong_pos = sum(1 for token in tokens if token in STRONG_POSITIVE_HINTS)
    strong_neg = sum(1 for token in tokens if token in STRONG_NEGATIVE_HINTS)

    emojis = extract_emojis(text, resources)
    alias_tokens: list[str] = []
    pos_emoji = 0.0
    neg_emoji = 0.0

    for emoji in emojis:
        alias = resources.alias_by_emoji.get(emoji)
        if alias:
            alias_tokens.append(f"emoji_{alias}")

        score = float(emoji_polarity.get(emoji, 0.0))
        if score > 0:
            pos_emoji += score
        elif score < 0:
            neg_emoji += -score

    exclamation = text.count("!")
    question = text.count("?") + text.count("？")

    return {
        "normalized": normalized,
        "token_count": len(tokens),
        "char_count": len(normalized),
        "pos_hint": pos_hint,
        "neg_hint": neg_hint,
        "strong_pos": strong_pos,
        "strong_neg": strong_neg,
        "pos_emoji": pos_emoji,
        "neg_emoji": neg_emoji,
        "emoji_count": len(emojis),
        "emoji_alias_tokens": alias_tokens,
        "exclamation": exclamation,
        "question": question,
    }


def sentiment_strength(signal: Dict[str, float | int | list[str]]) -> float:
    return (
        float(signal["pos_hint"])
        + 1.35 * float(signal["strong_pos"])
        + float(signal["pos_emoji"])
        - float(signal["neg_hint"])
        - 1.35 * float(signal["strong_neg"])
        - float(signal["neg_emoji"])
    )


def augment_text_for_model(
    text: str,
    resources: EmojiResources,
    emoji_polarity: Dict[str, float],
) -> str:
    signal = text_signal_features(text, resources, emoji_polarity)
    normalized = str(signal["normalized"])
    tags: list[str] = []

    pos_hint = int(signal["pos_hint"])
    neg_hint = int(signal["neg_hint"])
    strong_pos = int(signal["strong_pos"])
    strong_neg = int(signal["strong_neg"])
    pos_emoji = float(signal["pos_emoji"])
    neg_emoji = float(signal["neg_emoji"])

    tags.extend(["hint_pos"] * min(pos_hint, 4))
    tags.extend(["hint_neg"] * min(neg_hint, 4))
    tags.extend(["strong_pos"] * min(strong_pos, 3))
    tags.extend(["strong_neg"] * min(strong_neg, 3))

    if pos_emoji >= neg_emoji + 0.5:
        tags.append("emoji_tone_positive")
    elif neg_emoji >= pos_emoji + 0.5:
        tags.append("emoji_tone_negative")

    tags.extend(list(signal["emoji_alias_tokens"])[:20])

    if not tags:
        return normalized
    return f"{normalized} {' '.join(tags)}"


def build_dense_feature_matrix(
    texts: Iterable[str],
    resources: EmojiResources,
    emoji_polarity: Dict[str, float],
) -> np.ndarray:
    matrix: list[list[float]] = []

    for text in texts:
        signal = text_signal_features(text, resources, emoji_polarity)
        matrix.append(
            [
                float(signal["pos_hint"]),
                float(signal["neg_hint"]),
                float(signal["strong_pos"]),
                float(signal["strong_neg"]),
                float(signal["pos_emoji"]),
                float(signal["neg_emoji"]),
                float(signal["emoji_count"]),
                float(signal["exclamation"]),
                float(signal["question"]),
                float(signal["token_count"]),
                float(signal["char_count"]),
            ]
        )

    return np.asarray(matrix, dtype=np.float32)


def split_long_text(text: str, max_chars: int = 320) -> List[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    parts = [part.strip() for part in SPLIT_RE.split(cleaned) if part.strip()]
    if not parts:
        return [cleaned[:max_chars]]

    chunks: list[str] = []
    current = ""

    for part in parts:
        if len(part) > max_chars:
            words = part.split()
            temp = ""
            for word in words:
                candidate = f"{temp} {word}".strip()
                if len(candidate) <= max_chars:
                    temp = candidate
                else:
                    if temp:
                        if len(temp) > max_chars:
                            chunks.extend(
                                temp[i : i + max_chars] for i in range(0, len(temp), max_chars)
                            )
                        else:
                            chunks.append(temp)
                    temp = word

            if temp:
                if len(temp) > max_chars:
                    chunks.extend(temp[i : i + max_chars] for i in range(0, len(temp), max_chars))
                else:
                    chunks.append(temp)
            continue

        candidate = f"{current} {part}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = part

    if current:
        chunks.append(current)

    return chunks if chunks else [cleaned[:max_chars]]

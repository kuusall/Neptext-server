from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

from models.sentiment_preprocessing import (
    SENTIMENT_LABELS,
    SENTIMENT_TO_ID,
    augment_text_for_model,
    build_dense_feature_matrix,
    build_emoji_polarity,
    load_emoji_resources,
    read_labeled_dataset,
    sentiment_strength,
    text_signal_features,
)


def _read_five_class_dataset(csv_path: str | Path) -> list[tuple[str, int]]:
    path = Path(csv_path)
    if not path.exists():
        return []

    rows: list[tuple[str, int]] = []
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            text = str(row.get("text", "")).strip()
            label_raw = str(row.get("label", "")).strip()
            if not text:
                continue

            try:
                label = int(label_raw)
            except ValueError:
                continue

            if 0 <= label < len(SENTIMENT_LABELS):
                rows.append((text, label))

    return rows


def _upgrade_labels_to_five_classes(
    rows: list[tuple[str, int]],
    emoji_dict_path: str,
) -> tuple[list[str], np.ndarray, dict[str, object], dict[str, float]]:
    resources = load_emoji_resources(emoji_dict_path)
    emoji_polarity = build_emoji_polarity(rows, resources)

    strengths: list[tuple[str, int, float, int]] = []
    negative_strengths: list[float] = []
    positive_strengths: list[float] = []
    zero_signal_negative_token_counts: list[int] = []

    for text, base_label in rows:
        signal = text_signal_features(text, resources, emoji_polarity)
        signed_strength = sentiment_strength(signal)
        token_count = int(signal["token_count"])
        strengths.append((text, base_label, signed_strength, token_count))

        if base_label == 0:
            negative_strengths.append(max(0.0, -signed_strength))
            if abs(signed_strength) <= 0.05:
                zero_signal_negative_token_counts.append(token_count)
        elif base_label == 1:
            positive_strengths.append(max(0.0, signed_strength))

    # Two-stage thresholding for base negatives:
    # 1) strong-score split
    # 2) zero-signal split via comment length (shorter negative comments are often direct attacks)
    negative_cut = 0.55
    positive_cut = max(1.25, float(np.percentile(positive_strengths, 62))) if positive_strengths else 1.25
    negative_positive_conflict_cut = 0.35
    zero_signal_strength_band = 0.05
    zero_signal_token_cut = (
        int(np.percentile(zero_signal_negative_token_counts, 40))
        if zero_signal_negative_token_counts
        else 11
    )
    neutral_band = 1.35

    texts: list[str] = []
    labels: list[int] = []

    for text, base_label, signed_strength, token_count in strengths:
        if base_label == 0:
            if signed_strength > negative_positive_conflict_cut:
                label = "semi_negative"
            elif (-signed_strength) >= negative_cut:
                label = "negative"
            elif abs(signed_strength) <= zero_signal_strength_band:
                label = "negative" if token_count <= zero_signal_token_cut else "semi_negative"
            else:
                label = "semi_negative"
        elif base_label == 1:
            label = "positive" if signed_strength >= positive_cut else "semi_positive"
        else:
            if signed_strength <= -neutral_band:
                label = "semi_negative"
            elif signed_strength >= neutral_band:
                label = "semi_positive"
            else:
                label = "neutral"

        texts.append(text)
        labels.append(SENTIMENT_TO_ID[label])

    distribution = Counter(labels)
    distribution_named = {SENTIMENT_LABELS[idx]: count for idx, count in sorted(distribution.items())}

    stats = {
        "rows": len(texts),
        "five_class_distribution": distribution_named,
        "label_thresholds": {
            "negative_cut": round(negative_cut, 4),
            "positive_cut": round(positive_cut, 4),
            "negative_positive_conflict_cut": negative_positive_conflict_cut,
            "zero_signal_strength_band": zero_signal_strength_band,
            "zero_signal_token_cut": zero_signal_token_cut,
            "neutral_band": neutral_band,
        },
        "emoji_polarity_size": len(emoji_polarity),
    }

    return texts, np.asarray(labels, dtype=np.int32), stats, emoji_polarity


def train_sentiment_model(
    dataset_path: str,
    emoji_dict_path: str,
    output_path: str,
    output_pickle_path: str,
    metrics_path: str,
    five_class_supplement_path: str | None = None,
    five_class_supplement_weight: float = 0.35,
) -> dict[str, object]:
    rows = read_labeled_dataset(dataset_path)
    if len(rows) < 100:
        raise ValueError("dataset is too small for robust training")

    resources = load_emoji_resources(emoji_dict_path)
    if not resources.alias_by_emoji:
        raise ValueError("emoji dictionary could not be loaded")

    texts, labels, upgrade_stats, emoji_polarity = _upgrade_labels_to_five_classes(rows, emoji_dict_path)

    base_count = len(texts)

    supplement_rows: list[tuple[str, int]] = []
    if five_class_supplement_path:
        supplement_rows = _read_five_class_dataset(five_class_supplement_path)

    upgrade_stats["five_class_supplement_rows"] = int(len(supplement_rows))

    train_idx, test_idx = train_test_split(
        np.arange(len(texts)),
        test_size=0.15,
        random_state=42,
        stratify=labels,
    )

    train_texts = [texts[idx] for idx in train_idx]
    test_texts = [texts[idx] for idx in test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]
    w_train = np.ones(len(y_train), dtype=np.float32)

    if supplement_rows:
        train_texts.extend([text for text, _ in supplement_rows])
        y_train = np.concatenate(
            [y_train, np.asarray([label for _, label in supplement_rows], dtype=np.int32)]
        )
        w_train = np.concatenate(
            [w_train, np.full(len(supplement_rows), five_class_supplement_weight, dtype=np.float32)]
        )

    train_augmented = [augment_text_for_model(text, resources, emoji_polarity) for text in train_texts]
    test_augmented = [augment_text_for_model(text, resources, emoji_polarity) for text in test_texts]

    train_normalized = [signal["normalized"] for signal in (text_signal_features(t, resources, emoji_polarity) for t in train_texts)]
    test_normalized = [signal["normalized"] for signal in (text_signal_features(t, resources, emoji_polarity) for t in test_texts)]

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=140000,
        sublinear_tf=True,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=120000,
        sublinear_tf=True,
    )

    x_word_train = word_vectorizer.fit_transform(train_augmented)
    x_word_test = word_vectorizer.transform(test_augmented)

    x_char_train = char_vectorizer.fit_transform(train_normalized)
    x_char_test = char_vectorizer.transform(test_normalized)

    dense_train = build_dense_feature_matrix(train_texts, resources, emoji_polarity)
    dense_test = build_dense_feature_matrix(test_texts, resources, emoji_polarity)
    dense_scaler = MaxAbsScaler()
    x_dense_train = csr_matrix(dense_scaler.fit_transform(dense_train))
    x_dense_test = csr_matrix(dense_scaler.transform(dense_test))

    x_train = hstack([x_word_train, x_char_train, x_dense_train], format="csr")
    x_test = hstack([x_word_test, x_char_test, x_dense_test], format="csr")

    classifier = LogisticRegression(
        solver="saga",
        class_weight="balanced",
        max_iter=5200,
        C=2.0,
        random_state=42,
    )
    classifier.fit(x_train, y_train, sample_weight=w_train)

    predictions = classifier.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))
    macro_f1 = float(f1_score(y_test, predictions, average="macro"))
    weighted_f1 = float(f1_score(y_test, predictions, average="weighted"))

    report = classification_report(
        y_test,
        predictions,
        labels=list(range(len(SENTIMENT_LABELS))),
        target_names=list(SENTIMENT_LABELS),
        zero_division=0,
        output_dict=True,
    )

    artifact = {
        "artifact_version": "2.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label_names": list(SENTIMENT_LABELS),
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
        "dense_scaler": dense_scaler,
        "classifier": classifier,
        "emoji_polarity": emoji_polarity,
        "training_stats": {
            **upgrade_stats,
            "base_rows": int(base_count),
            "supplement_weight": float(five_class_supplement_weight),
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
        },
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_file)

    pickle_file = Path(output_pickle_path)
    pickle_file.parent.mkdir(parents=True, exist_ok=True)
    with pickle_file.open("wb") as file_obj:
        pickle.dump(artifact, file_obj, protocol=pickle.HIGHEST_PROTOCOL)

    metrics = {
        "summary": artifact["training_stats"],
        "artifacts": {
            "joblib": str(output_file),
            "pickle": str(pickle_file),
        },
        "classification_report": report,
    }

    metrics_file = Path(metrics_path)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 5-class Nepali social media sentiment model")
    parser.add_argument(
        "--dataset",
        default="model_dicts/collected_labeled_data.csv",
        help="Path to labeled csv data with columns text,label",
    )
    parser.add_argument(
        "--emoji-dict",
        default="model_dicts/Emoji_Dict.p",
        help="Path to pickled emoji dictionary",
    )
    parser.add_argument(
        "--output",
        default="model_dicts/sentiment_5class_model.joblib",
        help="Path for trained model artifact",
    )
    parser.add_argument(
        "--output-pickle",
        default="model_dicts/sentiment_5class_model.pkl",
        help="Path for trained model pickle artifact",
    )
    parser.add_argument(
        "--metrics",
        default="model_dicts/sentiment_5class_metrics.json",
        help="Path for training metrics JSON",
    )
    parser.add_argument(
        "--five-class-supplement",
        default="model_dicts/negative_sentences_5class.csv",
        help="Optional CSV path with columns text,label already encoded in 5-class IDs",
    )
    parser.add_argument(
        "--five-class-supplement-weight",
        type=float,
        default=0.35,
        help="Training sample weight applied to supplement rows",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = train_sentiment_model(
        dataset_path=args.dataset,
        emoji_dict_path=args.emoji_dict,
        output_path=args.output,
        output_pickle_path=args.output_pickle,
        metrics_path=args.metrics,
        five_class_supplement_path=args.five_class_supplement,
        five_class_supplement_weight=args.five_class_supplement_weight,
    )
    print(json.dumps(result.get("summary", {}), ensure_ascii=False, indent=2))

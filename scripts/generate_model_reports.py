from __future__ import annotations

import argparse
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.env import get_settings
from models.sentiment_preprocessing import (
    SENTIMENT_LABELS,
    augment_text_for_model,
    build_dense_feature_matrix,
    load_emoji_resources,
    read_labeled_dataset,
    text_signal_features,
)
from models.spell_checker import SpellChecker, is_dev_word
from models.train_sentiment_model import _upgrade_labels_to_five_classes
from models.word_perdictor import WordPredictor


matplotlib.use("Agg")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_confusion_matrix(
    matrix: np.ndarray,
    labels: Sequence[str],
    title: str,
    out_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    axis.figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(
                col,
                row,
                str(int(matrix[row, col])),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black",
                fontsize=9,
            )

    figure.tight_layout()
    figure.savefig(out_path, dpi=180)
    plt.close(figure)


def _save_key_values_image(title: str, rows: Sequence[Tuple[str, str]], out_path: Path) -> None:
    figure_height = max(4.5, 0.45 * len(rows) + 1.8)
    figure, axis = plt.subplots(figsize=(9, figure_height))
    axis.axis("off")

    table = axis.table(
        cellText=[[key, value] for key, value in rows],
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.3)
    axis.set_title(title, fontsize=13, pad=12)

    figure.tight_layout()
    figure.savefig(out_path, dpi=180)
    plt.close(figure)


def _save_summary_image(title: str, lines: Sequence[str], out_path: Path) -> None:
    figure_height = max(5.0, 0.42 * len(lines) + 1.8)
    figure, axis = plt.subplots(figsize=(10, figure_height))
    axis.axis("off")
    axis.set_title(title, fontsize=13, pad=10)
    axis.text(
        0.02,
        0.98,
        "\n".join(lines),
        fontsize=10,
        va="top",
        ha="left",
        family="monospace",
    )

    figure.tight_layout()
    figure.savefig(out_path, dpi=180)
    plt.close(figure)


def _load_sentiment_artifact(base_dir: Path) -> Dict[str, object]:
    model_dir = base_dir / get_settings().model_dicts_dir
    pickle_path = model_dir / "sentiment_5class_model.pkl"
    joblib_path = model_dir / "sentiment_5class_model.joblib"

    if pickle_path.exists():
        return joblib.load(pickle_path)
    if joblib_path.exists():
        return joblib.load(joblib_path)

    raise FileNotFoundError("sentiment artifact not found (.pkl or .joblib)")


def build_sentiment_report(base_dir: Path, output_dir: Path) -> None:
    model_dir = base_dir / get_settings().model_dicts_dir
    dataset_path = model_dir / "collected_labeled_data.csv"
    emoji_path = model_dir / "Emoji_Dict.p"

    rows = read_labeled_dataset(dataset_path)
    texts, labels, upgrade_stats, _emoji_polarity = _upgrade_labels_to_five_classes(
        rows,
        str(emoji_path),
    )

    train_idx, test_idx = train_test_split(
        np.arange(len(texts)),
        test_size=0.15,
        random_state=42,
        stratify=labels,
    )

    test_texts = [texts[idx] for idx in test_idx]
    y_test = labels[test_idx]

    artifact = _load_sentiment_artifact(base_dir)
    resources = load_emoji_resources(emoji_path)
    emoji_polarity = {str(key): float(value) for key, value in dict(artifact.get("emoji_polarity", {})).items()}

    word_vectorizer = artifact["word_vectorizer"]
    char_vectorizer = artifact["char_vectorizer"]
    dense_scaler = artifact["dense_scaler"]
    classifier = artifact["classifier"]

    test_augmented = [augment_text_for_model(text, resources, emoji_polarity) for text in test_texts]
    test_normalized = [
        str(text_signal_features(text, resources, emoji_polarity)["normalized"]) for text in test_texts
    ]

    x_word_test = word_vectorizer.transform(test_augmented)
    x_char_test = char_vectorizer.transform(test_normalized)
    dense_test = build_dense_feature_matrix(test_texts, resources, emoji_polarity)
    x_dense_test = csr_matrix(dense_scaler.transform(dense_test))
    x_test = hstack([x_word_test, x_char_test, x_dense_test], format="csr")

    y_pred = classifier.predict(x_test)

    matrix = confusion_matrix(y_test, y_pred, labels=np.arange(len(SENTIMENT_LABELS)))
    _save_confusion_matrix(
        matrix,
        labels=SENTIMENT_LABELS,
        title="Sentiment Model - Confusion Matrix",
        out_path=output_dir / "confusion_matrix.png",
    )

    macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)

    per_class = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=np.arange(len(SENTIMENT_LABELS)),
        zero_division=0,
    )

    rows_for_table: List[Tuple[str, str]] = [
        ("Accuracy", f"{accuracy:.4f}"),
        ("Macro Precision", f"{macro_precision:.4f}"),
        ("Macro Recall", f"{macro_recall:.4f}"),
        ("Macro F1", f"{macro_f1:.4f}"),
        ("Weighted F1", f"{weighted_f1:.4f}"),
        ("Holdout Samples", str(int(len(y_test)))),
    ]
    for index, label in enumerate(SENTIMENT_LABELS):
        rows_for_table.append((f"{label} precision", f"{per_class[0][index]:.4f}"))
        rows_for_table.append((f"{label} recall", f"{per_class[1][index]:.4f}"))
        rows_for_table.append((f"{label} f1", f"{per_class[2][index]:.4f}"))
        rows_for_table.append((f"{label} support", str(int(per_class[3][index]))))

    _save_key_values_image(
        "Sentiment Model - Evaluation Scores",
        rows_for_table,
        output_dir / "evaluation_scores.png",
    )

    class_names = list(getattr(classifier, "classes_", []))
    class_names_pretty = [SENTIMENT_LABELS[int(index)] for index in class_names if int(index) < len(SENTIMENT_LABELS)]
    feature_count = int(getattr(classifier, "coef_", np.zeros((1, 0))).shape[1])

    summary_lines = [
        "Model Type: LogisticRegression (multiclass via one-vs-rest)",
        f"Artifact Version: {artifact.get('artifact_version', 'unknown')}",
        f"Created At: {artifact.get('created_at', 'unknown')}",
        f"Word Vectorizer: Tfidf ngram_range={word_vectorizer.ngram_range}, max_features={word_vectorizer.max_features}",
        f"Char Vectorizer: Tfidf ngram_range={char_vectorizer.ngram_range}, max_features={char_vectorizer.max_features}",
        f"Dense Feature Count: {dense_test.shape[1]}",
        f"Total Feature Count: {feature_count}",
        f"Classifier C: {getattr(classifier, 'C', 'n/a')}",
        f"Classifier Solver: {getattr(classifier, 'solver', 'n/a')}",
        f"Classes: {', '.join(class_names_pretty)}",
        f"Train/Test split: {len(train_idx)} / {len(test_idx)}",
        f"Dataset rows used: {upgrade_stats.get('rows', len(texts))}",
        f"Five-class distribution: {upgrade_stats.get('five_class_distribution', {})}",
        f"Thresholds: {upgrade_stats.get('label_thresholds', {})}",
    ]
    _save_summary_image("Sentiment Model - Summary", summary_lines, output_dir / "model_summary.png")


def _inject_single_typo(word: str, lexicon: Dict[str, int], rng: random.Random) -> str | None:
    if len(word) < 2:
        return None

    candidates: List[str] = []

    idx = rng.randrange(0, len(word))
    candidates.append(word[:idx] + word[idx] + word[idx:])

    if len(word) > 2:
        idx_del = rng.randrange(0, len(word))
        candidates.append(word[:idx_del] + word[idx_del + 1 :])

    if len(word) > 2:
        idx_swap = rng.randrange(0, len(word) - 1)
        swapped = list(word)
        swapped[idx_swap], swapped[idx_swap + 1] = swapped[idx_swap + 1], swapped[idx_swap]
        candidates.append("".join(swapped))

    for typo in candidates:
        if typo and typo != word and typo not in lexicon:
            return typo

    return None


def _collect_spell_eval_samples(
    checker: SpellChecker,
    max_positive: int = 500,
    seed: int = 42,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    rng = random.Random(seed)
    positive: List[Tuple[str, str]] = []
    negative: List[Tuple[str, str]] = []

    tri_items = sorted(
        checker.ng["tri"].items(),
        key=lambda item: sum(item[1].values()),
        reverse=True,
    )

    for (prev2, prev1), next_words in tri_items:
        if len(positive) >= max_positive:
            break

        if not is_dev_word(prev2) or not is_dev_word(prev1):
            continue

        ranked = sorted(next_words.items(), key=lambda item: item[1], reverse=True)
        for target_word, _count in ranked[:4]:
            if len(positive) >= max_positive:
                break
            if not is_dev_word(target_word):
                continue
            if len(target_word) < 3:
                continue
            if target_word not in checker.lexicon:
                continue

            typo = _inject_single_typo(target_word, checker.lexicon, rng)
            if typo is None:
                continue

            positive.append((f"{prev2} {prev1} {typo}", target_word))
            negative.append((f"{prev2} {prev1} {target_word}", target_word))

    return positive, negative


def build_spell_report(output_dir: Path) -> None:
    checker = SpellChecker()
    positive_samples, negative_samples = _collect_spell_eval_samples(checker)

    y_true: List[int] = []
    y_pred: List[int] = []
    exact_match_positive = 0
    positive_count = 0
    suggestion_count = 0

    for sentence, expected in positive_samples:
        _corrected, suggestions = checker.correct(sentence, suggest_only=True)
        y_true.append(1)
        suggested = len(suggestions) > 0
        y_pred.append(1 if suggested else 0)
        if suggested:
            suggestion_count += 1
            if suggestions[0].get("suggest") == expected:
                exact_match_positive += 1
        positive_count += 1

    for sentence, _expected in negative_samples:
        _corrected, suggestions = checker.correct(sentence, suggest_only=True)
        y_true.append(0)
        y_pred.append(1 if len(suggestions) > 0 else 0)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    _save_confusion_matrix(
        matrix,
        labels=["No Correction", "Suggest Correction"],
        title="Spell Checker - Confusion Matrix",
        out_path=output_dir / "confusion_matrix.png",
    )

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    exact_match_rate = (exact_match_positive / positive_count) if positive_count else 0.0
    suggestion_precision = (exact_match_positive / suggestion_count) if suggestion_count else 0.0

    rows = [
        ("Samples (positive typos)", str(len(positive_samples))),
        ("Samples (negative clean)", str(len(negative_samples))),
        ("Detection Accuracy", f"{accuracy:.4f}"),
        ("Detection Precision", f"{precision:.4f}"),
        ("Detection Recall", f"{recall:.4f}"),
        ("Detection F1", f"{f1:.4f}"),
        ("Exact Correction Rate (positive)", f"{exact_match_rate:.4f}"),
        ("Exact Suggestion Precision", f"{suggestion_precision:.4f}"),
    ]
    _save_key_values_image(
        "Spell Checker - Evaluation Scores",
        rows,
        output_dir / "evaluation_scores.png",
    )

    summary_lines = [
        "Model Type: SymSpell-style candidate generation + tri/bi language-model reranking",
        "Inference Mode: suggest_only=True for evaluation",
        f"Lexicon Size: {len(checker.lexicon)}",
        f"Deletes Map Keys: {len(checker.deletes_map)}",
        f"Max Edit Distance: {checker.max_edit}",
        f"Trigram contexts: {len(checker.ng['tri'])}",
        f"Bigram contexts: {len(checker.ng['bi'])}",
        f"Benchmark positives: {len(positive_samples)}",
        f"Benchmark negatives: {len(negative_samples)}",
        "Note: Evaluation is synthetic typo injection over frequent trigram contexts.",
        "Note: Confusion matrix measures correction decision quality, not token-level language understanding.",
    ]
    _save_summary_image("Spell Checker - Summary", summary_lines, output_dir / "model_summary.png")


def _collect_word_eval_samples(
    predictor: WordPredictor,
    max_samples: int = 3000,
) -> List[Tuple[str, str]]:
    samples: List[Tuple[str, str]] = []
    tri_items = sorted(
        predictor.ng["tri"].items(),
        key=lambda item: sum(item[1].values()),
        reverse=True,
    )

    for (prev2, prev1), next_counts in tri_items:
        if len(samples) >= max_samples:
            break
        if not next_counts:
            continue

        true_next = max(next_counts.items(), key=lambda item: item[1])[0]
        if not true_next:
            continue

        context = f"{prev2} {prev1}"
        samples.append((context, true_next))

    return samples


def build_word_prediction_report(output_dir: Path) -> None:
    predictor = WordPredictor()
    samples = _collect_word_eval_samples(predictor)

    if not samples:
        raise RuntimeError("word prediction evaluation set is empty")

    top_words = [word for word, _count in Counter(target for _, target in samples).most_common(14)]
    other_label = "OTHER"
    labels = top_words + [other_label]

    def bucket(word: str) -> str:
        return word if word in top_words else other_label

    true_bucketed: List[str] = []
    pred_bucketed: List[str] = []
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    reciprocal_ranks: List[float] = []

    for context, true_next in samples:
        predictions = predictor.predict(context, top_k=5)
        predicted_words = [str(item.get("word", "")) for item in predictions]

        top1 = predicted_words[0] if predicted_words else ""
        true_bucketed.append(bucket(true_next))
        pred_bucketed.append(bucket(top1))

        if top1 == true_next:
            top1_hits += 1
        if true_next in predicted_words[:3]:
            top3_hits += 1
        if true_next in predicted_words:
            top5_hits += 1
            rank = predicted_words.index(true_next) + 1
            reciprocal_ranks.append(1.0 / float(rank))
        else:
            reciprocal_ranks.append(0.0)

    matrix = confusion_matrix(true_bucketed, pred_bucketed, labels=labels)
    _save_confusion_matrix(
        matrix,
        labels=labels,
        title="Word Prediction - Confusion Matrix (Top-1, bucketed)",
        out_path=output_dir / "confusion_matrix.png",
    )

    count = len(samples)
    top1_acc = top1_hits / count
    hit3 = top3_hits / count
    hit5 = top5_hits / count
    mrr = float(np.mean(reciprocal_ranks))
    macro_f1_bucketed = f1_score(true_bucketed, pred_bucketed, average="macro", zero_division=0)

    rows = [
        ("Evaluation Samples", str(count)),
        ("Top-1 Accuracy", f"{top1_acc:.4f}"),
        ("Hit@3", f"{hit3:.4f}"),
        ("Hit@5", f"{hit5:.4f}"),
        ("MRR@5", f"{mrr:.4f}"),
        ("Macro F1 (bucketed Top-1)", f"{macro_f1_bucketed:.4f}"),
        ("Confusion labels", f"Top-14 words + {other_label}"),
    ]
    _save_key_values_image(
        "Word Prediction - Evaluation Scores",
        rows,
        output_dir / "evaluation_scores.png",
    )

    summary_lines = [
        "Model Type: Context-aware trigram/bigram interpolation with lexical priors",
        f"Trigram Context Count: {len(predictor.ng['tri'])}",
        f"Bigram Context Count: {len(predictor.ng['bi'])}",
        f"Lexicon Size: {len(predictor.lexicon)}",
        f"Top Lexicon Cache Size: {len(predictor._top_lexicon_words)}",
        "Scoring Components:",
        "  - recency-decayed trigram probabilities",
        "  - recency-decayed bigram probabilities",
        "  - lexical prior bonus",
        "  - topical repetition boost and immediate-loop penalty",
        f"Evaluation samples: {len(samples)}",
        "Note: Confusion matrix is bucketed to top frequent true next words + OTHER.",
    ]
    _save_summary_image("Word Prediction - Summary", summary_lines, output_dir / "model_summary.png")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate confusion matrix, evaluation score, and model summary images for all models."
    )
    parser.add_argument(
        "--output-dir",
        default="model_reports",
        help="Directory where report image folders will be written.",
    )
    return parser.parse_args()


def main() -> None:
    random.seed(42)
    np.random.seed(42)

    args = _parse_args()
    base_dir = Path(__file__).resolve().parent.parent
    output_root = _ensure_dir(base_dir / args.output_dir)

    sentiment_dir = _ensure_dir(output_root / "sentiment")
    spell_dir = _ensure_dir(output_root / "spell_check")
    word_dir = _ensure_dir(output_root / "word_prediction")

    build_sentiment_report(base_dir, sentiment_dir)
    build_spell_report(spell_dir)
    build_word_prediction_report(word_dir)

    print(f"Saved report images to: {output_root}")


if __name__ == "__main__":
    main()
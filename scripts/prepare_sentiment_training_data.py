from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd


SPACE_RE = re.compile(r"\s+")


@dataclass
class Row3Class:
    text: str
    label: int
    source: str


def _clean_text(raw: object) -> str:
    text = str(raw or "").replace("\ufeff", " ").replace("\u200d", " ")
    return SPACE_RE.sub(" ", text).strip()


def _parse_int_label(raw: object) -> int | None:
    value = str(raw).strip()
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _read_csv_rows(path: Path, text_col: str, label_col: str, mapping: dict[int, int], source: str) -> List[Row3Class]:
    if not path.exists():
        return []

    df = pd.read_csv(path)
    rows: List[Row3Class] = []
    for _, row in df.iterrows():
        text = _clean_text(row.get(text_col, ""))
        if len(text) < 2:
            continue

        label_raw = _parse_int_label(row.get(label_col, ""))
        if label_raw is None or label_raw not in mapping:
            continue

        rows.append(Row3Class(text=text, label=mapping[label_raw], source=source))

    return rows


def _read_xlsx_rows(path: Path, text_col: str, label_col: str, mapping: dict[int, int], source: str) -> List[Row3Class]:
    if not path.exists():
        return []

    df = pd.read_excel(path)
    rows: List[Row3Class] = []
    for _, row in df.iterrows():
        text = _clean_text(row.get(text_col, ""))
        if len(text) < 2:
            continue

        label_raw = _parse_int_label(row.get(label_col, ""))
        if label_raw is None or label_raw not in mapping:
            continue

        rows.append(Row3Class(text=text, label=mapping[label_raw], source=source))

    return rows


def _normalize_key(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip().lower()


def _resolve_conflicts(rows: Iterable[Row3Class]) -> Tuple[List[Tuple[str, int]], int]:
    grouped: dict[str, list[Row3Class]] = defaultdict(list)
    for row in rows:
        grouped[_normalize_key(row.text)].append(row)

    source_priority = {
        "collected": 4,
        "train_dataset_1": 3,
        "train_dataset_2": 2,
        "train_dataset_3": 1,
    }
    conflict_count = 0
    merged: List[Tuple[str, int]] = []

    for _, group in grouped.items():
        labels = [item.label for item in group]
        counts = Counter(labels)
        if len(counts) > 1:
            conflict_count += 1

        max_count = max(counts.values())
        tied_labels = [label for label, count in counts.items() if count == max_count]

        if len(tied_labels) == 1:
            chosen = tied_labels[0]
        else:
            best_label = tied_labels[0]
            best_priority = -1
            for label in tied_labels:
                candidate_priority = max(
                    source_priority.get(item.source, 0) for item in group if item.label == label
                )
                if candidate_priority > best_priority:
                    best_priority = candidate_priority
                    best_label = label
            chosen = best_label

        merged.append((group[0].text, chosen))

    return merged, conflict_count


def _parse_negative_sentences_file(path: Path) -> List[Tuple[str, int, int]]:
    if not path.exists():
        return []

    # User mapping request:
    # raw 0 -> negative (0)
    # raw 1 -> semi_negative (1)
    # raw 2 -> negative (0)
    # raw 3 appears in file; treat as stronger negative (0)
    mapping = {0: 0, 1: 1, 2: 0, 3: 0}
    parsed: List[Tuple[str, int, int]] = []

    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.rstrip("\n")
            if not line:
                continue

            if "\t" not in line:
                continue

            sentence, raw_label_text = line.rsplit("\t", 1)
            text = _clean_text(sentence)
            if len(text) < 2:
                continue

            raw_label = _parse_int_label(raw_label_text)
            if raw_label is None or raw_label not in mapping:
                continue

            parsed.append((text, mapping[raw_label], raw_label))

    return parsed


def _build_negative_lexicon(xlsx_path: Path) -> set[str]:
    if not xlsx_path.exists():
        return set()

    df = pd.read_excel(xlsx_path)
    candidate_columns = [
        "RawRom",
        "RawNep",
        "NormNep",
        "NormRom",
    ]

    terms: set[str] = set()
    for column in candidate_columns:
        if column not in df.columns:
            continue

        for value in df[column].tolist():
            text = _clean_text(value)
            if len(text) < 2:
                continue

            terms.add(text)
            if " " in text:
                for token in text.split(" "):
                    token = _clean_text(token)
                    if len(token) >= 2:
                        terms.add(token)

    return terms


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare merged sentiment training data and negative lexicon.")
    parser.add_argument("--model-dicts-dir", default="model_dicts", help="Directory containing sentiment data files.")
    args = parser.parse_args()

    model_dir = Path(args.model_dicts_dir)

    rows: List[Row3Class] = []
    rows.extend(
        _read_csv_rows(
            model_dir / "collected_labeled_data.csv",
            text_col="text",
            label_col="label",
            mapping={0: 0, 1: 1, 2: 2},
            source="collected",
        )
    )
    rows.extend(
        _read_csv_rows(
            model_dir / "train dataset 1.csv",
            text_col="text",
            label_col="label",
            mapping={0: 0, 1: 1, 2: 2},
            source="train_dataset_1",
        )
    )
    rows.extend(
        _read_csv_rows(
            model_dir / "train dataset 2.csv",
            text_col="Sentences",
            label_col="Sentiment",
            mapping={-1: 0, 1: 1, 0: 2},
            source="train_dataset_2",
        )
    )
    rows.extend(
        _read_xlsx_rows(
            model_dir / "train dataset 3.xlsx",
            text_col="Reviews",
            label_col="Emotion",
            mapping={0: 0, 1: 1},
            source="train_dataset_3",
        )
    )

    merged_rows, conflicts = _resolve_conflicts(rows)
    merged_df = pd.DataFrame(merged_rows, columns=["text", "label"])
    merged_df = merged_df.drop_duplicates(subset=["text"], keep="first")
    merged_df = merged_df[merged_df["text"].astype(str).str.len() >= 2]

    merged_out = model_dir / "merged_labeled_data.csv"
    merged_df.to_csv(merged_out, index=False, encoding="utf-8")

    neg_rows = _parse_negative_sentences_file(model_dir / "negative sentences only.txt")
    neg_df = pd.DataFrame(neg_rows, columns=["text", "label", "raw_label"])
    neg_df = neg_df.drop_duplicates(subset=["text", "label"], keep="first")

    neg_out = model_dir / "negative_sentences_5class.csv"
    neg_df[["text", "label"]].to_csv(neg_out, index=False, encoding="utf-8")

    lexicon_terms = _build_negative_lexicon(model_dir / "Nepali negative speech words.xlsx")
    lexicon_out = model_dir / "negative_lexicon.txt"
    lexicon_out.write_text("\n".join(sorted(lexicon_terms)), encoding="utf-8")

    merged_counts = Counter(merged_df["label"].tolist())
    neg_counts = Counter(neg_df["label"].tolist())

    print(f"Merged dataset rows: {len(merged_df)}")
    print(f"Merged label distribution: {dict(sorted(merged_counts.items()))}")
    print(f"Conflicting texts resolved by vote/priority: {conflicts}")
    print(f"Supplement 5-class rows: {len(neg_df)}")
    print(f"Supplement label distribution: {dict(sorted(neg_counts.items()))}")
    print(f"Negative lexicon terms: {len(lexicon_terms)}")
    print(f"Wrote: {merged_out}")
    print(f"Wrote: {neg_out}")
    print(f"Wrote: {lexicon_out}")


if __name__ == "__main__":
    main()

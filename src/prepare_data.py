import ast
from typing import List, Tuple

import pandas as pd


def read_train_dataset(path: str):
    """Читаем тренировочный датасет (train_dataset.tsv)."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    df["text"] = df["text"].astype(str)
    df["target"] = df["target"].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) and str(x).strip() not in {"", "[]"} else []
    )
    return df


def read_test_dataset(path: str):
    """Читаем тестовый датасет (private_test_dataset.csv)."""
    df = pd.read_csv(path, dtype=str)
    df["text"] = df["text"].astype(str)
    return df


def spans_to_bio_tags(text: str, spans: List[Tuple[int, int, str]]) -> List[str]:
    """Преобразуем символьные span'ы (start, end, label) в BIO-теги."""
    text = str(text)
    bio_tags = ["O"] * len(text)
    occupied = set()

    for start, end, label in sorted(spans or [], key=lambda x: (x[0], x[1])):
        start = max(0, int(start))
        end = min(len(text), int(end))

        if start >= end:
            continue

        for i in range(start, end):
            if i in occupied:
                continue

            if i == start or bio_tags[i - 1] == "O":
                bio_tags[i] = f"B-{label}"
            else:
                bio_tags[i] = f"I-{label}"

            occupied.add(i)

    return bio_tags

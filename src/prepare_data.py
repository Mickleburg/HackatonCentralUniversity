import pandas as pd
import ast
from typing import List, Tuple

def read_train_dataset(path: str):
    df = pd.read_csv(path, sep="\t", dtype=str)
    df["target"] = df["target"].apply(lambda x: ast.literal_eval(x) if pd.notna(x) and x != "[]" else [])
    return df

def read_test_dataset(path: str):
    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str)
    return df

def train_test_split(df, test_size=0.2, seed=42):
    df = df.reset_index(drop=True)
    n = len(df)
    split_idx = int(n * (1 - test_size))
    train_df = df[:split_idx].reset_index(drop=True)
    valid_df = df[split_idx:].reset_index(drop=True)
    return train_df, valid_df

def spans_to_bio_tags(text: str, spans: List[Tuple[int, int, str]]) -> List[str]:
    """Преобразуем символьные span'ы в BIO-теги."""
    bio_tags = []
    span_dict = {}
    
    for start, end, label in spans:
        for i in range(start, end):
            span_dict[i] = label
    
    for i, char in enumerate(text):
        if i not in span_dict:
            bio_tags.append("O")
        else:
            label = span_dict[i]
            if i == 0 or span_dict.get(i - 1) != label:
                bio_tags.append(f"B-{label}")
            else:
                bio_tags.append(f"I-{label}")
    
    return bio_tags

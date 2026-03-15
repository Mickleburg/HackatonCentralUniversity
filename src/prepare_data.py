import ast
import pandas as pd


def safe_parse_target(value):
    """Парсим target как список кортежей [(start, end, label), ...]."""
    if pd.isna(value):
        return []
    
    value = str(value).strip()
    
    if not value or value == "[]" or value == "empty":
        return []
    
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            result = []
            for item in parsed:
                if isinstance(item, (tuple, list)) and len(item) >= 3:
                    start, end, label = item[0], item[1], item[2]
                    result.append((int(start), int(end), str(label)))
            return result
        return []
    except (ValueError, SyntaxError, TypeError):
        return []


def read_train_dataset(path: str):
    """Читаем train_dataset.tsv с target в формате [(start, end, label), ...]."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    df["text"] = df["text"].fillna("").astype(str)
    df["target"] = df["target"].apply(safe_parse_target)
    df = df.reset_index(drop=True)
    df["row_id"] = df.index
    return df


def read_test_dataset(path: str):
    """Читаем test dataset (может быть CSV или TSV)."""
    try:
        df = pd.read_csv(path, sep="\t", dtype=str)
    except:
        df = pd.read_csv(path, dtype=str)
    
    df["text"] = df["text"].fillna("").astype(str)
    
    if "id" not in df.columns:
        df = df.reset_index(drop=True)
        df["row_id"] = df.index
    
    return df


def save_processed(df, path: str):
    """Сохраняем датасет."""
    df.to_csv(path, index=False)


def spans_to_bio_by_offsets(offset_mapping, spans):
    """
    Конвертируем символьные spans в BIO-теги по токен-оффсетам.
    
    Args:
        offset_mapping: список (start, end) позиций каждого токена в исходном тексте
        spans: список (char_start, char_end, label)
    
    Returns:
        список BIO-тегов для каждого токена
    """
    tags = []
    
    for token_start, token_end in offset_mapping:
        if token_start == token_end:
            tags.append("O")
            continue
        
        tag = "O"
        
        for ent_start, ent_end, label in spans:
            if token_start >= ent_start and token_start < ent_end:
                if token_start == ent_start:
                    tag = f"B-{label}"
                else:
                    tag = f"I-{label}"
                break
        
        tags.append(tag)
    
    return tags

import argparse
import ast
import os

import pandas as pd


def parse_prediction(value):
    """
    Безопасно парсим prediction из CSV.
    Ожидаем строку вида:
    "[(0, 10, 'ФИО'), (15, 25, 'Email')]"
    """
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    text = str(value).strip()
    if text in {"", "[]", "nan", "None"}:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return parsed
        return []
    except Exception:
        return []


def normalize_spans(spans):
    """
    Нормализуем spans к формату:
    [(start, end, label), ...]
    """
    result = []

    for item in spans:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            continue

        start, end, label = item

        try:
            start = int(start)
            end = int(end)
            label = str(label)
        except Exception:
            continue

        if start < 0 or end <= start:
            continue

        result.append((start, end, label))

    result = sorted(set(result), key=lambda x: (x[0], x[1], x[2]))
    return result


def build_submission(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, dtype=str)

    if "prediction" not in df.columns:
        raise ValueError("Input CSV must contain 'prediction' column")

    predictions = []
    for value in df["prediction"]:
        spans = parse_prediction(value)
        spans = normalize_spans(spans)
        predictions.append(str(spans))

    submission = pd.DataFrame({
        "prediction": predictions
    })

    if "id" in df.columns:
        submission.insert(0, "id", df["id"])
    else:
        submission.insert(0, "id", range(len(submission)))

    submission.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")
    print(f"Rows: {len(submission)}")

    if len(submission) > 0:
        print("Sample row:")
        print(submission.iloc[0].to_dict())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build final submission file")
    parser.add_argument(
        "--input",
        default="data/answer/merged_predictions.csv",
        help="Path to predictions CSV"
    )
    parser.add_argument(
        "--output",
        default="data/answer/submission.csv",
        help="Path to save submission CSV"
    )

    args = parser.parse_args()
    build_submission(args.input, args.output)

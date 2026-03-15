import argparse
import ast
import os

import pandas as pd

from src.evaluate import compute_metrics
from src.merge_predictions import merge_predictions
from src.ner_model import NERModel
from src.prepare_data import read_test_dataset, read_train_dataset
from src.regex_detector import detect_pii
from src.utils import ensure_dirs


ensure_dirs()


def _parse_spans_cell(value):
    if isinstance(value, list):
        return value

    if pd.isna(value) or str(value).strip() in {"", "[]", "nan", "None"}:
        return []

    try:
        return ast.literal_eval(str(value))
    except Exception:
        return []


def prepare_command(args):
    """Копируем данные в папки."""
    print("Preparing data...")

    if os.path.exists("data/raw/train_dataset.tsv"):
        train_df = read_train_dataset("data/raw/train_dataset.tsv")
        train_df.to_csv("data/processed/train.csv", index=False)
        print(f"Train: {len(train_df)} samples")

    if os.path.exists("data/raw/private_test_dataset.csv"):
        test_df = read_test_dataset("data/raw/private_test_dataset.csv")
        test_df.to_csv("data/processed/test.csv", index=False)
        print(f"Test: {len(test_df)} samples")


def regex_command(args):
    """Запуск regex детектора."""
    input_file = getattr(args, "input", None) or "data/processed/test.csv"
    output_file = getattr(args, "output", None) or "data/answer/regex_predictions.csv"

    print(f"Running regex detector on {input_file}...")

    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return

    if input_file.endswith(".tsv"):
        df = pd.read_csv(input_file, sep="\t", dtype=str)
    else:
        df = pd.read_csv(input_file, dtype=str)

    df["text"] = df["text"].astype(str)
    predictions = [detect_pii(text) for text in df["text"]]

    result_df = pd.DataFrame({"prediction": predictions})

    if "id" in df.columns:
        result_df.insert(0, "id", df["id"])

    result_df["text"] = df["text"]
    result_df.to_csv(output_file, index=False)

    print(f"Saved to {output_file}")

    if "target" in df.columns:
        targets = [_parse_spans_cell(t) for t in df["target"]]
        metrics = compute_metrics(predictions, targets)
        print(f"Regex metrics: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['micro_f1']:.4f}")


def ner_train_command(args):
    """Обучение NER модели."""
    print("Training NER model...")

    if not os.path.exists("data/processed/train.csv"):
        print("Run 'python main.py prepare' first!")
        return

    train_df = pd.read_csv("data/processed/train.csv", dtype=str)
    train_df["text"] = train_df["text"].astype(str)
    train_df["target"] = train_df["target"].apply(_parse_spans_cell)

    model_name = getattr(args, "model_name", None) or "tiny"
    output_dir = getattr(args, "model_dir", None) or "ner_model"
    epochs = getattr(args, "epochs", 3)
    batch_size = getattr(args, "batch_size", 8)
    max_len = getattr(args, "max_len", 512)
    learning_rate = getattr(args, "learning_rate", 2e-5)

    print(f"Training on {len(train_df)} samples...")

    model = NERModel(model_name=model_name, output_dir=output_dir)
    model.train(
        train_df,
        epochs=epochs,
        batch_size=batch_size,
        max_len=max_len,
        learning_rate=learning_rate,
    )

    print("NER model trained and saved")


def ner_predict_command(args):
    """Предсказание NER на датасете."""
    input_file = getattr(args, "input", None) or "data/processed/test.csv"
    output_file = getattr(args, "output", None) or "data/answer/ner_predictions.csv"
    model_dir = getattr(args, "model_dir", None) or "ner_model"

    print(f"Running NER on {input_file}...")

    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return

    df = pd.read_csv(input_file, dtype=str)
    df["text"] = df["text"].astype(str)

    model = NERModel(output_dir=model_dir)
    model.load()

    print(f"Predicting on {len(df)} texts...")
    predictions = model.predict_batch(df["text"].tolist())

    result_df = pd.DataFrame({"prediction": predictions})

    if "id" in df.columns:
        result_df.insert(0, "id", df["id"])

    result_df["text"] = df["text"]
    result_df.to_csv(output_file, index=False)

    print(f"Saved to {output_file}")

    if "target" in df.columns:
        targets = [_parse_spans_cell(t) for t in df["target"]]
        metrics = compute_metrics(predictions, targets)
        print(f"NER metrics: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['micro_f1']:.4f}")


def merge_command(args):
    """Merge regex и NER predictions."""
    print("Merging regex and NER predictions...")

    regex_file = getattr(args, "regex", None) or "data/answer/regex_predictions.csv"
    ner_file = getattr(args, "ner", None) or "data/answer/ner_predictions.csv"
    output_file = getattr(args, "output", None) or "data/answer/merged_predictions.csv"

    if not os.path.exists(regex_file):
        print(f"File not found: {regex_file}")
        return

    if not os.path.exists(ner_file):
        print(f"File not found: {ner_file}")
        return

    regex_df = pd.read_csv(regex_file, dtype=str)
    ner_df = pd.read_csv(ner_file, dtype=str)

    merged = []
    for regex_pred, ner_pred in zip(regex_df["prediction"], ner_df["prediction"]):
        regex_spans = _parse_spans_cell(regex_pred)
        ner_spans = _parse_spans_cell(ner_pred)
        merged.append(merge_predictions(regex_spans, ner_spans))

    result_df = pd.DataFrame({"prediction": merged})

    if "id" in regex_df.columns:
        result_df.insert(0, "id", regex_df["id"])

    result_df["text"] = regex_df["text"]
    result_df.to_csv(output_file, index=False)

    print(f"Saved to {output_file}")


def all_command(args):
    """Полный pipeline."""
    print("=" * 60)
    print("Running full NER pipeline")
    print("=" * 60)

    print("\n[1/5] Preparing data...")
    prepare_command(args)

    print("\n[2/5] Training NER model...")
    ner_train_command(args)

    print("\n[3/5] Running regex detector...")
    regex_args = argparse.Namespace(**vars(args))
    regex_args.input = "data/processed/test.csv"
    regex_args.output = "data/answer/regex_predictions.csv"
    regex_command(regex_args)

    print("\n[4/5] Running NER predictions...")
    ner_args = argparse.Namespace(**vars(args))
    ner_args.input = "data/processed/test.csv"
    ner_args.output = "data/answer/ner_predictions.csv"
    ner_predict_command(ner_args)

    print("\n[5/5] Merging predictions...")
    merge_args = argparse.Namespace(**vars(args))
    merge_args.regex = "data/answer/regex_predictions.csv"
    merge_args.ner = "data/answer/ner_predictions.csv"
    merge_args.output = "data/answer/merged_predictions.csv"
    merge_command(merge_args)

    print("\n" + "=" * 60)
    print("Pipeline completed!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NER Pipeline for Russian PII detection")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    subparsers.add_parser("prepare", help="Prepare data")

    regex_parser = subparsers.add_parser("regex", help="Run regex detector")
    regex_parser.add_argument("--input", default=None, help="Input file path")
    regex_parser.add_argument("--output", default=None, help="Output file path")

    ner_train_parser = subparsers.add_parser("ner_train", help="Train NER model")
    ner_train_parser.add_argument("--model-name", dest="model_name", default="tiny")
    ner_train_parser.add_argument("--model-dir", dest="model_dir", default="ner_model")
    ner_train_parser.add_argument("--epochs", type=int, default=3)
    ner_train_parser.add_argument("--batch-size", dest="batch_size", type=int, default=8)
    ner_train_parser.add_argument("--max-len", dest="max_len", type=int, default=512)
    ner_train_parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=2e-5)

    ner_pred_parser = subparsers.add_parser("ner_predict", help="Run NER prediction")
    ner_pred_parser.add_argument("--input", default=None, help="Input file path")
    ner_pred_parser.add_argument("--output", default=None, help="Output file path")
    ner_pred_parser.add_argument("--model-dir", dest="model_dir", default="ner_model")

    merge_parser = subparsers.add_parser("merge", help="Merge regex and NER predictions")
    merge_parser.add_argument("--regex", default=None, help="Regex predictions file")
    merge_parser.add_argument("--ner", default=None, help="NER predictions file")
    merge_parser.add_argument("--output", default=None, help="Output file path")

    all_parser = subparsers.add_parser("all", help="Run full pipeline")
    all_parser.add_argument("--model-name", dest="model_name", default="tiny")
    all_parser.add_argument("--model-dir", dest="model_dir", default="ner_model")
    all_parser.add_argument("--epochs", type=int, default=3)
    all_parser.add_argument("--batch-size", dest="batch_size", type=int, default=8)
    all_parser.add_argument("--max-len", dest="max_len", type=int, default=512)
    all_parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=2e-5)

    args = parser.parse_args()

    if args.command == "prepare":
        prepare_command(args)
    elif args.command == "regex":
        regex_command(args)
    elif args.command == "ner_train":
        ner_train_command(args)
    elif args.command == "ner_predict":
        ner_predict_command(args)
    elif args.command == "merge":
        merge_command(args)
    elif args.command == "all":
        all_command(args)
    else:
        parser.print_help()

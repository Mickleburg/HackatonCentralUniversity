import argparse
import pandas as pd
import os
from pathlib import Path

from src.regex_detector import detect_pii
from src.ner_model import NERModel
from src.merge_predictions import merge_predictions
from src.prepare_data import read_train_dataset, read_test_dataset, train_test_split
from src.evaluate import compute_metrics, save_metrics
from src.utils import ensure_dirs

ensure_dirs()

def prepare_command(args):
    """Подготовка данных."""
    print("Preparing data...")
    train_df = read_train_dataset("data/raw/train_dataset.tsv")
    train_df, valid_df = train_test_split(train_df, test_size=0.2)
    
    train_df.to_csv("data/processed/train.csv", index=False)
    valid_df.to_csv("data/processed/valid.csv", index=False)
    
    test_df = read_test_dataset("data/raw/private_test_dataset.csv")
    test_df.to_csv("data/processed/test.csv", index=False)
    
    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

def regex_command(args):
    """Запуск regex детектора."""
    input_file = args.input or "data/raw/train_dataset.tsv"
    output_file = args.output or "data/answer/regex_predictions.csv"
    
    print(f"Running regex on {input_file}...")
    
    if input_file.endswith(".tsv"):
        df = pd.read_csv(input_file, sep="\t", dtype=str)
    else:
        df = pd.read_csv(input_file, dtype=str)
    
    df["text"] = df["text"].astype(str)
    predictions = []
    
    for text in df["text"]:
        pred = detect_pii(text)
        predictions.append(str(pred))
    
    result_df = pd.DataFrame({
        "text": df["text"],
        "prediction": predictions,
    })
    
    if "id" in df.columns:
        result_df.insert(0, "id", df["id"])
    
    result_df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")
    
    if "target" in df.columns:
        import ast
        targets = [ast.literal_eval(t) if pd.notna(t) and t != "[]" else [] for t in df["target"]]
        metrics = compute_metrics(
            [ast.literal_eval(p) for p in predictions],
            targets,
        )
        print(f"Regex metrics: {metrics}")

def ner_train_command(args):
    """Обучение NER."""
    print("Training NER...")
    
    if not os.path.exists("data/processed/train.csv"):
        print("Run 'python main.py prepare' first")
        return
    
    train_df = pd.read_csv("data/processed/train.csv")
    train_df["target"] = train_df["target"].apply(eval)
    
    model = NERModel()
    model.train(train_df, epochs=3, batch_size=8)
    print("NER model trained and saved")

def ner_predict_command(args):
    """Предсказание NER."""
    input_file = args.input or "data/raw/private_test_dataset.csv"
    output_file = args.output or "data/answer/ner_predictions.csv"
    
    print(f"Running NER on {input_file}...")
    
    df = pd.read_csv(input_file, dtype=str)
    df["text"] = df["text"].astype(str)
    
    model = NERModel()
    model.load()
    
    predictions = model.predict_batch(df["text"].tolist())
    predictions_str = [str(p) for p in predictions]
    
    result_df = pd.DataFrame({
        "text": df["text"],
        "prediction": predictions_str,
    })
    
    if "id" in df.columns:
        result_df.insert(0, "id", df["id"])
    
    result_df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

def merge_command(args):
    """Merge regex и NER."""
    print("Merging predictions...")
    
    regex_df = pd.read_csv("data/answer/regex_predictions.csv")
    ner_df = pd.read_csv("data/answer/ner_predictions.csv")
    
    merged = []
    for regex_pred, ner_pred in zip(regex_df["prediction"], ner_df["prediction"]):
        import ast
        regex_spans = ast.literal_eval(regex_pred) if regex_pred else []
        ner_spans = ast.literal_eval(ner_pred) if ner_pred else []
        merged_spans = merge_predictions(regex_spans, ner_spans)
        merged.append(str(merged_spans))
    
    result_df = pd.DataFrame({
        "text": regex_df["text"],
        "prediction": merged,
    })
    
    if "id" in regex_df.columns:
        result_df.insert(0, "id", regex_df["id"])
    
    result_df.to_csv("data/answer/merged_predictions.csv", index=False)
    print("Saved to data/answer/merged_predictions.csv")

def all_command(args):
    """Запуск всего."""
    print("Running full pipeline...")
    prepare_command(args)
    
    ner_train_command(args)
    
    regex_command({**vars(args), "input": "data/processed/train.csv", "output": None})
    regex_command({**vars(args), "input": "data/processed/valid.csv", "output": None})
    regex_command({**vars(args), "input": "data/processed/test.csv", "output": None})
    
    ner_predict_command({**vars(args), "input": "data/processed/train.csv", "output": None})
    ner_predict_command({**vars(args), "input": "data/processed/valid.csv", "output": None})
    ner_predict_command({**vars(args), "input": "data/processed/test.csv", "output": None})
    
    merge_command(args)
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    
    subparsers.add_parser("prepare")
    
    regex_parser = subparsers.add_parser("regex")
    regex_parser.add_argument("--input", default=None)
    regex_parser.add_argument("--output", default=None)
    
    subparsers.add_parser("ner_train")
    
    ner_pred_parser = subparsers.add_parser("ner_predict")
    ner_pred_parser.add_argument("--input", default=None)
    ner_pred_parser.add_argument("--output", default=None)
    
    subparsers.add_parser("merge")
    subparsers.add_parser("all")
    
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

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from src.labels import LABEL2ID, ID2LABEL
from src.prepare_data import spans_to_bio_tags


# ============================================================================
# КОНФИГУРАЦИЯ МОДЕЛЕЙ
# ============================================================================

MODEL_OPTIONS = {
    "tiny": "cointegrated/rubert-tiny2",
    "base": "cointegrated/rubert-base-cased",
    "large": "cointegrated/rubert-large-cased",
    "deeppavlov": "DeepPavlov/rubert-base-cased",
}


# ============================================================================
# NER ДАТАСЕТ
# ============================================================================

class NERDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        spans: List[List[Tuple[int, int, str]]],
        tokenizer,
        max_len: int = 512
    ):
        self.texts = texts
        self.spans = spans
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.encodings = []
        self._prepare()

    def _prepare(self):
        for text, span_list in zip(self.texts, self.spans):
            text = str(text)
            span_list = span_list or []

            bio_tags = spans_to_bio_tags(text, span_list)

            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_len,
                padding="max_length",
                return_offsets_mapping=True,
            )

            labels = []
            for start_char, end_char in encoding["offset_mapping"]:
                if start_char == end_char:
                    labels.append(-100)
                elif start_char < len(bio_tags):
                    tag = bio_tags[start_char]
                    labels.append(LABEL2ID.get(tag, LABEL2ID["O"]))
                else:
                    labels.append(LABEL2ID["O"])

            encoding["labels"] = labels
            encoding.pop("offset_mapping", None)
            self.encodings.append(encoding)

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> dict:
        enc = self.encodings[idx]
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(enc["labels"], dtype=torch.long),
        }


# ============================================================================
# NER МОДЕЛЬ
# ============================================================================

class NERModel:
    def __init__(
        self,
        model_name: str = "tiny",
        output_dir: str = "ner_model",
        device: str = None
    ):
        self.output_dir = output_dir
        self.model_name = MODEL_OPTIONS.get(model_name, model_name)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Using model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(LABEL2ID),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
        self.model.to(self.device)

    def train(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame = None,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        max_len: int = 512,
        save_steps: int = 100
    ) -> dict:
        train_dataset = NERDataset(
            train_df["text"].astype(str).tolist(),
            train_df["target"].tolist(),
            self.tokenizer,
            max_len=max_len,
        )

        eval_dataset = None
        if valid_df is not None and len(valid_df) > 0:
            eval_dataset = NERDataset(
                valid_df["text"].astype(str).tolist(),
                valid_df["target"].tolist(),
                self.tokenizer,
                max_len=max_len,
            )

        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=8 if torch.cuda.is_available() else None,
        )

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            save_steps=save_steps,
            save_total_limit=2,
            logging_steps=50,
            seed=42,
            fp16=torch.cuda.is_available(),
            eval_strategy="steps" if eval_dataset is not None else "no",
            save_strategy="steps",
            report_to="none",
            remove_unused_columns=False,
            optim="adamw_torch",
            dataloader_pin_memory=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        trainer.train()
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        print(f"Model saved to {self.output_dir}")
        return trainer.state.log_history

    def predict_text(self, text: str, max_len: int = 512) -> List[Tuple[int, int, str]]:
        text = str(text)

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        offset_mapping = encoding.pop("offset_mapping")[0].cpu().numpy()
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits[0].detach().cpu().numpy()

        pred_ids = np.argmax(logits, axis=1)

        spans = []
        current_start = None
        current_end = None
        current_label = None

        for pred_id, (start_char, end_char) in zip(pred_ids, offset_mapping):
            if start_char == end_char:
                continue

            label_tag = ID2LABEL.get(int(pred_id), "O")

            if label_tag == "O":
                if current_start is not None:
                    spans.append((current_start, current_end, current_label))
                    current_start = None
                    current_end = None
                    current_label = None
                continue

            label = label_tag.replace("B-", "").replace("I-", "")

            if label_tag.startswith("B-") or current_label != label:
                if current_start is not None:
                    spans.append((current_start, current_end, current_label))
                current_start = int(start_char)
                current_end = int(end_char)
                current_label = label
            else:
                current_end = int(end_char)

        if current_start is not None:
            spans.append((current_start, current_end, current_label))

        return sorted(set(spans), key=lambda x: (x[0], x[1], x[2]))

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 4
    ) -> List[List[Tuple[int, int, str]]]:
        predictions = []

        for i in tqdm(range(0, len(texts), batch_size), desc="NER prediction"):
            batch_texts = texts[i:i + batch_size]
            for text in batch_texts:
                predictions.append(self.predict_text(text))

        return predictions

    def load(self):
        if not os.path.exists(self.output_dir):
            raise FileNotFoundError(f"Model directory {self.output_dir} not found")

        self.tokenizer = AutoTokenizer.from_pretrained(self.output_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(self.output_dir)
        self.model.to(self.device)
        print(f"Model loaded from {self.output_dir}")

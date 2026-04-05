"""
Owner: Mayowa
Suggested model family: transformer / DistilBERT / strongest variant.

Shared reporting rules:
- data source: Codabench train/val/test files
- split policy: use the same Codabench split the class used
- reported metrics: validation accuracy and macro F1

This implementation fine-tunes DistilBERT on paired system-context and
user-utterance inputs, selects the best epoch by validation macro F1, and
exports Codabench-aligned metrics plus prediction artifacts for report use.
"""

from __future__ import annotations

import copy
import csv
import json
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

TRAIN_PATH = "codabench/bundle/starting_kit/data/train.csv"
VAL_PATH = "codabench/bundle/starting_kit/data/val.csv"
TEST_EVAL_PATH = "codabench/test.csv"
TEST_SUBMISSION_PATH = "codabench/bundle/starting_kit/data/test.csv"
OUTPUT_DIR = Path("models/results")
MODEL_NAME = "Mayowa DistilBERT (Codabench Split)"
OUTPUT_PREFIX = "mayowa_distilbert_codabench"
PRETRAINED_MODEL_NAME = "distilbert-base-uncased"
RANDOM_STATE = 42
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
GRADIENT_CLIP_NORM = 1.0

LABEL_MAP = {
    "REQUEST": 0,
    "INFORM_CONSTRAINT": 1,
    "CONFIRM_ACCEPT": 2,
    "CORRECT_CLARIFY": 3,
    "SOCIAL": 4,
}
ID_TO_LABEL = {value: key for key, value in LABEL_MAP.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_split(df: pd.DataFrame, require_labels: bool) -> pd.DataFrame:
    normalized = df.copy()
    normalized["system_context"] = normalized["system_context"].fillna("").astype(str)
    normalized["user_utterance"] = normalized["user_utterance"].fillna("").astype(str)
    normalized["text"] = (
        "system: "
        + normalized["system_context"]
        + " user: "
        + normalized["user_utterance"]
    )

    if require_labels:
        normalized["label_num"] = normalized["label"].map(LABEL_MAP)
        if normalized["label_num"].isna().any():
            raise ValueError("Found unknown label values in a labeled split.")

    return normalized


def load_codabench_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = normalize_split(pd.read_csv(TRAIN_PATH), require_labels=True)
    val_df = normalize_split(pd.read_csv(VAL_PATH), require_labels=True)
    test_eval_df = normalize_split(pd.read_csv(TEST_EVAL_PATH), require_labels=True)
    test_submission_df = normalize_split(pd.read_csv(TEST_SUBMISSION_PATH), require_labels=False)
    return train_df, val_df, test_eval_df, test_submission_df


class IntentDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int,
        with_labels: bool,
    ) -> None:
        self.system_contexts = df["system_context"].tolist()
        self.user_utterances = df["user_utterance"].tolist()
        self.instance_ids = df["instance_id"].tolist()
        self.dialogue_ids = df["dialogue_id"].tolist()
        self.turn_ids = df["turn_id"].tolist()
        self.texts = df["text"].tolist()
        self.labels = df["label_num"].astype(int).tolist() if with_labels else None
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_labels = with_labels

    def __len__(self) -> int:
        return len(self.instance_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        encoded = self.tokenizer(
            self.system_contexts[idx],
            self.user_utterances[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item: dict[str, torch.Tensor | str | int] = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "instance_id": self.instance_ids[idx],
            "dialogue_id": self.dialogue_ids[idx],
            "turn_id": int(self.turn_ids[idx]),
            "text": self.texts[idx],
        }
        if self.with_labels and self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def collate_batch(
    batch: list[dict[str, torch.Tensor | str | int]],
) -> dict[str, torch.Tensor | list[str] | list[int]]:
    payload: dict[str, torch.Tensor | list[str] | list[int]] = {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "instance_id": [str(item["instance_id"]) for item in batch],
        "dialogue_id": [str(item["dialogue_id"]) for item in batch],
        "turn_id": [int(item["turn_id"]) for item in batch],
        "text": [str(item["text"]) for item in batch],
    }
    if "labels" in batch[0]:
        payload["labels"] = torch.stack([item["labels"] for item in batch])
    return payload


def create_class_weights(train_df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    counts = train_df["label_num"].value_counts().sort_index()
    ordered_counts = np.array([counts[idx] for idx in range(len(LABEL_MAP))], dtype=np.float32)
    class_weights = ordered_counts.sum() / (len(ordered_counts) * ordered_counts)
    return torch.tensor(class_weights, dtype=torch.float32, device=device)


def evaluate_model(
    model: AutoModelForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[
    dict[str, float],
    list[int],
    list[int],
    list[str],
    list[str],
    list[str],
    list[int],
]:
    model.eval()
    predictions: list[int] = []
    gold_labels: list[int] = []
    instance_ids: list[str] = []
    texts: list[str] = []
    dialogue_ids: list[str] = []
    turn_ids: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_predictions = outputs.logits.argmax(dim=-1)

            predictions.extend(batch_predictions.detach().cpu().tolist())
            gold_labels.extend(labels.detach().cpu().tolist())
            instance_ids.extend(batch["instance_id"])
            texts.extend(batch["text"])
            dialogue_ids.extend(batch["dialogue_id"])
            turn_ids.extend(batch["turn_id"])

    metrics = {
        "accuracy": accuracy_score(gold_labels, predictions),
        "macro_f1": f1_score(gold_labels, predictions, average="macro", zero_division=0),
    }
    return metrics, predictions, gold_labels, instance_ids, texts, dialogue_ids, turn_ids


def predict_unlabeled(
    model: AutoModelForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[list[str], list[int], list[int]]:
    model.eval()
    predictions: list[int] = []
    dialogue_ids: list[str] = []
    turn_ids: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_predictions = outputs.logits.argmax(dim=-1)

            predictions.extend(batch_predictions.detach().cpu().tolist())
            dialogue_ids.extend(batch["dialogue_id"])
            turn_ids.extend(batch["turn_id"])

    return dialogue_ids, turn_ids, predictions


def train_and_select_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    device: torch.device,
) -> tuple[
    AutoModelForSequenceClassification,
    AutoTokenizer,
    list[dict[str, float]],
    dict[str, float],
    list[int],
    list[int],
    list[str],
    list[str],
    list[str],
    list[int],
]:
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_NAME,
        num_labels=len(LABEL_MAP),
        id2label=ID_TO_LABEL,
        label2id=LABEL_MAP,
    )
    model.to(device)

    train_dataset = IntentDataset(train_df, tokenizer, MAX_LENGTH, with_labels=True)
    val_dataset = IntentDataset(val_df, tokenizer, MAX_LENGTH, with_labels=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )
    loss_fn = CrossEntropyLoss(weight=create_class_weights(train_df, device))

    best_state = copy.deepcopy(model.state_dict())
    best_val_metrics = {"accuracy": 0.0, "macro_f1": -1.0, "epoch": 0}
    history: list[dict[str, float]] = []
    best_predictions: list[int] = []
    best_gold: list[int] = []
    best_instance_ids: list[str] = []
    best_texts: list[str] = []
    best_dialogue_ids: list[str] = []
    best_turn_ids: list[int] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        (
            val_metrics,
            val_predictions,
            val_gold,
            val_instance_ids,
            val_texts,
            val_dialogue_ids,
            val_turn_ids,
        ) = evaluate_model(model, val_loader, device)
        epoch_record = {
            "epoch": float(epoch),
            "train_loss": running_loss / max(len(train_loader), 1),
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(epoch_record)
        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={epoch_record['train_loss']:.4f} | "
            f"val_accuracy={val_metrics['accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_metrics["macro_f1"]:
            best_state = copy.deepcopy(model.state_dict())
            best_val_metrics = {
                "accuracy": val_metrics["accuracy"],
                "macro_f1": val_metrics["macro_f1"],
                "epoch": epoch,
            }
            best_predictions = val_predictions
            best_gold = val_gold
            best_instance_ids = val_instance_ids
            best_texts = val_texts
            best_dialogue_ids = val_dialogue_ids
            best_turn_ids = val_turn_ids

    model.load_state_dict(best_state)
    return (
        model,
        tokenizer,
        history,
        best_val_metrics,
        best_predictions,
        best_gold,
        best_instance_ids,
        best_texts,
        best_dialogue_ids,
        best_turn_ids,
    )


def export_metrics(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def export_confusion_matrix(path: Path, gold_labels: list[int], predictions: list[int]) -> None:
    matrix = confusion_matrix(
        gold_labels,
        predictions,
        labels=list(range(len(LABEL_MAP))),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gold_label", *[ID_TO_LABEL[idx] for idx in range(len(LABEL_MAP))]])
        for label_idx, row in enumerate(matrix):
            writer.writerow([ID_TO_LABEL[label_idx], *row.tolist()])


def export_predictions(
    path: Path,
    instance_ids: list[str],
    gold_labels: list[int],
    predictions: list[int],
    texts: list[str],
    dialogue_ids: list[str],
    turn_ids: list[int],
) -> None:
    payload = pd.DataFrame(
        {
            "instance_id": instance_ids,
            "dialogue_id": dialogue_ids,
            "turn_id": turn_ids,
            "gold_label": [ID_TO_LABEL[idx] for idx in gold_labels],
            "predicted_label": [ID_TO_LABEL[idx] for idx in predictions],
            "correct": [gold == pred for gold, pred in zip(gold_labels, predictions, strict=True)],
            "text": texts,
        }
    )
    payload.to_csv(path, index=False)


def export_submission(
    path: Path,
    predictions: list[int],
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for prediction in predictions:
            handle.write(f"{ID_TO_LABEL[prediction]}\n")


def export_submission_debug_csv(
    path: Path,
    dialogue_ids: list[str],
    turn_ids: list[int],
    predictions: list[int],
) -> None:
    payload = pd.DataFrame(
        {
            "dialogue_id": dialogue_ids,
            "turn_id": turn_ids,
            "predicted_label": [ID_TO_LABEL[idx] for idx in predictions],
        }
    )
    payload.to_csv(path, index=False)


def export_submission_zip(zip_path: Path, txt_path: Path) -> None:
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(txt_path, arcname="predictions.txt")


def main() -> None:
    set_seed(RANDOM_STATE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    train_df, val_df, test_eval_df, test_submission_df = load_codabench_splits()
    (
        model,
        tokenizer,
        history,
        best_val_metrics,
        val_predictions,
        val_gold,
        val_instance_ids,
        val_texts,
        val_dialogue_ids,
        val_turn_ids,
    ) = train_and_select_model(train_df, val_df, device)

    test_eval_dataset = IntentDataset(test_eval_df, tokenizer, MAX_LENGTH, with_labels=True)
    test_eval_loader = DataLoader(
        test_eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
    )
    (
        test_metrics,
        test_predictions,
        test_gold,
        test_instance_ids,
        test_texts,
        test_dialogue_ids,
        test_turn_ids,
    ) = evaluate_model(model, test_eval_loader, device)

    submission_dataset = IntentDataset(test_submission_df, tokenizer, MAX_LENGTH, with_labels=False)
    submission_loader = DataLoader(
        submission_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
    )
    submission_dialogue_ids, submission_turn_ids, submission_predictions = predict_unlabeled(
        model,
        submission_loader,
        device,
    )
    submission_txt_path = OUTPUT_DIR / "predictions.txt"
    submission_zip_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_submission.zip"

    pd.DataFrame(history).to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_history.csv", index=False)

    export_metrics(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_metrics.json",
        {
            "model_name": MODEL_NAME,
            "pretrained_model": PRETRAINED_MODEL_NAME,
            "device": str(device),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_eval_df),
            "train_path": TRAIN_PATH,
            "val_path": VAL_PATH,
            "test_eval_path": TEST_EVAL_PATH,
            "test_submission_path": TEST_SUBMISSION_PATH,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "best_val_epoch": int(best_val_metrics["epoch"]),
            "validation": {
                "accuracy": best_val_metrics["accuracy"],
                "macro_f1": best_val_metrics["macro_f1"],
            },
            "test": test_metrics,
        },
    )
    export_confusion_matrix(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_val_confusion_matrix.csv",
        val_gold,
        val_predictions,
    )
    export_predictions(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_val_predictions.csv",
        val_instance_ids,
        val_gold,
        val_predictions,
        val_texts,
        val_dialogue_ids,
        val_turn_ids,
    )
    export_predictions(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_test_predictions.csv",
        test_instance_ids,
        test_gold,
        test_predictions,
        test_texts,
        test_dialogue_ids,
        test_turn_ids,
    )
    export_submission_debug_csv(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_submission_debug.csv",
        submission_dialogue_ids,
        submission_turn_ids,
        submission_predictions,
    )
    export_submission(submission_txt_path, submission_predictions)
    export_submission_zip(submission_zip_path, submission_txt_path)

    print(MODEL_NAME)
    print(f"Train rows: {len(train_df)}")
    print(f"Val rows: {len(val_df)}")
    print(f"Test rows: {len(test_eval_df)}")
    print(f"Best validation epoch: {int(best_val_metrics['epoch'])}")
    print(f"Validation accuracy: {best_val_metrics['accuracy']:.4f}")
    print(f"Validation macro F1: {best_val_metrics['macro_f1']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Saved artifacts to: {OUTPUT_DIR}")
    print(f"Codabench predictions.txt: {submission_txt_path}")
    print(f"Codabench upload zip: {submission_zip_path}")


if __name__ == "__main__":
    main()

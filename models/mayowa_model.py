"""
Owner: Mayowa
Suggested model family: transformer / DistilBERT / strongest variant.

Do not change these shared reporting rules:
- data source: data/processed/final_gold_labels.csv
- split policy: dialogue-level 80/10/10 with random_state=42
- reported metrics: validation accuracy and macro F1

This is a runnable starter. It uses a temporary majority-class placeholder so
the scaffold works before the final model is implemented. Replace only the
TODO section with the real model code.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed/final_gold_labels.csv"
RANDOM_STATE = 42
MODEL_NAME = "Mayowa Model Placeholder"

LABEL_MAP = {
    "REQUEST": 0,
    "INFORM_CONSTRAINT": 1,
    "CONFIRM_ACCEPT": 2,
    "CORRECT_CLARIFY": 3,
    "SOCIAL": 4,
}


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["label_num"] = df["label"].map(LABEL_MAP)
    if df["label_num"].isna().any():
        raise ValueError("Found unknown label values in final_gold_labels.csv")
    return df


def build_dialogue_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dialogue_ids = df["dialogue_id"].unique()
    train_ids, temp_ids = train_test_split(
        dialogue_ids,
        train_size=0.8,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    val_ids, test_ids = train_test_split(
        temp_ids,
        train_size=0.5,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    train_df = df[df["dialogue_id"].isin(train_ids)].copy()
    val_df = df[df["dialogue_id"].isin(val_ids)].copy()
    test_df = df[df["dialogue_id"].isin(test_ids)].copy()

    for part in (train_df, val_df, test_df):
        part["text"] = (
            "system: "
            + part["system_context"].fillna("").astype(str)
            + " user: "
            + part["user_utterance"].fillna("").astype(str)
        )

    return train_df, val_df, test_df


def run_model(train_df: pd.DataFrame, val_df: pd.DataFrame) -> list[int]:
    """
    TODO(Mayowa):
    Replace this placeholder with your final model.
    Suggested direction: DistilBERT/BERT or the strongest variant you can run.
    Keep the shared split logic and reported metrics unchanged.
    """
    majority_label_num = int(train_df["label_num"].mode().iloc[0])
    return [majority_label_num] * len(val_df)


def main() -> None:
    df = load_dataset()
    train_df, val_df, test_df = build_dialogue_splits(df)
    val_pred = run_model(train_df, val_df)

    val_accuracy = accuracy_score(val_df["label_num"], val_pred)
    val_macro_f1 = f1_score(
        val_df["label_num"],
        val_pred,
        average="macro",
        zero_division=0,
    )

    print(MODEL_NAME)
    print(f"Train rows: {len(train_df)}")
    print(f"Val rows: {len(val_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Validation macro F1: {val_macro_f1:.4f}")


if __name__ == "__main__":
    main()

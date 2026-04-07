"""
Owner: Divij
Suggested model family: tuned classical model or shallow neural variant with a
clear improvement path over the current TF-IDF + logistic-regression baseline.

Do not change these shared reporting rules:
- data source: data/processed/final_gold_labels.csv
- split policy: dialogue-level 80/10/10 with random_state=42
- reported metrics: validation accuracy and macro F1

This is a runnable starter. It uses a temporary majority-class placeholder so
the scaffold works before the final model is implemented. Replace only the
TODO section with the real model code.
"""

from pathlib import Path
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC

RANDOM_STATE = 42
MODEL_NAME = "Divij Tuned TF-IDF + LinearSVC"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "codabench" / "bundle" / "starting_kit" / "data"
DATA_PATH = "codabench/bundle/starting_kit/data/"

LABEL_MAP = {
    "REQUEST": 0,
    "INFORM_CONSTRAINT": 1,
    "CONFIRM_ACCEPT": 2,
    "CORRECT_CLARIFY": 3,
    "SOCIAL": 4,
}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def load_splits():
    train_df = pd.read_csv(DATA_PATH + "train.csv")
    val_df = pd.read_csv(DATA_PATH + "val.csv")
    test_df = pd.read_csv(DATA_PATH + "test.csv")

    for part in (train_df, val_df, test_df):
        part["text"] = (
            "system: "
            + part["system_context"].fillna("").astype(str)
            + " user: "
            + part["user_utterance"].fillna("").astype(str)
        )

    train_df["label_num"] = train_df["label"].map(LABEL_MAP)
    val_df["label_num"] = val_df["label"].map(LABEL_MAP)

    return train_df, val_df, test_df

def run_model(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.98,
        max_features=40000,
        stop_words="english",
    )

    char_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=30000,
    )

    X_train_word = word_vectorizer.fit_transform(train_df["text"])
    X_val_word = word_vectorizer.transform(val_df["text"])
    X_test_word = word_vectorizer.transform(test_df["text"])

    X_train_char = char_vectorizer.fit_transform(train_df["text"])
    X_val_char = char_vectorizer.transform(val_df["text"])
    X_test_char = char_vectorizer.transform(test_df["text"])

    X_train = hstack([X_train_word, X_train_char], format="csr")
    X_val = hstack([X_val_word, X_val_char], format="csr")
    X_test = hstack([X_test_word, X_test_char], format="csr")

    normalizer = Normalizer(copy=False)
    X_train = normalizer.fit_transform(X_train)
    X_val = normalizer.transform(X_val)
    X_test = normalizer.transform(X_test)

    clf = LinearSVC(C=1.5, class_weight="balanced", random_state=RANDOM_STATE)
    clf.fit(X_train, train_df["label_num"])

    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)

    return val_pred, test_pred


def main():

    print("Loading from:", DATA_PATH)

    train_df, val_df, test_df = load_splits()

    print(f"Train rows: {len(train_df)}")
    print(f"Val rows: {len(val_df)}")
    print(f"Test rows: {len(test_df)}")

    val_pred, test_pred = run_model(train_df, val_df, test_df)

    val_accuracy = accuracy_score(val_df["label_num"], val_pred)
    val_macro_f1 = f1_score(val_df["label_num"], val_pred, average="macro", zero_division=0)

    print(MODEL_NAME)
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Validation macro F1: {val_macro_f1:.4f}")


if __name__ == "__main__":
    main()

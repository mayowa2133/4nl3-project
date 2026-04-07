"""
Owner: Divij
Suggested model family: tuned classical model or shallow neural variant with a
clear improvement path over the current TF-IDF + logistic-regression baseline.
Do not change these shared reporting rules:
- data source: data/processed/final_gold_labels.csv
- split policy: dialogue-level 80/10/10 with random_state=42
- reported metrics: validation accuracy and macro F1
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import Normalizer
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

RANDOM_STATE = 42
MODEL_NAME   = "Divij Tuned TF-IDF + Neural MLP"
DATA_PATH    = "codabench/bundle/starting_kit/data/"

LABEL_MAP = {
    "REQUEST": 0, "INFORM_CONSTRAINT": 1, "CONFIRM_ACCEPT": 2,
    "CORRECT_CLARIFY": 3, "SOCIAL": 4,
}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# Dataset wrapper

class DialogueDataset(Dataset):
    """Wraps a dense numpy matrix + label array for PyTorch."""
    def __init__(self, X: np.ndarray, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

# Model definition

class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)

# Data loading

def load_splits():
    train_df = pd.read_csv(DATA_PATH + "train.csv")
    val_df   = pd.read_csv(DATA_PATH + "val.csv")
    test_df  = pd.read_csv(DATA_PATH + "test.csv")

    for part in (train_df, val_df, test_df):
        part["text"] = (
            "system: " + part["system_context"].fillna("").astype(str)
            + " user: " + part["user_utterance"].fillna("").astype(str)
        )

    train_df["label_num"] = train_df["label"].map(LABEL_MAP)
    val_df["label_num"]   = val_df["label"].map(LABEL_MAP)
    return train_df, val_df, test_df

# Neural run_model

def run_model(train_df, val_df, test_df):
    # --- TF-IDF features (same vectorisers as before) ---
    word_vec = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", sublinear_tf=True,
        ngram_range=(1, 3), min_df=2, max_df=0.98,
        max_features=40_000, stop_words="english",
    )
    char_vec = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", sublinear_tf=True,
        analyzer="char_wb", ngram_range=(3, 5),
        min_df=2, max_features=30_000,
    )

    X_tr_w = word_vec.fit_transform(train_df["text"])
    X_va_w = word_vec.transform(val_df["text"])
    X_te_w = word_vec.transform(test_df["text"])

    X_tr_c = char_vec.fit_transform(train_df["text"])
    X_va_c = char_vec.transform(val_df["text"])
    X_te_c = char_vec.transform(test_df["text"])

    normalizer = Normalizer(copy=False)
    X_train = normalizer.fit_transform(hstack([X_tr_w, X_tr_c], format="csr"))
    X_val   = normalizer.transform(hstack([X_va_w, X_va_c], format="csr"))
    X_test  = normalizer.transform(hstack([X_te_w, X_te_c], format="csr"))

    # Convert sparse → dense numpy (MLP needs dense input)
    X_train = X_train.toarray()
    X_val   = X_val.toarray()
    X_test  = X_test.toarray()

    # Class weights for imbalanced labels
    y_train = train_df["label_num"].values
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float32)

    # DataLoaders
    train_loader = DataLoader(
        DialogueDataset(X_train, y_train), batch_size=256, shuffle=True
    )
    val_loader = DataLoader(
        DialogueDataset(X_val, val_df["label_num"].values), batch_size=512
    )
    test_loader = DataLoader(DialogueDataset(X_test), batch_size=512)

    # Model, loss, optimiser
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MLPClassifier(input_dim=X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # Training loop with early stopping on val macro-F1
    best_f1, best_state, patience_left = 0.0, None, 8
    for epoch in range(50):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        # Validation
        model.eval()
        val_preds = []
        with torch.no_grad():
            for xb, _ in val_loader:
                val_preds.append(model(xb.to(device)).argmax(1).cpu())
        val_pred = torch.cat(val_preds).numpy()
        val_f1   = f1_score(val_df["label_num"], val_pred,
                            average="macro", zero_division=0)

        scheduler.step(val_f1)
        if val_f1 > best_f1:
            best_f1, best_state, patience_left = val_f1, model.state_dict(), 8
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"  Early stop at epoch {epoch+1}")
                break

    # Restore best checkpoint
    model.load_state_dict(best_state)
    model.eval()

    # Final val preds (for metric reporting)
    val_preds = []
    with torch.no_grad():
        for xb, _ in val_loader:
            val_preds.append(model(xb.to(device)).argmax(1).cpu())
    val_pred = torch.cat(val_preds).numpy()

    # Test preds
    test_preds = []
    with torch.no_grad():
        for xb in test_loader:
            test_preds.append(model(xb.to(device)).argmax(1).cpu())
    test_pred = torch.cat(test_preds).numpy()

    return val_pred, test_pred

def main():
    torch.manual_seed(RANDOM_STATE)
    print("Loading from:", DATA_PATH)
    train_df, val_df, test_df = load_splits()
    print(f"Train rows: {len(train_df)}")
    print(f"Val rows:   {len(val_df)}")
    print(f"Test rows:  {len(test_df)}")

    val_pred, test_pred = run_model(train_df, val_df, test_df)

    val_accuracy = accuracy_score(val_df["label_num"], val_pred)
    val_macro_f1 = f1_score(val_df["label_num"], val_pred,
                            average="macro", zero_division=0)
    print(MODEL_NAME)
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Validation macro F1: {val_macro_f1:.4f}")
    # Save test predictions
    import os

    output_path = "models/results/mlp_predictions.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    test_pred_labels = [INV_LABEL_MAP[n] for n in test_pred]
    with open(output_path, "w") as f:
        f.write("\n".join(test_pred_labels))

    print(f"Test predictions written to {output_path}")

if __name__ == "__main__":
    main()


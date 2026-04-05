import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DATA_PATH = "codabench/bundle/starting_kit/data/"
RANDOM_STATE = 42
MODEL_NAME = "LinearSVC"
LABEL_MAP = {
    "REQUEST": 0,
    "INFORM_CONSTRAINT": 1,
    "CONFIRM_ACCEPT": 2,
    "CORRECT_CLARIFY": 3,
    "SOCIAL": 4,
}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# Load splits
train_df = pd.read_csv(DATA_PATH + "train.csv")
val_df   = pd.read_csv(DATA_PATH + "val.csv")
test_df  = pd.read_csv(DATA_PATH + "test.csv")  # no labels

# Build text feature
def make_text(df):
    return (
        "system: " + df["system_context"].fillna("").astype(str)
        + " user: " + df["user_utterance"].fillna("").astype(str)
    )

train_df["text"] = make_text(train_df)
val_df["text"]   = make_text(val_df)
test_df["text"]  = make_text(test_df)

train_df["label_num"] = train_df["label"].map(LABEL_MAP)
val_df["label_num"]   = val_df["label"].map(LABEL_MAP)

# Train
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf",   LinearSVC(random_state=RANDOM_STATE)),
])
pipeline.fit(train_df["text"], train_df["label_num"])

# Validate
val_pred = pipeline.predict(val_df["text"])
print(f"Model             : {MODEL_NAME}")
print(f"Validation accuracy : {accuracy_score(val_df['label_num'], val_pred):.4f}")
print(f"Validation macro F1 : {f1_score(val_df['label_num'], val_pred, average='macro', zero_division=0):.4f}")

# Predict on test and write results
test_pred_nums   = pipeline.predict(test_df["text"])
test_pred_labels = [INV_LABEL_MAP[n] for n in test_pred_nums]

with open("models/results/linear_svc_predictions.csv", "w") as f:
    f.write("\n".join(test_pred_labels))
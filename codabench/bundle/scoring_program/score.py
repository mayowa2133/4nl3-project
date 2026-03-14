import os
import sys
import json
from sklearn.metrics import f1_score, accuracy_score

VALID_LABELS = {"REQUEST", "INFORM_CONSTRAINT", "CONFIRM_ACCEPT", "CORRECT_CLARIFY", "SOCIAL"}

def load_labels(path):
    with open(path, "r") as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels

def main():
    input_dir  = sys.argv[1]
    output_dir = sys.argv[2]

    reference_path  = os.path.join(input_dir, "ref", "test_labels.txt")
    prediction_path = os.path.join(input_dir, "res", "predictions.txt")

    if not os.path.exists(prediction_path):
        raise FileNotFoundError(
            "predictions.txt not found in submission. "
            "Make sure your submission file is named predictions.txt."
        )

    gold = load_labels(reference_path)
    pred = load_labels(prediction_path)

    if len(pred) != len(gold):
        raise ValueError(
            f"Submission has {len(pred)} lines but expected {len(gold)}. "
            "Make sure your file has exactly one prediction per line with no header."
        )

    invalid = [p for p in pred if p not in VALID_LABELS]
    if invalid:
        raise ValueError(
            f"Invalid labels found: {set(invalid)}. "
            f"Valid labels are: {VALID_LABELS}"
        )

    macro_f1 = f1_score(gold, pred, average="macro", labels=sorted(VALID_LABELS))
    accuracy = accuracy_score(gold, pred)

    scores = {
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(accuracy, 4),
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "scores.json"), "w") as f:
        json.dump(scores, f, indent=2)

    print(f"Macro F1 : {scores['macro_f1']}")
    print(f"Accuracy : {scores['accuracy']}")

if __name__ == "__main__":
    main()
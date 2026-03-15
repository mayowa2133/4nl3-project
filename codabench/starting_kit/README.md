# Turn-Level User Intent Classification
## COMPSCI 4NL3 — Winter 2025

## Contents

| File | Description |
|---|---|
| `train.csv` | 841 labeled training instances |
| `val.csv` | 106 labeled validation instances |
| `test_utterances.csv` | 106 test instances (no labels) |
| `sample_predictions.txt` | Example submission file (106 lines) |
| `README.md` | This file |

---

## Task

Given a user utterance and the immediately preceding system turn, predict the
user's intent. This is a five-way classification task.

Valid labels:

- `REQUEST`
- `INFORM_CONSTRAINT`
- `CONFIRM_ACCEPT`
- `CORRECT_CLARIFY`
- `SOCIAL`

---

## Data Format

Each CSV file contains the following columns:

| Column | Description |
|---|---|
| `dialogue_id` | Source dialogue identifier |
| `turn_id` | Turn index within the dialogue |
| `system_context` | The preceding system utterance |
| `user_utterance` | The user turn to classify |
| `label` | Gold intent label (train/val only) |

---

## How to Submit

1. Train your model on `train.csv`
2. Tune your model on `val.csv`
3. Run your model on `test_utterances.csv`
4. Save predictions as `predictions.txt`
   - One label per line
   - No header row
   - Must be exactly 106 lines
   - See `sample_predictions.txt` for the correct format
5. Zip `predictions.txt` into a zip file
6. Upload the zip to the competition page

Example `predictions.txt` format:

```
REQUEST
INFORM_CONSTRAINT
CONFIRM_ACCEPT
...
```

---

## Evaluation Metrics

| Metric | Role |
|---|---|
| Macro F1 | Primary — used for leaderboard ranking |
| Accuracy | Secondary — reported only |

# Evaluation

## Task Introduction

Given a user utterance and one turn of system context from a MultiWOZ 2.2
dialogue, predict the user's primary communicative intent. This is a five-way
classification task over the following label set:

| Label | Description | Example Utterance |
|---|---|---|
| `REQUEST` | User asks for information or options | "Can you give me their phone number?" |
| `INFORM_CONSTRAINT` | User provides a constraint or preference | "I would like something in the cheap price range." |
| `CONFIRM_ACCEPT` | User confirms or accepts a suggestion | "Yes, that sounds great." |
| `CORRECT_CLARIFY` | User corrects or clarifies a prior turn | "Actually, I wanted a moderately priced place." |
| `SOCIAL` | Greeting, thanks, or social exchange | "Thank you, goodbye!" |

## Dataset Format

The dataset is provided as CSV files with the following columns:

| Column | Description |
|---|---|
| `dialogue_id` | Unique identifier for the source dialogue |
| `turn_id` | Turn index within the dialogue |
| `system_context` | The preceding system utterance |
| `user_utterance` | The user turn to classify |
| `label` | Gold-standard intent label (train/val only) |

Example row from `train.csv`:

```
dialogue_id,turn_id,system_context,user_utterance,label
MUL0001.json,2,"I can help you find a restaurant. What area are you looking in?","I am looking for something in the centre.",INFORM_CONSTRAINT
```

## Starting Kit

The starting kit is available for download under the Datasets tab. It contains:

- `train.csv` — 841 labeled training instances
- `val.csv` — 106 labeled validation instances
- `test_utterances.csv` - test utterances without labels to test the model
- `sample_predictions.txt` — example submission file
- `README.txt` — submission instructions

Train on `train.csv`, tune on `val.csv`, and generate predictions on `test_utterances.csv` to submit.

## Submission Format

Submit a plain text file named `predictions.txt` with one predicted label per
line, in the same order as the test instances. No header row. Zip the file before
uploading.

```
REQUEST
INFORM_CONSTRAINT
CONFIRM_ACCEPT
CORRECT_CLARIFY
SOCIAL
REQUEST
...
```

Valid labels: `REQUEST`, `INFORM_CONSTRAINT`, `CONFIRM_ACCEPT`,
`CORRECT_CLARIFY`, `SOCIAL`. Submissions with invalid labels or the wrong number
of rows will be rejected.

## Evaluation Metrics

| Metric | Role | Description |
|---|---|---|
| Macro F1 | Primary (used for ranking) | Averages F1 per class without weighting by support. Penalizes poor performance on minority classes. |
| Accuracy | Secondary (reported only) | Proportion of correctly predicted labels. |
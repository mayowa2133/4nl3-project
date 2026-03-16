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
| `instance_id` | Unique identifier for the instance |
| `dialogue_id` | Unique identifier for the source dialogue |
| `turn_id` | Turn index within the dialogue |
| `system_context` | The preceding system utterance |
| `user_utterance` | The user turn to classify |
| `label` | Gold-standard intent label (train/val only) |

Example row from `train.csv`:

```
instance_id,dialogue_id,turn_id,system_context,user_utterance,label
mw22_MUL0592.json_12,MUL0592.json,12,"Indeed I can book that for 4 people. Your booking was successful, the total fee is 40.4 GBP payable at the station. Reference number is : 2I1YOWD4","Great that's all that I need, thank you!",SOCIAL
```

## Starting Kit

The starting kit is available for download under the Datasets tab. It contains:

- `data/train.csv` — 840 labeled training instances
- `data/val.csv` — 105 labeled validation instances
- `data/test.csv` — 105 unlabeled test instances used for evaluation
- `baseline.py` — baseline models to test models against

Train on `data/train.csv`, tune on `data/val.csv`, and generate predictions on
`data/test.csv` to submit.

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
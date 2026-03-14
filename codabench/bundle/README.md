# Turn-Level User Intent Classification in Task-Oriented Dialogue

## Overview

Given a single user utterance and one turn of preceding system context, classify
the user's primary communicative intent into one of five categories. This task
tests a model's ability to understand dialogue pragmatics in real-world
task-oriented conversations.

The dataset is derived from MultiWOZ 2.2, a large-scale multi-domain
task-oriented dialogue corpus. Utterances were manually annotated by three
annotators using a structured label set and adjudication protocol, producing
gold-standard labels for 1,051 user turns.

**Input:** A user utterance (≤40 tokens) and the immediately preceding system
turn from a MultiWOZ 2.2 dialogue.

**Output:** One of five intent labels: `REQUEST`, `INFORM_CONSTRAINT`,
`CONFIRM_ACCEPT`, `CORRECT_CLARIFY`, or `SOCIAL`.

**Why it matters:** Accurate turn-level intent detection is a core component of
dialogue state tracking and response generation in task-oriented systems. This
benchmark provides a manually adjudicated gold standard for comparing intent
classifiers on realistic, ambiguous conversational data.

### Label Set

| Label | Description |
|---|---|
| `REQUEST` | User asks for information or options |
| `INFORM_CONSTRAINT` | User provides a constraint or preference |
| `CONFIRM_ACCEPT` | User confirms or accepts a system suggestion |
| `CORRECT_CLARIFY` | User corrects or clarifies a prior turn |
| `SOCIAL` | Greeting, thanks, or other social exchange |

---

## Data

### Source

User turns were sampled from MultiWOZ 2.2 dialogues and filtered to utterances
between 1 and 40 tokens. Each instance includes the user utterance and the
immediately preceding system turn as context. Instances were drawn from the
train split of MultiWOZ 2.2 to avoid overlap with standard evaluation sets.

### Ground Truth Method

Ground truth labels were determined through **adjudication**. Three annotators
each labeled 350 turns during an initial pass, with a 135-turn overlap subset
(45 per annotator pair) used to measure agreement. Disagreements on the overlap
set were resolved through manual adjudication: each contested instance was
reviewed by the full team and assigned a final label by consensus. The frozen
gold labels are stored in `final_gold_labels.csv`. Inter-annotator agreement was
computed using Krippendorff's Alpha (nominal) on the overlap subset.

### Label Distribution

*(Upload `final_label_distribution.png` here.)*

See `final_label_distribution.csv` for exact counts and percentages per class.
Note whether the distribution is balanced or imbalanced and describe any
dominant classes (e.g., "REQUEST accounts for X% of instances").

### Splits

The dataset of 1,051 labeled instances was split as follows:

| Split | Instances | Proportion |
|---|---|---|
| Train | 841 | ~80% |
| Validation | 106 | ~10% |
| Test | 106 | ~10% |

Splits are stratified by label to preserve class distribution. Participants may
download the train and validation sets. Test labels are withheld and used only
for scoring.

---

## Evaluation

### Metrics

Submissions are evaluated against the withheld test set labels. Two metrics are
reported:

- **Macro F1** *(primary)*: Averages F1 per class without weighting by support.
  Penalizes poor performance on minority classes. Used for leaderboard ranking.
- **Accuracy** *(secondary)*: Proportion of correctly predicted labels. Reported
  for reference alongside macro F1.

Macro F1 is the primary ranking metric because the label distribution may be
imbalanced. A model that ignores rare classes will be penalized even if its
overall accuracy is high.

### Submission Format

Upload a plain-text or CSV file with one predicted label per line, in the same
order as the test instances. Do not include a header row.
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

---

## Baselines

### Simple Baselines

**Majority class baseline (REQUEST)**
Always predicts the most frequent label in the training set, which is REQUEST.
Establishes a simple but strong no-learning baseline that exploits label
imbalance and defines the minimum performance a trained model must beat. Despite
a relatively high accuracy, macro F1 collapses because the model scores 0 on all
non-REQUEST classes.

| Metric | Score |
|---|---|
| Macro F1 | 0.1119 |
| Accuracy | 0.3883 |

**Random baseline (uniform)**
Predicts one of the five intent labels uniformly at random. Serves as a pure
chance baseline for a 5-way classification task.

| Metric | Score |
|---|---|
| Macro F1 | 0.2073 |
| Accuracy | 0.2233 |

### Trained Baseline

**Logistic regression + TF-IDF**
A logistic regression classifier trained on TF-IDF bag-of-words features
extracted from the user utterance (and optionally the system context turn).

This is the minimum performance target. Any submitted model should aim to exceed
this on macro F1.

| Metric | Score |
|---|---|
| Macro F1 | 0.4708 |
| Accuracy | 0.6699  |
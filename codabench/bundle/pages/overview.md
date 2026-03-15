# Turn-Level User Intent Classification in Task-Oriented Dialogue

Given a single user utterance and one turn of preceding system context, classify
the user's primary communicative intent into one of five categories. This task
tests a model's ability to understand dialogue pragmatics in real-world
task-oriented conversations.

The dataset is derived from MultiWOZ 2.2, a large-scale multi-domain task-oriented
dialogue corpus. Utterances were manually annotated by three annotators using a
structured label set and adjudication protocol, producing gold-standard labels for
1,051 user turns.

## Why It Matters

Accurate turn-level intent detection is a core component of dialogue state tracking
and response generation in task-oriented systems. This benchmark provides a manually
adjudicated gold standard for comparing intent classifiers on realistic, ambiguous
conversational data.

## Label Set

| Label | Description |
|---|---|
| `REQUEST` | User asks for information or options |
| `INFORM_CONSTRAINT` | User provides a constraint or preference |
| `CONFIRM_ACCEPT` | User confirms or accepts a system suggestion |
| `CORRECT_CLARIFY` | User corrects or clarifies a prior turn |
| `SOCIAL` | Greeting, thanks, or other social exchange |

## Baseline Performance

| Model | Macro F1 | Accuracy |
|---|---|---|
| Random (uniform) | 0.2073 | 0.2233 |
| Majority (REQUEST) | 0.1119 | 0.3883 |
| Logistic Regression + TF-IDF | 0.4708 | 0.6699 |
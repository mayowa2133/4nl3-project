# Turn-Level User Intent Classification in Task-Oriented Dialogue

Given a single user utterance and one turn of preceding system context, classify
the user's primary communicative intent into one of five categories. This task
tests a model's ability to understand dialogue pragmatics in real-world
task-oriented conversations.

The dataset is derived from MultiWOZ 2.2, a large-scale multi-domain task-oriented
dialogue corpus. Utterances were manually annotated by three annotators using a
structured label set and adjudication protocol, producing gold-standard labels for
1,050 user turns.

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

## Baselines

Three baselines are provided to establish a lower bound for performance on this task. All baselines are evaluated using macro F1 as the primary metric and accuracy as a secondary metric.

### Simple Baselines

**Random Uniform** assigns a label drawn uniformly at random from the five classes for each test instance. This ignores all input and serves as the weakest possible baseline. With five classes, it achieves roughly 0.20 macro F1 by chance.

**Majority Class** always predicts REQUEST, the most frequent label in the training set (40.76% of instances). Despite achieving relatively high accuracy (0.39) due to class imbalance, it scores only 0.11 macro F1 because it completely fails on the four minority classes. This highlights why macro F1 is used as the primary metric rather than accuracy. A model that ignores minority classes like CORRECT_CLARIFY is not useful, even if its accuracy looks reasonable.

The better of the two simple baselines by macro F1 is Random Uniform (0.2073), which is reported as the simple baseline lower bound.

### Trained Baseline

**Logistic Regression + TF-IDF** is the trained baseline. Each instance is represented by concatenating the system context and user utterance into a single text field, which is then vectorized using TF-IDF with unigrams, bigrams, and trigrams (top 5,000 features). A logistic regression classifier is trained on these features using the training split and evaluated on the validation split.

This baseline achieves 0.4708 macro F1 and 0.6699 accuracy, substantially outperforming both simple baselines. It confirms that surface-level lexical features carry meaningful information for this task, but also leaves significant room for improvement, particularly on minority classes like CORRECT_CLARIFY, where contextual understanding is needed beyond bag-of-words patterns.

Your submissions should aim to outperform the trained baseline. The starting kit includes all baseline code so you can reproduce these results before building your own model.

### Summary

| Model | Macro F1 | Accuracy |
|---|---|---|
| Random (uniform) | 0.2073 | 0.2233 |
| Majority (REQUEST) | 0.1119 | 0.3883 |
| Logistic Regression + TF-IDF | 0.4708 | 0.6699 |
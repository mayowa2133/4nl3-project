# Model name: Random (uniform)

1) Description: Predicts one of the 5 intent labels uniformly at random.

2) Validation metrics: 
- Accuracy = 0.2233
- Macro F1 = 0.2073.

3) Reason: Serves as a pure chance baseline for a 5-way classification task.

# Model name: Majority (REQUEST)

1) Description: Always predicts the most frequent label in the training set, which is REQUEST.

2) Validation metrics 
- Accuracy = 0.3883
- Macro F1 = 0.1119.

3) Reason: Establishes a simple but strong no-learning baseline that exploits label imbalance and defines the minimum performance a trained model must beat.


# Trained Model
## Baselines Results

**Simple baselines** (no learning required):
- Random: 22.33% accuracy, 20.73% macro F1
- Majority (always REQUEST): 38.83% accuracy, 11.19% macro F1 ✓ (floor)

**Trained baseline**: TF-IDF with 1-3 n-grams (5000 features) on combined system context + user utterance, followed by logistic regression. Achieves **66.99% accuracy** and **47.08% macro F1**, substantially outperforming the majority baseline and confirming the task is learnable.


| Baseline             | Description                                              | Val Accuracy | Val Macro F1 |
|----------------------|----------------------------------------------------------|--------------|--------------|
| random_uniform       | Predict uniform random label (1/5 chance per class)      | 0.2233       | 0.2073       |
| majority             | Always predict train majority: REQUEST                   | 0.3883       | 0.1119       |
| tfidf_lr             | TF-IDF (1-3grams, 5000 features) + Logistic Regression   | **0.6699**   | **0.4708**   |
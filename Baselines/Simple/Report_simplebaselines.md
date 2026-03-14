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
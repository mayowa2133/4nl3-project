print("Import libraries.")

import json
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

print("Running...")

print(os.listdir('/app/input/res/'))

# Directories
reference_dir = os.path.join('/app/input/', 'ref')
prediction_dir = os.path.join('/app/input/', 'res')
score_dir = '/app/output/'

# Load predictions and ground truth as strings
prediction = np.genfromtxt(os.path.join(prediction_dir, 'predictions.txt'), dtype=str)
truth = np.genfromtxt(os.path.join(reference_dir, 'test_labels.txt'), dtype=str)

# Safety check
if len(prediction) != len(truth):
    raise ValueError("Prediction length does not match ground truth length!")

# Compute metrics
accuracy = accuracy_score(truth, prediction)
f1 = f1_score(truth, prediction, average='macro')  # Macro F1 for multiclass

# Save scores
scores = {
    'accuracy': float(accuracy),
    'f1_macro': float(f1)
}

with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file:
    json.dump(scores, score_file)

print("Scores:", scores)
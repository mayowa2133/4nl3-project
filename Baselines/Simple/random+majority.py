import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Load your data
df = pd.read_csv('data/processed/final_gold_labels.csv')

print('Dataset loaded:', df.shape)
print('Label distribution:\n', df['label'].value_counts(normalize=True).round(3))

# Numeric labels
label_map = {'REQUEST': 0, 'INFORM_CONSTRAINT': 1, 'CONFIRM_ACCEPT': 2, 'CORRECT_CLARIFY': 3, 'SOCIAL': 4}
df['label_num'] = df['label'].map(label_map)

# Split by dialogue_id to avoid leakage (80/10/10)
dialogue_ids = df['dialogue_id'].unique()
train_ids, temp_ids = train_test_split(dialogue_ids, train_size=0.8, random_state=42, shuffle=True)
val_ids, test_ids = train_test_split(temp_ids, train_size=0.5, random_state=42, shuffle=True)
train_df = df[df['dialogue_id'].isin(train_ids)]
val_df = df[df['dialogue_id'].isin(val_ids)]
print(f'Train: {len(train_df)}, Val: {len(val_df)}')


# 1. Random baseline (uniform random over 5 classes)
np.random.seed(42)
val_pred_random = np.random.randint(0, 5, size=len(val_df))
acc_random = accuracy_score(val_df['label_num'], val_pred_random)
f1_random = f1_score(val_df['label_num'], val_pred_random, average='macro', zero_division=0)


# 2. Majority baseline (from train)
majority_num = Counter(train_df['label_num']).most_common(1)[0][0]
majority_label = list(label_map.keys())[majority_num]
val_pred_maj = np.full(len(val_df), majority_num)
acc_maj = accuracy_score(val_df['label_num'], val_pred_maj)
f1_maj = f1_score(val_df['label_num'], val_pred_maj, average='macro', zero_division=0)

# Results table
results = pd.DataFrame({
    'baseline': ['random_uniform', 'majority'],
    'description': ['Predict uniform random label (1/5 chance per class)', 
                    f'Always predict train majority: {majority_label}'],
    'val_accuracy': [acc_random, acc_maj],
    'val_f1_macro': [f1_random, f1_maj]
})
results.to_csv('Baselines/Simple/simple_baselines_results.csv', index=False)
print('\nReport Generated: Baselines/Simple/simple_baselines_results.csv')
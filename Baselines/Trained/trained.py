import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

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

# Add this to your existing code after loading df and splits
train_df['text'] = 'system: ' + train_df['system_context'].astype(str) + ' user: ' + train_df['user_utterance'].astype(str)
val_df['text'] = 'system: ' + val_df['system_context'].astype(str) + ' user: ' + val_df['user_utterance'].astype(str)

print('Sample texts:')
print(train_df[['text', 'label']].head())

# Setup TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000, 
    ngram_range=(1, 3),  # words + phrases
    stop_words='english'
)
X_train_tfidf = vectorizer.fit_transform(train_df['text'])
X_val_tfidf = vectorizer.transform(val_df['text'])
print('TF-IDF shape:', X_train_tfidf.shape)


# Train a simple logistic regression classifier
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_tfidf, train_df['label_num'])
val_pred_lr = lr.predict(X_val_tfidf)

# Evaluate
acc_lr = accuracy_score(val_df['label_num'], val_pred_lr)
f1_lr = f1_score(val_df['label_num'], val_pred_lr, average='macro')
print(f'Trained baseline: Acc={acc_lr:.4f}, F1={f1_lr:.4f}')

# Export
trained_result = pd.DataFrame({
    'baseline': ['tfidf_lr'],
    'description': ['TF-IDF (1-3grams, 5000 features) on system+user text + Logistic Regression'],
    'val_accuracy': [acc_lr],
    'val_f1_macro': [f1_lr]
})

# Save trained baseline CSV
trained_result.to_csv('Baselines/Trained/trained_baseline_results.csv', index=False)
print("Report Generated: Baselines/Trained/trained_baseline_results.csv")
print(trained_result.round(4))
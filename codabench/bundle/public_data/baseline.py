import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ── Data ──────────────────────────────────────────────────────────────────────
train_df = pd.read_csv('data/train.csv')
val_df   = pd.read_csv('data/val.csv')
test_df  = pd.read_csv('data/test.csv')  # no labels

print('Train:', train_df.shape, '| Val:', val_df.shape, '| Test:', test_df.shape)
print('Label distribution:\n', train_df['label'].value_counts(normalize=True).round(3))

label_map = {
    'REQUEST': 0,
    'INFORM_CONSTRAINT': 1,
    'CONFIRM_ACCEPT': 2,
    'CORRECT_CLARIFY': 3,
    'SOCIAL': 4
}
reverse_label_map = {v: k for k, v in label_map.items()}

train_df['label_num'] = train_df['label'].map(label_map)
val_df['label_num']   = val_df['label'].map(label_map)

# ── Feature construction ───────────────────────────────────────────────────────
# Combines system context and user utterance into a single text field.
# Feel free to modify or add features here.
def build_text(data):
    return ('system: ' + data['system_context'].astype(str)
            + ' user: '  + data['user_utterance'].astype(str))

train_df['text'] = build_text(train_df)
val_df['text']   = build_text(val_df)
test_df['text']  = build_text(test_df)

# ── Simple baselines ───────────────────────────────────────────────────────────
# Random uniform
np.random.seed(42)
val_pred_random = np.random.randint(0, 5, size=len(val_df))
acc_random = accuracy_score(val_df['label_num'], val_pred_random)
f1_random  = f1_score(val_df['label_num'], val_pred_random, average='macro', zero_division=0)
print(f'\n[Random Uniform]  Acc={acc_random:.4f}  Macro-F1={f1_random:.4f}')

# Majority class
majority_num   = Counter(train_df['label_num']).most_common(1)[0][0]
majority_label = reverse_label_map[majority_num]
val_pred_maj   = np.full(len(val_df), majority_num)
acc_maj = accuracy_score(val_df['label_num'], val_pred_maj)
f1_maj  = f1_score(val_df['label_num'], val_pred_maj, average='macro', zero_division=0)
print(f'[Majority ({majority_label})]  Acc={acc_maj:.4f}  Macro-F1={f1_maj:.4f}')

# ── Trained baseline: TF-IDF + Logistic Regression ────────────────────────────
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), stop_words='english')
X_train = vectorizer.fit_transform(train_df['text'])
X_val   = vectorizer.transform(val_df['text'])
X_test  = vectorizer.transform(test_df['text'])

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, train_df['label_num'])

val_pred_lr = lr.predict(X_val)
acc_lr = accuracy_score(val_df['label_num'], val_pred_lr)
f1_lr  = f1_score(val_df['label_num'], val_pred_lr, average='macro')
print(f'[TF-IDF + LR]     Acc={acc_lr:.4f}  Macro-F1={f1_lr:.4f}')

# ── Results summary ────────────────────────────────────────────────────────────
results = pd.DataFrame({
    'baseline':     ['random_uniform', f'majority_{majority_label}', 'tfidf_lr'],
    'description':  [
        'Uniform random prediction across 5 classes',
        f'Always predicts most common training label: {majority_label}',
        'TF-IDF (1-3grams, top 5000) + Logistic Regression'
    ],
    'val_accuracy': [acc_random, acc_maj, acc_lr],
    'val_f1_macro': [f1_random,  f1_maj,  f1_lr]
})
print('\nBaseline Results:')
print(results.round(4).to_string(index=False))

# ── Submission ─────────────────────────────────────────────────────────────────
# Replace lr.predict(X_test) with your own model's predictions.
# Submit predictions.csv zipped as submission.zip on the competition page.
test_pred_lr = lr.predict(X_test)

submission = pd.DataFrame({
    'id':    test_df['id'],
    'label': [reverse_label_map[p] for p in test_pred_lr]  # use string labels, not integers
})
submission.to_csv('predictions.csv', index=False)
print('\nSubmission saved: predictions.csv')
print(submission['label'].value_counts())

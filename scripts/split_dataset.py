import pandas as pd
from sklearn.model_selection import train_test_split

"""
Load the dataset split into the codabench folder.
"""

# Load the data
df = pd.read_csv('data/processed/final_gold_labels.csv')

# First split: 80% train, 20% temp (which will become val + test)
train, temp = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Second split: split the 20% temp evenly into 10% val and 10% test
val, test = train_test_split(temp, test_size=0.5, random_state=42)

# Write to new CSV files
train.to_csv('codabench/bundle/public_data/train.csv', index=False)
val.to_csv('codabench/bundle/public_data/val.csv', index=False)
test.to_csv('codabench/test.csv', index=False)

print(f"Train: {len(train)} rows")
print(f"Val:   {len(val)} rows")
print(f"Test:  {len(test)} rows")

test = pd.read_csv("codabench/test.csv")
test.drop(columns=["label"]).to_csv("codabench/starting_kit/test_utterances.csv", index=False)

print(f"Done. test_utterances.csv created with {len(test)} rows and columns: {list(test.drop(columns=['label']).columns)}")
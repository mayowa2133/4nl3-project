import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv('data/processed/final_gold_labels.csv')

# First split: 80% train, 20% temp (which will become val + test)
train, temp = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Second split: split the 20% temp evenly into 10% val and 10% test
val, test = train_test_split(temp, test_size=0.5, random_state=42)

# Write to new CSV files
train.to_csv('data/split/training.csv', index=False)
val.to_csv('data/split/validation.csv', index=False)
test.to_csv('data/split/testing.csv', index=False)

print(f"Train: {len(train)} rows")
print(f"Val:   {len(val)} rows")
print(f"Test:  {len(test)} rows")
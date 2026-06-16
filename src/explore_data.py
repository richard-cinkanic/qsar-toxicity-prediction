import pandas as pd

# Load datasets
train_df = pd.read_csv("data_train.csv")
test_df = pd.read_csv("smiles_test.csv")
sample_sub = pd.read_csv("sample_submission.csv")

# Basic information
print("TRAIN SHAPE:")
print(train_df.shape)

print("\nTEST SHAPE:")
print(test_df.shape)

print("\nSAMPLE SUBMISSION SHAPE:")
print(sample_sub.shape)

# Show first rows
print("\nTRAIN HEAD:")
print(train_df.head())

print("\nTEST HEAD:")
print(test_df.head())

print("\nSAMPLE SUBMISSION HEAD:")
print(sample_sub.head())

# Column names
print("\nTRAIN COLUMNS:")
print(train_df.columns)

# Label distribution
print("\nLABEL COUNTS:")
for col in train_df.columns[2:]:
    print(f"\n{col}")
    print(train_df[col].value_counts(dropna=False))

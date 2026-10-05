import pandas as pd

# Load the training data
df = pd.read_csv("train-test.csv")

print("Dataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)

print("\nDate range:")
print(df["date"].min())
print(df["date"].max())

print("\nEquipment types:")
print(df["equipment"].value_counts())

print("\nTarget statistics:")
print(df["posted_rate"].describe())
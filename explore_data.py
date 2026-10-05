import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("train-test.csv")

# Convert date
df["date"] = pd.to_datetime(df["date"])

print("\n===== BASIC INFORMATION =====")
print(df.info())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== NUMERICAL SUMMARY =====")
print(df.describe())

print("\n===== CATEGORICAL VALUES =====")
print("\nEquipment:")
print(df["equipment"].value_counts())

print("\nNumber of pickup locations:")
print(df["pickup"].nunique())

print("\nNumber of delivery locations:")
print(df["delivery"].nunique())

print("\n===== CORRELATION WITH POSTED RATE =====")

numeric_columns = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "posted_rate"
]

print(df[numeric_columns].corr()["posted_rate"].sort_values(ascending=False))

# Plot 1: Distance vs Rate

plt.figure(figsize=(8, 5))
plt.scatter(df["distance"], df["posted_rate"], alpha=0.2)
plt.xlabel("Distance")
plt.ylabel("Posted Rate")
plt.title("Distance vs Posted Rate")
plt.tight_layout()
plt.show()

# Plot 2: Rate by Equipment
df.boxplot(column="posted_rate", by="equipment", figsize=(8, 5))
plt.title("Posted Rate by Equipment")
plt.suptitle("")
plt.xlabel("Equipment")
plt.ylabel("Posted Rate")
plt.tight_layout()
plt.show()

# Plot 3: Rate over time
daily_rate = df.groupby("date")["posted_rate"].mean()

plt.figure(figsize=(10, 5))
plt.plot(daily_rate)
plt.xlabel("Date")
plt.ylabel("Average Posted Rate")
plt.title("Average Posted Rate Over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
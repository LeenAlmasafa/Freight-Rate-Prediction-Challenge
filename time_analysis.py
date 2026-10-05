import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("train-test.csv")
df["date"] = pd.to_datetime(df["date"])

# Average rate by month
monthly = df.groupby(df["date"].dt.to_period("M"))["posted_rate"].agg(
    ["mean", "std", "count"]
)

print("\n===== MONTHLY POSTED RATE =====")
print(monthly)

# Average market index by month
market = df.groupby(df["date"].dt.to_period("M"))["market_index"].mean()

print("\n===== MONTHLY MARKET INDEX =====")
print(market)

# Average quote signal by month
quote = df.groupby(df["date"].dt.to_period("M"))["quote_signal"].mean()

print("\n===== MONTHLY QUOTE SIGNAL =====")
print(quote)

# Plot monthly average rate
plt.figure(figsize=(10, 5))
plt.plot(monthly.index.astype(str), monthly["mean"], marker="o")
plt.xlabel("Month")
plt.ylabel("Average Posted Rate")
plt.title("Average Freight Rate by Month")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
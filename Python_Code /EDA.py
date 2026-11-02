import pandas as pd
import matplotlib.pyplot as plt

# Read CSV File
df = pd.read_csv("Microsoft.csv")

# Histogram
df["Close"].hist(bins=20)
plt.title("Histogram of Close Price")
plt.xlabel("Close Price")
plt.ylabel("Frequency")
plt.show()

# Box Plot
plt.boxplot(df["Close"])
plt.title("Box Plot of Close Price")
plt.show()

# Pie Chart
labels = ["Open", "High", "Low", "Close"]
values = [
    df["Open"].mean(),
    df["High"].mean(),
    df["Low"].mean(),
    df["Close"].mean()
]

plt.pie(values, labels=labels, autopct="%1.1f%%")
plt.title("Average Stock Price Distribution")
plt.show()

# Bar Chart
plt.bar(labels, values)
plt.title("Average Stock Price")
plt.xlabel("Stock Price Type")
plt.ylabel("Average Value")
plt.show()

# Correlation Matrix
print("\nCorrelation Matrix")
print(df.corr(numeric_only=True))

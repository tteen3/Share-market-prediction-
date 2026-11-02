import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import math

# Read CSV File
df = pd.read_csv("Microsoft.csv")

print("========== STOCK MARKET DATASET ==========")

# First 5 Records
print("\nFirst 5 Records")
print(df.head())

# Last 5 Records
print("\nLast 5 Records")
print(df.tail())

# Dataset Shape
print("\nDataset Shape")
print(df.shape)

# Column Names
print("\nColumn Names")
print(df.columns)

# Dataset Information
print("\nDataset Information")
print(df.info())

# Data Types
print("\nData Types")
print(df.dtypes)

# Missing Values
print("\nMissing Values")
print(df.isnull().sum())

# Duplicate Values
print("\nDuplicate Records")
print(df.duplicated().sum())

# Total Records
print("\nTotal Records :", len(df))

# Total Columns
print("Total Columns :", len(df.columns))

# Statistical Summary
print("\nStatistical Summary")
print(df.describe())

# Save Clean Dataset
df.to_csv("clean_stock_data.csv", index=False)

print("\nDataset Loaded Successfully")

#1

print("===== BEFORE CLEANING =====")
print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Records")
print(df.duplicated().sum())

# Remove Duplicate Records
df = df.drop_duplicates()

# Handle Missing Values
# Fill missing values using mean
numeric_columns = ["Close", "High", "Low", "Open", "Volume"]

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].mean())

# Remove Outliers (IQR Method)
for col in numeric_columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower) & (df[col] <= upper)]

print("\n===== AFTER CLEANING =====")

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Records")
print(df.duplicated().sum())

print("\nFinal Shape")
print(df.shape)

# Save Clean Dataset
#df.to_csv("clean_stock_data.csv", index=False)
#print("\nClean Dataset Saved Successfully")



# Remove Date Column
df = df.drop("Date", axis=1)

# Independent Variables
X = df[["Open", "High", "Low", "Volume"]]

# Dependent Variable
y = df["Close"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  : Build ML Model

model = LinearRegression()

model.fit(X_train, y_train)

print("Model Trained Successfully")

# Prediction
y_pred = model.predict(X_test)

print("\nActual Price:")
print(y_test.head())

print("\nPredicted Price:")
print(y_pred[:5])

#  : Model Evaluation

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)

print("\nR2 Score :", r2)
print("MSE :", mse)
print("RMSE :", rmse)

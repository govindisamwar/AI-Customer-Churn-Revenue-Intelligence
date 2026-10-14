import pandas as pd
from pathlib import Path


# -----------------------------
# Paths
# -----------------------------
INPUT_PATH = Path(
    "Data_Engineering/data/processed/customer_churn/"
    "telco_customer_churn_cleaned.csv"
)

OUTPUT_PATH = Path(
    "Data_Engineering/data/processed/customer_churn/"
    "telco_customer_churn_features.csv"
)


# -----------------------------
# Load cleaned dataset
# -----------------------------
df = pd.read_csv(INPUT_PATH)

print("Original shape:", df.shape)


# -----------------------------
# 1. Average monthly spend
# -----------------------------
df["AvgMonthlySpend"] = df["TotalCharges"] / df["tenure"]

# For customers with tenure = 0
df["AvgMonthlySpend"] = df["AvgMonthlySpend"].fillna(
    df["MonthlyCharges"]
)


# -----------------------------
# 2. Contract risk indicator
# -----------------------------
df["IsMonthToMonth"] = (
    df["Contract"] == "Month-to-month"
).astype(int)


# -----------------------------
# 3. Internet service indicator
# -----------------------------
df["IsFiberOptic"] = (
    df["InternetService"] == "Fiber optic"
).astype(int)


# -----------------------------
# 4. Customer service count
# -----------------------------
service_columns = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

df["ServiceCount"] = sum(
    (df[col] == "Yes").astype(int)
    for col in service_columns
)


# -----------------------------
# 5. Tenure groups
# -----------------------------
def tenure_group(tenure):
    if tenure <= 12:
        return "0-12"
    elif tenure <= 24:
        return "13-24"
    elif tenure <= 48:
        return "25-48"
    else:
        return "49-72"


df["TenureGroup"] = df["tenure"].apply(tenure_group)


# -----------------------------
# 6. Remove customer ID
# -----------------------------
df = df.drop(columns=["customerID"])


# -----------------------------
# Save feature-engineered dataset
# -----------------------------
df.to_csv(OUTPUT_PATH, index=False)


# -----------------------------
# Verification
# -----------------------------
print("\nFeature engineering completed successfully.")
print("Final shape:", df.shape)

print("\nNew features:")
print([
    "AvgMonthlySpend",
    "IsMonthToMonth",
    "IsFiberOptic",
    "ServiceCount",
    "TenureGroup"
])

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nOutput saved to:")
print(OUTPUT_PATH)

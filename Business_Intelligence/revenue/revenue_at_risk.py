import pandas as pd
import joblib

from pathlib import Path


# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = Path(
    "Data_Engineering/data/processed/customer_churn/"
    "telco_customer_churn_features.csv"
)

MODEL_PATH = Path(
    "ML_Modeling/models/churn_xgboost_model.joblib"
)

OUTPUT_DIR = Path(
    "Business_Intelligence/revenue"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    OUTPUT_DIR /
    "customer_revenue_risk.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(
    DATA_PATH
)

print("=" * 60)
print("CUSTOMER REVENUE AT RISK ANALYSIS")
print("=" * 60)

print("\nDataset shape:", df.shape)


# ============================================================
# 3. PREPARE FEATURES
# ============================================================

TARGET = "ChurnBinary"

X = df.drop(
    columns=[TARGET, "Churn"]
)


# ============================================================
# 4. LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained model...")

pipeline = joblib.load(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# 5. PREDICT CHURN PROBABILITY
# ============================================================

print("\nCalculating churn probability...")

churn_probability = pipeline.predict_proba(
    X
)[:, 1]


# ============================================================
# 6. CREATE BUSINESS DATASET
# ============================================================

result = df.copy()

result["ChurnProbability"] = (
    churn_probability
)

result["ExpectedMonthlyRevenueAtRisk"] = (
    result["MonthlyCharges"]
    * result["ChurnProbability"]
)


# ============================================================
# 7. RISK CATEGORY
# ============================================================

def risk_category(probability):

    if probability >= 0.70:
        return "High Risk"

    elif probability >= 0.40:
        return "Medium Risk"

    else:
        return "Low Risk"


result["RiskCategory"] = (
    result["ChurnProbability"]
    .apply(risk_category)
)


# ============================================================
# 8. EXPECTED ANNUAL REVENUE AT RISK
# ============================================================

result["ExpectedAnnualRevenueAtRisk"] = (
    result["ExpectedMonthlyRevenueAtRisk"]
    * 12
)


# ============================================================
# 9. SORT BY RISK
# ============================================================

result = result.sort_values(
    by="ExpectedMonthlyRevenueAtRisk",
    ascending=False
)


# ============================================================
# 10. SAVE RESULTS
# ============================================================

result.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nRevenue risk dataset saved:")
print(OUTPUT_PATH)


# ============================================================
# 11. BUSINESS SUMMARY
# ============================================================

total_customers = len(
    result
)

high_risk_customers = (
    result["RiskCategory"]
    .eq("High Risk")
    .sum()
)

medium_risk_customers = (
    result["RiskCategory"]
    .eq("Medium Risk")
    .sum()
)

low_risk_customers = (
    result["RiskCategory"]
    .eq("Low Risk")
    .sum()
)

total_monthly_revenue_at_risk = (
    result[
        "ExpectedMonthlyRevenueAtRisk"
    ].sum()
)

total_annual_revenue_at_risk = (
    result[
        "ExpectedAnnualRevenueAtRisk"
    ].sum()
)


# ============================================================
# 12. PRINT BUSINESS KPIs
# ============================================================

print("\n" + "=" * 60)
print("REVENUE RISK SUMMARY")
print("=" * 60)

print(
    f"\nTotal Customers        : {total_customers}"
)

print(
    f"High Risk Customers    : {high_risk_customers}"
)

print(
    f"Medium Risk Customers  : {medium_risk_customers}"
)

print(
    f"Low Risk Customers     : {low_risk_customers}"
)

print(
    f"\nExpected Monthly Revenue At Risk: "
    f"${total_monthly_revenue_at_risk:,.2f}"
)

print(
    f"Expected Annual Revenue At Risk : "
    f"${total_annual_revenue_at_risk:,.2f}"
)


# ============================================================
# 13. TOP 10 HIGH-VALUE AT-RISK CUSTOMERS
# ============================================================

print("\n" + "=" * 60)
print("TOP 10 CUSTOMERS BY REVENUE AT RISK")
print("=" * 60)

top_customers = result[
    [
        "MonthlyCharges",
        "ChurnProbability",
        "RiskCategory",
        "ExpectedMonthlyRevenueAtRisk",
        "ExpectedAnnualRevenueAtRisk"
    ]
].head(10)

print(
    top_customers.to_string(
        index=False
    )
)


# ============================================================
# 14. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("REVENUE AT RISK ANALYSIS COMPLETED")
print("=" * 60)
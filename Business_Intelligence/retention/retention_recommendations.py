import pandas as pd
from pathlib import Path


# ============================================================
# 1. PATHS
# ============================================================

INPUT_PATH = Path(
    "Business_Intelligence/revenue/"
    "customer_revenue_risk.csv"
)

OUTPUT_DIR = Path(
    "Business_Intelligence/retention"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    OUTPUT_DIR /
    "customer_retention_actions.csv"
)


# ============================================================
# 2. LOAD REVENUE RISK DATA
# ============================================================

df = pd.read_csv(INPUT_PATH)

print("=" * 60)
print("CUSTOMER RETENTION RECOMMENDATION ENGINE")
print("=" * 60)

print("\nDataset shape:", df.shape)


# ============================================================
# 3. RETENTION ACTION LOGIC
# ============================================================

def recommend_action(row):

    probability = row["ChurnProbability"]
    contract = row["Contract"]
    internet = row["InternetService"]
    support = row["TechSupport"]
    security = row["OnlineSecurity"]
    monthly_charge = row["MonthlyCharges"]
    tenure = row["tenure"]

    # Highest priority
    if probability >= 0.80:

        if contract == "Month-to-month":
            return "Offer contract upgrade incentive"

        elif support == "No":
            return "Offer technical support package"

        elif internet == "Fiber optic":
            return "Offer fiber service retention discount"

        elif monthly_charge >= 80:
            return "Offer personalized price incentive"

        else:
            return "Priority retention outreach"

    # Medium-high risk
    elif probability >= 0.60:

        if support == "No":
            return "Offer technical support"

        elif security == "No":
            return "Offer online security package"

        elif contract == "Month-to-month":
            return "Promote annual contract"

        elif tenure <= 12:
            return "Early-tenure onboarding campaign"

        else:
            return "Personalized retention communication"

    # Medium risk
    elif probability >= 0.40:

        if tenure <= 12:
            return "Improve onboarding engagement"

        elif contract == "Month-to-month":
            return "Contract upgrade campaign"

        else:
            return "Customer engagement campaign"

    # Low risk
    else:

        return "Maintain relationship and monitor"


# ============================================================
# 4. GENERATE RECOMMENDATIONS
# ============================================================

print("\nGenerating retention recommendations...")

df["RecommendedAction"] = (
    df.apply(
        recommend_action,
        axis=1
    )
)


# ============================================================
# 5. PRIORITY LEVEL
# ============================================================

def priority_level(probability):

    if probability >= 0.80:
        return "Critical"

    elif probability >= 0.60:
        return "High"

    elif probability >= 0.40:
        return "Medium"

    else:
        return "Low"


df["RetentionPriority"] = (
    df["ChurnProbability"]
    .apply(priority_level)
)


# ============================================================
# 6. ESTIMATED RETENTION VALUE
# ============================================================

df["PotentialAnnualRevenue"] = (
    df["MonthlyCharges"] * 12
)


# ============================================================
# 7. SORT BY PRIORITY AND REVENUE
# ============================================================

priority_order = {
    "Critical": 1,
    "High": 2,
    "Medium": 3,
    "Low": 4
}

df["PriorityOrder"] = (
    df["RetentionPriority"]
    .map(priority_order)
)

df = df.sort_values(
    by=[
        "PriorityOrder",
        "ExpectedAnnualRevenueAtRisk"
    ],
    ascending=[
        True,
        False
    ]
)

df = df.drop(
    columns=["PriorityOrder"]
)


# ============================================================
# 8. SAVE OUTPUT
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nRetention recommendation dataset saved:")

print(OUTPUT_PATH)


# ============================================================
# 9. RETENTION SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("RETENTION PRIORITY SUMMARY")
print("=" * 60)

priority_summary = (
    df["RetentionPriority"]
    .value_counts()
)

print(
    priority_summary
)


# ============================================================
# 10. ACTION SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("RECOMMENDED ACTION SUMMARY")
print("=" * 60)

action_summary = (
    df["RecommendedAction"]
    .value_counts()
)

print(
    action_summary
)


# ============================================================
# 11. TOP PRIORITY CUSTOMERS
# ============================================================

print("\n" + "=" * 60)
print("TOP 10 RETENTION PRIORITY CUSTOMERS")
print("=" * 60)

top_customers = df[
    [
        "MonthlyCharges",
        "ChurnProbability",
        "RiskCategory",
        "ExpectedMonthlyRevenueAtRisk",
        "ExpectedAnnualRevenueAtRisk",
        "RetentionPriority",
        "RecommendedAction"
    ]
].head(10)

print(
    top_customers.to_string(
        index=False
    )
)


# ============================================================
# 12. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("RETENTION RECOMMENDATION ENGINE COMPLETED")
print("=" * 60)
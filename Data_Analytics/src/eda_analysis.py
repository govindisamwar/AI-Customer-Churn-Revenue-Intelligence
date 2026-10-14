import pandas as pd

INPUT = "Data_Engineering/data/processed/telco_customer_churn_processed.csv"
OUTPUT = "Data_Analytics/reports/churn_eda_summary.csv"

df = pd.read_csv(INPUT)

summary = []

for column in ["Contract", "TenureGroup", "InternetService", "PaymentMethod", "SeniorCitizen"]:
    if column not in df.columns:
        if column == "TenureGroup":
            df[column] = pd.cut(
                df["tenure"],
                bins=[-1, 12, 24, 48, 72],
                labels=["0-12", "13-24", "25-48", "49-72"]
            )
        elif column == "SeniorCitizen":
            df["SeniorCitizenLabel"] = df["SeniorCitizen"].map({0: "Non-Senior", 1: "Senior"})
            column = "SeniorCitizenLabel"

    result = (
        df.groupby(column, observed=False)["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="ChurnRate")
    )

    result["Feature"] = column
    summary.append(result)

final = pd.concat(summary, ignore_index=True)
final.to_csv(OUTPUT, index=False)

print("EDA summary created successfully")
print("Rows:", len(final))

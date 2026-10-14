from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "Data_Engineering"
    / "data"
    / "processed"
    / "customer_churn"
    / "telco_customer_churn_cleaned.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "Data_Engineering"
    / "reports"
    / "eda"
)


def load_data():
    """Load the cleaned customer churn dataset."""
    return pd.read_csv(INPUT_FILE)


def create_output_directory():
    """Create the EDA output directory."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def basic_summary(df):
    """Print basic dataset information."""
    print("\n========== DATASET SUMMARY ==========")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n========== DATA TYPES ==========")
    print(df.dtypes)

    print("\n========== CHURN DISTRIBUTION ==========")
    print(df["Churn"].value_counts())

    churn_rate = df["ChurnBinary"].mean() * 100

    print(f"\nOverall churn rate: {churn_rate:.2f}%")


def analyze_contract_churn(df):
    """Analyze churn by contract type."""
    result = (
        df.groupby("Contract")["ChurnBinary"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    print("\n========== CHURN BY CONTRACT ==========")
    print(result)

    ax = result.plot(
        kind="bar",
        figsize=(8, 5)
    )

    ax.set_title("Churn Rate by Contract Type")
    ax.set_xlabel("Contract Type")
    ax.set_ylabel("Churn Rate (%)")

    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "churn_by_contract.png",
        dpi=150
    )

    plt.close()


def analyze_internet_service_churn(df):
    """Analyze churn by internet service."""
    result = (
        df.groupby("InternetService")["ChurnBinary"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    print("\n========== CHURN BY INTERNET SERVICE ==========")
    print(result)

    ax = result.plot(
        kind="bar",
        figsize=(8, 5)
    )

    ax.set_title("Churn Rate by Internet Service")
    ax.set_xlabel("Internet Service")
    ax.set_ylabel("Churn Rate (%)")

    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "churn_by_internet_service.png",
        dpi=150
    )

    plt.close()


def analyze_tenure(df):
    """Analyze churn across tenure groups."""
    df = df.copy()

    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=[
            "0-12 months",
            "13-24 months",
            "25-48 months",
            "49-72 months"
        ]
    )

    result = (
        df.groupby(
            "TenureGroup",
            observed=False
        )["ChurnBinary"]
        .mean()
        .mul(100)
    )

    print("\n========== CHURN BY TENURE ==========")
    print(result)

    ax = result.plot(
        kind="bar",
        figsize=(8, 5)
    )

    ax.set_title("Churn Rate by Customer Tenure")
    ax.set_xlabel("Tenure Group")
    ax.set_ylabel("Churn Rate (%)")

    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "churn_by_tenure.png",
        dpi=150
    )

    plt.close()


def analyze_monthly_charges(df):
    """Compare monthly charges between churned and retained customers."""
    result = (
        df.groupby("Churn")["MonthlyCharges"]
        .agg(["mean", "median", "min", "max"])
    )

    print("\n========== MONTHLY CHARGES BY CHURN ==========")
    print(result)

    ax = (
        df.boxplot(
            column="MonthlyCharges",
            by="Churn",
            figsize=(8, 5)
        )
    )

    ax.set_title("Monthly Charges by Churn Status")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Monthly Charges")

    plt.suptitle("")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "monthly_charges_by_churn.png",
        dpi=150
    )

    plt.close()


def main():
    print("Loading cleaned dataset...")

    df = load_data()

    create_output_directory()

    basic_summary(df)

    analyze_contract_churn(df)

    analyze_internet_service_churn(df)

    analyze_tenure(df)

    analyze_monthly_charges(df)

    print("\n========================================")
    print("EDA completed successfully.")
    print(f"Reports saved to:\n{OUTPUT_DIR}")
    print("========================================")


if __name__ == "__main__":
    main()
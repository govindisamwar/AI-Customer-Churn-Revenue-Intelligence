from pathlib import Path
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
    PROJECT_ROOT
    / "Data_Engineering"
    / "data"
    / "raw"
    / "customer_churn"
    / "telco_customer_churn.csv"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "Data_Engineering"
    / "data"
    / "processed"
    / "customer_churn"
)

OUTPUT_FILE = PROCESSED_DIR / "telco_customer_churn_cleaned.csv"


def clean_customer_data():
    print("Loading raw customer churn dataset...")

    df = pd.read_csv(RAW_FILE)

    print(f"Raw dataset shape: {df.shape}")

    # Remove duplicate rows
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows found: {duplicates}")

    df = df.drop_duplicates().copy()

    # Convert TotalCharges to numeric.
    # Invalid/blank values become NaN.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    missing_total_charges = df["TotalCharges"].isna().sum()
    print(
        f"Missing TotalCharges after conversion: "
        f"{missing_total_charges}"
    )

    # Fill missing TotalCharges with 0 only for customers
    # with zero tenure; otherwise use the column median.
    zero_tenure_mask = (
        df["TotalCharges"].isna()
        & (df["tenure"] == 0)
    )

    df.loc[zero_tenure_mask, "TotalCharges"] = 0

    if df["TotalCharges"].isna().any():
        median_total_charges = df["TotalCharges"].median()

        df["TotalCharges"] = df["TotalCharges"].fillna(
            median_total_charges
        )

    # Remove leading/trailing spaces from text columns
    text_columns = df.select_dtypes(include="object").columns

    for column in text_columns:
        df[column] = df[column].str.strip()

    # Create binary target for future ML models
    df["ChurnBinary"] = df["Churn"].map(
        {
            "No": 0,
            "Yes": 1,
        }
    )

    # Final validation
    print(f"Cleaned dataset shape: {df.shape}")
    print(
        f"Total missing values: "
        f"{df.isna().sum().sum()}"
    )

    print("\nChurn distribution:")
    print(df["Churn"].value_counts())

    # Save processed dataset
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"\nCleaned dataset saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    clean_customer_data()
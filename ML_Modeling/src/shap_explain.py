import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

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
    "ML_Modeling/explainability"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("SHAP EXPLAINABILITY ANALYSIS")
print("=" * 60)

print("\nDataset shape:", df.shape)


# ============================================================
# 3. PREPARE FEATURES
# ============================================================

TARGET = "ChurnBinary"

X = df.drop(
    columns=[TARGET, "Churn"]
)

print("\nFeature shape:", X.shape)


# ============================================================
# 4. LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained model...")

pipeline = joblib.load(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# 5. TRANSFORM DATA
# ============================================================

print("\nTransforming features...")

preprocessor = pipeline.named_steps[
    "preprocessor"
]

model = pipeline.named_steps[
    "model"
]

X_transformed = preprocessor.transform(
    X
)

print(
    "Transformed feature shape:",
    X_transformed.shape
)


# ============================================================
# 6. GET FEATURE NAMES
# ============================================================

feature_names = (
    preprocessor.get_feature_names_out()
)

print(
    "Number of feature names:",
    len(feature_names)
)


# ============================================================
# 7. CREATE SHAP EXPLAINER
# ============================================================

print("\nCreating SHAP TreeExplainer...")

explainer = shap.TreeExplainer(
    model
)

print("SHAP explainer created successfully.")


# ============================================================
# 8. CALCULATE SHAP VALUES
# ============================================================

print("\nCalculating SHAP values...")

shap_values = explainer.shap_values(
    X_transformed
)

print(
    "SHAP calculation completed."
)


# ============================================================
# 9. SHAP SUMMARY PLOT
# ============================================================

print("\nCreating SHAP summary plot...")

plt.figure(
    figsize=(12, 8)
)

shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()

summary_path = (
    OUTPUT_DIR /
    "shap_summary.png"
)

plt.savefig(
    summary_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nSHAP summary saved:"
)

print(summary_path)


# ============================================================
# 10. SHAP BAR PLOT
# ============================================================

print("\nCreating SHAP feature importance plot...")

plt.figure(
    figsize=(12, 8)
)

shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)

plt.tight_layout()

bar_path = (
    OUTPUT_DIR /
    "shap_feature_importance.png"
)

plt.savefig(
    bar_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nSHAP feature importance saved:"
)

print(bar_path)


# ============================================================
# 11. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("SHAP ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Customer Churn & Revenue Intelligence",
    layout="wide"
)

st.title("Customer Churn & Revenue Intelligence")
st.caption("AI-powered customer risk and revenue monitoring dashboard")

df = pd.read_csv("Revenue_Intelligence/revenue_at_risk.csv")

# Sidebar filters
st.sidebar.header("Dashboard Filters")

risk_options = ["All"] + sorted(df["Risk_Category"].dropna().unique().tolist())
selected_risk = st.sidebar.selectbox("Risk Category", risk_options)

contract_options = ["All"] + sorted(df["Contract"].dropna().unique().tolist())
selected_contract = st.sidebar.selectbox("Contract Type", contract_options)

min_prob = float(df["Churn_Probability"].min())
max_prob = float(df["Churn_Probability"].max())

prob_range = st.sidebar.slider(
    "Churn Probability",
    min_value=0.0,
    max_value=1.0,
    value=(min_prob, max_prob),
    step=0.01
)

# Apply filters
filtered = df.copy()

if selected_risk != "All":
    filtered = filtered[
        filtered["Risk_Category"] == selected_risk
    ]

if selected_contract != "All":
    filtered = filtered[
        filtered["Contract"] == selected_contract
    ]

filtered = filtered[
    (filtered["Churn_Probability"] >= prob_range[0]) &
    (filtered["Churn_Probability"] <= prob_range[1])
]

# KPIs
total_customers = len(filtered)

predicted_churners = int(
    filtered["Predicted_Churn"].sum()
)

high_risk = int(
    (filtered["Risk_Category"] == "High Risk").sum()
)

avg_churn = (
    filtered["Churn_Probability"].mean()
    if len(filtered) > 0
    else 0
)

monthly_risk = filtered[
    "Monthly_Revenue_At_Risk"
].sum()

annual_risk = filtered[
    "Annual_Revenue_At_Risk"
].sum()

# KPI cards
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Customers",
    f"{total_customers:,}"
)

col2.metric(
    "Predicted Churners",
    f"{predicted_churners:,}"
)

col3.metric(
    "High Risk Customers",
    f"{high_risk:,}"
)

col4.metric(
    "Avg Churn Probability",
    f"{avg_churn:.2%}"
)

st.divider()

# Revenue KPIs
col5, col6 = st.columns(2)

col5.metric(
    "Monthly Revenue at Risk",
    f"${monthly_risk:,.2f}"
)

col6.metric(
    "Annual Revenue at Risk",
    f"${annual_risk:,.2f}"
)

st.divider()

# Customer risk distribution
st.subheader("Customer Risk Distribution")

risk_counts = (
    filtered["Risk_Category"]
    .value_counts()
    .reindex(
        ["Low Risk", "Medium Risk", "High Risk"]
    )
    .fillna(0)
)

st.bar_chart(risk_counts)

st.divider()

# Annual revenue at risk
st.subheader("Annual Revenue at Risk by Customer Risk")

risk_revenue = (
    filtered
    .groupby(
        "Risk_Category",
        observed=True
    )["Annual_Revenue_At_Risk"]
    .sum()
    .reindex(
        ["Low Risk", "Medium Risk", "High Risk"]
    )
    .fillna(0)
)

st.bar_chart(risk_revenue)

st.divider()

# Top customers
st.subheader("Top Customers by Revenue at Risk")

top_customers = (
    filtered
    .sort_values(
        "Monthly_Revenue_At_Risk",
        ascending=False
    )
    .head(10)
)

st.dataframe(
    top_customers[
        [
            "customerID",
            "Contract",
            "tenure",
            "Churn_Probability",
            "Risk_Category",
            "Monthly_Revenue_At_Risk",
            "Annual_Revenue_At_Risk"
        ]
    ],
    width="stretch"
)

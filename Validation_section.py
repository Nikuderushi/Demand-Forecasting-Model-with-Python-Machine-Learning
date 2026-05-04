import streamlit as st
import pandas as pd
import numpy as np
import pickle

@st.cache_resource
def load_artifacts():
    with open("xgboost_demand_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

model, label_encoder = load_artifacts()

# ── App Construction ──────────────────────────────────────────────────────────
st.title("📦 Demand Forecasting App")

# ── Input Features ────────────────────────────────────────────────────────────
st.divider()
st.header("Input Features")

price              = st.number_input("Price",            min_value=0.0,  value=50.0)
discount           = st.number_input("Discount (%)",     min_value=0,    max_value=100, value=10)
inventory_level    = st.number_input("Inventory Level",  min_value=0,    value=100)
promotion          = st.selectbox("Promotion",           [0, 1])
competitor_pricing = st.number_input("Competitor Price", min_value=0.0,  value=50.0)
category           = st.selectbox(
    "Category",
    label_encoder["Category"].classes_.tolist()
)

# ── Build Input DataFrame ─────────────────────────────────────────────────────
input_data = pd.DataFrame({
    "Price":              [price],
    "Discount":           [discount],
    "Inventory Level":    [inventory_level],
    "Promotion":          [promotion],
    "Competitor Pricing": [competitor_pricing],
    "Category":           [category],
})

for col, encoder in label_encoder.items():
    if col in input_data.columns:
        input_data[col] = encoder.transform(input_data[col])

# ── Prediction ────────────────────────────────────────────────────────────────
st.divider()

if st.button("Predict Demand", use_container_width=True):
    try:
        prediction = model.predict(input_data)[0]
        st.session_state["prediction"] = int(prediction)
        st.success(f"### Predicted Demand: {int(prediction)} units")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# ── Prediction Validation ─────────────────────────────────────────────────────
if "prediction" in st.session_state:
    st.divider()
    st.header("🔍 Validate Prediction")
    st.caption("Enter the actual observed demand to check how accurate the prediction was.")

    actual = st.number_input(
        "Actual Demand (units)",
        min_value=0,
        value=0,
        step=1,
        help="Enter the real demand value to compare against the model's prediction."
    )

    if st.button("Check Accuracy", use_container_width=True):
        predicted = st.session_state["prediction"]

        if actual == 0:
            st.warning("Please enter the actual demand value before checking accuracy.")
        else:
            error     = abs(predicted - actual)
            pct_error = (error / actual) * 100
            accuracy  = max(0.0, 100 - pct_error)

            st.subheader("📊 Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted",      f"{predicted} units")
            col2.metric("Actual",         f"{actual} units")
            col3.metric("Absolute Error", f"{error} units")

            st.divider()
            col4, col5 = st.columns(2)
            col4.metric("Error Rate", f"{pct_error:.2f}%")
            col5.metric("Accuracy",   f"{accuracy:.2f}%")

            # ── Verdict ───────────────────────────────────────────────────────
            if pct_error <= 5:
                st.success("✅ Excellent prediction! Error is within 5%.")
            elif pct_error <= 15:
                st.info("🟡 Good prediction. Error is between 5–15%.")
            elif pct_error <= 30:
                st.warning("🟠 Fair prediction. Error is between 15–30%.")
            else:
                st.error("🔴 Poor prediction. Error exceeds 30%. Consider retraining the model with more recent data.")
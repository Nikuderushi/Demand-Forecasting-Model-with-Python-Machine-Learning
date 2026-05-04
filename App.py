import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Page config ---
st.set_page_config(page_title="Demand Forecasting", page_icon="📦", layout="centered")

# --- Load model and encoder ---
@st.cache_resource
def load_artifacts():
    with open("xgboost_demand_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

model, label_encoder = load_artifacts()

# --- UI ---
st.title("📦 Demand Forecasting App")
st.markdown("Enter product and store details below to predict demand.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    price = st.number_input("Price ($)", min_value=0.0, value=50.0, step=0.5)
    discount = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
    inventory_level = st.number_input("Inventory Level (units)", min_value=0, value=200, step=1)

with col2:
    promotion = st.selectbox("Promotion Active?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    competitor_pricing = st.number_input("Competitor Pricing ($)", min_value=0.0, value=55.0, step=0.5)
    category = st.selectbox("Category", options=label_encoder["Category"].classes_)

st.divider()

if st.button("🔮 Predict Demand", use_container_width=True):
    # Encode category
    category_encoded = label_encoder["Category"].transform([category])[0]

    # Build input DataFrame (must match training feature order)
    input_data = pd.DataFrame([{
        "Price": price,
        "Discount": discount,
        "Inventory Level": inventory_level,
        "Promotion": promotion,
        "Competitor Pricing": competitor_pricing,
        "Category": category_encoded,
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"### Predicted Demand: **{prediction:,.0f} units**")
    st.caption("Prediction made using an XGBoost model trained on historical demand data.")
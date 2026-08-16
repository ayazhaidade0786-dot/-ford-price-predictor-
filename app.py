import pandas as pd
import streamlit as st
import joblib

# ---------- Load trained artifacts ----------
model = joblib.load("ford_price_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")
dropdown_options = joblib.load("dropdown_options.pkl")

NUMERICAL_COLS = ["year", "mileage", "tax", "mpg", "engineSize"]

st.set_page_config(page_title="Ford Price Predictor", page_icon="🚗")
st.title("🚗 Ford Car Price Predictor")
st.write("Enter the car's details below to estimate its resale price.")

# ---------- Input form ----------
col1, col2 = st.columns(2)

with col1:
    model_name = st.selectbox("Model", dropdown_options["model"])
    year = st.number_input("Year", min_value=1996, max_value=2026, value=2018, step=1)
    transmission = st.selectbox("Transmission", dropdown_options["transmission"])
    mileage = st.number_input("Mileage (miles)", min_value=0, value=15000, step=500)

with col2:
    fuel_type = st.selectbox("Fuel Type", dropdown_options["fuelType"])
    tax = st.number_input("Tax (£)", min_value=0, value=145, step=5)
    mpg = st.number_input("MPG", min_value=0.0, value=55.0, step=0.1)
    engine_size = st.number_input("Engine Size (L)", min_value=0.0, value=1.0, step=0.1)

if st.button("Predict Price", type="primary"):
    # Build a single-row DataFrame exactly like the training data (pre-encoding)
    input_df = pd.DataFrame([{
        "model": model_name,
        "year": year,
        "transmission": transmission,
        "mileage": mileage,
        "fuelType": fuel_type,
        "tax": tax,
        "mpg": mpg,
        "engineSize": engine_size,
    }])

    # One-hot encode the same way training did
    input_encoded = pd.get_dummies(
        input_df, columns=["model", "transmission", "fuelType"], drop_first=True
    )

    # Align columns to what the model was trained on (fills any missing dummy cols with 0)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Scale numeric columns with the SAME scaler fitted during training
    input_encoded[NUMERICAL_COLS] = scaler.transform(input_encoded[NUMERICAL_COLS])

    # Predict
    predicted_price = model.predict(input_encoded)[0]
    predicted_price = max(predicted_price, 0)  # price can't be negative

    st.success(f"### Estimated Price: £{predicted_price:,.2f}")
    st.caption("This is a linear regression estimate (R² ≈ 0.84 on test data) — treat it as a ballpark figure, not an exact valuation.")

st.divider()
st.caption("Model trained on UK Ford used-car listings · Linear Regression · One-hot encoded categorical features")

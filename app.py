#!/usr/bin/env python
"""
Streamlit Web App for Car Price Prediction.
Run: streamlit run app.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from src.config import get_config
from src.models.predict import predict_price, predict_from_raw


# Page config
st.set_page_config(
    page_title="Car Price Predictor 🚗💰",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    .info-box {
        background: #f3f4f6;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg., #1e40af 0%, #2563eb 100%);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_encoder():
    """Load trained model and feature engineer."""
    model_dir = Path("models")
    model_path = model_dir / "best_model.joblib"
    fe_path = model_dir / "feature_engineer.joblib"

    if not model_path.exists():
        return None, None, "Model not found. Train first with `python train.py`."

    try:
        model = joblib.load(model_path)
        fe = joblib.load(fe_path) if fe_path.exists() else None
        return model, fe, None
    except Exception as e:
        return None, None, f"Error loading model: {e}"


def get_brand_options():
    """Get list of brands from training data or config."""
    # Try to load from data
    data_path = Path("data/raw/car_prices.csv")
    if data_path.exists():
        try:
            df = pd.read_csv(data_path)
            if "Name" in df.columns:
                brands = df["Name"].apply(lambda x: str(x).split()[0]).unique()
                return sorted([b for b in brands if b != "unknown"])
        except Exception:
            pass

    # Fallback brands
    return [
        "Maruti", "Hyundai", "Honda", "Toyota", "Mahindra",
        "Tata", "Ford", "Volkswagen", "Skoda", "Renault",
        "Kia", "MG", "Jeep", "Nissan", "BMW", "Mercedes",
        "Audi", "Volvo", "Land Rover", "Porsche"
    ]


def main():
    # Header
    st.markdown('<div class="main-header">🚗 Car Price Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Estimate used car prices instantly — powered by CatBoost ML model</div>', unsafe_allow_html=True)

    # Load model
    model, fe, error = load_model_and_encoder()

    if error:
        st.error(f"⚠️ {error}")
        st.info("💡 Run `python train.py` in the project root to train the model first.")
        return

    # Model info
    st.markdown("""
    <div class="info-box">
        <strong>Model:</strong> CatBoost (optimized with Optuna)<br>
        <strong>Test R²:</strong> ~0.86 | <strong>Test RMSE:</strong> ~₹89,000<br>
        <strong>Training data:</strong> CarDekho used car listings
    </div>
    """, unsafe_allow_html=True)

    # Input form
    st.markdown("### Enter Car Details")

    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox(
            "Brand *",
            options=get_brand_options(),
            index=0,
            help="Car manufacturer"
        )
        year = st.number_input(
            "Year *",
            min_value=1990,
            max_value=2026,
            value=2019,
            help="Manufacturing year"
        )
        km_driven = st.number_input(
            "Kilometers Driven *",
            min_value=0,
            max_value=500000,
            value=30000,
            step=1000,
            help="Total km driven"
        )
        fuel = st.selectbox(
            "Fuel Type *",
            options=["Petrol", "Diesel", "CNG", "LPG", "Electric"],
            index=0
        )
        seller_type = st.selectbox(
            "Seller Type *",
            options=["Individual", "Dealer", "Trustmark Dealer"],
            index=0
        )

    with col2:
        transmission = st.selectbox(
            "Transmission *",
            options=["Manual", "Automatic", "CVT", "AMT"],
            index=0
        )
        owner = st.selectbox(
            "Owner *",
            options=["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"],
            index=0
        )
        mileage = st.number_input(
            "Mileage (kmpl) *",
            min_value=5.0,
            max_value=40.0,
            value=20.0,
            step=0.5,
            help="Fuel efficiency in km per liter"
        )
        engine = st.number_input(
            "Engine (CC) *",
            min_value=600,
            max_value=5000,
            value=1200,
            step=100,
            help="Engine displacement in cubic centimeters"
        )
        max_power = st.number_input(
            "Max Power (bhp) *",
            min_value=30.0,
            max_value=500.0,
            value=85.0,
            step=5.0,
            help="Maximum power in brake horsepower"
        )
        seats = st.selectbox(
            "Seats *",
            options=[2, 4, 5, 6, 7, 8, 9],
            index=2
        )

    # Predict button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔮 Predict Price", type="primary"):
        with st.spinner("Predicting..."):
            try:
                price = predict_price(
                    year=int(year),
                    km_driven=int(km_driven),
                    fuel=fuel,
                    seller_type=seller_type,
                    transmission=transmission,
                    owner=owner,
                    mileage=float(mileage),
                    engine=int(engine),
                    max_power=float(max_power),
                    seats=int(seats),
                    brand=brand,
                )

                # Format price
                if price >= 10000000:  # 1 crore+
                    price_str = f"₹{price/10000000:.2f} Cr"
                elif price >= 100000:  # 1 lakh+
                    price_str = f"₹{price/100000:.2f} Lakh"
                else:
                    price_str = f"₹{price:,.0f}"

                # Display result
                st.markdown(f"""
                <div class="prediction-box">
                    Estimated Price: {price_str}
                </div>
                """, unsafe_allow_html=True)

                # Additional context
                car_age = 2026 - int(year)
                st.markdown(f"""
                <div class="info-box">
                    <strong>Car Age:</strong> {car_age} years |
                    <strong>Usage:</strong> {km_driven/car_age if car_age > 0 else 0:,.0f} km/year |
                    <strong>Power-to-Weight:</strong> {max_power/(engine/1000):.1f} bhp/L
                </div>
                """, unsafe_allow_html=True)

                # Disclaimer
                st.caption("⚠️ This is an ML estimate based on historical data. Actual prices vary by condition, location, negotiation, and market factors.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #9ca3af; font-size: 0.85rem;">
        Built with Streamlit · CatBoost · Optuna · CarDekho Dataset<br>
        <a href="https://github.com/djkalyanprakash-gif/car-prediction" target="_blank" style="color: #3b82f6;">View Source Code</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
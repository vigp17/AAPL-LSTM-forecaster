# app/app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import yfinance as yf
import os

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Page config
st.set_page_config(page_title="AAPL LSTM Forecaster", layout="wide")
st.title("Apple Stock Price Predictor – LSTM (Live)")
st.markdown("Trained on 2015-2025 data · Test RMSE $6.67 · MAPE 2.32%")

# Load model & scaler once
@st.cache_resource
def load_artifacts():
    try:
        model = load_model("models/lstm_aapl.h5", compile=False)
        model.compile(optimizer='adam', loss='mean_squared_error')
        scaler = joblib.load("models/scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, scaler = load_artifacts()

if model is None or scaler is None:
    st.stop()

# Live data + prediction
ticker = "AAPL"
try:
    # Download only Close price
    df = yf.download(ticker, period="3y", progress=False)
    
    # Handle multi-index columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close']
    else:
        df = df['Close']
    
    df = df.dropna()
    
    # Ensure it's a proper 2D array for scaler
    close_values = df.values.reshape(-1, 1)
    
    # Transform
    scaled = scaler.transform(close_values)
    
    # Create sequence for prediction (last 60 days)
    if len(scaled) < 60:
        st.error("Not enough historical data. Need at least 60 days.")
        st.stop()
    
    sequence = scaled[-60:].reshape(1, 60, 1)
    
    # Predict
    pred_scaled = model.predict(sequence, verbose=0)
    pred_price = scaler.inverse_transform(pred_scaled)[0][0]
    
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current AAPL Price", f"${df.iloc[-1]:.2f}")
    with col2:
        st.metric("Next-Day LSTM Prediction", f"${pred_price:.2f}", 
                  delta=f"${pred_price - df.iloc[-1]:+.2f}")
    
    # Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index[-120:], df[-120:], label="Historical Close", linewidth=2, color='#1f77b4')
    ax.axvline(df.index[-1], color="gray", linestyle="--", alpha=0.5)
    ax.plot(df.index[-1], pred_price, "o", color="#ff7f0e", markersize=12, label="Tomorrow's Prediction", zorder=5)
    ax.set_title("AAPL – Last 120 Days + Tomorrow's LSTM Forecast", fontsize=14, fontweight='bold')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    
except Exception as e:
    st.error(f"Error: {e}")
    st.info("Make sure the model and scaler files are in the 'models/' directory.")
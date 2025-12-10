# app/app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import yfinance as yf
import os

# Page config
st.set_page_config(page_title="AAPL LSTM Forecaster", layout="wide")
st.title("Apple Stock Price Predictor – LSTM (Live)")
st.markdown("Trained on 2015-2025 data · Test RMSE $6.67 · MAPE 2.32%")

# Load model & scaler once
@st.cache_resource
def load_artifacts():
    model = load_model("models/lstm_aapl.h5")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# Live data + prediction
ticker = "AAPL"
df = yf.download(ticker, period="3y", progress=False)['Close'].dropna()
scaled = scaler.transform(df.values.reshape(-1, 1))

sequence = scaled[-60:].reshape(1, 60, 1)
pred_scaled = model.predict(sequence, verbose=0)
pred_price = scaler.inverse_transform(pred_scaled)[0][0]

col1, col2 = st.columns(2)
with col1:
    st.metric("Current AAPL Price", f"${df.iloc[-1]:.2f}")
with col2:
    st.metric("Next-Day LSTM Prediction", f"${pred_price:.2f}", 
              delta=f"${pred_price - df.iloc[-1]:+.2f}")

# Chart
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index[-120:], df[-120:], label="Historical Close", linewidth=2)
ax.axvline(df.index[-1], color="gray", linestyle="--")
ax.plot(df.index[-1:], pred_price, "o", color="#ff7f0e", markersize=10, label="Tomorrow’s Prediction")
ax.set_title("AAPL – Last 120 Days + Tomorrow’s LSTM Forecast")
ax.legend()
st.pyplot(fig)
# app/app.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GRPC_VERBOSITY'] = 'ERROR'

# Set these BEFORE importing tensorflow
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

import tensorflow as tf
# Force CPU mode (this is still safe to do)
tf.config.set_visible_devices([], 'GPU')

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import yfinance as yf
from datetime import datetime, timedelta

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

# Cache data download for 1 hour
@st.cache_data(ttl=3600)
def download_stock_data(ticker, period="3y"):
    """Download stock data with caching and retry logic"""
    import time
    max_retries = 5
    
    for attempt in range(max_retries):
        try:
            # Method 1: Use Ticker object
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if not df.empty and 'Close' in df.columns:
                return df['Close'].dropna()
            
            # Method 2: Direct download as fallback
            df_alt = yf.download(ticker, period=period, progress=False, 
                                 threads=False, ignore_tz=True)
            if not df_alt.empty:
                if 'Close' in df_alt.columns:
                    return df_alt['Close'].dropna()
                elif isinstance(df_alt.columns, pd.MultiIndex):
                    return df_alt['Close'][ticker].dropna()
            
            # Wait before retry
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))  # Exponential backoff
            
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Download failed after {max_retries} attempts: {str(e)}")
                return None
            time.sleep(2 * (attempt + 1))
    
    return None

model, scaler = load_artifacts()

if model is None or scaler is None:
    st.stop()

# Download data
ticker = "AAPL"
with st.spinner("Downloading latest AAPL data..."):
    df = download_stock_data(ticker)

if df is None or len(df) == 0:
    st.error("❌ Could not download stock data. Please try again later.")
    st.info("💡 Yahoo Finance may be experiencing issues or rate limiting. Try refreshing in a minute.")
    st.stop()

# Success - show data info
st.success(f"✓ Downloaded {len(df)} days of data ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")

try:
    # Prepare data for prediction
    close_values = df.values.reshape(-1, 1)
    
    # Transform
    scaled = scaler.transform(close_values)
    
    # Create sequence for prediction (last 60 days)
    if len(scaled) < 60:
        st.error("Not enough historical data. Need at least 60 days.")
        st.stop()
    
    sequence = scaled[-60:].reshape(1, 60, 1)
    
    # Predict
    with st.spinner("Running LSTM prediction..."):
        with tf.device('/CPU:0'):
            pred_scaled = model.predict(sequence, verbose=0, batch_size=8)
        pred_price = scaler.inverse_transform(pred_scaled)[0][0]
    
    # Display metrics
    current_price = df.iloc[-1]
    price_change = pred_price - current_price
    percent_change = (price_change / current_price) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current AAPL Price", f"${current_price:.2f}")
    with col2:
        st.metric("Next-Day LSTM Prediction", f"${pred_price:.2f}", 
                  delta=f"${price_change:+.2f}")
    with col3:
        st.metric("Predicted Change", f"{percent_change:+.2f}%")
    
    # Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot last 120 days
    plot_days = min(120, len(df))
    ax.plot(df.index[-plot_days:], df[-plot_days:], 
            label="Historical Close", linewidth=2, color='#1f77b4')
    
    # Add vertical line at current day
    ax.axvline(df.index[-1], color="gray", linestyle="--", alpha=0.5)
    
    # Plot prediction
    next_day = df.index[-1] + timedelta(days=1)
    ax.plot(next_day, pred_price, "o", color="#ff7f0e", 
            markersize=12, label="Tomorrow's Prediction", zorder=5)
    
    ax.set_title(f"AAPL – Last {plot_days} Days + Tomorrow's LSTM Forecast", 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    
    st.pyplot(fig)
    
    # Additional info
    st.markdown("---")
    st.markdown("**Note:** This is a prediction based on historical patterns. "
                "Actual stock prices are influenced by many factors not captured by this model. "
                "Not financial advice.")
    
except Exception as e:
    st.error(f"Error during prediction: {e}")
    import traceback
    st.code(traceback.format_exc())
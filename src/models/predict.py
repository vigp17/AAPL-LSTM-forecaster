# src/models/predict.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GRPC_VERBOSITY'] = 'ERROR'

import tensorflow as tf

# Aggressive CPU-only configuration for M1
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from src.config import config
from src.features.build_features import prepare_data

plt.style.use("seaborn-v0_8")

def load_trained_model():
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(f"No trained model found at {config.MODEL_PATH}")
    model = load_model(config.MODEL_PATH)
    scaler = joblib.load(config.PROJECT_ROOT / "models" / "scaler.pkl")
    return model, scaler

def make_predictions():
    print("Loading trained model and scaler...")
    model, scaler = load_trained_model()
    
    # Compile the model explicitly
    print("Compiling model...")
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print("Preparing fresh data (same splits as training)...")
    data = prepare_data()
    
    # Debug info
    print(f"X_test shape: {data['X_test'].shape}")
    print(f"y_test shape: {data['y_test'].shape}")
    print(f"X_test dtype: {data['X_test'].dtype}")
    
    # Predict on test set
    print("\n🔄 Making predictions on test set...")
    try:
        with tf.device('/CPU:0'):
            y_pred_scaled = model.predict(
                data['X_test'], 
                verbose=1, 
                batch_size=8
            )
        print(f"Predictions complete! Shape: {y_pred_scaled.shape}")
    except Exception as e:
        print(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("Inverse transforming predictions...")
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(data['y_test'])
    
    # Build date index for test period
    full_dates = data['raw_df'].index[config.SEQUENCE_LENGTH:]
    test_dates = full_dates[-len(y_true):]
    
    # Metrics
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"\nTest RMSE : ${rmse:.2f}")
    print(f"Test MAPE : {mape:.2f}%")
    
    # Plot
    plt.figure(figsize=(15, 7))
    plt.plot(test_dates, y_true, label="Actual Price", color="#1f77b4", linewidth=2)
    plt.plot(test_dates, y_pred, label="LSTM Prediction", color="#ff7f0e", linewidth=2)
    plt.title(f"AAPL Stock Price – LSTM Forecast vs Actual (Test Set)\nRMSE = ${rmse:.2f} | MAPE = {mape:.2f}%", 
              fontsize=16, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    plt.savefig(config.FIGURES_DIR / "final_prediction_vs_actual.png", dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nPlot saved → {config.FIGURES_DIR}/final_prediction_vs_actual.png")

if __name__ == "__main__":
    make_predictions()
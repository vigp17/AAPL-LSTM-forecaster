# src/features/build_features.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.config import config
from src.data.data_loader import load_raw_data, get_processed_path
import joblib
import os

def create_sequences(data: np.ndarray, seq_length: int = config.SEQUENCE_LENGTH):
    """Convert array of values into X/y sequences for LSTM"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def prepare_data() -> dict:
    """
    Full pipeline:
    1. Load raw data
    2. Add simple features
    3. Scale everything 0–1
    4. Create sequences
    5. Train/validation/test split (time-ordered!)
    6. Save scaler + processed data
    Returns dictionary with everything needed for training
    """
    df = load_raw_data()

    # Use only Close price for this simple but powerful model
    close_prices = df['Close'].values.reshape(-1, 1)

    # Scale 0 to 1
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices)

    # Create sequences
    X, y = create_sequences(scaled_data, config.SEQUENCE_LENGTH)

    # Time-based split (no shuffling!)
    total_samples = len(X)
    train_end = int(total_samples * (1 - config.TEST_SIZE - 0.1))   # 70% train
    val_end = int(total_samples * (1 - config.TEST_SIZE))      # 10% val

    X_train, y_train = X[:train_end], y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:], y[val_end:]

    print(f"Sequences created!")
    print(f"Train: {X_train.shape} → {y_train.shape}")
    print(f"Val:   {X_val.shape}   → {y_val.shape}")
    print(f"Test:  {X_test.shape}  → {y_test.shape}")

    # Save scaler for inverse transform later
    scaler_path = config.PROJECT_ROOT / "models" / "scaler.pkl"
    joblib.dump(scaler, scaler_path)

    result = {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "scaler":  scaler,
        "raw_df":  df
    }

    # Optional: save processed sequences for later debugging
    processed_path = get_processed_path()
    np.savez_compressed(
        processed_path.replace(".csv", ""),
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )

    return result

if __name__ == "__main__":
    data_dict = prepare_data()
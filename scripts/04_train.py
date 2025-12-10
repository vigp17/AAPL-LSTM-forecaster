# scripts/04_train.py
from src.models.train import train_model

if __name__ == "__main__":
    print("="*60)
    print("LSTM TRAINING – AAPL STOCK PRICE FORECASTER")
    print("="*60)
    train_model()
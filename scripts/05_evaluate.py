# scripts/05_evaluate.py
from src.models.predict import make_predictions

if __name__ == "__main__":
    print("="*70)
    print("FINAL MODEL EVALUATION – AAPL LSTM")
    print("="*70)
    make_predictions()
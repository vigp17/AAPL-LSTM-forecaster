# scripts/03_prepare_data.py
from src.features.build_features import prepare_data

if __name__ == "__main__":
    data = prepare_data()
    print("\nData preparation complete!")
    print("Scaler saved → models/scaler.pkl")
    print("Sequences saved → data/processed/")
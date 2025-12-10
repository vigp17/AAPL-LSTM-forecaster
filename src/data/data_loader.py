# src/data/data_loader.py
import pandas as pd
from src.config import config
import os

def load_raw_data() -> pd.DataFrame:
    """Load the raw CSV we just downloaded"""
    if not os.path.exists(config.RAW_DATA):
        raise FileNotFoundError(f"Raw data not found at {config.RAW_DATA}\nRun 01_download_data.py first!")
    
    df = pd.read_csv(config.RAW_DATA, index_col=0, parse_dates=True)
    print(f"Loaded raw data: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df

def get_processed_path(ticker: str = config.TICKER) -> str:
    """Where we will save the final ready-to-train data"""
    return os.path.join(config.PROCESSED_DIR, f"processed_{ticker}.csv")

if __name__ == "__main__":
    # Quick test
    df = load_raw_data()
    print(df.head())
    print(df.tail())
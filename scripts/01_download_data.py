# scripts/01_download_data.py
import yfinance as yf
from src.config import config
import pandas as pd

def download_and_save():
    print(f"Downloading {config.TICKER} from {config.START_DATE} to {config.END_DATE}...")

    # NEW correct way in 2025 yfinance
    df = yf.download(
        tickers=config.TICKER,
        start=config.START_DATE,
        end=config.END_DATE,
        progress=False,
        auto_adjust=False,
        actions=True
    )

    if df.empty:
        raise ValueError("No data returned. Check internet or ticker.")

    # Force column names (sometimes yfinance returns multi-level columns)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only the columns we actually need
    expected_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        print(f"Available columns: {list(df.columns)}")
        raise KeyError(f"Missing columns: {missing}")

    df = df[expected_columns].copy()
    df = df.round(4)
    df.dropna(inplace=True)

    # Save
    config.RAW_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RAW_DATA)

    print(f"Success! {len(df):,} rows saved")
    print(f"→ {config.RAW_DATA}")

if __name__ == "__main__":
    download_and_save()
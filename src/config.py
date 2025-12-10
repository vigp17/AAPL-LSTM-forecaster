from pathlib import Path

class Config:
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    RAW_DATA = PROJECT_ROOT / "data" / "raw" / "raw_aapl.csv"
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    MODEL_PATH = PROJECT_ROOT / "models" / "lstm_aapl.h5"
    FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

    # Data settings
    TICKER = "AAPL"
    START_DATE = "2015-01-01"
    END_DATE = "2025-12-09"
    SEQUENCE_LENGTH = 60
    TEST_SIZE = 0.2

    # Model settings
    LSTM_UNITS = [100, 50]
    DROPOUT = 0.2
    EPOCHS = 50
    BATCH_SIZE = 32
    RANDOM_STATE = 42

# Create one global config object
config = Config()

# Create folders if they don't exist
config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
(Path(config.MODEL_PATH).parent).mkdir(parents=True, exist_ok=True)
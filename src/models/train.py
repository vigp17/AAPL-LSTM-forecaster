# src/models/train.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.features.build_features import prepare_data
from src.config import config
import os

def build_model():
    model = Sequential([
        LSTM(units=config.LSTM_UNITS[0], return_sequences=True,
             input_shape=(config.SEQUENCE_LENGTH, 1)),
        Dropout(config.DROPOUT),
        
        LSTM(units=config.LSTM_UNITS[1], return_sequences=False),
        Dropout(config.DROPOUT),
        
        Dense(units=25),
        Dense(units=1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )
    return model

def train_model():
    print("Preparing data...")
    data = prepare_data()
    
    print("Building LSTM model...")
    model = build_model()
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(config.MODEL_PATH, save_best_only=True, monitor='val_loss')
    ]
    
    print("Training started...")
    history = model.fit(
        data['X_train'], data['y_train'],
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=(data['X_val'], data['y_val']),
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"Model trained and saved to → {config.MODEL_PATH}")
    return model, history, data

if __name__ == "__main__":
    trained_model, hist, dataset = train_model()
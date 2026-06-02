"""
Startup script to ensure model is available before running the app.
Trains the model if it doesn't exist.
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import MODEL_PATH
from train import main as train_model

def ensure_model_exists():
    """Train model if it doesn't exist."""
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Training model...")
        train_model()
        print("Model training completed!")
    else:
        print("Model found!")

if __name__ == "__main__":
    ensure_model_exists()

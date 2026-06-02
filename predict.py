#!/usr/bin/env python3
"""
Batch prediction script for credit card fraud detection.

Usage:
    python predict.py --input data/new_transactions.csv --output predictions.csv
    python predict.py --input data/new_transactions.csv --output predictions.csv --threshold 0.3
"""

import argparse
import pandas as pd
import joblib
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import MODEL_PATH, SCALER_PATH, FEATURES_PATH
from src.preprocessor import FraudPreprocessor
from src.utils import validate_input_data


def load_model_and_preprocessor():
    """Load the trained model and preprocessing artifacts."""
    print("Loading model and preprocessor...")

    model = joblib.load(MODEL_PATH)

    preprocessor = FraudPreprocessor()
    preprocessor.load_artifacts()

    print("Model and preprocessor loaded successfully.")
    return model, preprocessor


def predict_batch(input_path: str, output_path: str, threshold: float = 0.5):
    """
    Run batch predictions on a CSV file.

    Args:
        input_path: Path to input CSV with transaction data
        output_path: Path to save predictions CSV
        threshold: Probability threshold for fraud classification
    """
    # Load model and preprocessor
    model, preprocessor = load_model_and_preprocessor()

    # Load input data
    print(f"\nLoading input data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} transactions")

    # Preprocess
    print("Preprocessing data...")
    X = preprocessor.transform_new_data(df)

    # Predict
    print("Making predictions...")
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    # Add results to dataframe
    df['Fraud_Probability'] = probabilities
    df['Fraud_Prediction'] = predictions
    df['Risk_Level'] = df['Fraud_Probability'].apply(
        lambda p: 'High' if p >= 0.7 else ('Medium' if p >= 0.3 else 'Low')
    )

    # Save results
    df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")

    # Summary
    n_fraud = predictions.sum()
    n_legitimate = len(predictions) - n_fraud
    print(f"\nPrediction Summary:")
    print(f"  Total transactions: {len(predictions)}")
    print(f"  Predicted fraud: {n_fraud} ({n_fraud/len(predictions)*100:.2f}%)")
    print(f"  Predicted legitimate: {n_legitimate} ({n_legitimate/len(predictions)*100:.2f}%)")
    print(f"  Average fraud probability: {probabilities.mean():.4f}")
    print(f"  Max fraud probability: {probabilities.max():.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Credit Card Fraud Detection - Batch Prediction'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to input CSV file with transaction data'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Path to save predictions CSV file'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.5,
        help='Probability threshold for fraud classification (default: 0.5)'
    )

    args = parser.parse_args()

    predict_batch(args.input, args.output, args.threshold)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
End-to-end training script for credit card fraud detection.

Usage:
    python train.py

This script:
    1. Loads and validates the dataset
    2. Preprocesses the data (feature engineering, scaling, SMOTE)
    3. Trains a Random Forest classifier
    4. Evaluates the model with full metrics
    5. Saves the model and all artifacts
    6. Generates evaluation reports
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from src.config import DATASET_PATH


def main():
    print("=" * 70)
    print("CREDIT CARD FRAUD DETECTION - MODEL TRAINING")
    print("=" * 70)

    # Initialize trainer
    trainer = ModelTrainer(dataset_path=DATASET_PATH)

    # Run full pipeline
    model, preprocessor, X_test, y_test, data_summary = trainer.run()

    # Evaluate on test set
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    # Get predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluate
    evaluator = ModelEvaluator(model_name="Random Forest")
    evaluator.generate_full_report(
        y_test, y_pred, y_prob,
        importances_df=model.get_feature_importance(preprocessor.feature_names)
    )

    print("\n" + "-" * 70)
    print("DATASET SUMMARY")
    print("-" * 70)
    for key, value in data_summary.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Run the Streamlit app:  streamlit run app.py")
    print("  2. Make batch predictions: python predict.py --input data/sample.csv --output predictions.csv")


if __name__ == '__main__':
    main()

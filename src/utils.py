"""
Utility functions for the fraud detection project.
"""

import pandas as pd
import numpy as np
import logging
from typing import Iterable
from pandas.api.types import is_numeric_dtype

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_input_data(
    df: pd.DataFrame,
    expected_features: Iterable[str],
    *,
    allow_extra: bool = True,
    numeric_only: bool = True
) -> bool:
    """
    Validate that input data contains all required features.

    Args:
        df: Input dataframe
        expected_features: List of required column names

    Returns:
        bool: True if valid, raises ValueError otherwise
    """
    missing = [col for col in expected_features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    if not allow_extra:
        extra = [col for col in df.columns if col not in expected_features]
        if extra:
            raise ValueError(f"Unexpected extra features: {extra}")

    if numeric_only:
        non_numeric = [
            col for col in expected_features
            if col in df.columns and not is_numeric_dtype(df[col])
        ]
        if non_numeric:
            raise ValueError(
                "Non-numeric values detected in numeric feature columns: "
                f"{non_numeric}"
            )

    if df.empty:
        raise ValueError("Input dataframe is empty")

    logger.info(f"Input validation passed: {len(df)} rows, {len(df.columns)} columns")
    return True


def create_sample_data(n_samples: int = 100, fraud_rate: float = 0.01) -> pd.DataFrame:
    """
    Create synthetic sample data for testing the pipeline.

    Args:
        n_samples: Number of samples to generate
        fraud_rate: Proportion of fraudulent transactions

    Returns:
        pd.DataFrame: Synthetic data matching the creditcard.csv structure
    """
    np.random.seed(42)

    n_fraud = int(n_samples * fraud_rate)
    n_legitimate = n_samples - n_fraud

    # Generate V1-V28 (PCA components) - normally distributed
    data = {}
    for i in range(1, 29):
        data[f'V{i}'] = np.random.normal(0, 1, n_samples)

    # Time (seconds) - uniform over 48 hours
    data['Time'] = np.random.uniform(0, 172800, n_samples)

    # Amount - log-normal distribution (skewed like real data)
    data['Amount'] = np.random.lognormal(3, 1.5, n_samples)

    # Class
    data['Class'] = [1] * n_fraud + [0] * n_legitimate

    df = pd.DataFrame(data)

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    logger.info(f"Created sample data: {n_samples} samples ({n_fraud} fraud, {n_legitimate} legitimate)")
    return df


def format_prediction_result(probability: float, threshold: float = 0.5) -> dict:
    """
    Format a single prediction result for display.

    Args:
        probability: Predicted probability of fraud
        threshold: Classification threshold

    Returns:
        dict: Formatted result with prediction, probability, and risk level
    """
    prediction = 1 if probability >= threshold else 0

    if probability < 0.3:
        risk_level = "Low"
        color = "green"
    elif probability < 0.7:
        risk_level = "Medium"
        color = "orange"
    else:
        risk_level = "High"
        color = "red"

    return {
        'prediction': int(prediction),
        'probability': float(probability),
        'risk_level': risk_level,
        'color': color,
        'is_fraud': bool(prediction == 1)
    }

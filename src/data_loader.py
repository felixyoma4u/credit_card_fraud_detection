"""
Data loading and validation module.
Handles loading the creditcard.csv dataset with integrity checks.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading and initial validation of the credit card fraud dataset.

    Attributes:
        filepath (str): Path to the CSV file
        df (pd.DataFrame): Loaded dataframe
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None

    def load(self) -> pd.DataFrame:
        """
        Load the dataset from CSV with validation.

        Returns:
            pd.DataFrame: Loaded and validated dataset

        Raises:
            FileNotFoundError: If dataset file does not exist
            ValueError: If dataset fails validation checks
        """
        logger.info(f"Loading dataset from: {self.filepath}")

        try:
            self.df = pd.read_csv(self.filepath)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Dataset not found at {self.filepath}.\n"
                "Please download from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
            )

        self._validate()
        logger.info(f"Dataset loaded successfully: {self.df.shape}")
        return self.df

    def _validate(self):
        """Run validation checks on the loaded dataset."""

        # Check required columns
        required_cols = ['Class', 'Amount', 'Time'] + [f'V{i}' for i in range(1, 29)]
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Check target column
        if 'Class' not in self.df.columns:
            raise ValueError("Target column 'Class' not found in dataset")

        if not set(self.df['Class'].unique()).issubset({0, 1}):
            raise ValueError("Target column 'Class' must contain only 0 and 1")

        # Check for NaN in target
        if self.df['Class'].isnull().sum() > 0:
            raise ValueError("Target column contains NaN values")

        # Log dataset statistics
        fraud_rate = self.df['Class'].mean()
        logger.info(f"Dataset shape: {self.df.shape}")
        logger.info(f"Fraud rate: {fraud_rate:.4f} ({fraud_rate*100:.3f}%)")
        logger.info(f"Duplicates: {self.df.duplicated().sum()}")
        logger.info(f"Missing values: {self.df.isnull().sum().sum()}")

    def get_summary(self) -> dict:
        """Return a summary dictionary of the dataset."""
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load() first.")

        return {
            'shape': self.df.shape,
            'fraud_count': int(self.df['Class'].sum()),
            'legitimate_count': int((self.df['Class'] == 0).sum()),
            'fraud_rate': float(self.df['Class'].mean()),
            'duplicates': int(self.df.duplicated().sum()),
            'missing_values': int(self.df.isnull().sum().sum()),
            'amount_stats': {
                'mean': float(self.df['Amount'].mean()),
                'median': float(self.df['Amount'].median()),
                'max': float(self.df['Amount'].max()),
                'min': float(self.df['Amount'].min())
            }
        }

    def clean(self) -> pd.DataFrame:
        """
        Remove duplicates and handle any data quality issues.

        Returns:
            pd.DataFrame: Cleaned dataset
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load() first.")

        initial_shape = self.df.shape

        # Remove duplicates
        self.df = self.df.drop_duplicates()

        # Reset index
        self.df = self.df.reset_index(drop=True)

        logger.info(f"Cleaning complete. Shape: {initial_shape} -> {self.df.shape}")
        return self.df

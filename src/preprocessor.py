"""
Data preprocessing pipeline for credit card fraud detection.
Handles feature engineering, scaling, and train-test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from typing import Tuple
import joblib
import logging

from src.utils import validate_input_data

from src.config import (
    TEST_SIZE, RANDOM_STATE, SMOTE_PARAMS, 
    DROP_FEATURES, TARGET_COLUMN, SCALER_PATH, FEATURES_PATH
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FraudPreprocessor:
    """
    Complete preprocessing pipeline for credit card fraud data.

    Pipeline steps:
    1. Feature engineering (Hour, Log_Amount)
    2. Feature selection (drop redundant columns)
    3. Train-test split with stratification
    4. Scaling with RobustScaler
    5. SMOTE for class imbalance (training set only)
    """

    def __init__(self):
        self.scaler = RobustScaler()
        self.smote = SMOTE(**SMOTE_PARAMS)
        self.feature_names = None

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features from raw data.

        Args:
            df: Raw dataframe with Time and Amount columns

        Returns:
            DataFrame with new engineered features
        """
        df = df.copy()

        required_base_cols = ['Time', 'Amount']
        missing = [col for col in required_base_cols if col not in df.columns]
        if missing:
            raise ValueError(
                "Missing required base columns for feature engineering: "
                f"{missing}"
            )

        df[required_base_cols] = df[required_base_cols].apply(pd.to_numeric, errors='coerce')
        if df[required_base_cols].isnull().any().any():
            raise ValueError("Non-numeric values detected in Time/Amount columns after coercion")

        # Hour of day from Time (seconds)
        df['Hour'] = (df['Time'] // 3600) % 24

        # Log-transform Amount to reduce skewness
        # log1p handles zero values safely (log(1+x) instead of log(x))
        df['Log_Amount'] = np.log1p(df['Amount'])

        logger.info("Feature engineering complete. New features: Hour, Log_Amount")
        return df

    def select_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target, drop redundant columns.

        Args:
            df: DataFrame with all columns

        Returns:
            Tuple of (X features, y target)
        """
        # Drop original Amount and Time (replaced by engineered features)
        feature_cols = [col for col in df.columns if col not in DROP_FEATURES + [TARGET_COLUMN]]

        X = df[feature_cols]
        y = df[TARGET_COLUMN]

        self.feature_names = feature_cols

        logger.info(f"Feature selection complete. Using {len(feature_cols)} features")
        return X, y

    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """
        Split data into train and test sets with stratification.

        Stratification ensures both sets maintain the same fraud rate,
        which is critical for imbalanced datasets.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

        logger.info(f"Train set: {X_train.shape[0]} samples (fraud rate: {y_train.mean():.4f})")
        logger.info(f"Test set: {X_test.shape[0]} samples (fraud rate: {y_test.mean():.4f})")

        return X_train, X_test, y_train, y_test

    def fit_scale(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """
        Fit scaler on training data and transform it.

        Args:
            X_train: Training features

        Returns:
            Scaled training features
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(
            X_train_scaled,
            columns=X_train.columns,
            index=X_train.index
        )

        logger.info("Scaler fitted on training data")
        return X_train_scaled

    def transform_scale(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using the already-fitted scaler.

        Args:
            X: Features to transform

        Returns:
            Scaled features
        """
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(
            X_scaled,
            columns=X.columns,
            index=X.index
        )
        return X_scaled

    def apply_smote(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE to balance the training set.

        SMOTE creates synthetic minority class examples by interpolating
        between existing fraud cases. This is applied ONLY to the training
        set to prevent data leakage into the test set.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Balanced training data
        """
        try:
            X_resampled, y_resampled = self.smote.fit_resample(X_train, y_train)
        except ValueError as exc:
            logger.warning(
                "SMOTE could not be applied (%s). Proceeding without resampling.",
                exc
            )
            return X_train, y_train

        logger.info(f"SMOTE applied. Before: {len(y_train)} samples, After: {len(y_resampled)} samples")
        logger.info(f"Class distribution after SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}")

        return X_resampled, y_resampled

    def fit_transform(self, df: pd.DataFrame) -> Tuple:
        """
        Run the full preprocessing pipeline on raw data.

        Args:
            df: Raw dataframe from data loader

        Returns:
            Tuple of (X_train_resampled, X_test_scaled, y_train_resampled, y_test)
        """
        logger.info("Starting preprocessing pipeline...")

        # Step 1: Feature engineering
        df = self.engineer_features(df)

        # Step 2: Feature selection
        X, y = self.select_features(df)

        # Step 3: Train-test split
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # Step 4: Scale training data (fit scaler)
        X_train_scaled = self.fit_scale(X_train)

        # Step 5: Scale test data (transform only - no fitting)
        X_test_scaled = self.transform_scale(X_test)

        # Step 6: Apply SMOTE to training data only
        X_train_resampled, y_train_resampled = self.apply_smote(X_train_scaled, y_train)

        logger.info("Preprocessing pipeline complete")
        return X_train_resampled, X_test_scaled, y_train_resampled, y_test

    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess new data for prediction using the fitted scaler.

        Args:
            df: New raw dataframe

        Returns:
            Preprocessed features ready for prediction
        """
        df = self.engineer_features(df)

        if self.feature_names is None:
            raise ValueError("Preprocessor artifacts missing. Load artifacts before transforming new data.")

        validate_input_data(df, self.feature_names, allow_extra=True)

        missing_cols = [col for col in self.feature_names if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing engineered feature columns: {missing_cols}. Did you include Time and Amount?")

        X = df[self.feature_names]

        X_scaled = self.transform_scale(X)
        return X_scaled

    def save_artifacts(self):
        """Save scaler and feature names for later use."""
        joblib.dump(self.scaler, SCALER_PATH)
        joblib.dump(self.feature_names, FEATURES_PATH)
        logger.info(f"Scaler saved to: {SCALER_PATH}")
        logger.info(f"Feature names saved to: {FEATURES_PATH}")

    def load_artifacts(self):
        """Load previously saved scaler and feature names."""
        self.scaler = joblib.load(SCALER_PATH)
        self.feature_names = joblib.load(FEATURES_PATH)
        logger.info("Preprocessor artifacts loaded")

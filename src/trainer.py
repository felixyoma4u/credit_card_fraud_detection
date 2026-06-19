"""
Model training orchestrator.
Coordinates data loading, preprocessing, model training, and saving.
"""

import os
import joblib
import logging
from typing import Tuple, Dict, Any

from src.config import MODEL_PATH, DATASET_PATH, SUMMARY_PATH
from src.data_loader import DataLoader
from src.preprocessor import FraudPreprocessor
from src.model import FraudDetectionModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Orchestrates the end-to-end training pipeline.

    Usage:
        trainer = ModelTrainer()
        model, preprocessor = trainer.run()
    """

    def __init__(self, dataset_path: str = DATASET_PATH, model_path: str = MODEL_PATH):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.data_loader = DataLoader(dataset_path)
        self.preprocessor = FraudPreprocessor()
        self.model = FraudDetectionModel()

    def run(self) -> Tuple[FraudDetectionModel, FraudPreprocessor, Any, Any, Dict[str, Any]]:
        """
        Execute the full training pipeline.

        Steps:
        1. Load and clean data
        2. Preprocess (engineer features, split, scale, SMOTE)
        3. Train Random Forest
        4. Save model and artifacts

        Returns:
            Tuple of (trained_model, fitted_preprocessor)
        """
        logger.info("=" * 60)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("=" * 60)

        # Step 1: Load and clean data
        df = self.data_loader.load()
        df = self.data_loader.clean()
        summary = self.data_loader.get_summary()

        # Step 2: Preprocess
        X_train, X_test, y_train, y_test = self.preprocessor.fit_transform(df)

        # Step 3: Train model
        self.model.fit(X_train, y_train)

        # Step 4: Save artifacts
        self._save_model()
        self.preprocessor.save_artifacts()
        self._save_summary(summary)

        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info("=" * 60)

        return self.model, self.preprocessor, X_test, y_test, summary

    def _save_model(self):
        """Save the trained model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logger.info(f"Model saved to: {self.model_path}")

    def _save_summary(self, summary: Dict[str, Any]):
        os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
        joblib.dump(summary, SUMMARY_PATH)
        logger.info(f"Dataset summary saved to: {SUMMARY_PATH}")

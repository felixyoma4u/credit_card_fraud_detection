"""
Random Forest model definition for credit card fraud detection.
Encapsulates the model architecture and hyperparameters.
"""

from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any
import logging

from src.config import RF_PARAMS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FraudDetectionModel:
    """
    Random Forest classifier for fraud detection.

    Why Random Forest?
    - Handles non-linear relationships well (fraud patterns are complex)
    - Robust to outliers (important since outliers may be fraud signals)
    - Provides feature importance rankings
    - Works well on tabular data with engineered features
    - Less prone to overfitting than single decision trees

    Attributes:
        model: The underlying sklearn RandomForestClassifier
        is_trained: Boolean flag indicating if model has been fitted
    """

    def __init__(self, custom_params: Dict[str, Any] = None):
        """
        Initialize the Random Forest model.

        Args:
            custom_params: Optional dictionary to override default parameters
        """
        params = custom_params or RF_PARAMS
        self.model = RandomForestClassifier(**params)
        self.is_trained = False

        logger.info(f"Model initialized with parameters: {params}")

    def fit(self, X_train, y_train):
        """
        Train the model on the provided data.

        Args:
            X_train: Training features (should be preprocessed and SMOTE-balanced)
            y_train: Training labels

        Returns:
            self: For method chaining
        """
        logger.info(f"Training model on {len(X_train)} samples...")
        try:
            self.model.fit(X_train, y_train)
        except ValueError as exc:
            raise ValueError(
                "Model training failed. Ensure input features are numeric and match the training schema."
            ) from exc
        self.is_trained = True

        # Log OOB score if available
        if hasattr(self.model, 'oob_score_') and self.model.oob_score:
            logger.info(f"Out-of-bag score: {self.model.oob_score_:.4f}")

        logger.info("Model training complete")
        return self

    def predict(self, X):
        """
        Make binary predictions (0 or 1).

        Args:
            X: Features to predict on

        Returns:
            Array of predicted classes
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        if X is None or len(X) == 0:
            raise ValueError("Input data for prediction is empty.")
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X: Features to predict on

        Returns:
            Array of shape (n_samples, 2) with probabilities for each class
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        if X is None or len(X) == 0:
            raise ValueError("Input data for prediction is empty.")
        return self.model.predict_proba(X)

    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance rankings from the trained model.

        Args:
            feature_names: List of feature names (optional)

        Returns:
            DataFrame with features ranked by importance
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")

        importances = self.model.feature_importances_

        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(importances))]

        importance_df = (
            __import__('pandas', fromlist=['DataFrame'])
            .DataFrame({
                'Feature': feature_names,
                'Importance': importances
            })
            .sort_values('Importance', ascending=False)
        )

        return importance_df

    def get_params(self) -> Dict[str, Any]:
        """Return the model's current parameters."""
        return self.model.get_params()

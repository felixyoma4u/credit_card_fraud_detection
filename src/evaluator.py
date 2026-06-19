"""
Model evaluation module.
Computes and visualizes all required classification metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
from typing import Dict, List
import os
import logging
import joblib

from src.config import REPORTS_DIR, METRICS_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive evaluation of fraud detection models.

    For fraud detection, we prioritize:
    - Recall: Catching actual frauds (minimizing false negatives)
    - Precision: Avoiding false alarms (minimizing false positives)
    - F1-Score: Balance between precision and recall
    - ROC-AUC: Overall discriminative ability
    - PR-AUC: More reliable than ROC for imbalanced data
    """

    def __init__(self, model_name: str = "Model"):
        self.model_name = model_name
        self.results = {}

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
        """
        Compute all evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels (binary)
            y_prob: Predicted probabilities for the positive class

        Returns:
            Dictionary of all metrics
        """
        metrics = {
            'Model': self.model_name,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1-Score': f1_score(y_true, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(y_true, y_prob),
            'PR-AUC': average_precision_score(y_true, y_prob)
        }

        self.results = metrics

        logger.info(f"\n{'='*50}")
        logger.info(f"EVALUATION RESULTS: {self.model_name}")
        logger.info(f"{'='*50}")
        for key, value in metrics.items():
            if key != 'Model':
                logger.info(f"{key:12s}: {value:.4f}")

        joblib.dump(metrics, METRICS_PATH)
        logger.info(f"Metrics saved to: {METRICS_PATH}")

        return metrics

    def print_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Print detailed per-class metrics."""
        report = classification_report(
            y_true, y_pred, 
            target_names=['Legitimate', 'Fraud'],
            zero_division=0
        )
        logger.info(f"\nClassification Report:\n{report}")
        return report

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
        """
        Plot and optionally save confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)

        disp = ConfusionMatrixDisplay(
            cm, 
            display_labels=['Legitimate', 'Fraud']
        )
        disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=True)
        ax.set_title(f'Confusion Matrix - {self.model_name}', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to: {save_path}")

        plt.show()
        return fig

    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, save_path: str = None):
        """
        Plot ROC curve.

        The ROC curve shows the trade-off between True Positive Rate (Recall)
        and False Positive Rate at various thresholds.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)

        ax.plot(fpr, tpr, linewidth=2.5, 
                label=f'{self.model_name} (AUC = {auc_score:.4f})', color='#2ecc71')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier', color='#e74c3c')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to: {save_path}")

        plt.show()
        return fig

    def plot_precision_recall_curve(self, y_true: np.ndarray, y_prob: np.ndarray, save_path: str = None):
        """
        Plot Precision-Recall curve.

        PR curves are more informative than ROC for imbalanced datasets
        because they focus on the minority class performance.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_prec = average_precision_score(y_true, y_prob)

        ax.plot(recall, precision, linewidth=2.5,
                label=f'{self.model_name} (AP = {avg_prec:.4f})', color='#3498db')

        # Random baseline
        baseline = y_true.mean()
        ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1.5,
                   label=f'Random Baseline ({baseline:.4f})')

        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PR curve saved to: {save_path}")

        plt.show()
        return fig

    def plot_feature_importance(self, importances_df: pd.DataFrame, top_n: int = 15, save_path: str = None):
        """
        Plot feature importance rankings.

        Args:
            importances_df: DataFrame with 'Feature' and 'Importance' columns
            top_n: Number of top features to display
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        top_features = importances_df.head(top_n)

        sns.barplot(
            x='Importance', 
            y='Feature', 
            data=top_features,
            palette='viridis',
            ax=ax
        )

        ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to: {save_path}")

        plt.show()
        return fig

    def generate_full_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             y_prob: np.ndarray, importances_df: pd.DataFrame = None):
        """
        Generate a complete evaluation report with all plots and metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            importances_df: Optional feature importance DataFrame
        """
        os.makedirs(REPORTS_DIR, exist_ok=True)

        # Metrics
        self.evaluate(y_true, y_pred, y_prob)
        self.print_classification_report(y_true, y_pred)

        # Plots
        self.plot_confusion_matrix(
            y_true, y_pred,
            save_path=os.path.join(REPORTS_DIR, 'confusion_matrix.png')
        )

        self.plot_roc_curve(
            y_true, y_prob,
            save_path=os.path.join(REPORTS_DIR, 'roc_curve.png')
        )

        self.plot_precision_recall_curve(
            y_true, y_prob,
            save_path=os.path.join(REPORTS_DIR, 'precision_recall_curve.png')
        )

        if importances_df is not None:
            self.plot_feature_importance(
                importances_df,
                save_path=os.path.join(REPORTS_DIR, 'feature_importance.png')
            )

        logger.info(f"\nFull report generated in: {REPORTS_DIR}")

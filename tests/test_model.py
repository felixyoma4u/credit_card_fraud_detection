"""
Unit tests for the fraud detection pipeline.
Run with: pytest tests/
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import create_sample_data, validate_input_data, format_prediction_result
from src.config import RF_PARAMS


class TestDataUtils:
    """Tests for data utility functions."""

    def test_create_sample_data_shape(self):
        """Test that sample data has correct shape."""
        df = create_sample_data(n_samples=100, fraud_rate=0.1)
        assert df.shape == (100, 31)  # 28 V + Time + Amount + Class
        assert 'Class' in df.columns
        assert 'Amount' in df.columns
        assert 'Time' in df.columns

    def test_create_sample_data_fraud_rate(self):
        """Test that fraud rate is approximately correct."""
        df = create_sample_data(n_samples=1000, fraud_rate=0.05)
        actual_rate = df['Class'].mean()
        assert abs(actual_rate - 0.05) < 0.02  # Allow small variance

    def test_format_prediction_result(self):
        """Test prediction result formatting."""
        result = format_prediction_result(0.8)
        assert result['prediction'] == 1
        assert result['is_fraud'] == True
        assert result['risk_level'] == 'High'

        result = format_prediction_result(0.2)
        assert result['prediction'] == 0
        assert result['is_fraud'] == False
        assert result['risk_level'] == 'Low'

    def test_validate_input_data(self):
        """Test input validation."""
        df = create_sample_data(n_samples=10)
        expected = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
        assert validate_input_data(df, expected) == True

        with pytest.raises(ValueError):
            validate_input_data(df, ['NonExistentColumn'])

        bad_df = df.copy()
        bad_df['Time'] = 'not numeric'
        with pytest.raises(ValueError):
            validate_input_data(bad_df, expected)


class TestConfig:
    """Tests for configuration."""

    def test_rf_params(self):
        """Test that RF parameters are properly defined."""
        assert 'n_estimators' in RF_PARAMS
        assert 'max_depth' in RF_PARAMS
        assert 'class_weight' in RF_PARAMS
        assert RF_PARAMS['class_weight'] == 'balanced'
        assert RF_PARAMS['n_estimators'] == 200


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

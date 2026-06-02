"""
Configuration settings for the Credit Card Fraud Detection project.
All hyperparameters and file paths are centralized here.
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

# Dataset
DATASET_PATH = os.path.join(DATA_DIR, 'creditcard.csv')

# Model
MODEL_PATH = os.path.join(MODELS_DIR, 'fraud_detection_rf_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'robust_scaler.pkl')
FEATURES_PATH = os.path.join(MODELS_DIR, 'feature_names.pkl')

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Random Forest parameters
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'bootstrap': True,
    'oob_score': True
}

# SMOTE parameters
SMOTE_PARAMS = {
    'random_state': RANDOM_STATE,
    'sampling_strategy': 'auto',
    'k_neighbors': 5
}

# Features to drop (will be replaced by engineered features)
DROP_FEATURES = ['Amount', 'Time']

# Target column
TARGET_COLUMN = 'Class'

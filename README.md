# Credit Card Fraud Detection System

A production-ready machine learning application for detecting fraudulent credit card transactions using Random Forest classifier.

## Project Structure

```
CREDIT_CARD_FRAUD_DETECTION/
├── data/                       # Dataset storage
├── models/                     # Saved trained models
├── notebooks/                  # Jupyter notebooks for exploration
├── reports/                    # Analysis reports and visualizations
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── data_loader.py         # Data loading and validation
│   ├── preprocessor.py        # Data preprocessing pipeline
│   ├── model.py               # RandomForest model definition
│   ├── trainer.py             # Model training logic
│   ├── evaluator.py           # Model evaluation metrics
│   └── utils.py               # Utility functions
├── tests/                      # Unit tests
├── app.py                      # Streamlit interactive application
├── train.py                    # End-to-end training script
├── predict.py                  # Batch prediction script
├── requirements.txt
└── README.md
```

## Dataset

This project uses the **Credit Card Fraud Detection Dataset** from Kaggle:
- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions × 31 features
- **Fraud Rate:** 0.172% (492 frauds out of 284,807 transactions)
- **Features:** V1-V28 (PCA-transformed), Time, Amount, Class (target)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

Option A: Manual download from Kaggle and place `creditcard.csv` in the `data/` folder.

Option B: Use Kaggle API (requires API token):
```bash
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/
```

### 3. Train the Model

```bash
python train.py
```

This will:
- Load and validate the dataset
- Preprocess the data (scaling, feature engineering)
- Train a Random Forest classifier
- Evaluate the model with full metrics
- Save the trained model to `models/`
- Generate evaluation reports in `reports/`

### 4. Run Interactive App

```bash
streamlit run app.py
```

The Streamlit app allows you to:
- Upload transaction data for batch prediction
- Input individual transaction features manually
- View model performance metrics
- See feature importance rankings

### 5. Make Batch Predictions

```bash
python predict.py --input data/new_transactions.csv --output predictions.csv
```

## Model Architecture

- **Algorithm:** Random Forest Classifier
- **Key Parameters:**
  - n_estimators: 200
  - max_depth: 20
  - min_samples_split: 5
  - class_weight: 'balanced'
- **Preprocessing:** RobustScaler for numerical features
- **Imbalance Handling:** SMOTE for training data
- **Evaluation Metrics:** Precision, Recall, F1-Score, ROC-AUC, PR-AUC

## Performance

Expected performance on the test set:
- ROC-AUC: ~0.95-0.98
- Precision: ~0.85-0.95
- Recall: ~0.80-0.90
- F1-Score: ~0.82-0.92

## License

This project is for educational purposes as part of the TechCrush AI/ML Bootcamp Capstone Project.

## References

1. Kaggle. (2013). *Credit Card Fraud Detection Dataset*. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique.
3. Scikit-learn Documentation. https://scikit-learn.org/

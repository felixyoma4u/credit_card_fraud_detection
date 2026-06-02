import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import MODEL_PATH, SCALER_PATH, FEATURES_PATH
from src.preprocessor import FraudPreprocessor
from src.utils import create_sample_data, format_prediction_result

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_preprocessor():
    """Load model and preprocessor with caching for performance."""
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = FraudPreprocessor()
        preprocessor.load_artifacts()
        return model, preprocessor
    except FileNotFoundError:
        return None, None


def predict_single(features_dict, model, preprocessor):
    """Make prediction on a single transaction."""
    df = pd.DataFrame([features_dict])
    X = preprocessor.transform_new_data(df)
    prob = model.predict_proba(X)[0, 1]
    return format_prediction_result(prob)


def predict_batch(df, model, preprocessor):
    """Make predictions on a batch of transactions."""
    X = preprocessor.transform_new_data(df)
    probs = model.predict_proba(X)[:, 1]
    return probs


def main():
    # Header
    st.markdown('<div class="main-header">💳 Credit Card Fraud Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Fraud Detection System | TechCrush AI/ML Bootcamp Capstone</div>', unsafe_allow_html=True)

    # Load model
    model, preprocessor = load_model_and_preprocessor()

    if model is None:
        st.error("Model not found! Please train the model first by running: python train.py")
        return

    st.success("Model loaded successfully!")

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Mode:",
        ["Home", "Single Transaction Check", "Batch Prediction", "Model Insights"]
    )

    # Threshold slider
    threshold = st.sidebar.slider(
        "Fraud Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Adjust the probability threshold for classifying a transaction as fraud"
    )

    st.sidebar.markdown("---")
    st.sidebar.info("This application uses a Random Forest classifier trained on the Kaggle Credit Card Fraud Detection dataset.")

    # Home Page
    if page == "Home":
        st.header("Welcome to the Fraud Detection System")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Dataset Size", "284,807", "transactions")
        with col2:
            st.metric("Fraud Rate", "0.172%", "492 frauds")
        with col3:
            st.metric("Model Type", "Random Forest", "Ensemble")

        st.markdown("""
        ### How to Use This Application

        1. **Single Transaction Check**: Enter transaction details manually to get an instant fraud risk assessment.
        2. **Batch Prediction**: Upload a CSV file with multiple transactions for batch analysis.
        3. **Model Insights**: View model performance metrics and feature importance rankings.

        ### About the Model

        The fraud detection model is a **Random Forest Classifier** trained on the European credit card fraud dataset. 

        **Preprocessing Pipeline:**
        - Feature engineering (Hour of day, Log-transformed Amount)
        - RobustScaler normalization (resistant to outliers)
        - SMOTE for handling class imbalance

        **Model Parameters:**
        - 200 estimators (trees)
        - Max depth: 20
        - Min samples split: 5
        - Class weight: balanced
        """)

    # Single Transaction Check
    elif page == "Single Transaction Check":
        st.header("Single Transaction Fraud Check")
        st.markdown("Enter transaction details below to analyze fraud risk.")

        tab1, tab2 = st.tabs(["Manual Input (PCA Features)", "Use Sample Data"])

        with tab1:
            st.info("Enter the PCA-transformed features (V1-V28), Time, and Amount for the transaction.")

            col1, col2, col3 = st.columns(3)

            features = {}

            with col1:
                st.subheader("V1 - V10")
                for i in range(1, 11):
                    features[f'V{i}'] = st.number_input(f'V{i}', value=0.0, format="%.6f", key=f'v{i}')

            with col2:
                st.subheader("V11 - V20")
                for i in range(11, 21):
                    features[f'V{i}'] = st.number_input(f'V{i}', value=0.0, format="%.6f", key=f'v{i}')

            with col3:
                st.subheader("V21 - V28 + Meta")
                for i in range(21, 29):
                    features[f'V{i}'] = st.number_input(f'V{i}', value=0.0, format="%.6f", key=f'v{i}')

                features['Time'] = st.number_input('Time (seconds)', value=0.0, min_value=0.0, format="%.1f", key='time')
                features['Amount'] = st.number_input('Amount (€)', value=100.0, min_value=0.0, format="%.2f", key='amount')

            if st.button("Analyze Transaction", type="primary", use_container_width=True):
                with st.spinner("Analyzing transaction..."):
                    result = predict_single(features, model, preprocessor)

                st.markdown("---")
                st.subheader("Analysis Result")

                col_result1, col_result2, col_result3 = st.columns(3)

                with col_result1:
                    if result['is_fraud']:
                        st.error("FRAUD DETECTED")
                    else:
                        st.success("LEGITIMATE")

                with col_result2:
                    st.metric("Fraud Probability", f"{result['probability']:.2%}")

                with col_result3:
                    st.metric("Risk Level", result['risk_level'])

                st.progress(min(result['probability'], 1.0))

                if result['is_fraud']:
                    st.error(f"High Risk Transaction: {result['probability']:.2%} fraud probability. Recommendation: Flag for manual review.")
                else:
                    st.success(f"Low Risk Transaction: {result['probability']:.2%} fraud probability. Recommendation: Approve transaction.")

        with tab2:
            st.info("Generate a random sample transaction for testing.")

            sample_type = st.radio("Sample Type:", ["Legitimate-like", "Fraud-like"], horizontal=True)

            if st.button("Generate Sample", type="primary"):
                with st.spinner("Generating sample..."):
                    sample_df = create_sample_data(n_samples=1, fraud_rate=1.0 if sample_type == "Fraud-like" else 0.0)
                    sample_dict = sample_df.iloc[0].to_dict()

                st.write("Generated Sample:")
                st.json(sample_dict)

                result = predict_single(sample_dict, model, preprocessor)

                st.markdown("---")
                st.subheader("Prediction Result")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Prediction", "FRAUD" if result['is_fraud'] else "LEGITIMATE")
                with col2:
                    st.metric("Probability", f"{result['probability']:.2%}")

    # Batch Prediction
    elif page == "Batch Prediction":
        st.header("Batch Prediction")
        st.markdown("Upload a CSV file with multiple transactions for batch fraud detection.")

        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="CSV should contain columns: V1-V28, Time, Amount (and optionally Class)"
        )

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            st.write(f"Uploaded {len(df)} transactions")
            st.dataframe(df.head(10), use_container_width=True)

            if st.button("Run Batch Prediction", type="primary", use_container_width=True):
                with st.spinner("Processing transactions..."):
                    probs = predict_batch(df, model, preprocessor)
                    preds = (probs >= threshold).astype(int)

                results_df = df.copy()
                results_df['Fraud_Probability'] = probs
                results_df['Fraud_Prediction'] = preds
                results_df['Risk_Level'] = pd.cut(probs, bins=[0, 0.3, 0.7, 1.0], labels=['Low', 'Medium', 'High'])

                st.markdown("---")
                st.subheader("Prediction Summary")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total", len(results_df))
                with col2:
                    st.metric("Predicted Fraud", int(preds.sum()))
                with col3:
                    st.metric("Fraud Rate", f"{preds.mean()*100:.2f}%")
                with col4:
                    st.metric("Avg Probability", f"{probs.mean():.4f}")

                st.subheader("Fraud Probability Distribution")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.hist(probs, bins=50, color='#3498db', edgecolor='white', alpha=0.7)
                ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
                ax.set_xlabel('Fraud Probability')
                ax.set_ylabel('Count')
                ax.set_title('Distribution of Fraud Probabilities')
                ax.legend()
                st.pyplot(fig)

                st.subheader("Detailed Results")
                st.dataframe(results_df, use_container_width=True)

                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions CSV",
                    data=csv,
                    file_name="fraud_predictions.csv",
                    mime="text/csv"
                )

                high_risk = results_df[results_df['Risk_Level'] == 'High']
                if len(high_risk) > 0:
                    st.markdown("---")
                    st.subheader(f"High Risk Transactions ({len(high_risk)} found)")
                    st.dataframe(high_risk, use_container_width=True)

    # Model Insights
    elif page == "Model Insights":
        st.header("Model Performance Insights")

        reports_dir = os.path.join(os.path.dirname(__file__), 'reports')

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Confusion Matrix")
            cm_path = os.path.join(reports_dir, 'confusion_matrix.png')
            if os.path.exists(cm_path):
                st.image(cm_path, use_container_width=True)
            else:
                st.info("Train the model to generate evaluation reports.")

        with col2:
            st.subheader("ROC Curve")
            roc_path = os.path.join(reports_dir, 'roc_curve.png')
            if os.path.exists(roc_path):
                st.image(roc_path, use_container_width=True)
            else:
                st.info("Train the model to generate evaluation reports.")

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Precision-Recall Curve")
            pr_path = os.path.join(reports_dir, 'precision_recall_curve.png')
            if os.path.exists(pr_path):
                st.image(pr_path, use_container_width=True)
            else:
                st.info("Train the model to generate evaluation reports.")

        with col4:
            st.subheader("Feature Importance")
            fi_path = os.path.join(reports_dir, 'feature_importance.png')
            if os.path.exists(fi_path):
                st.image(fi_path, use_container_width=True)
            else:
                st.info("Train the model to generate evaluation reports.")

        st.markdown("---")
        st.subheader("Model Configuration")

        params = model.get_params()
        params_df = pd.DataFrame([
            {'Parameter': k, 'Value': str(v)} 
            for k, v in params.items()
        ])
        st.dataframe(params_df, use_container_width=True, hide_index=True)


if __name__ == '__main__':
    main()

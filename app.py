from src.utils import create_sample_data, format_prediction_result, validate_input_data
from src.preprocessor import FraudPreprocessor
from src.config import (
    MODEL_PATH,
    SCALER_PATH,
    FEATURES_PATH,
    SUMMARY_PATH,
    METRICS_PATH,
    REPORTS_DIR
)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    :root {
        --brand: #0a6cff;
        --brand-dark: #003a99;
        --bg: linear-gradient(180deg, #f0f4ff 0%, #ffffff 65%);
        --card: rgba(255,255,255,0.88);
        --card-border: rgba(12, 60, 130, 0.08);
        --muted: #5c6e8f;
        --accent: #ff8a65;
        --success: #0dbc79;
        --warning: #f5b301;
        --danger: #ff4b5c;
    }
    html, body, [class*="css"], .stApp {
        background: var(--bg) !important;
        font-family: "Manrope", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: #1b2431;
    }
    .stMainBlockContainer {
        padding-top: 4.5rem;
    }
    .glass {
        background: var(--card);
        border-radius: 18px;
        border: 1px solid var(--card-border);
        box-shadow: 0 25px 45px rgba(30, 41, 83, 0.08);
        backdrop-filter: blur(12px);
    }
    .hero {
        display: flex;
        justify-content: center;
        padding: 1.6rem 0 1.8rem 0;
    }
    .hero-card {
        max-width: 1120px;
        width: 100%;
        padding: 24px 32px;
        position: relative;
        overflow: hidden;
    }
    .hero-card::after {
        content: "";
        position: absolute;
        inset: -30% 40% auto -25%;
        height: 160%;
        background: radial-gradient(circle at center, rgba(10,108,255,0.18), transparent 60%);
        z-index: 0;
        transform: rotate(8deg);
    }
    .hero-title {
        position: relative;
        z-index: 1;
        font-size: 2.35rem;
        font-weight: 700;
        color: var(--brand-dark);
        letter-spacing: -0.6px;
    }
    .hero-subtitle {
        position: relative;
        z-index: 1;
        margin-top: 8px;
        font-size: 1.05rem;
        color: var(--muted);
        max-width: 720px;
    }
    .quick-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 16px;
        margin-top: 16px;
        position: relative;
        z-index: 1;
    }
    .stat-card {
        border-radius: 16px;
        padding: 16px 18px;
        background: rgba(255,255,255,0.85);
        border: 1px solid rgba(10,108,255,0.08);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.6);
    }
    .stat-label { font-size: 0.85rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
    .stat-value { font-size: 1.8rem; font-weight: 700; color: var(--brand-dark); margin-top: 6px; }
    .stat-sub { font-size: 0.9rem; color: var(--muted); margin-top: 2px; }
    .section {
        margin: 0 auto;
        max-width: 1120px;
        width: 100%;
        padding: 0 1.2rem;
    }
    .section-title {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
        color: var(--brand-dark);
    }
    .section-title span {
        font-size: 1.65rem;
    }
    .feature-grid {
        display: grid;
        gap: 16px;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        margin-top: 8px;
    }
    .feature-card {
        padding: 18px 20px;
        border-radius: 16px;
        background: rgba(255,255,255,0.7);
        border: 1px solid rgba(11,108,255,0.12);
        display: grid;
        gap: 10px;
    }
    .feature-icon {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        display: grid;
        place-items: center;
        font-size: 1.35rem;
        background: rgba(10,108,255,0.1);
        color: var(--brand-dark);
    }
    .feature-title { font-weight: 600; font-size: 1.05rem; }
    .feature-desc { color: var(--muted); font-size: 0.96rem; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 12px 12px 0 0;
        background: rgba(255,255,255,0.7);
        border: 1px solid rgba(10,108,255,0.1);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.5);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: rgba(10,108,255,0.15);
        border-bottom-color: transparent;
        color: var(--brand-dark);
    }
    .stButton>button {
        border-radius: 12px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        border: none;
        background: linear-gradient(135deg, var(--brand) 0%, #3a8dff 100%);
        box-shadow: 0 12px 24px rgba(10,108,255,0.26);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 18px 30px rgba(10,108,255,0.32);
    }
    .pill {
        border-radius: 999px;
        padding: 6px 12px;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
        background: rgba(10,108,255,0.12);
        color: var(--brand-dark);
    }
    .risk-low { color: var(--success); }
    .risk-medium { color: var(--warning); }
    .risk-high { color: var(--danger); }
    .stMetric > div {
        border-radius: 14px;
        padding: 10px;
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid rgba(16, 97, 184, 0.12);
        backdrop-filter: blur(6px);
    }
    .stDataFrame, .stImage {
        border-radius: 14px;
        border: 1px solid rgba(10,108,255,0.08);
        overflow: hidden;
    }
    .form-grid {
        display: grid;
        gap: 12px 16px;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        margin-top: 12px;
    }
    @media (max-width: 900px) {
        .form-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    @media (max-width: 640px) {
        .form-grid {
            grid-template-columns: repeat(1, minmax(0, 1fr));
        }
    }
    .footer-note {
        margin-top: 2rem;
        font-size: 0.85rem;
        color: var(--muted);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model_and_preprocessor():
    """Load model, preprocessor, and dataset summary with caching."""
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = FraudPreprocessor()
        preprocessor.load_artifacts()
        data_summary = joblib.load(
            SUMMARY_PATH) if os.path.exists(SUMMARY_PATH) else None
        metrics = joblib.load(METRICS_PATH) if os.path.exists(
            METRICS_PATH) else None
        return model, preprocessor, data_summary, metrics
    except FileNotFoundError:
        return None, None, None, None


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
    # Header (hero)
    st.markdown(
        '''
        <div class="hero">
            <div class="hero-content">
                <div class="title">💳 CardGuard</div>
                <div class="subtitle">Fraud Detection System — TechCrush AI/ML Bootcamp Capstone</div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )

    # Load model
    model, preprocessor, data_summary, metrics = load_model_and_preprocessor()

    if model is None:
        st.error(
            "Model not found! Please train the model first by running: python train.py")
        return

    st.success("Model loaded successfully!")

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Mode:",
        ["Overview", "Single Check", "Batch Analysis", "Model Insights"]
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
    st.sidebar.info(
        "This application uses a Random Forest classifier trained on the Kaggle Credit Card Fraud Detection dataset.")

    # Home Page
    if page == "Overview":
        with st.container():
            st.markdown("""
                <div class="section">
                    <div class="hero glass hero-card">
                        <div>
                            <div class="pill">Real-Time Risk Intelligence</div>
                            <div class="hero-title">CardGuard Fraud Detection Platform</div>
                            <div class="hero-subtitle">
                                Monitor and score transactions instantly. Explore model insights, tune thresholds, and ship decisions with confidence.
                            </div>
                            <div class="quick-stats">
                                <div class="stat-card">
                                    <div class="stat-label">Model</div>
                                    <div class="stat-value">Random Forest</div>
                                    <div class="stat-sub">Balanced ensemble</div>
                                </div>
                                <div class="stat-card">
                                    <div class="stat-label">Threshold</div>
                                    <div class="stat-value">{threshold:.2f}</div>
                                    <div class="stat-sub">adjustable in sidebar</div>
                                </div>
                                <div class="stat-card">
                                    <div class="stat-label">Status</div>
                                    <div class="stat-value">Deployed</div>
                                    <div class="stat-sub">Ready for scoring</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            """.format(threshold=threshold), unsafe_allow_html=True)

        st.markdown("""
            <div class="section glass" style="padding: 18px 24px; margin-top: 1rem;">
                <div class="section-title"><span>🧭</span> Workflow Shortcuts</div>
                <div class="feature-grid">
                    <div class="feature-card">
                        <div class="feature-icon">🔍</div>
                        <div class="feature-title">Single Transaction Check</div>
                        <div class="feature-desc">Manually key-in PCA features, time, and amount to assess fraud risk instantly.</div>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">📤</div>
                        <div class="feature-title">Batch Scoring</div>
                        <div class="feature-desc">Upload CSV files, review distribution plots, and download annotated predictions.</div>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">📈</div>
                        <div class="feature-title">Model Insights</div>
                        <div class="feature-desc">Inspect confusion matrix, ROC/PR curves, and feature importance to explain behavior.</div>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">⚙️</div>
                        <div class="feature-title">Threshold Tuning</div>
                        <div class="feature-desc">Use the sidebar to calibrate risk tolerance and observe precision/recall trade-offs.</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if data_summary:
            st.markdown("""
                <div class="section" style="margin-top: 1rem;">
                    <div class="section-title"><span>📊</span> Dataset Snapshot</div>
                </div>
            """, unsafe_allow_html=True)

            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Total Rows", f"{data_summary['shape'][0]:,}")
            col_b.metric("Features", f"{data_summary['shape'][1]:,}")
            fraud_rate = data_summary.get('fraud_rate', 0)
            col_c.metric("Fraud Rate", f"{fraud_rate*100:.3f}%")
            col_d.metric("Frauds", f"{data_summary.get('fraud_count', 0):,}")

            with st.expander("Explore dataset details", expanded=False):
                st.write({k: v for k, v in data_summary.items() if k not in {
                         'shape', 'fraud_rate', 'fraud_count'}})

        st.markdown("""
            <div class="section glass" style="padding: 18px 24px; margin-top: 1rem;">
                <div class="section-title"><span>🚀</span> Next Actions</div>
                <ol style="margin-left: 1rem; color: #3b4861;">
                    <li><strong>Audit thresholds:</strong> Move the sidebar slider to see how sensitivity shifts.</li>
                    <li><strong>Validate live data:</strong> Use “Batch Analysis” to score production samples.</li>
                    <li><strong>Share insights:</strong> Download plots under “Model Insights” for reporting.</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)

    # Single Transaction Check
    elif page == "Single Check":
        st.markdown("""
            <div class="section">
                <div class="section-title"><span>🔍</span> Single Transaction Check</div>
            </div>
        """, unsafe_allow_html=True)

        with st.container():
            tabs = st.tabs(["Manual Input", "Use Sample"])

        with tabs[0]:
            st.info(
                "Provide PCA features V1-V28 plus Time and Amount to score a single transaction.")

            features = {}
            with st.container():
                st.markdown("""
                    <div class="glass" style="padding:18px;">
                        <div class="section-title" style="font-size:1.1rem; margin-bottom:0;"><span>⚙️</span> Feature Entry</div>
                    </div>
                """, unsafe_allow_html=True)

            feature_groups = [
                ("V1 - V10", range(1, 11)),
                ("V11 - V20", range(11, 21)),
                ("V21 - V28", range(21, 29)),
            ]

            form_placeholder = st.empty()
            with form_placeholder.form("manual_input"):
                cols = st.columns(3)
                for (title, indices), col in zip(feature_groups, cols):
                    col.subheader(title)
                    for i in indices:
                        features[f'V{i}'] = col.number_input(
                            f'V{i}', value=0.0, format="%.6f")

                meta_col1, meta_col2 = st.columns(2)
                features['Time'] = meta_col1.number_input(
                    'Time (seconds)', value=0.0, min_value=0.0, format="%.1f")
                features['Amount'] = meta_col2.number_input(
                    'Amount (€)', value=25.0, min_value=0.0, format="%.2f")

                submitted = st.form_submit_button("Analyze Transaction")

            if submitted:
                with st.spinner("Analyzing transaction..."):
                    try:
                        result = predict_single(features, model, preprocessor)
                    except ValueError as exc:
                        st.error(f"Validation failed: {exc}")
                        result = None

                if result:
                    st.markdown("""
                        <div class="section" style="margin-top:1.2rem;">
                            <div class="section-title"><span>🧮</span> Risk Assessment</div>
                        </div>
                    """, unsafe_allow_html=True)

                    col_result1, col_result2, col_result3 = st.columns(3)

                    with col_result1:
                        if result['is_fraud']:
                            st.error("Fraud Detected")
                        else:
                            st.success("Legitimate")

                    with col_result2:
                        st.metric("Fraud Probability",
                                  f"{result['probability']:.2%}")

                    with col_result3:
                        st.metric("Risk Level", result['risk_level'])

                    st.progress(min(result['probability'], 1.0))

                    if result['is_fraud']:
                        st.error(
                            f"High Risk Transaction: {result['probability']:.2%} fraud probability. Recommendation: Flag for manual review.")
                    else:
                        st.success(
                            f"Low Risk Transaction: {result['probability']:.2%} fraud probability. Recommendation: Approve transaction.")

        with tabs[1]:
            st.info("Generate a synthetic transaction to test the pipeline quickly.")

            sample_type = st.radio(
                "Sample Type:", ["Legitimate-like", "Fraud-like"], horizontal=True)

            if st.button("Generate & Score Sample", type="primary"):
                with st.spinner("Creating sample transaction..."):
                    sample_df = create_sample_data(
                        n_samples=1, fraud_rate=1.0 if sample_type == "Fraud-like" else 0.0)
                    sample_dict = sample_df.iloc[0].to_dict()

                st.json(sample_dict)

                with st.spinner("Scoring sample..."):
                    result = predict_single(sample_dict, model, preprocessor)

                st.markdown("""
                    <div class="section" style="margin-top:1rem;">
                        <div class="section-title"><span>🧮</span> Prediction Result</div>
                    </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Prediction",
                              "FRAUD" if result['is_fraud'] else "LEGITIMATE")
                with col2:
                    st.metric("Probability", f"{result['probability']:.2%}")

    # Batch Prediction
    elif page == "Batch Analysis":
        st.markdown("""
            <div class="section">
                <div class="section-title"><span>📤</span> Batch Transaction Analysis</div>
            </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Required columns: V1-V28, Time, Amount. Optional: Class"
        )

        if uploaded_file is not None:
            with st.spinner("Validating CSV..."):
                df = pd.read_csv(uploaded_file)
                try:
                    expected = [f'V{i}' for i in range(
                        1, 29)] + ['Time', 'Amount']
                    validate_input_data(df, expected, allow_extra=True)
                except ValueError as exc:
                    st.error(f"Validation failed: {exc}")
                    return

            st.write(f"Uploaded {len(df)} transactions")
            st.dataframe(df.head(10), use_container_width=True)

            if st.button("Run Batch Prediction", type="primary", use_container_width=True):
                with st.spinner("Processing transactions..."):
                    probs = predict_batch(df, model, preprocessor)
                    preds = (probs >= threshold).astype(int)

                results_df = df.copy()
                results_df['Fraud_Probability'] = probs
                results_df['Fraud_Prediction'] = preds
                results_df['Risk_Level'] = pd.cut(
                    probs, bins=[0, 0.3, 0.7, 1.0], labels=['Low', 'Medium', 'High'])

                st.markdown("""
                    <div class="section" style="margin-top:1rem;">
                        <div class="section-title"><span>📊</span> Prediction Summary</div>
                    </div>
                """, unsafe_allow_html=True)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total", len(results_df))
                with col2:
                    st.metric("Predicted Fraud", int(preds.sum()))
                with col3:
                    st.metric("Fraud Rate", f"{preds.mean()*100:.2f}%")
                with col4:
                    st.metric("Avg Probability", f"{probs.mean():.4f}")

                st.markdown("""
                    <div class="section glass" style="padding: 18px 24px; margin-top: 1rem;">
                        <div class="section-title"><span>📈</span> Fraud Probability Distribution</div>
                    </div>
                """, unsafe_allow_html=True)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.hist(probs, bins=50, color='#3b82f6',
                        edgecolor='white', alpha=0.72)
                ax.axvline(x=threshold, color='#ef4444', linestyle='--',
                           linewidth=2, label=f'Threshold ({threshold})')
                ax.set_xlabel('Fraud Probability')
                ax.set_ylabel('Count')
                ax.set_title('Distribution of Fraud Probabilities')
                ax.legend()
                st.pyplot(fig)

                st.markdown("""
                    <div class="section" style="margin-top:1rem;">
                        <div class="section-title"><span>📑</span> Detailed Results</div>
                    </div>
                """, unsafe_allow_html=True)
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
                    st.markdown("""
                        <div class="section glass" style="padding:18px 24px; margin-top:1rem;">
                            <div class="section-title"><span>⚠️</span> High-Risk Transactions</div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.dataframe(high_risk, use_container_width=True)

    # Model Insights
    elif page == "Model Insights":
        st.markdown("""
            <div class="section">
                <div class="section-title"><span>📈</span> Model Performance Insights</div>
            </div>
        """, unsafe_allow_html=True)

        col_text, col_metrics = st.columns([2, 1])
        with col_text:
            st.markdown("""
                Monitor how the model balances fraud detection and false positives. Use the plots to explain performance to stakeholders and validate threshold decisions.
            """)
        with col_metrics:
            if metrics:
                st.metric("ROC-AUC", f"{metrics['ROC-AUC']:.3f}")
                st.metric("PR-AUC", f"{metrics['PR-AUC']:.3f}")
                st.metric("F1-Score", f"{metrics['F1-Score']:.3f}")
            else:
                st.info("Run training to populate metrics.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Confusion Matrix")
            cm_path = os.path.join(REPORTS_DIR, 'confusion_matrix.png')
            if os.path.exists(cm_path):
                st.image(cm_path, use_container_width=True)
            else:
                st.info("Train the model to generate evaluation reports.")

        with col2:
            st.subheader("ROC Curve")
            roc_path = os.path.join(REPORTS_DIR, 'roc_curve.png')
            if os.path.exists(roc_path):
                st.image(roc_path, use_container_width=True)
            else:
                st.info("Train the model to generate evaluation reports.")

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Precision-Recall Curve")
            pr_path = os.path.join(REPORTS_DIR, 'precision_recall_curve.png')
            if os.path.exists(pr_path):
                st.image(pr_path, use_container_width=True)
            else:
                st.info("Train the model to generate evaluation reports.")

        with col4:
            st.subheader("Feature Importance")
            fi_path = os.path.join(REPORTS_DIR, 'feature_importance.png')
            if os.path.exists(fi_path):
                st.image(fi_path, use_container_width=True)
            else:
                st.info("Train the model to generate evaluation reports.")

        st.markdown("""
            <div class="section" style="margin-top:1rem;">
                <div class="section-title"><span>⚙️</span> Model Configuration</div>
            </div>
        """, unsafe_allow_html=True)

        params = model.get_params()
        params_df = pd.DataFrame([
            {'Parameter': k, 'Value': str(v)}
            for k, v in params.items()
        ])
        st.dataframe(params_df, use_container_width=True, hide_index=True)


if __name__ == '__main__':
    main()

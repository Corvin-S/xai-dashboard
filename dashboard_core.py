"""
dashboard_core.py — Core module for the Credit Risk XAI Dashboard.

This module implements the main dashboard interface for the bachelor thesis
"Uncertainty Visualization in XAI Dashboards: Effects on User Trust".

The dashboard integrates three components for each credit application:
    1. Applicant details with human-readable feature labels
    2. Model prediction (Low Risk / High Risk) with confidence visualization
    3. SHAP-based local feature attribution explanation

Technical stack:
    - Streamlit (web framework)
    - scikit-learn (Random Forest classifier)
    - SHAP values (pre-computed locally via TreeExplainer)
    - Plotly (gauge chart and SHAP visualization)
"""

import streamlit as st
import numpy as np
import pandas as pd
from components.gauge_chart import show_gauge
from components.plain_text import show_text_confidence
from components.shap_plot import show_shap_chart
from components.feature_mapping import decode_case, FEATURE_LABELS


def run_dashboard(condition: str):
    """Run the credit risk dashboard with a fixed visualization condition.

    The condition parameter is fixed per app instance — participants
    cannot switch between formats. This supports the within-subject
    study design where each participant sees both conditions in
    separate sessions.

    Args:
        condition: Either "gauge" (semicircular gauge chart) or "text"
                   (plain text with verbal confidence label).
    """

    st.set_page_config(
        page_title="Credit Risk Dashboard",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    st.markdown("""
            <style>
                div[data-testid="stVerticalBlockBorderWrapper"] {
                    height: 100%;
                    min-height: 400px;
                }
                .main .block-container {
                    max-width: 100%;
                    padding-left: 2rem;
                    padding-right: 2rem;
                    padding-top: 1rem;
                }
            </style>
        """, unsafe_allow_html=True)

    @st.cache_resource
    def load_model():
        """Train the Random Forest classifier and load pre-computed SHAP values.

        The model is trained at runtime for cross-platform compatibility.
        SHAP values are pre-computed locally and loaded from file to avoid
        the SHAP C-extension dependency in cloud environments.

        Returns:
            rf: Trained RandomForestClassifier (100 trees)
            shap_values: Pre-computed SHAP values for all test instances
            X_test: Test set features (200 instances, 20 features)
            y_test: Test set labels (0 = good risk, 1 = bad risk)
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        col_names = [
            "checking_account", "duration", "credit_history", "purpose",
            "credit_amount", "savings_account", "employment", "installment_rate",
            "personal_status", "other_debtors", "residence_since", "property",
            "age", "other_installments", "housing", "existing_credits",
            "job", "liable_people", "telephone", "foreign_worker", "target"
        ]

        df = pd.read_csv("data/german.data", sep=" ", header=None, names=col_names)
        df["target"] = df["target"].map({1: 0, 2: 1})

        cat_cols = df.select_dtypes(include="object").columns.tolist()
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

        X = df.drop(columns=["target"])
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        shap_values = np.load("model/shap_values.npy")

        return rf, shap_values, X_test, y_test

    rf, shap_values, X_test, y_test = load_model()

    # ── Case Selection ────────────────────────────────────────────
    # Stratified, disjoint case blocks for the within-subject design.
    # Each block contains 2 LOW, 2 MED, 2 HIGH confidence cases to
    # ensure comparable distributions across conditions.
    GAUGE_CASES = [3, 65, 37, 4, 15, 10]
    TEXT_CASES = [47, 7, 61, 6, 17, 12]

    case_list = GAUGE_CASES if condition == "gauge" else TEXT_CASES

    if "case_idx" not in st.session_state:
        st.session_state.case_idx = 0

    # Map session index to actual test set index
    idx = case_list[st.session_state.case_idx]
    case = X_test.iloc[idx]

    proba = rf.predict_proba(case.values.reshape(1, -1))[0]
    pred = np.argmax(proba)
    confidence = proba[pred]
    pred_label = "Low Risk" if pred == 0 else "High Risk"

    # ── Dashboard Header ──────────────────────────────────────────
    st.title("Credit Risk Dashboard")
    st.caption(
        "This dashboard presents individual credit risk predictions generated by a "
        "Random Forest classifier trained on the German Credit Dataset (Hofmann, 1994). "
        "Each prediction is accompanied by a SHAP-based local explanation and an "
        "uncertainty visualization showing the model's confidence."
    )
    st.markdown("---")

    # ── Applicant Box ─────────────────────────────────────────────
    with st.container(border=True):
        nav1, nav2, nav3 = st.columns([4, 1, 1])
        with nav1:
            st.markdown(f"### Applicant {st.session_state.case_idx + 1} of {len(case_list)}")
        with nav2:
            if st.button("← Previous", use_container_width=True):
                st.session_state.case_idx = max(0, st.session_state.case_idx - 1)
        with nav3:
            if st.button("Next →", use_container_width=True):
                st.session_state.case_idx = min(len(case_list) - 1, st.session_state.case_idx + 1)

        st.markdown("---")

        decoded = decode_case(case)
        sections = {
            "Financial": [
                "Checking Account", "Savings Account", "Credit History",
                "Credit Amount (DM)", "Other Installments",
                "Existing Credits", "Installment Rate (%)"
            ],
            "Personal": [
                "Age (years)", "Personal Status", "Employment",
                "Job", "Liable People", "Telephone",
                "Foreign Worker", "Residence Since (years)"
            ],
            "Loan": [
                "Duration (months)", "Purpose", "Other Debtors",
                "Property", "Housing"
            ]
        }

        sec_cols = st.columns(3)
        for col, (section_name, features) in zip(sec_cols, sections.items()):
            with col:
                st.markdown(f"**{section_name}**")
                for feat in features:
                    if feat in decoded:
                        st.markdown(
                            f"<span style='color:#4A90D9; font-size:13px;'>{feat}:</span> "
                            f"<span style='font-size:13px;'>{decoded[feat]}</span>",
                            unsafe_allow_html=True
                        )

    st.markdown("---")

    # ── Prediction + Confidence ───────────────────────────────────
    pred_col, conf_col = st.columns([1, 2])

    with pred_col:
        with st.container(border=True):
            st.subheader("Model Prediction")
            st.markdown(f"## {pred_label}")
            st.caption(
                "The model predicts whether this applicant represents a "
                "low or high credit risk based on the features above."
            )

    with conf_col:
        with st.container(border=True):
            st.subheader("Model Confidence")
            st.caption(
                "The confidence score shows how certain the model is about "
                "this prediction. A higher value means the model is more "
                "sure that its assessment is correct."
            )

            if condition == "gauge":
                fig = show_gauge(confidence)
                st.plotly_chart(fig, use_container_width=True)
            else:
                show_text_confidence(confidence)

    st.markdown("---")

    # ── SHAP Explanation ──────────────────────────────────────────
    with st.expander("Prediction Explanation (SHAP) — click to expand"):
        st.caption(
            "The chart below shows which features had the most impact on this "
            "prediction and how much each feature contributed in percent."
        )

        readable_feature_names = [FEATURE_LABELS.get(f, f) for f in X_test.columns]

        case_shap = shap_values[idx, :, pred]

        fig = show_shap_chart(
            shap_values=case_shap,
            feature_names=readable_feature_names,
            pred_label=pred_label,
            max_display=10
        )
        st.plotly_chart(fig, use_container_width=True)
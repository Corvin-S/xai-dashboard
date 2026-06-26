"""
app_onboarding.py — Step-by-step onboarding module for the user study.

Introduces participants to the dashboard components one at a time
before the actual measurement begins. The confidence display is
deliberately shown as a placeholder to avoid priming participants
with either the gauge or text format before the study.

All data (features, prediction, SHAP values) is hardcoded from a
pre-computed example case to avoid runtime dependencies on SHAP
or model training, ensuring the module works in cloud environments.
"""

import streamlit as st
import numpy as np
from components.shap_plot import show_shap_chart
from components.feature_mapping import FEATURE_LABELS

# ── Page Configuration ────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Dashboard — Introduction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
        div[data-testid="stVerticalBlockBorderWrapper"] {
            height: 100%;
            min-height: 400px;
        }
    </style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────
if "step" not in st.session_state:
    st.session_state.step = 1

step = st.session_state.step

# ── Pre-computed Example Case ─────────────────────────────────────
# Hardcoded values from Applicant 1 in the test set to avoid
# any runtime dependency on the model or SHAP library.
example_features = {
    "Financial": {
        "Checking Account": "< 0 DM",
        "Savings Account": "500 - 1000 DM",
        "Credit History": "Existing Credits Paid Back",
        "Credit Amount (DM)": 1913,
        "Other Installments": "Bank",
        "Existing Credits": 1,
        "Installment Rate (%)": 3
    },
    "Personal": {
        "Age (years)": 36,
        "Personal Status": "Male: Married / Widowed",
        "Employment": "< 1 Year",
        "Job": "Skilled Employee",
        "Liable People": 1,
        "Telephone": "Yes",
        "Foreign Worker": "No",
        "Residence Since (years)": 3
    },
    "Loan": {
        "Duration (months)": 18,
        "Purpose": "Business",
        "Other Debtors": "None",
        "Property": "Real Estate",
        "Housing": "Own"
    }
}

example_pred_label = "Low Risk"

# Pre-computed SHAP values for this case (from shap_values.npy)
example_shap = np.array([
    0.02, -0.03, 0.01, -0.015, 0.005,
    0.08, -0.06, 0.025, -0.01, 0.005,
    0.015, 0.03, 0.02, -0.055, 0.01,
    0.005, 0.015, -0.005, 0.01, -0.005
])

example_feature_names = [
    "Checking Account", "Duration (months)", "Credit History", "Purpose",
    "Credit Amount (DM)", "Savings Account", "Employment", "Installment Rate (%)",
    "Personal Status", "Other Debtors", "Residence Since (years)", "Property",
    "Age (years)", "Other Installments", "Housing", "Existing Credits",
    "Job", "Liable People", "Telephone", "Foreign Worker"
]

# ── Header ────────────────────────────────────────────────────────
st.title("Credit Risk Dashboard — Introduction")
st.caption(
    "This introduction will guide you through the dashboard step by step. "
    "Each element will be explained before you begin the study."
)
st.progress(step / 5)
st.markdown("---")

# ── Step 1+: Applicant Features ───────────────────────────────────
if step >= 1:
    with st.container(border=True):
        st.markdown("### Example Applicant")
        st.markdown("---")

        sec_cols = st.columns(3)
        for col, (section_name, features) in zip(sec_cols, example_features.items()):
            with col:
                st.markdown(f"**{section_name}**")
                for feat, value in features.items():
                    st.markdown(
                        f"<span style='color:#4A90D9; font-size:13px;'>{feat}:</span> "
                        f"<span style='font-size:13px;'>{value}</span>",
                        unsafe_allow_html=True
                    )

    if step == 1:
        st.info(
            "**Merkmale des Kreditantragstellers**\n\n"
            "Hier sehen Sie die Merkmale eines Kreditantragstellers, aufgeteilt in "
            "drei Kategorien: finanzielle Informationen, persönliche Angaben und "
            "Kreditdetails. Diese Merkmale bilden die Grundlage für die Vorhersage "
            "des Modells."
        )

st.markdown("---")

# ── Step 2+: Prediction ──────────────────────────────────────────
if step >= 2:
    pred_col, conf_col = st.columns([1, 2])

    with pred_col:
        with st.container(border=True):
            st.subheader("Model Prediction")
            st.markdown(f"## {example_pred_label}")
            st.caption(
                "The model predicts whether this applicant represents a "
                "low or high credit risk based on the features above."
            )

    # Step 4+: Confidence placeholder or actual placeholder
    with conf_col:
        if step >= 4:
            with st.container(border=True):
                st.subheader("Model Confidence")
                st.markdown(
                    """
                    <div style="
                        background-color: #2a2a2a;
                        border: 2px dashed #666666;
                        border-radius: 8px;
                        padding: 60px 20px;
                        text-align: center;
                        margin: 10px 0;
                    ">
                        <p style="font-size: 18px; color: #999999; margin: 0;">
                            Hier wird Ihnen gleich angezeigt, wie sicher sich das
                            Modell bei seiner Vorhersage ist.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.empty()

    if step == 2:
        st.info(
            "**Modellvorhersage**\n\n"
            "Das Modell bewertet den Kreditantragsteller als **Low Risk** (geringes Risiko) "
            "oder **High Risk** (hohes Risiko). Diese Einschätzung basiert auf den "
            "Merkmalen des Antragstellers, die links dargestellt sind."
        )

# ── Step 3+: SHAP Explanation ─────────────────────────────────────
if step >= 3:
    with st.expander("Prediction Explanation (SHAP) — click to expand", expanded=True):
        st.caption(
            "The chart below shows which features had the most impact on this "
            "prediction and how much each feature contributed in percent."
        )

        fig = show_shap_chart(
            shap_values=example_shap,
            feature_names=example_feature_names,
            pred_label=example_pred_label,
            max_display=10
        )
        st.plotly_chart(fig, use_container_width=True)

    if step == 3:
        st.info(
            "**Erklärung der Vorhersage (SHAP)**\n\n"
            "Dieses Diagramm zeigt, welche Merkmale den größten Einfluss auf die "
            "Vorhersage hatten.\n\n"
            "- **Grüne Balken** unterstützen die Vorhersage — sie sprechen dafür, "
            "dass der Antragsteller das vorhergesagte Risikoprofil hat.\n"
            "- **Rote Balken** sprechen dagegen — sie deuten in die entgegengesetzte "
            "Richtung.\n"
            "- **Längere Balken** bedeuten einen stärkeren Einfluss auf die Vorhersage.\n\n"
            "Die Prozentwerte zeigen, wie viel jedes Merkmal relativ zur Gesamtvorhersage beigetragen hat."
        )

# ── Step 4: Confidence Placeholder Info ───────────────────────────
if step == 4:
    st.info(
        "**Modellkonfidenz (Platzhalter)**\n\n"
        "An der markierten Stelle wird Ihnen gleich angezeigt, wie sicher sich das "
        "Modell bei seiner Vorhersage ist. Diese Darstellung ist der zentrale "
        "Gegenstand der Studie — Sie werden sie anschließend bewerten.\n\n"
        "Die Darstellungsform wird Ihnen in der Studie gezeigt."
    )

# ── Step 5: Transition ────────────────────────────────────────────
if step == 5:
    st.success(
        "**Sie haben alle Elemente des Dashboards kennengelernt!**\n\n"
        "In der Studie sehen Sie mehrere Kreditfälle. Für jeden Fall zeigt das "
        "Dashboard die gleichen Elemente, die Sie gerade kennengelernt haben — "
        "zusätzlich wird die Modellkonfidenz in einer bestimmten Darstellungsform "
        "angezeigt. Bitte bewerten Sie diese nach jeder Runde im Fragebogen."
    )

# ── Navigation Button ─────────────────────────────────────────────
st.markdown("---")
if step < 5:
    if st.button("Weiter →", use_container_width=True, type="primary"):
        st.session_state.step += 1
        st.rerun()
else:
    if st.button("Studie starten →", use_container_width=True, type="primary"):
        st.markdown(
            """
            <meta http-equiv="refresh" content="0; url=https://YOUR-GAUGE-LINK.streamlit.app">
            """,
            unsafe_allow_html=True
        )
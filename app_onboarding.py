"""
app_onboarding.py — Step-by-step onboarding module for the user study.

Introduces participants to the dashboard components one at a time
before the actual measurement begins. The confidence display is
deliberately shown as a placeholder to avoid priming participants
with either the gauge or text format before the study.
"""

import streamlit as st
import numpy as np
import pickle
from components.shap_plot import show_shap_chart
from components.feature_mapping import decode_case, FEATURE_LABELS

# ── Page Configuration ────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Dashboard — Einleitung",
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

# ── Load pre-computed data (no SHAP library needed) ───────────────
@st.cache_resource
def load_example():
    with open("model/X_test.pkl", "rb") as f:
        X_test = pickle.load(f)
    shap_values = np.load("model/shap_values.npy")
    return X_test, shap_values

X_test, shap_values = load_example()

idx = 0
case = X_test.iloc[idx]
pred = 0
pred_label = "Low Risk"

# ── Session State ─────────────────────────────────────────────────
if "step" not in st.session_state:
    st.session_state.step = 1

step = st.session_state.step

# ── Header ────────────────────────────────────────────────────────
st.title("Credit Risk Dashboard — Einleitung")
st.caption(
    "This introduction will guide you through the dashboard step by step. "
    "Each element will be explained before you begin the study."
)
st.progress(step / 7)
st.markdown("---")

# ── Step 1: Use Case Introduction ─────────────────────────────────
if step == 1:
    st.markdown("### Ihr Szenario")

    st.info(
        "**Stellen Sie sich vor, Sie arbeiten als Kreditsachbearbeiter bei einer Bank.**\n\n"
        "Ihre Aufgabe: Sie prüfen Kreditanträge und entscheiden, ob ein Antrag "
        "**bewilligt** oder **abgelehnt** wird.\n\n"
        "Um Sie bei dieser Entscheidung zu unterstützen, steht Ihnen ein "
        "**KI-gestütztes Dashboard** zur Verfügung. Für jeden Antragsteller zeigt "
        "das Dashboard drei Informationen:\n\n"
        "1. **Die Merkmale des Antragstellers** — z. B. Einkommen, Alter, "
        "bestehende Kredite\n"
        "2. **Die Vorhersage des KI-Modells** — ob der Antragsteller ein "
        "niedriges oder hohes Kreditrisiko darstellt\n"
        "3. **Wie sicher sich das Modell bei dieser Vorhersage ist** — "
        "die sogenannte Modellkonfidenz\n\n"
        "Das Dashboard ist ein Hilfsmittel — **die Entscheidung liegt bei Ihnen.** "
        "Das Modell kann sich irren. Die Konfidenzanzeige zeigt Ihnen, "
        "wie sehr Sie sich auf die Einschätzung des Modells verlassen können.\n\n"
        "In der folgenden Einführung lernen Sie die einzelnen Elemente des "
        "Dashboards Schritt für Schritt kennen."
    )

# ── Step 2+: Applicant Features ───────────────────────────────────
if step >= 2:
    with st.container(border=True):
        nav1, nav2, nav3 = st.columns([4, 1, 1])
        with nav1:
            st.markdown("### Example Applicant")
        if step >= 3:
            with nav2:
                st.button("← Previous", disabled=True, use_container_width=True)
            with nav3:
                st.button("Next →", disabled=True, use_container_width=True)

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

if step == 2:
    st.info(
        "**Merkmale des Kreditantragstellers**\n\n"
        "Hier sehen Sie die Merkmale eines Kreditantragstellers, aufgeteilt in "
        "drei Kategorien: finanzielle Informationen, persönliche Angaben und "
        "Kreditdetails. Diese Merkmale bilden die Grundlage für die Vorhersage "
        "des Modells."
    )

if step == 3:
    st.info(
        "**Navigation zwischen den Kreditfällen**\n\n"
        "In der Studie sehen Sie sechs Kreditfälle pro Runde. "
        "Mit den Schaltflächen **← Previous** und **Next →** oben rechts können Sie "
        "zwischen den einzelnen Fällen hin- und herwechseln.\n\n"
        "Bitte klicken Sie sich durch **alle sechs Fälle**, bevor "
        "Sie zum Fragebogen zurückkehren."
    )

st.markdown("---")

# ── Step 4+: Prediction + Confidence area ─────────────────────────
if step >= 4:
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
        if step >= 6:
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

if step == 4:
    st.info(
        "**Modellvorhersage**\n\n"
        "Das Modell bewertet den Kreditantragsteller als **Low Risk** (geringes Risiko) "
        "oder **High Risk** (hohes Risiko). Diese Einschätzung basiert auf den "
        "Merkmalen des Antragstellers, die oben dargestellt sind."
    )
elif step == 6:
    st.info(
        "**Modellkonfidenz (Platzhalter)**\n\n"
        "An der markierten Stelle wird Ihnen gleich angezeigt, wie sicher sich das "
        "Modell bei seiner Vorhersage ist. Diese Darstellung ist der zentrale "
        "Gegenstand der Studie — Sie werden sie anschließend bewerten.\n\n"
        "Die Konfidenz des Modells bewegt sich immer zwischen 50 % und 100 %. "
        "Ein Wert von 50 % bedeutet, dass das Modell maximal unsicher ist — es hat "
        "praktisch geraten. Ein Wert nahe 100 % bedeutet, dass sich das Modell bei "
        "seiner Einschätzung sehr sicher ist. Werte unter 50 % sind bei diesem Modell "
        "nicht möglich, da es sich immer für die wahrscheinlichere Klasse entscheidet.\n\n"
        "Es werden Ihnen zwei unterschiedliche Visualisierungsformen gezeigt. "
        "Bei einer davon wird der Bereich unter 50 % grau dargestellt, um zu verdeutlichen, "
        "dass dieser Bereich vom Modell nicht erreicht werden kann. "
        "Die eigentliche Konfidenz beginnt erst ab 50 % und wird farblich hervorgehoben:\n\n"
        "- **Rot** (50–67 %): niedrige Konfidenz\n"
        "- **Orange** (67–83 %): mittlere Konfidenz\n"
        "- **Grün** (83–100 %): hohe Konfidenz"
    )

# ── Step 5+: SHAP Explanation ─────────────────────────────────────
if step >= 5:
    st.markdown("---")

    with st.expander("Prediction Explanation (SHAP) — click to expand", expanded=True):
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

if step == 5:
    st.info(
        "**Erklärung der Vorhersage (SHAP)**\n\n"
        "Dieses Diagramm zeigt, welche Merkmale den größten Einfluss auf die "
        "Vorhersage hatten. Hierbei werden die wichtigsten **zehn** Merkmale für "
        "die Vorhersage angezeigt.\n\n"
        "- **Rote Balken** erhöhen die Vorhersage\n"
        "- **Blaue Balken** senken die Vorhersage\n"
        "- **Längere Balken** bedeuten einen stärkeren Einfluss auf die Vorhersage.\n\n"
        "Der SHAP-Wert zeigt, wie stark ein einzelnes Merkmal die Vorhersage für "
        "diesen konkreten Antragsteller beeinflusst hat — im Vergleich zur "
        "durchschnittlichen Vorhersage über alle Antragsteller. Ein hoher Wert "
        "bedeutet einen höheren Einfluss.\n\n"
        "Wenn Sie mit der Maus über einen Balken fahren, sehen Sie den genauen "
        "SHAP-Wert und den relativen prozentualen Beitrag zur Gesamtvorhersage.\n\n"
        "Für die Durchführung der Studie wird diese Ansicht zunächst zugeklappt sein. "
        "Ihnen ist überlassen, ob Sie diese nutzen oder nicht."
    )

# ── Step 7: Transition ────────────────────────────────────────────
if step == 7:
    st.success(
        "**Sie haben alle Elemente des Dashboards kennengelernt!**\n\n"
        "In der Studie sehen Sie mehrere Kreditfälle. Für jeden Fall zeigt das "
        "Dashboard die gleichen Elemente, die Sie gerade kennengelernt haben — "
        "zusätzlich wird die Modellkonfidenz in einer bestimmten Darstellungsform "
        "angezeigt. Bitte bewerten Sie diese nach jeder Runde im Fragebogen."
    )

# ── Navigation Button ─────────────────────────────────────────────
st.markdown("---")
if step < 7:
    if st.button("Weiter →", use_container_width=True, type="primary"):
        st.session_state.step += 1
        st.rerun()
else:
    st.markdown("### Bitte wählen Sie Ihre Gruppe:")
    group_col1, group_col2 = st.columns(2)

    with group_col1:
        st.link_button(
            "Gruppe 1 — Studie starten →",
            "https://docs.google.com/forms/d/e/1FAIpQLSdUEdIAYjKTsuBkX0YaLr_-lzM5ZQxUNaZJc1gy6HggNKRPlA/viewform?usp=dialog",
            use_container_width=True,
            type="primary"
        )

    with group_col2:
        st.link_button(
            "Gruppe 2 — Studie starten →",
            "https://docs.google.com/forms/d/e/1FAIpQLSd5xQp_RLAludkGZLXEknoxMJ2TTYYpNCI83x72Z8vi52O7og/viewform?usp=publish-editor",
            use_container_width=True,
            type="primary"
        )
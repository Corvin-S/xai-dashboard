import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ── Modell & Daten laden ─────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model/rf_model.pkl", "rb") as f:
        rf = pickle.load(f)
    with open("model/shap_explainer.pkl", "rb") as f:
        explainer = pickle.load(f)
    with open("model/X_test.pkl", "rb") as f:
        X_test = pickle.load(f)
    with open("model/y_test.pkl", "rb") as f:
        y_test = pickle.load(f)
    return rf, explainer, X_test, y_test

rf, explainer, X_test, y_test = load_model()

# ── Session State ────────────────────────────────────────────────
if "case_idx" not in st.session_state:
    st.session_state.case_idx = 0

# ── Header ───────────────────────────────────────────────────────
st.title("Kreditrisiko-Dashboard")
st.markdown("---")

# ── Case Navigation ──────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("← Zurück"):
        st.session_state.case_idx = max(0, st.session_state.case_idx - 1)
with col2:
    st.markdown(f"**Antragsteller {st.session_state.case_idx + 1} von {len(X_test)}**")
with col3:
    if st.button("Weiter →"):
        st.session_state.case_idx = min(len(X_test) - 1, st.session_state.case_idx + 1)

st.markdown("---")

# ── Aktueller Case ───────────────────────────────────────────────
idx = st.session_state.case_idx
case = X_test.iloc[idx]

# ── Prediction ───────────────────────────────────────────────────
proba = rf.predict_proba(case.values.reshape(1, -1))[0]
pred = np.argmax(proba)
confidence = proba[pred]

pred_label = "✅ Geringes Risiko" if pred == 0 else "❌ Hohes Risiko"
st.subheader(f"Vorhersage: {pred_label}")
st.metric("Modell-Konfidenz", f"{confidence:.0%}")

st.markdown("---")

# ── Antragsteller Details ────────────────────────────────────────
st.subheader("Antragsteller-Details")
st.dataframe(case.to_frame().T, use_container_width=True)

st.markdown("---")

# ── SHAP Erklärung ───────────────────────────────────────────────
st.subheader("Erklärung der Vorhersage")
shap_values = explainer.shap_values(case.values.reshape(1, -1))
fig, ax = plt.subplots(figsize=(10, 5))
shap.bar_plot(shap_values[0, :, pred], feature_names=list(X_test.columns), max_display=10, show=False)
st.pyplot(fig)
plt.close()

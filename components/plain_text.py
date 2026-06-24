"""
plain_text.py — Plain text visualization for model confidence.

Displays the model's predictive confidence as a styled text box
with a verbal confidence label (low/medium/high).
Confidence levels are based on three equal thirds of the 50-100% range:
    Low:    50-67%
    Medium: 67-83%
    High:   83-100%
"""

import streamlit as st


def show_text_confidence(confidence: float):
    percent = int(confidence * 100)

    if percent >= 83:
        level = "high"
        color = "#44bb44"
    elif percent >= 67:
        level = "medium"
        color = "#ffaa00"
    else:
        level = "low"
        color = "#ff4444"

    st.markdown(
        f"""
        <div style="
            background-color: #1e1e1e;
            border-left: 6px solid {color};
            padding: 20px;
            border-radius: 8px;
            margin: 10px 0;
        ">
            <p style="font-size: 16px; color: #cccccc; margin: 0;">Model Confidence</p>
            <p style="font-size: 42px; font-weight: bold; color: {color}; margin: 5px 0;">
                {percent}%
            </p>
            <p style="font-size: 16px; color: #cccccc; margin: 0;">
                The model is <strong style="color:{color};">{level}</strong> confident in this prediction.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

import streamlit as st

def show_text_confidence(confidence: float):
    percent = int(confidence * 100)

    if percent >= 86.7:
        level = "high"
        color = "#44bb44"
    elif percent >= 66.6:
        level = "medium"
        color = "#ffaa00"
    else:
        level = "low"
        color = "#ff4444"

    st.markdown(
        f"""
        <div style="
            background-color: #ffffff;
            border-left: 6px solid {color};
            padding: 20px;
            border-radius: 8px;
            margin: 10px 0;
        ">
            <p style="font-size: 42px; font-weight: bold; color: {color}; margin: 5px 0;">
                {percent}%
            </p>
            <p style="font-size: 16px; color: ##000000; margin: 0;">
                The model is <strong style="color:{color};">{level}</strong> confident in this prediction.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
"""
gauge_chart.py — Gauge chart visualization for model confidence.

Displays the model's predictive confidence as a semicircular gauge.
The scale starts at 50% because a binary Random Forest classifier
always assigns the predicted class a probability of at least 50%.
The 50-100% range is divided into three equal thirds:
    Low:    50-67%
    Medium: 67-83%
    High:   83-100%
"""

import plotly.graph_objects as go


def show_gauge(confidence: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={"text": "Model Confidence", "font": {"size": 20}},
        number={"suffix": "%", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "dtick": 10, "tick0": 0},
            "bar": {"color": "#1f77b4"},
            "steps": [
                {"range": [0, 50], "color": "#808080 "},
                {"range": [50, 67], "color": "#ff4444"},
                {"range": [67, 83], "color": "#ffaa00"},
                {"range": [83, 100], "color": "#44bb44"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": confidence * 100,
            },
        },
    ))
    fig.update_layout(height=350, margin=dict(t=50, b=30, l=50, r=50))
    return fig
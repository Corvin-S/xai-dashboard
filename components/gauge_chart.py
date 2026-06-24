"""
app_gauge.py — Entry point for the gauge chart condition.

Launches the Credit Risk Dashboard with the gauge chart as the
fixed uncertainty visualization. Participants using this version
see only the gauge and cannot switch to the text condition.
"""

from results.dashboard_core import run_dashboard

run_dashboard(condition="gauge")
import plotly.graph_objects as go

def show_gauge(confidence: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={"text": "Model Confidence", "font": {"size": 20}},
        number={"suffix": "%", "font": {"size": 36}},
        gauge={
            "axis": {"range": [50, 100], "tickwidth": 1},
            "bar": {"color": "#1f77b4"},
            "steps": [
                {"range": [50, 66.7], "color": "#ff4444"},
                {"range": [66.7, 83.3], "color": "#ffaa00"},
                {"range": [83.3, 100], "color": "#44bb44"},
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
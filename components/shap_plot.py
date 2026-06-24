"""
shap_plot.py — Custom SHAP visualization

Converts raw SHAP values into percentage-based feature contributions
and renders them as a horizontal Plotly bar chart. This representation
is more accessible to non-expert users than the default SHAP bar plot,
which displays abstract decimal values on the x-axis.

The raw SHAP values remain accessible via hover tooltip to preserve
technical transparency alongside the simplified percentage view.
"""

import plotly.graph_objects as go
import numpy as np


def show_shap_chart(shap_values, feature_names, pred_label, max_display=10):
    """Display a percentage-based SHAP feature importance chart.

    Each feature's contribution is expressed as a percentage of the
    total absolute SHAP impact, making it intuitively interpretable
    for lay users (e.g., "Checking Account contributes 35% to this
    prediction"). The raw SHAP value is shown on hover for technical
    transparency.

    Args:
        shap_values: 1D array of SHAP values for the predicted class.
        feature_names: List of human-readable feature names.
        pred_label: The predicted class label ("Low Risk" / "High Risk").
        max_display: Number of top features to display (default: 10).
    """
    # Calculate percentage contribution per feature
    abs_values = np.abs(shap_values)
    total = abs_values.sum()
    percentages = (abs_values / total) * 100

    # Sort by absolute impact and select top features
    sorted_idx = np.argsort(abs_values)[::-1][:max_display]
    # Reverse for bottom-to-top display (most important at top)
    sorted_idx = sorted_idx[::-1]

    top_names = [feature_names[i] for i in sorted_idx]
    top_percentages = [percentages[i] for i in sorted_idx]
    top_shap = [shap_values[i] for i in sorted_idx]

    # Color based on direction: green supports prediction, red opposes it
    colors = [
        "#2ecc71" if v >= 0 else "#e74c3c"
        for v in top_shap
    ]

    # Hover labels: direction relative to predicted class
    hover_directions = [
        f"Supports {pred_label}" if v >= 0 else f"Opposes {pred_label}"
        for v in top_shap
    ]

    fig = go.Figure(go.Bar(
        x=top_percentages,
        y=top_names,
        orientation="h",
        marker=dict(color=colors),
        text=[f"{p:.0f}%" for p in top_percentages],
        textposition="outside",
        textfont=dict(size=12),
        customdata=list(zip(top_shap, hover_directions)),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Contribution: %{x:.1f}%<br>"
            "SHAP Value: %{customdata[0]:.4f}<br>"
            "%{customdata[1]}"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        xaxis_title="Impact on Prediction (%)",
        yaxis_title="",
        height=350,
        margin=dict(t=40, b=40, l=150, r=50),
        xaxis=dict(range=[0, max(top_percentages) * 1.15]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        annotations=[
            dict(
                text=f"<span style='color:#2ecc71'>■</span> Supports {pred_label}    "
                     f"<span style='color:#e74c3c'>■</span> Opposes {pred_label}",
                xref="paper", yref="paper",
                x=0.5, y=1.08,
                showarrow=False,
                font=dict(size=13),
            )
        ]
    )

    return fig
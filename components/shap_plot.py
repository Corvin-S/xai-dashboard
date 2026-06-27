"""
shap_plot.py — SHAP feature-contribution chart

Renders a horizontal bar chart of raw SHAP values using the conventional
SHAP color scheme: red for positive contributions (pushing the prediction
higher) and blue for negative contributions (pushing it lower).

Percentage-based relative importance is available on hover, preserving
accessibility for non-expert users without sacrificing the additive
semantics of Shapley values on the primary axis.
"""

import plotly.graph_objects as go
import numpy as np


def show_shap_chart(shap_values, feature_names, pred_label, max_display=10):
    """Display a SHAP feature-contribution chart in the standard format.

    The x-axis shows raw SHAP values, preserving the additive semantics
    of Shapley values (Lundberg & Lee, 2017). On hover, each feature's
    percentage share of the total absolute impact is displayed alongside
    the raw value, providing an accessible secondary reading for lay users.

    Args:
        shap_values: 1D array of SHAP values for the predicted class.
        feature_names: List of human-readable feature names.
        pred_label: The predicted class label ("Low Risk" / "High Risk").
        max_display: Number of top features to display (default: 10).
    """
    # Calculate percentage contribution per feature (for hover only)
    abs_values = np.abs(shap_values)
    total = abs_values.sum()
    percentages = (abs_values / total) * 100

    # Sort by absolute impact and select top features
    sorted_idx = np.argsort(abs_values)[::-1][:max_display]
    # Reverse for bottom-to-top display (most important at top)
    sorted_idx = sorted_idx[::-1]

    top_names = [feature_names[i] for i in sorted_idx]
    top_shap = [shap_values[i] for i in sorted_idx]
    top_percentages = [percentages[i] for i in sorted_idx]

    # Standard SHAP color scheme: red for positive, blue for negative
    colors = [
        "#FF0051" if v >= 0 else "#008BFB"
        for v in top_shap
    ]

    fig = go.Figure(go.Bar(
        x=top_shap,
        y=top_names,
        orientation="h",
        marker=dict(color=colors),
        customdata=list(zip(top_percentages,)),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "SHAP Value: %{x:.4f}<br>"
            "Relative Impact: %{customdata[0]:.1f}%"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        xaxis_title="SHAP Value (Impact on Model Output)",
        yaxis_title="",
        height=350,
        margin=dict(t=40, b=40, l=150, r=50),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        annotations=[
            dict(
                text=(
                    f"<span style='color:#FF0051'>■</span> Increases prediction    "
                    f"<span style='color:#008BFB'>■</span> Decreases prediction"
                ),
                xref="paper", yref="paper",
                x=0.5, y=1.08,
                showarrow=False,
                font=dict(size=13),
            )
        ]
    )

    return fig
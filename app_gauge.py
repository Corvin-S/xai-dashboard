
"""
app_gauge.py — Entry point for the gauge chart condition.

Launches the Credit Risk Dashboard with the gauge chart as the
fixed uncertainty visualization. Participants using this version
see only the gauge and cannot switch to the text condition.
"""
from dashboard_core import run_dashboard

run_dashboard(condition="gauge")
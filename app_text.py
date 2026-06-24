
"""
app_text.py — Entry point for the plain text condition.

Launches the Credit Risk Dashboard with the plain text description
as the fixed uncertainty visualization. Participants using this
version see only the text and cannot switch to the gauge condition.
"""
from dashboard_core import run_dashboard

run_dashboard(condition="text")

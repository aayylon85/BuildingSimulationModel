"""
Output and reporting.

This module contains visualization and reporting tools:
- plot_simulation_results: Generate simulation plots
- generate_report_from_results: Generate markdown reports
"""

from bsm.output.plotting import plot_simulation_results
from bsm.output.report_generator import generate_report_from_results

__all__ = [
    "plot_simulation_results",
    "generate_report_from_results",
]

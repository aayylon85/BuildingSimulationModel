"""
Building Simulation Model (BSM)

A physics-based single-zone building thermal simulation with LLM-powered occupant agents.

This package provides:
- Core simulation engine for thermal zone modeling
- Heat transfer calculations (convection, radiation, conduction)
- Solar radiation calculations
- Weather data handling
- Building component models (windows, HVAC, air exchange)
- Rule-based and LLM-powered occupant behavior modeling
"""

__version__ = "1.0.0"

# Core simulation classes
from bsm.core.zone import Zone
from bsm.core.zone_solver import ZoneHeatBalanceSolver
from bsm.core.constants import (
    AIR_DENSITY_KG_M3,
    AIR_SPECIFIC_HEAT_J_KG_K,
    STEFAN_BOLTZMANN,
)

# Main entry point
from bsm.runner import run_simulation_from_config

__all__ = [
    "__version__",
    "Zone",
    "ZoneHeatBalanceSolver",
    "AIR_DENSITY_KG_M3",
    "AIR_SPECIFIC_HEAT_J_KG_K",
    "STEFAN_BOLTZMANN",
    "run_simulation_from_config",
]

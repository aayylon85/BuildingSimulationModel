"""
Core simulation engine.

This module contains the central simulation components:
- Zone: Main thermal zone model
- ZoneHeatBalanceSolver: Coupled heat balance solver
- CondFDSolver: Conduction finite-difference solver for constructions
"""

from bsm.core.zone import Zone
from bsm.core.zone_solver import ZoneHeatBalanceSolver
from bsm.core.fabric_heat_transfer import CondFDSolver
from bsm.core.constants import (
    AIR_DENSITY_KG_M3,
    AIR_SPECIFIC_HEAT_J_KG_K,
    STEFAN_BOLTZMANN,
    ROUGHNESS_MULTIPLIERS,
)

__all__ = [
    "Zone",
    "ZoneHeatBalanceSolver",
    "CondFDSolver",
    "AIR_DENSITY_KG_M3",
    "AIR_SPECIFIC_HEAT_J_KG_K",
    "STEFAN_BOLTZMANN",
    "ROUGHNESS_MULTIPLIERS",
]

"""
Boundary conditions.

This module contains boundary condition generators:
- create_boundary_conditions: Build setpoint and gains schedules
- Occupant: Rule-based occupant behavior model
"""

from bsm.boundary.conditions import create_boundary_conditions
from bsm.boundary.occupants import Occupant

__all__ = [
    "create_boundary_conditions",
    "Occupant",
]

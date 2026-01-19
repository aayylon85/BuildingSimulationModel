"""
Heat transfer calculations.

This module contains convection and radiation models:
- AdaptiveConvectionAlgorithm: Exterior convection coefficient calculation
- InternalAdaptiveConvection: Interior convection coefficient calculation
- ExternalLongwaveRadiation: Exterior longwave radiation exchange
- InteriorLongwaveRadiation: Interior surface radiation exchange
"""

from bsm.heat_transfer.exterior_convection import AdaptiveConvectionAlgorithm
from bsm.heat_transfer.interior_convection import InternalAdaptiveConvection
from bsm.heat_transfer.exterior_longwave import ExternalLongwaveRadiation
from bsm.heat_transfer.interior_longwave import InteriorLongwaveRadiation

# Shared convection correlations
from bsm.heat_transfer.convection import (
    walton_unstable_horizontal,
    walton_stable_horizontal,
    ashrae_vertical_wall,
)

__all__ = [
    "AdaptiveConvectionAlgorithm",
    "InternalAdaptiveConvection",
    "ExternalLongwaveRadiation",
    "InteriorLongwaveRadiation",
    "walton_unstable_horizontal",
    "walton_stable_horizontal",
    "ashrae_vertical_wall",
]

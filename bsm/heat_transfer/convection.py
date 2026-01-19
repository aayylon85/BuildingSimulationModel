"""
Shared convection correlation functions.

These correlations are used by both interior and exterior convection
algorithms. Consolidating them here reduces code duplication.

References:
    - Walton, G.N. 1983. "Thermal Analysis Research Program Reference Manual"
      NBSIR 83-2655. National Bureau of Standards.
"""
import math


def walton_unstable_horizontal(delta_t_abs: float, tilt_deg: float) -> float:
    """
    Walton (1983) correlation for unstable horizontal or tilted surfaces.

    Used for:
    - Heated floors (warm surface facing up)
    - Cooled ceilings (cool surface facing down)
    - Tilted surfaces with unstable stratification

    Args:
        delta_t_abs: Absolute temperature difference |T_surface - T_air| (K or °C)
        tilt_deg: Surface tilt from horizontal (degrees). 0=horizontal facing up, 180=facing down.

    Returns:
        Natural convection coefficient (W/m²·K)
    """
    if delta_t_abs < 1e-6:
        return 0.0
    cos_sigma = abs(math.cos(math.radians(tilt_deg)))
    return (9.482 * (delta_t_abs ** (1.0 / 3.0))) / (7.238 - cos_sigma)


def walton_stable_horizontal(delta_t_abs: float, tilt_deg: float) -> float:
    """
    Walton (1983) correlation for stable horizontal or tilted surfaces.

    Used for:
    - Cooled floors (cool surface facing up)
    - Heated ceilings (warm surface facing down)
    - Tilted surfaces with stable stratification

    Args:
        delta_t_abs: Absolute temperature difference |T_surface - T_air| (K or °C)
        tilt_deg: Surface tilt from horizontal (degrees). 0=horizontal facing up, 180=facing down.

    Returns:
        Natural convection coefficient (W/m²·K)
    """
    if delta_t_abs < 1e-6:
        return 0.0
    cos_sigma = abs(math.cos(math.radians(tilt_deg)))
    return (1.810 * (delta_t_abs ** (1.0 / 3.0))) / (1.382 + cos_sigma)


def ashrae_vertical_wall(delta_t_abs: float) -> float:
    """
    ASHRAE vertical wall correlation (Walton 1983).

    Simple correlation for vertical surfaces where tilt is approximately 90 degrees.

    Args:
        delta_t_abs: Absolute temperature difference |T_surface - T_air| (K or °C)

    Returns:
        Natural convection coefficient (W/m²·K)
    """
    if delta_t_abs < 1e-6:
        return 0.0
    return 1.31 * (delta_t_abs ** (1.0 / 3.0))

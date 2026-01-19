"""
Implements the Adaptive Convection Algorithm for calculating the INTERIOR heat
transfer coefficient (h_c) on building zone surfaces.

References:
    - Walton, G.N. 1983. "Thermal Analysis Research Program Reference Manual"
    - Alamdari, F. and G.P. Hammond. 1983. "Improved data correlations for
      buoyancy-driven convection in rooms." Building Services Engineering
      Research & Technology. Vol. 4, No. 3.
    - Awbi, H.B. and A. Hatton. 1999. "Natural convection from heated room
      surfaces." Energy and Buildings 30 (1999) 233-244.
    - EnergyPlus Engineering Reference, Section "Inside Convection"
"""

import math


class InternalAdaptiveConvection:
    """
    Calculates the inside face heat transfer coefficient (h_c) using an
    adaptive algorithm that classifies surfaces based on orientation and
    heat flow direction.

    Supports multiple convection algorithms:
    - Walton (1983): Default for most surfaces
    - Awbi-Hatton (1999): Specific to heated floors
    - Alamdari-Hammond (1983): Improved horizontal surface correlations
    """
    def __init__(self, model_assignments, hydraulic_diameter=None):
        """
        Initializes the algorithm with user-defined model assignments.

        Args:
            model_assignments (dict): Maps surface classifications to algorithm names.
            hydraulic_diameter (float, optional): Zone hydraulic diameter in meters.
                Used by Awbi-Hatton and Alamdari-Hammond algorithms.
                If not provided, defaults to 4.0 m (typical residential zone).
        """
        self.models = model_assignments
        # Default hydraulic diameter for BESTEST zone: 4 × Area / Perimeter
        # For 8m × 6m floor: D_h = 4 × 48 / 28 ≈ 6.86 m
        self.hydraulic_diameter = hydraulic_diameter if hydraulic_diameter else 6.86

    def calculate_h_c(self, surface_data, zone_air_temp_c):
        """
        Calculates the interior convective heat transfer coefficient (h_c).
        """
        classification = self._classify_surface(surface_data, zone_air_temp_c)
        model_name = self.models.get(classification)
        
        if not model_name:
            raise ValueError(f"No interior convection model assigned for classification: {classification}")

        return self._calculate_h(model_name, surface_data, zone_air_temp_c)
        
    def _classify_surface(self, surface_data, zone_air_temp_c):
        """
        Determines the surface classification based on orientation and heat flow.

        Classifications:
            - VerticalWall: Vertical surfaces (tilt = 90°)
            - HeatedFloor: Floor with surface warmer than air (tilt = 180°, delta_T > 0)
            - CooledFloor: Floor with surface cooler than air (tilt = 180°, delta_T < 0)
            - HeatedCeiling: Ceiling with surface warmer than air (tilt = 0°, delta_T > 0)
            - CooledCeiling: Ceiling with surface cooler than air (tilt = 0°, delta_T < 0)
            - StableHorizontal: Stable horizontal (generic fallback)
            - UnstableHorizontal: Unstable horizontal (generic fallback)
            - StableTilted: Tilted surface with stable stratification
            - UnstableTilted: Tilted surface with unstable stratification
        """
        surface_temp = surface_data['solver'].nodes[0]['T']
        surface_props = surface_data['props']
        delta_t = surface_temp - zone_air_temp_c
        tilt = surface_props['tilt']

        if 90 > tilt > 0 or 180 > tilt > 90:  # Tilted Surface
            if delta_t > 0:  # Heated surface
                return 'UnstableTilted' if tilt < 90 else 'StableTilted'
            else:  # Cooled surface
                return 'StableTilted' if tilt < 90 else 'UnstableTilted'

        elif tilt == 90:  # Vertical Wall
            return 'VerticalWall'

        elif tilt == 180:  # Floor (facing down into zone)
            # Heated floor = stable stratification (warm air rises, stays near ceiling)
            # Cooled floor = unstable stratification (air descends, enhances mixing)
            if delta_t > 0:
                return 'HeatedFloor'  # NEW: specific classification for heated floors
            else:
                return 'CooledFloor'  # NEW: specific classification for cooled floors

        elif tilt == 0:  # Ceiling (facing up into zone)
            # Heated ceiling = unstable stratification (warm air descends)
            # Cooled ceiling = stable stratification (cold air stays at top)
            if delta_t > 0:
                return 'HeatedCeiling'
            else:
                return 'CooledCeiling'

        else:  # Fallback for other horizontal surfaces
            if delta_t > 0:
                return 'StableHorizontal' if tilt > 90 else 'UnstableHorizontal'
            else:
                return 'UnstableHorizontal' if tilt > 90 else 'StableHorizontal'

    def _calculate_h(self, model_name, surface_data, zone_air_temp_c):
        """Dispatcher for the convection models."""
        dispatch_map = {
            'ASHRAEVerticalWall': self._h_ashrae_vertical,
            'WaltonUnstableHorizontalOrTilt': self._h_walton_unstable,
            'WaltonStableHorizontalOrTilt': self._h_walton_stable,
            # NEW: Awbi-Hatton correlations (1999)
            'AwbiHattonHeatedFloor': self._h_awbi_hatton_heated_floor,
            'AwbiHattonHeatedWall': self._h_awbi_hatton_heated_wall,
            # NEW: Alamdari-Hammond correlations (1983)
            'AlamdariHammondStableHorizontal': self._h_alamdari_hammond_stable,
            'AlamdariHammondUnstableHorizontal': self._h_alamdari_hammond_unstable,
            'AlamdariHammondVertical': self._h_alamdari_hammond_vertical,
        }
        if model_name not in dispatch_map:
            raise ValueError(f"Unknown interior convection model: {model_name}")

        surface_temp = surface_data['solver'].nodes[0]['T']
        tilt = surface_data['props']['tilt']
        delta_t_abs = abs(surface_temp - zone_air_temp_c)

        return dispatch_map[model_name](delta_t_abs, tilt)

    # --- Walton (1983) Correlations ---

    def _h_ashrae_vertical(self, delta_t_abs, tilt):
        """ASHRAE vertical wall correlation (Walton 1983)."""
        if delta_t_abs < 1e-6:
            return 0.0
        return 1.31 * (delta_t_abs ** (1./3.))

    def _h_walton_unstable(self, delta_t_abs, tilt):
        """Walton (1983) unstable horizontal or tilted surface."""
        if delta_t_abs < 1e-6:
            return 0.0
        cos_sigma = abs(math.cos(math.radians(tilt)))
        return (9.482 * (delta_t_abs ** (1./3.))) / (7.238 - cos_sigma)

    def _h_walton_stable(self, delta_t_abs, tilt):
        """Walton (1983) stable horizontal or tilted surface."""
        if delta_t_abs < 1e-6:
            return 0.0
        cos_sigma = abs(math.cos(math.radians(tilt)))
        return (1.810 * (delta_t_abs ** (1./3.))) / (1.382 + cos_sigma)

    # --- Awbi-Hatton (1999) Correlations ---
    # Reference: Energy and Buildings 30 (1999) 233-244

    def _h_awbi_hatton_heated_floor(self, delta_t_abs, tilt):
        """
        Awbi-Hatton (1999) correlation for heated floors.

        Equation 15: h_c = 2.175 × |ΔT|^0.308 / D_h^0.076

        This correlation gives significantly higher convection coefficients
        than Walton for heated floors, better capturing the natural convection
        physics when solar radiation heats the floor surface.

        For D_h ≤ 1.0 m, the diameter term is effectively constant.
        """
        if delta_t_abs < 1e-6:
            return 0.0

        D_h = self.hydraulic_diameter

        if D_h <= 1.0:
            # For small zones, diameter term is constant
            return 2.175 * (delta_t_abs ** 0.308)
        else:
            return 2.175 * (delta_t_abs ** 0.308) / (D_h ** 0.076)

    def _h_awbi_hatton_heated_wall(self, delta_t_abs, tilt):
        """
        Awbi-Hatton (1999) correlation for heated vertical walls.

        Equation 12: h_c = 1.823 × |ΔT|^0.293 / max(D_h, 1.0)^0.121
        """
        if delta_t_abs < 1e-6:
            return 0.0

        D_h = max(self.hydraulic_diameter, 1.0)
        return 1.823 * (delta_t_abs ** 0.293) / (D_h ** 0.121)

    # --- Alamdari-Hammond (1983) Correlations ---
    # Reference: Building Services Engineering Research & Technology Vol. 4, No. 3

    def _h_alamdari_hammond_stable(self, delta_t_abs, tilt):
        """
        Alamdari-Hammond (1983) stable horizontal surface.

        h_c = 0.6 × |ΔT / D_h²|^0.2

        For stable stratification (e.g., cooled ceiling, heated floor).
        """
        if delta_t_abs < 1e-6:
            return 0.0

        D_h = self.hydraulic_diameter
        if D_h < 0.1:
            D_h = 0.1  # Avoid division issues

        return 0.6 * ((delta_t_abs / (D_h ** 2)) ** 0.2)

    def _h_alamdari_hammond_unstable(self, delta_t_abs, tilt):
        """
        Alamdari-Hammond (1983) unstable horizontal surface.

        h_c = [(1.4 × |ΔT/D_h|^0.25)^6 + (1.63 × |ΔT|^(1/3))^6]^(1/6)

        For unstable stratification (e.g., heated ceiling, cooled floor).
        Combines buoyancy force term with thermal effect term.
        """
        if delta_t_abs < 1e-6:
            return 0.0

        D_h = self.hydraulic_diameter
        if D_h < 0.1:
            D_h = 0.1

        term1 = (1.4 * ((delta_t_abs / D_h) ** 0.25)) ** 6
        term2 = (1.63 * (delta_t_abs ** (1./3.))) ** 6

        return (term1 + term2) ** (1./6.)

    def _h_alamdari_hammond_vertical(self, delta_t_abs, tilt):
        """
        Alamdari-Hammond (1983) vertical wall correlation.

        h_c = [(1.5 × |ΔT/H|^0.25)^6 + (1.23 × |ΔT|^(1/3))^6]^(1/6)

        where H is the wall height. Uses hydraulic_diameter as characteristic length.
        """
        if delta_t_abs < 1e-6:
            return 0.0

        H = self.hydraulic_diameter  # Use as characteristic height
        if H < 0.1:
            H = 0.1

        term1 = (1.5 * ((delta_t_abs / H) ** 0.25)) ** 6
        term2 = (1.23 * (delta_t_abs ** (1./3.))) ** 6

        return (term1 + term2) ** (1./6.)



"""
Implements the Adaptive Convection Algorithm for calculating the INTERIOR heat 
transfer coefficient (h_c) on building zone surfaces.
"""

import math

class InternalAdaptiveConvection:
    """
    Calculates the inside face heat transfer coefficient (h_c) using an
    adaptive algorithm that classifies surfaces based on orientation and
    heat flow direction.
    """
    def __init__(self, model_assignments):
        """
        Initializes the algorithm with user-defined model assignments.
        """
        self.models = model_assignments

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
        """Determines the surface classification."""
        surface_temp = surface_data['solver'].nodes[0]['T']
        surface_props = surface_data['props'] # Access the nested properties dictionary
        delta_t = surface_temp - zone_air_temp_c
        tilt = surface_props['tilt'] # Correctly access the 'tilt' value

        if 90 > tilt > 0 or 180 > tilt > 90: # Tilted Surface
             if delta_t > 0: # Heated surface
                 return 'UnstableTilted' if tilt < 90 else 'StableTilted'
             else: # Cooled surface
                 return 'StableTilted' if tilt < 90 else 'UnstableTilted'
        
        elif tilt == 90: # Vertical Wall
            return 'VerticalWall'

        else: # Horizontal Surface (Floor or Roof/Ceiling)
            surface_type = surface_props['type'].lower()
            if delta_t > 0: # Heated surface
                return 'UnstableHorizontal' if surface_type in ['floor', 'roof'] else 'StableHorizontal'
            else: # Cooled surface
                return 'StableHorizontal' if surface_type in ['floor', 'roof'] else 'UnstableHorizontal'

    def _calculate_h(self, model_name, surface_data, zone_air_temp_c):
        """Dispatcher for the convection models."""
        dispatch_map = {
            'ASHRAEVerticalWall': self._h_ashrae_vertical,
            'WaltonUnstableHorizontalOrTilt': self._h_walton_unstable,
            'WaltonStableHorizontalOrTilt': self._h_walton_stable,
        }
        if model_name not in dispatch_map:
            raise ValueError(f"Unknown interior convection model: {model_name}")

        surface_temp = surface_data['solver'].nodes[0]['T']
        tilt = surface_data['props']['tilt'] # Also needed here
        delta_t_abs = abs(surface_temp - zone_air_temp_c)
        
        if delta_t_abs < 1e-6:
            return 0.0

        return dispatch_map[model_name](delta_t_abs, tilt)

    def _h_ashrae_vertical(self, delta_t_abs, tilt):
        return 1.31 * (delta_t_abs ** (1./3.))

    def _h_walton_unstable(self, delta_t_abs, tilt):
        cos_sigma = abs(math.cos(math.radians(tilt)))
        return (9.482 * (delta_t_abs ** (1./3.))) / (7.238 - cos_sigma)

    def _h_walton_stable(self, delta_t_abs, tilt):
        cos_sigma = abs(math.cos(math.radians(tilt)))
        return (1.810 * (delta_t_abs ** (1./3.))) / (1.382 + cos_sigma)



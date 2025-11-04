"""
Implements the Adaptive Convection Algorithm for calculating the exterior heat 
transfer coefficient (hc) on building surfaces.
"""

import math

# Constants: Surface Roughness Multipliers (Walton 1981)
ROUGHNESS_MULTIPLIERS = {
    1: 2.17,  # Very Rough (Stucco)
    2: 1.67,  # Rough (Brick)
    3: 1.52,  # Medium Rough (Concrete)
    4: 1.13,  # Medium Smooth (Clear pine)
    5: 1.11,  # Smooth (Smooth Plaster)
    6: 1.00,  # Very Smooth (Glass)
}

class AdaptiveConvectionAlgorithm:
    """
    A class to calculate the outside face heat transfer coefficient (hc) using
    the Adaptive Convection Algorithm.

    The algorithm classifies a surface based on its orientation, heat flow direction,
    and wind direction, then applies specific model equations for forced (hf) and
    natural (hn) convection.
    """

    def __init__(self, hf_model_assignments, hn_model_assignments):
        """
        Initializes the algorithm with user-defined model assignments.

        Args:
            hf_model_assignments (dict): A dictionary mapping surface classifications
                to the desired forced convection model.
                Example: {'VerticalWallWindward': 'SparrowWindward', ...}
            hn_model_assignments (dict): A dictionary mapping surface classifications
                to the desired natural convection model.
                Example: {'VerticalWallWindward': 'ASHRAEVerticalWall', ...}
        """
        self.hf_models = hf_model_assignments
        self.hn_models = hn_model_assignments
        self._combined_models = [
            'MoWITTWindward', 'MoWITTLeeward', 'NusseltJurges', 
            'McAdams'
        ]

    def calculate_hc(self, surface, weather):
        """
        Calculates the total convective heat transfer coefficient (hc).

        Args:
            surface (dict): A dictionary of surface properties.
                - type (str): 'wall' or 'roof'
                - tilt (float): Surface tilt angle in degrees (0=horizontal, 90=vertical)
                - azimuth (float): Surface azimuth in degrees (0-360, 180=S)
                - area (float): Surface area in m^2
                - perimeter (float): Surface perimeter in m
                - roughness_index (int): 1-6, corresponding to ROUGHNESS_MULTIPLIERS
                - surface_temp_c (float): Outside surface temperature in Celsius
                - building_volume_m3 (float, optional): Total building volume for Mitchell model.
            weather (dict): A dictionary of weather conditions.
                - wind_speed_local_ms (float): Wind speed at surface height in m/s
                - wind_speed_10m_ms (float): Wind speed at weather station (10m) in m/s
                - wind_direction_deg (float): Wind direction in degrees (0-360, 180=from S)
                - air_temp_c (float): Outside air temperature in Celsius

        Returns:
            float: The total convective heat transfer coefficient, hc (W/m^2.K).
        """
        classification = self._classify_surface(surface, weather)
        hf_model_name = self.hf_models.get(classification)
        
        if not hf_model_name:
            raise ValueError(f"No forced convection model assigned for classification: {classification}")
            
        if hf_model_name in self._combined_models:
            return self._calculate_h_combined(hf_model_name, surface, weather)

        hn_model_name = self.hn_models.get(classification)
        if not hn_model_name:
             raise ValueError(f"No natural convection model assigned for classification: {classification}")

        hf = self._calculate_hf(hf_model_name, surface, weather)
        hn = self._calculate_hn(hn_model_name, surface, weather)
        hc = ((hf**3)+(hn**3))**(1./3.)
        
        return hc

    def _classify_surface(self, surface, weather):
        """Determines the surface classification."""
        if surface['type'].lower() == 'roof':
            return 'RoofUnstable' if surface['surface_temp_c'] > weather['air_temp_c'] else 'RoofStable'
        
        elif surface['type'].lower() == 'wall':
            wind_angle_diff = abs(weather['wind_direction_deg'] - surface['azimuth'])
            incidence_angle = min(wind_angle_diff, 360 - wind_angle_diff)
            return 'VerticalWallWindward' if incidence_angle <= 90 else 'VerticalWallLeeward'
        
        else:
            raise ValueError("Invalid surface type. Must be 'roof' or 'wall'.")
            
    def _calculate_h_combined(self, model_name, surface, weather):
        """Dispatcher for combined convection models."""
        models = {
            'MoWITTWindward': self._hc_mowitt_windward,
            'MoWITTLeeward': self._hc_mowitt_leeward,
            'NusseltJurges': self._hc_nusselt_jurges,
            'McAdams': self._hc_mcadams,
        }
        if model_name not in models:
            raise ValueError(f"Unknown combined convection model: {model_name}")
        return models[model_name](surface, weather)
                
    def _calculate_hf(self, model_name, surface, weather):
        """Dispatcher for forced convection models."""
        models = {
            'SparrowWindward': self._hf_sparrow_windward,
            'SparrowLeeward': self._hf_sparrow_leeward,
            'BlockenWindward': self._hf_blocken_windward,
            'EmmelVertical': self._hf_emmel_vertical,
            'EmmelRoof': self._hf_emmel_roof,
            'Mitchell': self._hf_mitchell,
        }
        if model_name not in models:
            raise ValueError(f"Unknown forced convection model: {model_name}")
        return models[model_name](surface, weather)
    
    def _hc_mowitt_windward(self, surface, weather):
        # Eq. 3.97 (with updated coefficients from Table 3.9)
        delta_t = abs(surface['surface_temp_c'] - weather['air_temp_c'])
        v_z = weather['wind_speed_local_ms']
        c_t = 0.84; a = 3.26; b = 0.89
        hn_term_sq = (c_t * (delta_t ** (1./3.))) ** 2
        hf_term_sq = (a * (v_z ** b)) ** 2
        return math.sqrt(hn_term_sq + hf_term_sq)

    def _hc_mowitt_leeward(self, surface, weather):
        # Eq. 3.98 (with updated coefficients from Table 3.9)
        delta_t = abs(surface['surface_temp_c'] - weather['air_temp_c'])
        v_z = weather['wind_speed_local_ms']
        c_t = 0.84; a = 3.55; b = 0.617
        hn_term_sq = (c_t * (delta_t ** (1./3.))) ** 2
        hf_term_sq = (a * (v_z ** b)) ** 2
        return math.sqrt(hn_term_sq + hf_term_sq)

    def _hc_nusselt_jurges(self, surface, weather):
        # Eq. 3.103
        return 5.8 + 3.94 * weather['wind_speed_local_ms']
        
    def _hc_mcadams(self, surface, weather):
        # Eq. 3.104
        return 5.7 + 3.8 * weather['wind_speed_local_ms']

    def _hf_sparrow_windward(self, surface, weather):
        rf = ROUGHNESS_MULTIPLIERS.get(surface['roughness_index'], 1.0)
        pv_over_a = (surface['perimeter'] * weather['wind_speed_local_ms']) / surface['area']
        return 2.537 * rf * math.sqrt(pv_over_a) if pv_over_a > 0 else 0

    def _hf_sparrow_leeward(self, surface, weather):
        return 0.5 * self._hf_sparrow_windward(surface, weather)

    def _hf_blocken_windward(self, surface, weather):
        v_10m = weather['wind_speed_10m_ms']
        wind_angle_diff = abs(weather['wind_direction_deg'] - surface['azimuth'])
        theta = min(wind_angle_diff, 360 - wind_angle_diff)
        
        if theta <= 11.25:
            return 4.6 * (v_10m ** 0.89)
        elif 11.25 < theta <= 33.75:
            return 5.0 * (v_10m ** 0.80)
        elif 33.75 < theta <= 56.25:
            return 4.6 * (v_10m ** 0.84)
        else: # 56.25 < theta <= 90
            return 4.5 * (v_10m ** 0.81)
            
    def _hf_emmel_vertical(self, surface, weather):
        v_10m = weather['wind_speed_10m_ms']
        wind_angle_diff = abs(weather['wind_direction_deg'] - surface['azimuth'])
        theta = min(wind_angle_diff, 360 - wind_angle_diff)
        
        if theta <= 22.5:
            return 3.34 * (v_10m ** 0.84)
        elif 22.5 < theta <= 67.5:
            return 4.78 * (v_10m ** 0.71)
        elif 67.5 < theta <= 112.5:
            return 4.05 * (v_10m ** 0.77)
        elif 112.5 < theta <= 157.5:
            return 3.54 * (v_10m ** 0.16) 
        else: # 157.5 < theta <= 180.0
            return 3.34 * (v_10m ** 0.84)
            
    def _hf_emmel_roof(self, surface, weather):
        v_10m = weather['wind_speed_10m_ms']
        wind_angle_diff = abs(weather['wind_direction_deg'] - surface['azimuth'])
        theta = min(wind_angle_diff, 360 - wind_angle_diff)
        
        if theta <= 22.5:
            return 5.11 * (v_10m ** 0.78)
        elif 22.5 < theta <= 67.5:
            return 4.60 * (v_10m ** 0.79)
        else: # 67.5 < theta <= 90
            return 3.67 * (v_10m ** 0.85)

    def _hf_mitchell(self, surface, weather):
        if 'building_volume_m3' not in surface or surface['building_volume_m3'] <= 0:
            raise ValueError("Building volume must be provided for Mitchell model.")
        l_char = surface['building_volume_m3'] ** (1./3.)
        return (8.6 * (weather['wind_speed_local_ms'] ** 0.6)) / (l_char ** 0.4)

    def _calculate_hn(self, model_name, surface, weather):
        """Dispatcher for natural convection models."""
        models = {
            'WaltonUnstableHorizontalOrTilt': self._hn_walton_unstable,
            'WaltonStableHorizontalOrTilt': self._hn_walton_stable,
            'ASHRAEVerticalWall': self._hn_ashrae_vertical,
        }
        if model_name not in models:
            raise ValueError(f"Unknown natural convection model: {model_name}")
        delta_t = abs(surface['surface_temp_c'] - weather['air_temp_c'])
        return models[model_name](delta_t, surface['tilt'])

    def _hn_walton_unstable(self, delta_t, tilt):
        cos_sigma = abs(math.cos(math.radians(tilt)))
        return (9.482 * (delta_t ** (1./3.))) / (7.238 - cos_sigma)

    def _hn_walton_stable(self, delta_t, tilt):
        cos_sigma = abs(math.cos(math.radians(tilt)))
        return (1.810 * (delta_t ** (1./3.))) / (1.382 + cos_sigma)

    def _hn_ashrae_vertical(self, delta_t, tilt):
        return 1.31 * (delta_t ** (1./3.))




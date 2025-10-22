"""
Handles infiltration and ventilation calculations for the zone.
"""
import math

# Constants for air properties at standard conditions
AIR_DENSITY_KG_M3 = 1.225
AIR_SPECIFIC_HEAT_J_KG_K = 1006

class AirExchangeManager:
    """Manages all air exchange processes for a zone."""
    def __init__(self, air_exchange_data, zone_volume_m3):
        self.infiltration_config = air_exchange_data.get('infiltration', {})
        self.ventilation_config = air_exchange_data.get('ventilation', {})
        self.zone_volume_m3 = zone_volume_m3

    def get_mass_flow_rate_coeff_w_k(self, T_zone_c, T_out_c, wind_speed_ms,
                                      window_open_fraction=0.0, hvac_is_heating=False):
        """
        Calculates the total thermal conductance (m_dot * C_p) from all air
        exchange processes.
        """
        # Infiltration is suppressed when HVAC is actively heating
        infiltration_m3_s = 0.0
        if not hvac_is_heating:
            infiltration_m3_s = self._get_infiltration_rate_m3_s(T_zone_c, T_out_c, wind_speed_ms)

        # Ventilation is active when the window is open
        ventilation_m3_s = self._get_ventilation_rate_m3_s(window_open_fraction)
        
        # Combine the volumetric flow rates
        total_vol_flow_m3_s = math.sqrt(infiltration_m3_s**2 + ventilation_m3_s**2)
        
        # Convert to thermal conductance
        mass_flow_kg_s = total_vol_flow_m3_s * AIR_DENSITY_KG_M3
        return mass_flow_kg_s * AIR_SPECIFIC_HEAT_J_KG_K
        
    def _get_ventilation_rate_m3_s(self, window_open_fraction):
        """Calculates ventilation rate based on Air Changes per Hour (ACH)."""
        if window_open_fraction > 0:
            # This is the corrected line:
            ach = self.ventilation_config.get("open_window_ach", 0.0)
            vol_flow_m3_hr = self.zone_volume_m3 * ach
            return vol_flow_m3_hr / 3600.0 * window_open_fraction
        return 0.0

    def _get_infiltration_rate_m3_s(self, T_zone_c, T_out_c, wind_speed_ms):
        """Implements the AIM-2 Infiltration Model."""
        c = self.infiltration_config.get('flow_coefficient_m3_s_Pa_n', 0.0)
        n = self.infiltration_config.get('pressure_exponent_n', 0.65)
        Cs = self.infiltration_config.get('stack_coeff_Pa_K', 0.0)
        Cw = self.infiltration_config.get('wind_coeff_Pa_s2_m2', 0.0)
        s = self.infiltration_config.get('shelter_factor_s', 1.0)

        delta_T = abs(T_zone_c - T_out_c)
        
        # Calculate pressure difference from stack and wind effects
        delta_p_stack = Cs * delta_T
        delta_p_wind = Cw * ((s * wind_speed_ms)**2)
        
        # Combine pressures
        delta_p_total = (delta_p_stack**2 + delta_p_wind**2)**0.5
        
        # Calculate flow rate using the power law
        infiltration_rate = c * (delta_p_total**n)
        return infiltration_rate




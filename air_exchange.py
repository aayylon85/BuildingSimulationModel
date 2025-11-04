"""
Manages air exchange, including infiltration and ventilation,
and calculates the sensible heat load.
"""
import math
from constants import AIR_DENSITY_KG_M3, AIR_SPECIFIC_HEAT_J_KG_K

class AirExchangeManager:
    """
    Manages infiltration and ventilation models and combines their effects.
    """
    def __init__(self, air_exchange_data, zone_volume):
        self.zone_volume_m3 = zone_volume
        self.infiltration = InfiltrationFlowCoefficient(air_exchange_data['infiltration'])
        self.ventilation_config = air_exchange_data['ventilation']

    def get_mass_flow_rate_coeff_w_k(self, T_zone_c, T_ext_c, wind_speed_ms,
                                      window_open_fraction, hvac_is_heating):
        """
        Calculates the combined mass flow rate coefficient (m_dot * C_p)
        from all air exchange processes.
        """
        infiltration_rate_m3_s = self.infiltration.calculate_flow_rate(
            T_zone_c, T_ext_c, wind_speed_ms
        )
        
        ventilation_rate_m3_s = self._get_ventilation_rate_m3_s(window_open_fraction)

        # Superposition principle to combine flows
        q_n_sq = infiltration_rate_m3_s**2
        q_v_sq = ventilation_rate_m3_s**2
        
        total_flow_m3_s = (q_n_sq + q_v_sq)**0.5
        
        m_dot_kg_s = total_flow_m3_s * AIR_DENSITY_KG_M3
        return m_dot_kg_s * AIR_SPECIFIC_HEAT_J_KG_K

    def _get_ventilation_rate_m3_s(self, window_open_fraction):
        """Calculates the ventilation rate based on the window state."""
        if window_open_fraction <= 0:
            return 0.0
        
        ach = self.ventilation_config.get('open_window_ach', 0.0)
        volume_m3_hr = self.zone_volume_m3 * ach
        return (volume_m3_hr / 3600.0) * window_open_fraction


class InfiltrationFlowCoefficient:
    """
    Implements the ASHRAE "Flow Coefficient" (AIM-2) model.
    """
    def __init__(self, config):
        self.c = config['flow_coefficient_m3_s_Pa_n']
        self.n = config['pressure_exponent_n']
        self.Cs = config['stack_coeff_Pa_K']
        self.Cw = config['wind_coeff_Pa_s2_m2']
        self.s = config['shelter_factor_s']

    def calculate_flow_rate(self, T_zone_c, T_ext_c, wind_speed_ms):
        """Calculates the infiltration flow rate."""
        delta_T_abs = abs(T_zone_c - T_ext_c)
        
        delta_P_stack = self.Cs * delta_T_abs
        delta_P_wind = self.Cw * (self.s * wind_speed_ms)**2
        
        delta_P_total = (delta_P_stack**2 + delta_P_wind**2)**0.5
        
        flow_rate_m3_s = self.c * (delta_P_total**self.n)
        return flow_rate_m3_s



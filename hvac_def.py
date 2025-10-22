"""
Defines the HVAC system model. This module is responsible for calculating
the heating or cooling power required to meet the zone's setpoints.
"""

class VerySimpleHVAC:
    """
    A simple proportional control HVAC model.
    """
    def __init__(self, heating_capacity_w, cooling_capacity_w, proportional_gain_w_k):
        """
        Initializes the HVAC system with its maximum power capacities and control gain.
        """
        self.heating_capacity_w = heating_capacity_w
        self.cooling_capacity_w = cooling_capacity_w
        self.proportional_gain_w_k = proportional_gain_w_k

    def calculate_hvac_power(self, T_air_prev, T_heating_setpoint, T_cooling_setpoint):
        """
        Calculates the required HVAC power for the current timestep using
        proportional control.

        Returns:
            float: HVAC power in Watts (positive for heating, negative for cooling).
        """
        q_hvac = 0.0

        # Heating demand with proportional control
        heating_error = T_heating_setpoint - T_air_prev
        if heating_error > 0:
            # The ideal heating power is proportional to the error
            ideal_heating_power = self.proportional_gain_w_k * heating_error
            # The actual power is capped by the system's capacity
            q_hvac = min(ideal_heating_power, self.heating_capacity_w)
        
        # Cooling demand with proportional control
        cooling_error = T_air_prev - T_cooling_setpoint
        if cooling_error > 0:
            ideal_cooling_power = self.proportional_gain_w_k * cooling_error
            q_hvac = -min(ideal_cooling_power, self.cooling_capacity_w)
        
        return q_hvac




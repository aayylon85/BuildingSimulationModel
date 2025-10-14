"""
Defines the HVAC system model. This module is responsible for calculating
the heating or cooling power required to meet the zone's setpoints.
"""

class VerySimpleHVAC:
    """
    A simple predictive HVAC model that calculates the energy needed to
    bring the zone to its setpoint over the next timestep.
    """
    def __init__(self, heating_capacity_w, cooling_capacity_w):
        """
        Initializes the HVAC system with its maximum power capacities.
        """
        self.heating_capacity_w = heating_capacity_w
        self.cooling_capacity_w = cooling_capacity_w

    def calculate_hvac_power(self, T_air_prev, T_setpoint, total_passive_loss,
                               total_passive_gain, air_thermal_mass, dt_sec):
        """
        Calculates the required HVAC power for the current timestep.

        Returns:
            float: HVAC power in Watts (positive for heating, negative for cooling).
        """
        # Power needed to change the air temperature to the setpoint
        power_to_reach_setpoint = air_thermal_mass * (T_setpoint - T_air_prev) / dt_sec
        
        # Net power flow from passive sources (gains are positive)
        net_passive_power = total_passive_gain - total_passive_loss

        # Ideal HVAC power must counteract the net passive flow AND drive the temp change
        ideal_hvac_power = power_to_reach_setpoint - net_passive_power

        q_hvac = 0.0
        if ideal_hvac_power > 0: # Heating is required
            q_hvac = min(ideal_hvac_power, self.heating_capacity_w)
        elif ideal_hvac_power < 0: # Cooling is required
            q_hvac = -min(abs(ideal_hvac_power), self.cooling_capacity_w)
        
        return q_hvac


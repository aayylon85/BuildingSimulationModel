"""
Defines a simple window model based on U-value and SHGC.
"""

class SimpleWindow:
    """
    Represents a window with simplified thermal properties.
    """
    def __init__(self, area, u_value, shgc):
        """
        Initializes the simple window.
        """
        if not 0 < shgc <= 1:
            raise ValueError("SHGC must be between 0 and 1.")
        if u_value <= 0:
            raise ValueError("U-value must be positive.")
            
        self.area = area
        self.u_value = u_value
        self.shgc = shgc

    def calculate_heat_flow(self, t_inside_air, t_outside_air, solar_irradiance):
        """
        Calculates the conductive heat flow and solar gain through the window.

        Returns:
            tuple (float, float): 
                - q_conductive (W): Conductive flow. Positive=loss.
                - q_solar_gain (W): Solar gain component. Always positive.
        """
        # Conductive heat flow (positive for loss out of the zone)
        q_conductive = self.u_value * self.area * (t_inside_air - t_outside_air)
        
        # Solar heat gain (always a gain into the zone)
        q_solar_gain = self.shgc * self.area * solar_irradiance
        
        return q_conductive, q_solar_gain


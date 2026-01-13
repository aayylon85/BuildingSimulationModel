"""
Defines window models based on U-value and SHGC.

Includes both SimpleWindow (constant SHGC) and AngularDependentWindow
(ASHRAE 140 Table 7-12 angular-dependent SHGC).
"""
from window_angular_transmittance import AngularTransmittanceModel


class SimpleWindow:
    """
    Represents a window with simplified thermal properties.
    """
    def __init__(self, area, u_value, shgc, solar_distribution_ratios, glass_fraction=1.0):
        """
        Initializes the simple window.

        Args:
            area: Total window area including frame (m²)
            u_value: Overall U-value (W/m²·K)
            shgc: Solar Heat Gain Coefficient (center-of-glass value)
            solar_distribution_ratios: Dict mapping surface names to distribution fractions
            glass_fraction: Fraction of window area that is glazing vs frame (default: 1.0 = no frame)
        """
        if not 0 < shgc <= 1:
            raise ValueError("SHGC must be between 0 and 1.")
        if u_value <= 0:
            raise ValueError("U-value must be positive.")
        if not 0 < glass_fraction <= 1:
            raise ValueError("glass_fraction must be between 0 and 1.")

        self.area = area
        self.u_value = u_value
        self.shgc = shgc
        self.ratios = solar_distribution_ratios
        self.glass_fraction = glass_fraction

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
        # Only glass area transmits solar radiation; frame is opaque
        q_solar_gain = self.shgc * (self.area * self.glass_fraction) * solar_irradiance

        return q_conductive, q_solar_gain


class AngularDependentWindow(SimpleWindow):
    """
    Window with angular-dependent SHGC per ASHRAE 140-2023 Table 7-12.

    Implements angular dependence for direct beam radiation while using
    hemispherical-averaged SHGC for diffuse radiation. Accounts for solar
    lost (reradiated back out window) per ASHRAE 140 Table 7-13.
    """

    def __init__(self, area, u_value, normal_shgc,
                 solar_distribution_ratios, angular_model_config,
                 glass_fraction=1.0):
        """
        Initialize angular-dependent window.

        Args:
            area: Window area (m²)
            u_value: Overall U-value (W/m²·K)
            normal_shgc: SHGC at normal incidence (0° angle)
            solar_distribution_ratios: Dict of surface solar distribution fractions
            angular_model_config: Dict with angular model settings:
                - 'enabled': bool (must be True)
                - 'hemispherical_avg_transmittance': float (optional, auto-calculated if not provided)
                - 'solar_lost_fraction': float (optional, fraction reradiated back out, default 0.0)
            glass_fraction: Glazing fraction (default: 1.0 = no frame)
        """
        # Initialize base class
        super().__init__(area, u_value, normal_shgc,
                        solar_distribution_ratios, glass_fraction)

        # Create angular transmittance model
        self.angular_model = AngularTransmittanceModel(normal_shgc)

        # Get hemispherical-averaged SHGC for diffuse
        self.shgc_hemispherical = angular_model_config.get(
            'hemispherical_avg_transmittance',
            self.angular_model.calculate_hemispherical_average_transmittance()
        )

        # Solar lost fraction (reradiated back out window)
        # Per ASHRAE 140 Table 7-13, this is typically 0.035 (3.5%)
        self.solar_lost_fraction = angular_model_config.get('solar_lost_fraction', 0.0)

    def calculate_heat_flow(self, t_inside_air, t_outside_air,
                          irradiance_components, incident_angle_deg=None):
        """
        Calculate window heat flows with angular-dependent SHGC.

        Args:
            t_inside_air: Inside air temperature (°C)
            t_outside_air: Outside air temperature (°C)
            irradiance_components: Can be either:
                - dict with 'beam', 'diffuse_sky', 'ground_reflected' keys (angular mode)
                - float (total irradiance, for backward compatibility)
            incident_angle_deg: Incident angle for beam component (degrees)
                Only used if irradiance_components is a dict

        Returns:
            tuple: (q_conductive, q_solar_gain) in Watts
        """
        # Conductive heat flow (same as SimpleWindow)
        q_conductive = self.u_value * self.area * (t_inside_air - t_outside_air)

        # Check if we're using angular dependence or fallback to simple mode
        if isinstance(irradiance_components, dict):
            # Angular-dependent mode: separate beam and diffuse
            beam = irradiance_components.get('beam', 0.0)
            diffuse_sky = irradiance_components.get('diffuse_sky', 0.0)
            ground_reflected = irradiance_components.get('ground_reflected', 0.0)

            # Get incident angle (default to 90° if not provided)
            if incident_angle_deg is None:
                incident_angle_deg = 90.0

            # Apply angular-dependent SHGC to direct beam
            shgc_beam = self.angular_model.calculate_transmittance(incident_angle_deg)
            q_solar_beam = shgc_beam * (self.area * self.glass_fraction) * beam

            # Apply hemispherical-averaged SHGC to diffuse and ground-reflected
            q_solar_diffuse = self.shgc_hemispherical * (self.area * self.glass_fraction) * (diffuse_sky + ground_reflected)

            # Total transmitted solar gain
            q_solar_gain_total = q_solar_beam + q_solar_diffuse

        else:
            # Fallback mode: treat as total irradiance with normal SHGC
            # This maintains backward compatibility
            total_irradiance = float(irradiance_components)
            q_solar_gain_total = self.shgc * (self.area * self.glass_fraction) * total_irradiance

        # Account for solar lost (reradiated back out window)
        # Net solar gain to zone = Total transmitted × (1 - solar_lost_fraction)
        q_solar_gain = q_solar_gain_total * (1.0 - self.solar_lost_fraction)

        return q_conductive, q_solar_gain


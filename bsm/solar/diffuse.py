"""
Solar diffuse radiation models for tilted surfaces.

Implements the Perez anisotropic sky model and simpler isotropic model
for calculating diffuse sky radiation on tilted surfaces.
"""

import math
from typing import Tuple


class PerezDiffuseSkyModel:
    """
    Perez anisotropic sky model for diffuse irradiance on tilted surfaces.

    Based on Perez et al. (1990) model that decomposes sky diffuse radiation into:
    1. Isotropic dome component - uniform sky radiance
    2. Circumsolar brightening - concentrated near the sun
    3. Horizon brightening - linear source at the horizon

    Reference: Perez, R., et al. (1990). "Modeling Daylight Availability and
    Irradiance Components from Direct and Global Irradiance."
    Solar Energy, 44(5), 271-289.
    """

    # Perez model empirical coefficients F11, F12, F13, F21, F22, F23
    # Organized by sky clearness (epsilon) bins
    PEREZ_COEFFICIENTS = {
        # epsilon range: [low, high), coefficients: [F11, F12, F13, F21, F22, F23]
        1: (1.000, 1.065, [-0.008, 0.588, -0.062, -0.060, 0.072, -0.022]),
        2: (1.065, 1.230, [0.130, 0.683, -0.151, -0.019, 0.066, -0.029]),
        3: (1.230, 1.500, [0.330, 0.487, -0.221, 0.055, -0.064, -0.026]),
        4: (1.500, 1.950, [0.568, 0.187, -0.295, 0.109, -0.152, -0.014]),
        5: (1.950, 2.800, [0.873, -0.392, -0.362, 0.226, -0.462, 0.001]),
        6: (2.800, 4.500, [1.132, -1.237, -0.412, 0.288, -0.823, 0.056]),
        7: (4.500, 6.200, [1.060, -1.600, -0.359, 0.264, -1.127, 0.131]),
        8: (6.200, float('inf'), [0.678, -0.327, -0.250, 0.156, -1.377, 0.251])
    }

    # Solar constant for extraterrestrial irradiance
    SOLAR_CONSTANT = 1361.0  # W/m²

    def calculate_diffuse_on_tilted_surface(
        self,
        dhi_w_m2: float,
        dni_w_m2: float,
        ghi_w_m2: float,
        sun_zenith_deg: float,
        sun_azimuth_deg: float,
        surface_tilt_deg: float,
        surface_azimuth_deg: float,
        extraterrestrial_w_m2: float
    ) -> float:
        """
        Calculate diffuse irradiance on a tilted surface using Perez model.

        Args:
            dhi_w_m2: Diffuse horizontal irradiance (W/m²)
            dni_w_m2: Direct normal irradiance (W/m²)
            ghi_w_m2: Global horizontal irradiance (W/m²)
            sun_zenith_deg: Solar zenith angle (degrees)
            sun_azimuth_deg: Solar azimuth angle (degrees)
            surface_tilt_deg: Surface tilt from horizontal (degrees)
            surface_azimuth_deg: Surface azimuth (degrees)
            extraterrestrial_w_m2: Extraterrestrial irradiance (W/m²)

        Returns:
            diffuse_irradiance_W_m2: Diffuse irradiance on the tilted surface
        """
        # Handle nighttime or zero diffuse
        if dhi_w_m2 <= 0 or sun_zenith_deg >= 90:
            return 0.0

        # Convert angles to radians
        zenith_rad = math.radians(sun_zenith_deg)
        tilt_rad = math.radians(surface_tilt_deg)

        # Calculate incident angle between sun and surface
        from bsm.solar.incident_angle import IncidentAngleCalculator
        calc = IncidentAngleCalculator()
        incident_angle_deg = calc.calculate_incident_angle(
            sun_zenith_deg, sun_azimuth_deg,
            surface_tilt_deg, surface_azimuth_deg
        )
        incident_rad = math.radians(incident_angle_deg)

        # Calculate air mass using Kasten-Young formula (smooth, continuous)
        # Reference: Kasten, F. and Young, A. T. (1989)
        # "Revised optical air mass tables and approximation formula"
        if sun_zenith_deg >= 90:
            # Sun below horizon
            air_mass = 38.0
        else:
            # Kasten-Young formula: AM = 1 / [cos(θ) + 0.50572*(96.07995-θ)^(-1.6364)]
            # This provides smooth, continuous values from zenith=0° to horizon
            cos_z = math.cos(zenith_rad)
            exponent = -1.6364
            denominator = cos_z + 0.50572 * (96.07995 - sun_zenith_deg)**exponent
            air_mass = 1.0 / max(0.01, denominator)
            # Physical bounds: air mass ranges from 1.0 (overhead) to ~38 (horizon)
            air_mass = min(38.0, max(1.0, air_mass))

        # Calculate sky brightness (Delta)
        delta = dhi_w_m2 * air_mass / extraterrestrial_w_m2

        # Calculate sky clearness (epsilon)
        # epsilon = [(DHI + DNI)/DHI + kappa * zenith^3] / [1 + kappa * zenith^3]
        kappa = 1.041  # For zenith in radians
        zenith_cubed = zenith_rad ** 3

        if dhi_w_m2 > 0:
            epsilon = ((dhi_w_m2 + dni_w_m2) / dhi_w_m2 + kappa * zenith_cubed) / (1.0 + kappa * zenith_cubed)
        else:
            epsilon = 1.0

        # Get Perez coefficients for this epsilon bin
        F11, F12, F13, F21, F22, F23 = self._get_perez_coefficients(epsilon)

        # Calculate F1 and F2 coefficients
        F1 = max(0.0, F11 + F12 * delta + F13 * zenith_rad)
        F2 = F21 + F22 * delta + F23 * zenith_rad

        # Calculate geometric factors
        # a = max(0, cos(incident_angle))
        # b = max(0.087, cos(zenith))  # 0.087 rad = 5° minimum
        a = max(0.0, math.cos(incident_rad))
        b = max(0.087, math.cos(zenith_rad))

        # Calculate three Perez components

        # 1. Isotropic dome component
        I_dome = dhi_w_m2 * (1 - F1) * (1 + math.cos(tilt_rad)) / 2.0

        # 2. Circumsolar component
        if b > 0:
            I_circumsolar = dhi_w_m2 * F1 * a / b
        else:
            I_circumsolar = 0.0

        # 3. Horizon brightening component
        I_horizon = dhi_w_m2 * F2 * math.sin(tilt_rad)

        # Total diffuse irradiance
        total_diffuse = I_dome + I_circumsolar + I_horizon

        # Ensure non-negative
        return max(0.0, total_diffuse)

    def _get_perez_coefficients(self, epsilon: float) -> Tuple[float, float, float, float, float, float]:
        """
        Get Perez coefficients for a given sky clearness value.

        Args:
            epsilon: Sky clearness parameter

        Returns:
            Tuple of (F11, F12, F13, F21, F22, F23)
        """
        # Find appropriate epsilon bin
        for bin_num in range(1, 9):
            eps_low, eps_high, coeffs = self.PEREZ_COEFFICIENTS[bin_num]
            if eps_low <= epsilon < eps_high:
                return tuple(coeffs)

        # Default to bin 1 if out of range
        return tuple(self.PEREZ_COEFFICIENTS[1][2])


class IsotropicSkyModel:
    """
    Simple isotropic sky model for diffuse irradiance on tilted surfaces.

    Assumes uniform sky radiance distribution (no circumsolar or horizon brightening).
    Formula: DHI_tilted = DHI_horizontal × (1 + cos(tilt)) / 2

    This is computationally efficient and reasonably accurate for overcast conditions.
    """

    def calculate_diffuse_on_tilted_surface(
        self,
        dhi_w_m2: float,
        surface_tilt_deg: float
    ) -> float:
        """
        Calculate diffuse irradiance on a tilted surface using isotropic assumption.

        Args:
            dhi_w_m2: Diffuse horizontal irradiance (W/m²)
            surface_tilt_deg: Surface tilt from horizontal (degrees)

        Returns:
            diffuse_irradiance_W_m2: Diffuse irradiance on the tilted surface
        """
        if dhi_w_m2 <= 0:
            return 0.0

        # Isotropic view factor: (1 + cos(tilt)) / 2
        tilt_rad = math.radians(surface_tilt_deg)
        view_factor_sky = (1.0 + math.cos(tilt_rad)) / 2.0

        return dhi_w_m2 * view_factor_sky


class GroundReflectedRadiation:
    """
    Calculates ground-reflected solar radiation onto tilted surfaces.

    Assumes isotropic ground reflection (Lambertian surface).
    """

    def __init__(self, ground_reflectance: float = 0.2):
        """
        Initialize ground reflection calculator.

        Args:
            ground_reflectance: Ground albedo (0-1)
                               Typical values:
                               - 0.2: Grass, soil, asphalt
                               - 0.3-0.4: Concrete
                               - 0.6-0.8: Fresh snow
        """
        if not 0.0 <= ground_reflectance <= 1.0:
            raise ValueError("Ground reflectance must be between 0 and 1")
        self.rho_ground = ground_reflectance

    def calculate_reflected_irradiance(
        self,
        ghi_w_m2: float,
        surface_tilt_deg: float
    ) -> float:
        """
        Calculate ground-reflected irradiance on a tilted surface.

        Formula: I_ground = GHI × ρ_ground × (1 - cos(tilt)) / 2

        Args:
            ghi_w_m2: Global horizontal irradiance (W/m²)
            surface_tilt_deg: Surface tilt from horizontal (degrees)

        Returns:
            reflected_irradiance_W_m2: Ground-reflected irradiance
        """
        if ghi_w_m2 <= 0:
            return 0.0

        # Ground view factor: (1 - cos(tilt)) / 2
        tilt_rad = math.radians(surface_tilt_deg)
        view_factor_ground = (1.0 - math.cos(tilt_rad)) / 2.0

        return ghi_w_m2 * self.rho_ground * view_factor_ground


def test_diffuse_models():
    """Test diffuse radiation models with example scenarios."""
    print("=" * 70)
    print("Solar Diffuse Radiation Models - Test Cases")
    print("=" * 70)

    # Test conditions
    dhi = 100.0  # W/m²
    dni = 800.0  # W/m²
    ghi = 900.0  # W/m²
    sun_zenith = 30.0  # degrees
    sun_azimuth = 180.0  # South
    extraterrestrial = 1361.0  # W/m²

    # Test 1: Horizontal surface (should equal DHI)
    print("\nTest 1: Horizontal Surface (tilt=0°)")
    iso_model = IsotropicSkyModel()
    iso_result = iso_model.calculate_diffuse_on_tilted_surface(dhi, 0)
    print(f"  Isotropic model: {iso_result:.1f} W/m² (expect {dhi:.1f} W/m²)")

    # Test 2: Vertical south-facing wall
    print("\nTest 2: Vertical South-Facing Wall (tilt=90°, azimuth=180°)")
    iso_result = iso_model.calculate_diffuse_on_tilted_surface(dhi, 90)
    print(f"  Isotropic model: {iso_result:.1f} W/m² (expect ~{dhi/2:.1f} W/m²)")

    perez_model = PerezDiffuseSkyModel()
    perez_result = perez_model.calculate_diffuse_on_tilted_surface(
        dhi, dni, ghi, sun_zenith, sun_azimuth, 90, 180, extraterrestrial
    )
    print(f"  Perez model: {perez_result:.1f} W/m²")

    # Test 3: Ground reflection
    print("\nTest 3: Ground-Reflected Radiation")
    ground_calc = GroundReflectedRadiation(ground_reflectance=0.2)

    # Horizontal surface (should see no ground)
    ground_horiz = ground_calc.calculate_reflected_irradiance(ghi, 0)
    print(f"  Horizontal surface: {ground_horiz:.1f} W/m² (expect 0)")

    # Vertical surface (should see half of ground)
    ground_vert = ground_calc.calculate_reflected_irradiance(ghi, 90)
    print(f"  Vertical surface: {ground_vert:.1f} W/m² (expect ~{ghi*0.2*0.5:.1f} W/m²)")

    # Test 4: Total irradiance comparison
    print("\nTest 4: Total Irradiance (South wall, sunny day)")
    from bsm.solar.incident_angle import IncidentAngleCalculator
    calc = IncidentAngleCalculator()
    incident_angle = calc.calculate_incident_angle(sun_zenith, sun_azimuth, 90, 180)
    beam_on_surface = calc.calculate_beam_irradiance_on_surface(dni, incident_angle)

    print(f"  Beam: {beam_on_surface:.1f} W/m²")
    print(f"  Diffuse (Perez): {perez_result:.1f} W/m²")
    print(f"  Ground-reflected: {ground_vert:.1f} W/m²")
    print(f"  Total: {beam_on_surface + perez_result + ground_vert:.1f} W/m²")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_diffuse_models()

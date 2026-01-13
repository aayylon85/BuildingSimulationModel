"""
Solar irradiance orchestrator - combines beam, diffuse, and ground-reflected components.

This module provides a unified interface for calculating total solar irradiance
on tilted surfaces, integrating:
- Direct beam radiation (from solar_incident_angle.py)
- Diffuse sky radiation (from solar_diffuse.py - Perez or isotropic)
- Ground-reflected radiation (from solar_diffuse.py)
"""

from solar_incident_angle import IncidentAngleCalculator
from solar_diffuse import PerezDiffuseSkyModel, IsotropicSkyModel, GroundReflectedRadiation
from typing import Dict


class SolarIrradianceCalculator:
    """
    Main interface for calculating solar irradiance on tilted surfaces.

    Combines beam, diffuse sky, and ground-reflected components into
    total surface irradiance.
    """

    def __init__(
        self,
        diffuse_model: str = 'perez',
        ground_reflectance: float = 0.2
    ):
        """
        Initialize solar irradiance calculator.

        Args:
            diffuse_model: Model for diffuse sky radiation
                          'perez' - Anisotropic Perez model (more accurate)
                          'isotropic' - Simple isotropic model (faster)
            ground_reflectance: Ground albedo (0-1, default 0.2 for typical ground)
        """
        self.incident_calc = IncidentAngleCalculator()

        if diffuse_model.lower() == 'perez':
            self.diffuse_model = PerezDiffuseSkyModel()
            self.diffuse_model_type = 'perez'
        elif diffuse_model.lower() == 'isotropic':
            self.diffuse_model = IsotropicSkyModel()
            self.diffuse_model_type = 'isotropic'
        else:
            raise ValueError(f"Unknown diffuse model: {diffuse_model}. Use 'perez' or 'isotropic'")

        self.ground_reflection = GroundReflectedRadiation(ground_reflectance)

    def calculate_surface_irradiance(
        self,
        weather_data: Dict,
        surface_tilt_deg: float,
        surface_azimuth_deg: float
    ) -> Dict[str, float]:
        """
        Calculate total solar irradiance on a tilted surface.

        Args:
            weather_data: Dictionary containing:
                - 'solar_dni_w_m2': Direct normal irradiance
                - 'solar_dhi_w_m2': Diffuse horizontal irradiance
                - 'solar_ghi_w_m2': Global horizontal irradiance
                - 'sun_zenith_deg': Solar zenith angle
                - 'sun_azimuth_deg': Solar azimuth angle
                - 'extraterrestrial_w_m2': Extraterrestrial irradiance
                - 'is_sunlit': Boolean, True if sun is above horizon
            surface_tilt_deg: Surface tilt from horizontal (0=horizontal, 90=vertical)
            surface_azimuth_deg: Surface azimuth (0=North, 90=East, 180=South, 270=West)

        Returns:
            Dictionary with irradiance components:
                - 'beam': Direct beam irradiance on surface (W/m²)
                - 'diffuse_sky': Diffuse sky irradiance on surface (W/m²)
                - 'ground_reflected': Ground-reflected irradiance (W/m²)
                - 'total': Total irradiance on surface (W/m²)
                - 'incident_angle_deg': Incident angle for beam component
        """
        # Extract weather data
        dni = weather_data.get('solar_dni_w_m2', 0)
        dhi = weather_data.get('solar_dhi_w_m2', 0)
        ghi = weather_data.get('solar_ghi_w_m2', 0)
        sun_zenith = weather_data.get('sun_zenith_deg', 90)
        sun_azimuth = weather_data.get('sun_azimuth_deg', 180)
        extraterrestrial = weather_data.get('extraterrestrial_w_m2', 1361)
        is_sunlit = weather_data.get('is_sunlit', False)

        # Initialize result
        result = {
            'beam': 0.0,
            'diffuse_sky': 0.0,
            'ground_reflected': 0.0,
            'total': 0.0,
            'incident_angle_deg': 90.0
        }

        # If sun is not up or no radiation, return zeros
        if not is_sunlit or (dni == 0 and dhi == 0 and ghi == 0):
            return result

        # 1. Calculate beam component
        incident_angle = self.incident_calc.calculate_incident_angle(
            sun_zenith, sun_azimuth,
            surface_tilt_deg, surface_azimuth_deg
        )
        result['incident_angle_deg'] = incident_angle

        beam_on_surface = self.incident_calc.calculate_beam_irradiance_on_surface(
            dni, incident_angle
        )
        result['beam'] = beam_on_surface

        # 2. Calculate diffuse sky component
        if self.diffuse_model_type == 'perez':
            diffuse_sky = self.diffuse_model.calculate_diffuse_on_tilted_surface(
                dhi, dni, ghi,
                sun_zenith, sun_azimuth,
                surface_tilt_deg, surface_azimuth_deg,
                extraterrestrial
            )
        else:  # isotropic
            diffuse_sky = self.diffuse_model.calculate_diffuse_on_tilted_surface(
                dhi, surface_tilt_deg
            )
        result['diffuse_sky'] = diffuse_sky

        # 3. Calculate ground-reflected component
        ground_reflected = self.ground_reflection.calculate_reflected_irradiance(
            ghi, surface_tilt_deg
        )
        result['ground_reflected'] = ground_reflected

        # 4. Calculate total
        result['total'] = beam_on_surface + diffuse_sky + ground_reflected

        return result

    def calculate_all_surfaces(
        self,
        weather_data: Dict,
        surface_definitions: Dict
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate irradiance for all surfaces in a building.

        Args:
            weather_data: Weather dictionary (same as calculate_surface_irradiance)
            surface_definitions: Dictionary of surface properties
                Example: {'south_wall': {'tilt': 90, 'azimuth': 180}, ...}

        Returns:
            Dictionary mapping surface names to irradiance results
        """
        results = {}

        for surface_name, surface_props in surface_definitions.items():
            tilt = surface_props.get('tilt', 90)
            azimuth = surface_props.get('azimuth', 180)

            results[surface_name] = self.calculate_surface_irradiance(
                weather_data, tilt, azimuth
            )

        return results


def test_irradiance_calculator():
    """Test the solar irradiance orchestrator."""
    print("=" * 70)
    print("Solar Irradiance Calculator - Test")
    print("=" * 70)

    # Example weather data (sunny day at solar noon)
    weather = {
        'solar_dni_w_m2': 800.0,
        'solar_dhi_w_m2': 100.0,
        'solar_ghi_w_m2': 900.0,
        'sun_zenith_deg': 30.0,
        'sun_azimuth_deg': 180.0,
        'extraterrestrial_w_m2': 1361.0,
        'is_sunlit': True
    }

    # Test surfaces
    surfaces = {
        'roof': {'tilt': 0, 'azimuth': 0},
        'south_wall': {'tilt': 90, 'azimuth': 180},
        'north_wall': {'tilt': 90, 'azimuth': 0},
        'floor': {'tilt': 180, 'azimuth': 0}
    }

    # Test with Perez model
    print("\n--- Using Perez Anisotropic Sky Model ---")
    calc_perez = SolarIrradianceCalculator(diffuse_model='perez', ground_reflectance=0.2)
    results_perez = calc_perez.calculate_all_surfaces(weather, surfaces)

    for surface_name, result in results_perez.items():
        print(f"\n{surface_name}:")
        print(f"  Beam:              {result['beam']:6.1f} W/m²")
        print(f"  Diffuse sky:       {result['diffuse_sky']:6.1f} W/m²")
        print(f"  Ground reflected:  {result['ground_reflected']:6.1f} W/m²")
        print(f"  Total:             {result['total']:6.1f} W/m²")
        print(f"  Incident angle:    {result['incident_angle_deg']:6.1f}°")

    # Test with isotropic model for comparison
    print("\n--- Using Isotropic Sky Model (for comparison) ---")
    calc_iso = SolarIrradianceCalculator(diffuse_model='isotropic', ground_reflectance=0.2)
    results_iso = calc_iso.calculate_all_surfaces(weather, surfaces)

    print(f"\nSouth wall diffuse comparison:")
    print(f"  Perez:     {results_perez['south_wall']['diffuse_sky']:.1f} W/m²")
    print(f"  Isotropic: {results_iso['south_wall']['diffuse_sky']:.1f} W/m²")
    print(f"  Difference: {results_perez['south_wall']['diffuse_sky'] - results_iso['south_wall']['diffuse_sky']:.1f} W/m² (Perez accounts for circumsolar)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_irradiance_calculator()

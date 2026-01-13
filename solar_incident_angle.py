"""
Solar incident angle calculations for tilted surfaces.

Calculates the angle between incoming solar radiation and the surface normal,
which determines how much solar radiation is received by a tilted surface.
"""

import math
from typing import Tuple


class IncidentAngleCalculator:
    """
    Calculates incident angle between sun rays and surface normal using spherical geometry.

    The incident angle (θ) is the angle between the sun's beam direction and the
    surface normal vector. It determines the effective solar irradiance on the surface:
    Effective irradiance = DNI × cos(θ) for θ < 90°, else 0
    """

    @staticmethod
    def calculate_incident_angle(
        sun_zenith_deg: float,
        sun_azimuth_deg: float,
        surface_tilt_deg: float,
        surface_azimuth_deg: float
    ) -> float:
        """
        Calculate incident angle between sun and surface normal using spherical trigonometry.

        Args:
            sun_zenith_deg: Solar zenith angle in degrees (0° = overhead, 90° = horizon)
            sun_azimuth_deg: Solar azimuth in degrees (0° = North, 90° = East, 180° = South)
            surface_tilt_deg: Surface tilt from horizontal (0° = horizontal, 90° = vertical, 180° = downward)
            surface_azimuth_deg: Surface azimuth in degrees (0° = North, 90° = East, 180° = South)

        Returns:
            incident_angle_deg: Angle between sun ray and surface normal in degrees
                               (0° = perpendicular/normal, 90° = parallel/grazing)

        Formula (Duffie & Beckman):
            cos(θ) = cos(Z)cos(Σ) + sin(Z)sin(Σ)cos(ψ_sun - ψ_surf)

            where:
                θ = incident angle
                Z = solar zenith angle
                Σ = surface tilt angle from horizontal
                ψ_sun = solar azimuth
                ψ_surf = surface azimuth
        """
        # Convert to radians
        zenith_rad = math.radians(sun_zenith_deg)
        sun_azimuth_rad = math.radians(sun_azimuth_deg)
        tilt_rad = math.radians(surface_tilt_deg)
        surf_azimuth_rad = math.radians(surface_azimuth_deg)

        # Calculate cos(incident angle) using spherical trigonometry
        cos_incident = (
            math.cos(zenith_rad) * math.cos(tilt_rad)
            + math.sin(zenith_rad) * math.sin(tilt_rad)
            * math.cos(sun_azimuth_rad - surf_azimuth_rad)
        )

        # Clamp to valid range [-1, 1] to handle numerical errors
        cos_incident = max(-1.0, min(1.0, cos_incident))

        # Calculate incident angle
        incident_angle_rad = math.acos(cos_incident)
        incident_angle_deg = math.degrees(incident_angle_rad)

        return incident_angle_deg

    @staticmethod
    def calculate_beam_irradiance_on_surface(
        direct_normal_irradiance: float,
        incident_angle_deg: float
    ) -> float:
        """
        Calculate beam solar irradiance on a tilted surface.

        Args:
            direct_normal_irradiance: Direct normal irradiance (DNI) in W/m²
            incident_angle_deg: Incident angle in degrees

        Returns:
            beam_irradiance_W_m2: Beam irradiance on the tilted surface in W/m²

        Formula:
            I_beam = DNI × cos(θ) for θ < 90°
            I_beam = 0 for θ >= 90° (sun behind surface)
        """
        # If sun is behind surface (incident angle >= 90°), no beam irradiance
        if incident_angle_deg >= 90.0:
            return 0.0

        # Project DNI onto surface
        incident_angle_rad = math.radians(incident_angle_deg)
        beam_irradiance = direct_normal_irradiance * math.cos(incident_angle_rad)

        # Ensure non-negative (handle numerical errors)
        return max(0.0, beam_irradiance)

    @staticmethod
    def is_surface_sunlit(incident_angle_deg: float) -> bool:
        """
        Determine if a surface is receiving direct sunlight.

        Args:
            incident_angle_deg: Incident angle in degrees

        Returns:
            is_sunlit: True if surface receives direct sunlight (incident angle < 90°)
        """
        return incident_angle_deg < 90.0


def test_incident_angle():
    """
    Test incident angle calculator with known geometries.
    """
    calc = IncidentAngleCalculator()

    print("=" * 70)
    print("Solar Incident Angle Calculator Tests")
    print("=" * 70)

    # Test 1: Sun directly overhead (zenith=0°), horizontal surface (tilt=0°)
    # Should have incident angle = 0° (perpendicular)
    incident = calc.calculate_incident_angle(
        sun_zenith_deg=0,
        sun_azimuth_deg=180,
        surface_tilt_deg=0,
        surface_azimuth_deg=180
    )
    print(f"\nTest 1: Sun overhead, horizontal surface")
    print(f"  Sun: zenith=0°, azimuth=180° (South)")
    print(f"  Surface: tilt=0° (horizontal), azimuth=180° (South)")
    print(f"  Incident angle: {incident:.2f}° (expect 0° - perpendicular)")

    # Test 2: Sun at 45° zenith facing south, south-facing vertical wall
    # Should have incident angle = 45°
    incident = calc.calculate_incident_angle(
        sun_zenith_deg=45,
        sun_azimuth_deg=180,
        surface_tilt_deg=90,
        surface_azimuth_deg=180
    )
    beam = calc.calculate_beam_irradiance_on_surface(800, incident)
    print(f"\nTest 2: Sun at 45° elevation, south-facing vertical wall")
    print(f"  Sun: zenith=45°, azimuth=180° (South)")
    print(f"  Surface: tilt=90° (vertical), azimuth=180° (South)")
    print(f"  Incident angle: {incident:.2f}° (expect 45°)")
    print(f"  Beam irradiance: {beam:.1f} W/m² (800 W/m² DNI × cos(45°) = 565.7 W/m²)")

    # Test 3: Sun facing south, north-facing wall (sun behind surface)
    # Should have incident angle > 90°, no beam irradiance
    incident = calc.calculate_incident_angle(
        sun_zenith_deg=45,
        sun_azimuth_deg=180,
        surface_tilt_deg=90,
        surface_azimuth_deg=0
    )
    beam = calc.calculate_beam_irradiance_on_surface(800, incident)
    is_sunlit = calc.is_surface_sunlit(incident)
    print(f"\nTest 3: Sun facing south, north-facing wall")
    print(f"  Sun: zenith=45°, azimuth=180° (South)")
    print(f"  Surface: tilt=90° (vertical), azimuth=0° (North)")
    print(f"  Incident angle: {incident:.2f}° (expect >90° - sun behind surface)")
    print(f"  Is sunlit: {is_sunlit} (expect False)")
    print(f"  Beam irradiance: {beam:.1f} W/m² (expect 0)")

    # Test 4: Summer solstice solar noon in Denver (lat 39.7°)
    # Sun zenith ≈ 16° facing south, south-facing vertical wall
    # Incident angle should be 74° (90° - 16° for vertical south wall at solar noon)
    incident = calc.calculate_incident_angle(
        sun_zenith_deg=16,
        sun_azimuth_deg=180,
        surface_tilt_deg=90,
        surface_azimuth_deg=180
    )
    beam = calc.calculate_beam_irradiance_on_surface(900, incident)
    print(f"\nTest 4: Denver summer solstice solar noon, south vertical wall")
    print(f"  Sun: zenith=16° (high in sky), azimuth=180° (South)")
    print(f"  Surface: tilt=90° (vertical), azimuth=180° (South)")
    print(f"  Incident angle: {incident:.2f}° (expect 74° = 90° - 16°)")
    print(f"  Beam irradiance: {beam:.1f} W/m²")

    # Test 5: Horizontal roof in Denver summer
    incident = calc.calculate_incident_angle(
        sun_zenith_deg=16,
        sun_azimuth_deg=180,
        surface_tilt_deg=0,
        surface_azimuth_deg=0
    )
    beam = calc.calculate_beam_irradiance_on_surface(900, incident)
    print(f"\nTest 5: Denver summer solstice solar noon, horizontal roof")
    print(f"  Sun: zenith=16° (high in sky), azimuth=180°")
    print(f"  Surface: tilt=0° (horizontal)")
    print(f"  Incident angle: {incident:.2f}° (expect 16° - same as zenith)")
    print(f"  Beam irradiance: {beam:.1f} W/m² (expect ~864 W/m²)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_incident_angle()

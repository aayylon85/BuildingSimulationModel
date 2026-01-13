"""
Solar position calculations using simplified PSA algorithm.

Calculates solar zenith angle, azimuth angle, and extraterrestrial irradiance
for a given location and time. Results are cached for performance.
"""

import math
from datetime import datetime, timezone
from typing import Tuple


class SolarPositionCalculator:
    """
    Calculates solar position angles using simplified solar position algorithm.

    Results are cached by (latitude, longitude, timestamp) tuple for performance
    in building simulations where multiple surfaces share the same solar position.
    """

    # Solar constant (W/m²)
    SOLAR_CONSTANT = 1361.0

    def __init__(self, latitude: float, longitude: float, timezone_offset_hours: float = 0):
        """
        Initialize solar position calculator for a specific location.

        Args:
            latitude: Latitude in degrees (-90 to 90, positive = North)
            longitude: Longitude in degrees (-180 to 180, positive = East)
            timezone_offset_hours: Timezone offset from UTC in hours (e.g., -7 for MST)
        """
        if not -90 <= latitude <= 90:
            raise ValueError("Latitude must be between -90 and 90 degrees")
        if not -180 <= longitude <= 180:
            raise ValueError("Longitude must be between -180 and 180 degrees")

        self.latitude = latitude
        self.longitude = longitude
        self.tz_offset = timezone_offset_hours
        self._cache = {}

    def calculate_sun_position(self, dt: datetime) -> Tuple[float, float, float, bool]:
        """
        Calculate solar position for a given datetime.

        Args:
            dt: Datetime object (assumed to be in local time matching tz_offset)

        Returns:
            Tuple of (zenith_deg, azimuth_deg, extraterrestrial_irrad_W_m2, is_sunlit)
            - zenith_deg: Solar zenith angle in degrees (0° = overhead, 90° = horizon)
            - azimuth_deg: Solar azimuth in degrees (0° = North, 90° = East, 180° = South, 270° = West)
            - extraterrestrial_irrad_W_m2: Extraterrestrial irradiance on surface normal to sun
            - is_sunlit: True if sun is above horizon (zenith < 90°)
        """
        # Create cache key
        cache_key = (dt.year, dt.month, dt.day, dt.hour, dt.minute)

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Calculate solar position
        zenith, azimuth, extraterrestrial, is_sunlit = self._calculate_position(dt)

        # Cache result
        self._cache[cache_key] = (zenith, azimuth, extraterrestrial, is_sunlit)

        # Limit cache size to prevent memory issues
        if len(self._cache) > 10000:
            # Remove oldest 20% of entries
            keys_to_remove = list(self._cache.keys())[:2000]
            for key in keys_to_remove:
                del self._cache[key]

        return zenith, azimuth, extraterrestrial, is_sunlit

    def _calculate_position(self, dt: datetime) -> Tuple[float, float, float, bool]:
        """
        Internal method to calculate solar position using simplified algorithm.

        Based on simplified equations from Duffie & Beckman "Solar Engineering
        of Thermal Processes" (accuracy ±1° for typical applications).
        """
        # Convert to day of year
        day_of_year = dt.timetuple().tm_yday

        # Calculate declination angle (δ) - Spencer equation
        # Accounts for Earth's axial tilt variation through the year
        day_angle = 2.0 * math.pi * (day_of_year - 1) / 365.0
        declination_rad = (
            0.006918
            - 0.399912 * math.cos(day_angle)
            + 0.070257 * math.sin(day_angle)
            - 0.006758 * math.cos(2 * day_angle)
            + 0.000907 * math.sin(2 * day_angle)
            - 0.002697 * math.cos(3 * day_angle)
            + 0.00148 * math.sin(3 * day_angle)
        )

        # Calculate equation of time (minutes) - accounts for orbital eccentricity
        eot_minutes = (
            229.18 * (
                0.000075
                + 0.001868 * math.cos(day_angle)
                - 0.032077 * math.sin(day_angle)
                - 0.014615 * math.cos(2 * day_angle)
                - 0.040849 * math.sin(2 * day_angle)
            )
        )

        # Calculate solar time
        # Convert local time to solar time accounting for longitude and equation of time
        decimal_hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0

        # Time correction for longitude and equation of time
        # Standard meridian for time zone (15° per hour from Greenwich)
        # Negative tz_offset (west of Greenwich) gives negative standard meridian
        standard_meridian = 15.0 * self.tz_offset

        # Longitude correction: 4 minutes per degree from standard meridian
        longitude_correction = 4.0 * (standard_meridian - self.longitude)

        # Total time correction
        time_correction = longitude_correction + eot_minutes
        solar_time_hours = decimal_hour + time_correction / 60.0

        # Hour angle (ω) - angular distance from solar noon
        hour_angle_deg = 15.0 * (solar_time_hours - 12.0)
        hour_angle_rad = math.radians(hour_angle_deg)

        # Convert latitude and declination to radians
        lat_rad = math.radians(self.latitude)

        # Calculate solar zenith angle (θz) using spherical trigonometry
        # cos(θz) = sin(φ)sin(δ) + cos(φ)cos(δ)cos(ω)
        # where φ = latitude, δ = declination, ω = hour angle
        cos_zenith = (
            math.sin(lat_rad) * math.sin(declination_rad)
            + math.cos(lat_rad) * math.cos(declination_rad) * math.cos(hour_angle_rad)
        )

        # Clamp to valid range to handle numerical errors
        cos_zenith = max(-1.0, min(1.0, cos_zenith))
        zenith_rad = math.acos(cos_zenith)
        zenith_deg = math.degrees(zenith_rad)

        # Determine if sun is above horizon
        is_sunlit = zenith_deg < 90.0

        # Calculate solar azimuth angle (ψ) measured from North
        # Azimuth calculation uses different formulas for different cases
        if is_sunlit:
            sin_declination = math.sin(declination_rad)
            cos_declination = math.cos(declination_rad)
            sin_hour_angle = math.sin(hour_angle_rad)
            cos_latitude = math.cos(lat_rad)
            sin_zenith = math.sin(zenith_rad)

            # sin(ψ) = cos(δ)sin(ω) / sin(θz)
            if abs(sin_zenith) > 1e-6:
                sin_azimuth = cos_declination * sin_hour_angle / sin_zenith
                sin_azimuth = max(-1.0, min(1.0, sin_azimuth))

                # cos(ψ) = (sin(δ)cos(φ) - cos(δ)sin(φ)cos(ω)) / sin(θz)
                cos_azimuth = (sin_declination * cos_latitude -
                              cos_declination * math.sin(lat_rad) * math.cos(hour_angle_rad)) / sin_zenith
                cos_azimuth = max(-1.0, min(1.0, cos_azimuth))

                # Use atan2 to get correct quadrant
                azimuth_rad = math.atan2(sin_azimuth, cos_azimuth)
                azimuth_deg = math.degrees(azimuth_rad)

                # Convert to 0-360° range (0° = North, 90° = East, 180° = South, 270° = West)
                if azimuth_deg < 0:
                    azimuth_deg += 360.0
            else:
                # Sun directly overhead - azimuth undefined, use 0 (North)
                azimuth_deg = 0.0
        else:
            # Sun below horizon - set azimuth to 0
            azimuth_deg = 0.0

        # Calculate extraterrestrial irradiance
        # Accounts for Earth-Sun distance variation (eccentricity correction)
        eccentricity_correction = 1.0 + 0.033 * math.cos(2.0 * math.pi * day_of_year / 365.0)
        extraterrestrial_irrad = self.SOLAR_CONSTANT * eccentricity_correction

        return zenith_deg, azimuth_deg, extraterrestrial_irrad, is_sunlit

    def clear_cache(self):
        """Clear the position calculation cache."""
        self._cache.clear()


def test_solar_position():
    """
    Test solar position calculator with known values.

    Example: Denver, CO at solar noon on summer solstice should have:
    - Low zenith angle (sun high in sky)
    - Azimuth near 180° (south)
    """
    # Denver, CO coordinates
    latitude = 39.7392
    longitude = -104.9903

    calc = SolarPositionCalculator(latitude, longitude, timezone_offset_hours=-7)

    # Test at 1:00 PM MST (13:00) on June 21 (close to solar noon for Denver)
    # Solar noon occurs when hour angle = 0, which happens around 12:56 MST for Denver
    test_dt = datetime(2024, 6, 21, 13, 0, 0)

    # Debug: manually check solar position calculation
    day_of_year = test_dt.timetuple().tm_yday
    print(f"Debug - Day of year: {day_of_year}")

    zenith, azimuth, extraterrestrial, is_sunlit = calc.calculate_sun_position(test_dt)

    print(f"\nDenver, CO - Summer Solstice at ~1PM MST:")
    print(f"  Zenith angle: {zenith:.2f}° (expect ~16° for latitude 39.7°)")
    print(f"  Azimuth: {azimuth:.2f}° (expect ~180° near solar noon)")
    print(f"  Extraterrestrial irradiance: {extraterrestrial:.1f} W/m²")
    print(f"  Is sunlit: {is_sunlit}")

    # Winter solstice at 1:00 PM MST
    test_dt = datetime(2024, 12, 21, 13, 0, 0)

    zenith, azimuth, extraterrestrial, is_sunlit = calc.calculate_sun_position(test_dt)

    print(f"\nDenver, CO - Winter Solstice at ~1PM MST:")
    print(f"  Zenith angle: {zenith:.2f}° (expect ~63° for latitude 39.7°)")
    print(f"  Azimuth: {azimuth:.2f}° (expect ~180° near solar noon)")
    print(f"  Extraterrestrial irradiance: {extraterrestrial:.1f} W/m²")
    print(f"  Is sunlit: {is_sunlit}")

    # Test early morning (should have high zenith, eastern azimuth)
    test_dt = datetime(2024, 6, 21, 7, 0, 0)
    zenith, azimuth, extraterrestrial, is_sunlit = calc.calculate_sun_position(test_dt)

    print(f"\nDenver, CO - Summer Solstice at 7AM MST:")
    print(f"  Zenith angle: {zenith:.2f}° (expect ~70-80° - sun low in sky)")
    print(f"  Azimuth: {azimuth:.2f}° (expect ~60-90° - eastern sky)")
    print(f"  Is sunlit: {is_sunlit}")


if __name__ == "__main__":
    test_solar_position()

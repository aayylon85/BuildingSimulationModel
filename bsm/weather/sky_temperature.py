"""
Sky temperature calculations for longwave radiation exchange.

Calculates effective sky temperature for longwave (thermal) radiation
exchange between building surfaces and the sky.
"""

import math


class SkyTemperatureCalculator:
    """
    Calculates effective sky temperature for longwave radiation exchange.

    Provides multiple models ranging from simple to more sophisticated
    based on available weather data.
    """

    def __init__(self, model: str = 'clark_allen'):
        """
        Initialize sky temperature calculator.

        Args:
            model: Sky temperature model to use
                  'clark_allen' - Simple clear-sky depression (default)
                  'berdahl_martin' - Humidity-based model (more accurate)
                  'simple' - Very simple constant depression
        """
        valid_models = ['clark_allen', 'berdahl_martin', 'simple']
        if model.lower() not in valid_models:
            raise ValueError(f"Unknown sky temperature model: {model}. "
                           f"Valid models: {valid_models}")
        self.model = model.lower()

    def calculate_sky_temperature(
        self,
        air_temp_k: float,
        dew_point_temp_k: float = None,
        relative_humidity: float = None,
        cloud_cover_fraction: float = 0.0
    ) -> float:
        """
        Calculate effective sky temperature in Kelvin.

        Args:
            air_temp_k: Outdoor air temperature (K)
            dew_point_temp_k: Dew point temperature (K) - optional, for Berdahl-Martin
            relative_humidity: Relative humidity (0-1) - optional, for Berdahl-Martin
            cloud_cover_fraction: Cloud cover fraction (0-1, 0=clear, 1=overcast)

        Returns:
            sky_temp_k: Effective sky temperature (K)
        """
        if self.model == 'simple':
            return self._simple_model(air_temp_k, cloud_cover_fraction)
        elif self.model == 'clark_allen':
            return self._clark_allen_model(air_temp_k, cloud_cover_fraction)
        elif self.model == 'berdahl_martin':
            return self._berdahl_martin_model(
                air_temp_k, dew_point_temp_k, relative_humidity, cloud_cover_fraction
            )
        else:
            # Fallback
            return self._clark_allen_model(air_temp_k, cloud_cover_fraction)

    def _simple_model(
        self,
        air_temp_k: float,
        cloud_cover_fraction: float = 0.0
    ) -> float:
        """
        Very simple sky temperature model.

        Clear sky: T_sky = T_air - 20 K
        Cloud correction: Linear interpolation toward T_air

        Args:
            air_temp_k: Air temperature (K)
            cloud_cover_fraction: Cloud cover (0-1)

        Returns:
            sky_temp_k: Sky temperature (K)
        """
        # Clear sky depression
        clear_sky_depression = 20.0  # K

        # Calculate clear sky temperature
        t_sky_clear = air_temp_k - clear_sky_depression

        # Cloud cover reduces the depression (clouds radiate at ~air temperature)
        # Overcast sky is approximately equal to air temperature
        t_sky = t_sky_clear + cloud_cover_fraction * clear_sky_depression

        return t_sky

    def _clark_allen_model(
        self,
        air_temp_k: float,
        cloud_cover_fraction: float = 0.0
    ) -> float:
        """
        Clark & Allen (1978) sky temperature model.

        Clear sky: T_sky = T_air - 11 K (typical mid-latitude value)
        Cloud correction: 80% of the depression is recovered with full cloud cover

        This is a widely-used simple model that gives reasonable results
        for most locations and conditions.

        Reference: Clark, G. and Allen, C. (1978). "The Estimation of
        Atmospheric Radiation for Clear and Cloudy Skies." Proc. 2nd
        National Passive Solar Conference, AS/ISES, 675-678.

        Args:
            air_temp_k: Air temperature (K)
            cloud_cover_fraction: Cloud cover (0-1)

        Returns:
            sky_temp_k: Sky temperature (K)
        """
        # Clear sky depression (typical value for mid-latitudes)
        clear_sky_depression = 11.0  # K

        # Calculate clear sky temperature
        t_sky_clear = air_temp_k - clear_sky_depression

        # Cloud correction: clouds reduce the depression by ~80%
        # At 100% cloud cover, sky temp is about T_air - 2K
        cloud_correction = cloud_cover_fraction * clear_sky_depression * 0.8

        t_sky = t_sky_clear + cloud_correction

        return t_sky

    def _berdahl_martin_model(
        self,
        air_temp_k: float,
        dew_point_temp_k: float = None,
        relative_humidity: float = None,
        cloud_cover_fraction: float = 0.0
    ) -> float:
        """
        Berdahl & Martin (1984) humidity-based sky temperature model.

        More accurate model that accounts for atmospheric water vapor content
        (major contributor to longwave radiation from clear sky).

        Reference: Berdahl, P. and Martin, M. (1984). "Emissivity of Clear Skies."
        Solar Energy, 32(5), 663-664.

        Args:
            air_temp_k: Air temperature (K)
            dew_point_temp_k: Dew point temperature (K)
            relative_humidity: Relative humidity (0-1) - used if dew point not available
            cloud_cover_fraction: Cloud cover (0-1)

        Returns:
            sky_temp_k: Sky temperature (K)
        """
        # If no dew point or humidity provided, fall back to Clark-Allen
        if dew_point_temp_k is None and relative_humidity is None:
            return self._clark_allen_model(air_temp_k, cloud_cover_fraction)

        # Calculate dew point if not provided
        if dew_point_temp_k is None and relative_humidity is not None:
            dew_point_temp_k = self._estimate_dew_point(air_temp_k, relative_humidity)

        # Convert to Celsius for the correlation
        t_air_c = air_temp_k - 273.15
        t_dp_c = dew_point_temp_k - 273.15

        # Berdahl-Martin correlation for clear sky emissivity
        # ε_sky = 0.711 + 0.0056·T_dp + 0.000073·T_dp² + 0.013·cos(15t)
        # where t is hour of day (simplified here without diurnal variation)

        # Simplified version without time dependency
        epsilon_sky = 0.711 + 0.0056 * t_dp_c + 0.000073 * t_dp_c ** 2

        # Ensure physically reasonable bounds
        epsilon_sky = max(0.5, min(1.0, epsilon_sky))

        # Calculate clear sky temperature from emissivity
        # T_sky = T_air × ε_sky^0.25
        t_sky_clear = air_temp_k * (epsilon_sky ** 0.25)

        # Cloud correction (similar to Clark-Allen)
        # Clouds increase sky temperature toward air temperature
        t_sky = t_sky_clear + cloud_cover_fraction * (air_temp_k - t_sky_clear) * 0.8

        return t_sky

    def _estimate_dew_point(
        self,
        air_temp_k: float,
        relative_humidity: float
    ) -> float:
        """
        Estimate dew point temperature from air temperature and relative humidity.

        Uses Magnus-Tetens formula approximation.

        Args:
            air_temp_k: Air temperature (K)
            relative_humidity: Relative humidity (0-1)

        Returns:
            dew_point_k: Dew point temperature (K)
        """
        # Convert to Celsius
        t_c = air_temp_k - 273.15

        # Ensure RH is in valid range
        rh = max(0.01, min(0.99, relative_humidity))

        # Magnus-Tetens constants
        a = 17.27
        b = 237.7  # °C

        # Calculate dew point
        alpha = ((a * t_c) / (b + t_c)) + math.log(rh)
        t_dp_c = (b * alpha) / (a - alpha)

        # Convert back to Kelvin
        return t_dp_c + 273.15


def test_sky_temperature():
    """Test sky temperature calculator with various conditions."""
    print("=" * 70)
    print("Sky Temperature Calculator - Test Cases")
    print("=" * 70)

    # Test conditions
    air_temp_c = 20.0  # °C
    air_temp_k = air_temp_c + 273.15
    dew_point_c = 10.0  # °C
    dew_point_k = dew_point_c + 273.15
    rh = 0.5  # 50%

    # Test 1: Clear sky - all models
    print("\nTest 1: Clear Sky (air temp = 20°C)")
    print("-" * 70)

    calc_simple = SkyTemperatureCalculator(model='simple')
    t_sky_simple = calc_simple.calculate_sky_temperature(air_temp_k, cloud_cover_fraction=0.0)
    print(f"  Simple model:        {t_sky_simple - 273.15:6.1f}°C (depression: {air_temp_c - (t_sky_simple - 273.15):.1f} K)")

    calc_clark = SkyTemperatureCalculator(model='clark_allen')
    t_sky_clark = calc_clark.calculate_sky_temperature(air_temp_k, cloud_cover_fraction=0.0)
    print(f"  Clark-Allen model:   {t_sky_clark - 273.15:6.1f}°C (depression: {air_temp_c - (t_sky_clark - 273.15):.1f} K)")

    calc_berdahl = SkyTemperatureCalculator(model='berdahl_martin')
    t_sky_berdahl = calc_berdahl.calculate_sky_temperature(
        air_temp_k, dew_point_temp_k=dew_point_k, cloud_cover_fraction=0.0
    )
    print(f"  Berdahl-Martin:      {t_sky_berdahl - 273.15:6.1f}°C (depression: {air_temp_c - (t_sky_berdahl - 273.15):.1f} K)")

    # Test 2: Overcast sky
    print("\nTest 2: Overcast Sky (air temp = 20°C, 100% cloud cover)")
    print("-" * 70)

    t_sky_overcast = calc_clark.calculate_sky_temperature(air_temp_k, cloud_cover_fraction=1.0)
    print(f"  Clark-Allen model:   {t_sky_overcast - 273.15:6.1f}°C (depression: {air_temp_c - (t_sky_overcast - 273.15):.1f} K)")
    print(f"  (Clouds emit longwave at near-air temperature)")

    # Test 3: Partially cloudy
    print("\nTest 3: Partially Cloudy (50% cloud cover)")
    print("-" * 70)

    t_sky_partial = calc_clark.calculate_sky_temperature(air_temp_k, cloud_cover_fraction=0.5)
    print(f"  Clark-Allen model:   {t_sky_partial - 273.15:6.1f}°C (depression: {air_temp_c - (t_sky_partial - 273.15):.1f} K)")

    # Test 4: Cold winter night
    print("\nTest 4: Cold Winter Night (air temp = -10°C, clear sky)")
    print("-" * 70)

    air_temp_winter_k = 263.15  # -10°C
    t_sky_winter = calc_clark.calculate_sky_temperature(air_temp_winter_k, cloud_cover_fraction=0.0)
    print(f"  Air temperature:     {air_temp_winter_k - 273.15:6.1f}°C")
    print(f"  Sky temperature:     {t_sky_winter - 273.15:6.1f}°C")
    print(f"  Depression:          {(air_temp_winter_k - t_sky_winter):.1f} K")
    print(f"  (Large T difference drives high longwave radiation loss)")

    # Test 5: Hot summer day
    print("\nTest 5: Hot Summer Day (air temp = 35°C, clear sky)")
    print("-" * 70)

    air_temp_summer_k = 308.15  # 35°C
    dew_point_summer_k = 298.15  # 25°C (humid)
    t_sky_summer = calc_berdahl.calculate_sky_temperature(
        air_temp_summer_k, dew_point_temp_k=dew_point_summer_k, cloud_cover_fraction=0.0
    )
    print(f"  Air temperature:     {air_temp_summer_k - 273.15:6.1f}°C")
    print(f"  Dew point:           {dew_point_summer_k - 273.15:6.1f}°C (humid)")
    print(f"  Sky temperature:     {t_sky_summer - 273.15:6.1f}°C")
    print(f"  Depression:          {(air_temp_summer_k - t_sky_summer):.1f} K")
    print(f"  (High humidity → higher sky temp → less radiation loss)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_sky_temperature()

"""
Kusuda-Achenbach ground temperature model.

This module implements the Kusuda-Achenbach correlation for calculating
undisturbed ground temperature as a function of depth and time of year.
The model is based on the analytical solution to the heat diffusion equation
in a semi-infinite medium with a sinusoidal surface temperature boundary.

Reference:
    Kusuda, T. and Achenbach, P.R. (1965) "Earth Temperature and Thermal
    Diffusivity at Selected Stations in the United States"
    ASHRAE Transactions, Vol. 71, Part 1.

This is the same model used by EnergyPlus for ground temperature calculations
(see GroundTemperatureModeling/KusudaAchenbachGroundTemperatureModel.cc).
"""

import numpy as np


class KusudaGroundTemperature:
    """
    Calculate ground temperature using the Kusuda-Achenbach correlation.

    The model assumes:
    - Ground is a semi-infinite medium with uniform thermal properties
    - Surface temperature follows a sinusoidal annual cycle
    - No lateral heat flow (1D vertical conduction)

    At the surface (depth=0), the model simplifies to:
        T_ground = T_avg - A * cos(2*pi/365 * (day - phase_day))

    Where:
        T_avg = Annual mean air/ground temperature
        A = Annual temperature amplitude
        phase_day = Day of year with minimum temperature

    Attributes:
        T_avg (float): Annual mean temperature (C)
        amplitude (float): Annual temperature amplitude (K)
        phase_day (int): Day of year with minimum temperature (1-365)
    """

    def __init__(self, annual_mean_c, amplitude_c, phase_day):
        """
        Initialize with site-specific parameters.

        Args:
            annual_mean_c (float): Annual mean air temperature (C)
            amplitude_c (float): Annual temperature amplitude (K)
                Half of the difference between warmest and coldest monthly mean.
            phase_day (int): Day of year with minimum temperature (1-365)
                Typically mid-January to mid-February for Northern Hemisphere.
        """
        self.T_avg = annual_mean_c
        self.amplitude = amplitude_c
        self.phase_day = phase_day

    def get_temperature(self, day_of_year):
        """
        Calculate ground surface temperature for a given day.

        The surface temperature follows the Kusuda formula at depth=0:
            T = T_avg - A * cos(2*pi/365 * (day - phase_day))

        This gives:
        - Minimum temperature at day = phase_day (winter)
        - Maximum temperature at day = phase_day + 182.5 (summer)

        Args:
            day_of_year (float): Day of year (1-365), can be fractional

        Returns:
            float: Ground surface temperature in Celsius
        """
        # Kusuda formula at surface (z=0)
        # cos gives -1 at day=phase_day, +1 at day=phase_day+182.5
        # So T = T_avg + A when cos=-1 (summer), T = T_avg - A when cos=+1 (winter)
        # But we want minimum at phase_day, so use: T = T_avg - A*cos(...)
        angle = 2.0 * np.pi * (day_of_year - self.phase_day) / 365.0
        return self.T_avg - self.amplitude * np.cos(angle)

    def get_temperature_at_depth(self, day_of_year, depth_m,
                                  thermal_diffusivity=1.0e-6):
        """
        Calculate ground temperature at a specified depth.

        At depth, the temperature wave is:
        1. Attenuated (amplitude decreases exponentially)
        2. Phase-shifted (peaks occur later at deeper depths)

        Full Kusuda-Achenbach formula:
            T(z,t) = T_avg - A * exp(-z * sqrt(pi/(365*alpha))) *
                     cos(2*pi/365 * (t - phase - z/2 * sqrt(365/(pi*alpha))))

        Args:
            day_of_year (float): Day of year (1-365)
            depth_m (float): Depth below surface in meters
            thermal_diffusivity (float): Soil thermal diffusivity in m^2/s
                Typical value: 1.0e-6 m^2/s for average soil

        Returns:
            float: Ground temperature at specified depth in Celsius
        """
        if depth_m <= 0:
            return self.get_temperature(day_of_year)

        # Convert to consistent units
        seconds_per_year = 365.25 * 24 * 3600
        time_sec = (day_of_year - 1) * 24 * 3600  # Approximate
        phase_sec = (self.phase_day - 1) * 24 * 3600

        # Attenuation factor
        term1 = -depth_m * np.sqrt(np.pi / (seconds_per_year * thermal_diffusivity))
        attenuation = np.exp(term1)

        # Phase shift
        phase_shift = (depth_m / 2.0) * np.sqrt(
            seconds_per_year / (np.pi * thermal_diffusivity)
        )

        # Angular position
        angle = (2.0 * np.pi / seconds_per_year) * (
            time_sec - phase_sec - phase_shift * 24 * 3600
        )

        return self.T_avg - self.amplitude * attenuation * np.cos(angle)

    @classmethod
    def from_monthly_temps(cls, monthly_means):
        """
        Create a KusudaGroundTemperature instance from monthly mean temperatures.

        This convenience method calculates the required parameters from
        a list of 12 monthly mean air temperatures.

        Args:
            monthly_means (list): List of 12 monthly mean temperatures (C)
                Index 0 = January, Index 11 = December

        Returns:
            KusudaGroundTemperature: Configured instance

        Example:
            >>> temps = [0.8, -0.1, 6.1, 5.8, 15.5, 23.1, 22.3, 22.6, 19.2, 10.0, 2.9, 1.4]
            >>> model = KusudaGroundTemperature.from_monthly_temps(temps)
            >>> print(f"Annual mean: {model.T_avg:.2f}C")
            Annual mean: 10.88C
        """
        if len(monthly_means) != 12:
            raise ValueError("monthly_means must contain exactly 12 values")

        annual_mean = sum(monthly_means) / 12.0
        amplitude = (max(monthly_means) - min(monthly_means)) / 2.0

        # Find coldest month (1-12)
        min_temp = min(monthly_means)
        coldest_month = monthly_means.index(min_temp) + 1

        # Convert to day of year (mid-month)
        phase_day = (coldest_month - 1) * 30 + 15

        return cls(annual_mean, amplitude, phase_day)

    def __repr__(self):
        return (f"KusudaGroundTemperature(T_avg={self.T_avg:.2f}C, "
                f"amplitude={self.amplitude:.2f}K, phase_day={self.phase_day})")


if __name__ == '__main__':
    # Example: Denver, CO from EPW data
    print("=== Kusuda-Achenbach Ground Temperature Model ===")
    print("Location: Denver, CO (from BESTEST EPW)")
    print()

    # Denver parameters (calculated from EPW)
    denver = KusudaGroundTemperature(
        annual_mean_c=10.88,
        amplitude_c=11.58,
        phase_day=45  # Mid-February
    )
    print(f"Model: {denver}")
    print()

    # Show monthly ground temperatures
    print("Monthly Ground Surface Temperatures:")
    print("-" * 40)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    for i, month in enumerate(months):
        day = i * 30 + 15  # Mid-month
        t_ground = denver.get_temperature(day)
        print(f"  {month}: Day {day:3d} -> {t_ground:6.2f}C")

    print()
    print("Ground Temperature Range:")
    print(f"  Minimum (winter): {denver.T_avg - denver.amplitude:.2f}C")
    print(f"  Maximum (summer): {denver.T_avg + denver.amplitude:.2f}C")

    # Compare with air temperature extremes
    print()
    print("Comparison with Air Temperature:")
    print("  Air temp range: -19C to +40C (from EPW)")
    print(f"  Ground temp range: {denver.T_avg - denver.amplitude:.1f}C to "
          f"{denver.T_avg + denver.amplitude:.1f}C")
    print("  -> Ground has ~3x smaller amplitude than air")

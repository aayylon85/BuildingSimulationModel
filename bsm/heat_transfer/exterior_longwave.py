"""
This module calculates the exterior surface longwave radiation flux.

The model considers radiation exchange between an exterior surface and the ground,
sky, air, and surrounding surfaces. It can handle multiple ground and surrounding
surfaces, each with their own view factor and temperature.
"""

import math
from typing import List, Tuple

from bsm.core.constants import STEFAN_BOLTZMANN

class ExternalLongwaveRadiation:
    """
    A class to calculate the longwave radiation flux for a specific exterior surface.

    The properties of the surface (emissivity and tilt) are set during
    initialization. The flux can then be calculated for varying temperature
    conditions, which is typical for time-step based building simulations.
    """

    def __init__(self, surface_emissivity: float, surface_tilt_angle_deg: float):
        """
        Initializes the ExternalLongwaveRadiation calculator for a surface.

        Args:
            surface_emissivity (float): The long-wave emissivity of the surface.
                                        A value between 0 and 1.
            surface_tilt_angle_deg (float): The tilt angle of the surface in degrees.
                                            0 = horizontal roof (facing up)
                                            90 = vertical wall
                                            180 = horizontal floor (facing down)
        """
        if not 0.0 <= surface_emissivity <= 1.0:
            raise ValueError("Surface emissivity must be between 0 and 1.")
        if not 0.0 <= surface_tilt_angle_deg <= 180.0:
            raise ValueError("Surface tilt angle must be between 0 and 180 degrees.")

        self.emissivity = surface_emissivity
        self.tilt_rad = math.radians(surface_tilt_angle_deg)

        # Pre-calculate view factor to the sky and beta factor.
        # See equations 3.65 and 3.66 in the reference document.
        # The user is responsible for providing ground/surrounding view factors
        # that sum with f_sky to 1.
        cos_phi = math.cos(self.tilt_rad)
        self.f_sky = 0.5 * (1.0 + cos_phi)
        
        # Beta is the factor that splits sky radiation between sky and air
        if self.f_sky > 0:
            self.beta = math.sqrt(self.f_sky)
        else:
            self.beta = 0.0

    def calculate_flux(self,
                       surface_temperature_K: float,
                       air_temperature_K: float,
                       sky_temperature_K: float,
                       ground_surfaces: List[Tuple[float, float]] = None,
                       surrounding_surfaces: List[Tuple[float, float]] = None
                      ) -> float:
        """
        Calculates the total exterior longwave radiation heat flux.

        This method applies the Stefan-Boltzmann law for radiation exchange
        with the ground, sky, air, and surrounding surfaces.

        Args:
            surface_temperature_K (float): The outside surface temperature in Kelvin.
            air_temperature_K (float): The outside air temperature in Kelvin.
            sky_temperature_K (float): The effective sky temperature in Kelvin.
            ground_surfaces (list of tuples, optional): A list where each tuple
                contains (view_factor, temperature_K) for a ground surface.
                Defaults to None.
            surrounding_surfaces (list of tuples, optional): A list where each
                tuple contains (view_factor, temperature_K) for a surrounding
                surface. Defaults to None.

        Returns:
            float: The net longwave radiation flux in W/m^2.
                   A positive value indicates heat gain to the surface,
                   a negative value indicates heat loss from the surface.
        """
        if surface_temperature_K < 0 or air_temperature_K < 0 or sky_temperature_K < 0:
            raise ValueError("Temperatures must be in Kelvin and non-negative.")

        # Handle optional arguments
        ground_surfaces = ground_surfaces or []
        surrounding_surfaces = surrounding_surfaces or []

        # --- View Factor Sanity Check ---
        total_ground_f = sum(f for f, t in ground_surfaces)
        total_surrounding_f = sum(f for f, t in surrounding_surfaces)
        total_f = self.f_sky + total_ground_f + total_surrounding_f
        if not math.isclose(total_f, 1.0, rel_tol=1e-5):
            raise ValueError(
                f"The sum of view factors must be 1.0. "
                f"Current sum: {total_f} (Sky: {self.f_sky}, "
                f"Ground: {total_ground_f}, Surrounding: {total_surrounding_f})"
            )
            
        # Calculate the fourth power of temperatures
        t_surf_4 = surface_temperature_K ** 4
        t_air_4 = air_temperature_K ** 4
        t_sky_4 = sky_temperature_K ** 4

        # --- Calculate flux components ---

        # 1. Radiation exchange with the sky (pure sky radiation)
        q_sky = self.emissivity * STEFAN_BOLTZMANN * self.f_sky * self.beta * (t_sky_4 - t_surf_4)

        # 2. Radiation exchange with air (from the sky view component)
        q_air = self.emissivity * STEFAN_BOLTZMANN * self.f_sky * (1.0 - self.beta) * (t_air_4 - t_surf_4)
        
        # 3. Radiation exchange with multiple ground surfaces (Eq. 3.71)
        q_ground = 0.0
        for f_gnd, t_gnd_K in ground_surfaces:
            if t_gnd_K < 0:
                raise ValueError("Ground temperatures must be non-negative Kelvin.")
            q_ground += self.emissivity * STEFAN_BOLTZMANN * f_gnd * (t_gnd_K**4 - t_surf_4)

        # 4. Radiation exchange with multiple surrounding surfaces (Eq. 3.76)
        q_surrounding = 0.0
        for f_srd, t_srd_K in surrounding_surfaces:
            if t_srd_K < 0:
                raise ValueError("Surrounding surface temperatures must be non-negative Kelvin.")
            q_surrounding += self.emissivity * STEFAN_BOLTZMANN * f_srd * (t_srd_K**4 - t_surf_4)

        # Total flux is the sum of all components
        total_flux = q_sky + q_air + q_ground + q_surrounding

        return total_flux

    def calculate_linearized_coefficient(
        self,
        surface_temperature_K: float,
        air_temperature_K: float,
        sky_temperature_K: float,
        ground_surfaces=None,
        surrounding_surfaces=None
    ) -> float:
        """
        Calculate linearized radiation heat transfer coefficient for implicit treatment.

        Instead of calculating q_lwr = ε·σ·F·(T_other^4 - T_surf^4), this method
        calculates a linearized coefficient h_r such that:
            q_lwr ≈ h_r · (T_other - T_surf)

        The linearization is done using the previous iteration's temperature:
            h_r = ε·σ·F·(T_other + T_surf)·(T_other² + T_surf²)

        This allows h_r to be added to the A matrix diagonal, making the radiation
        term implicit and greatly improving convergence.

        Args:
            surface_temperature_K: Exterior surface temperature (K) from previous iteration
            air_temperature_K: Outdoor air temperature (K)
            sky_temperature_K: Effective sky temperature (K)
            ground_surfaces: List of tuples (view_factor, temperature_K) for ground surfaces
            surrounding_surfaces: List of tuples (view_factor, temperature_K) for surrounding surfaces

        Returns:
            h_r: Linearized radiation heat transfer coefficient (W/m²·K)

        References:
            - EnergyPlus Engineering Reference, Section "Exterior Surface Heat Balance"
            - ASHRAE Fundamentals 2021, Chapter 25 "Heat Transfer"
        """
        ground_surfaces = ground_surfaces or []
        surrounding_surfaces = surrounding_surfaces or []

        # View factor sanity check (same as calculate_flux)
        total_ground_f = sum(f for f, t in ground_surfaces)
        total_surrounding_f = sum(f for f, t in surrounding_surfaces)
        total_f = self.f_sky + total_ground_f + total_surrounding_f
        if not math.isclose(total_f, 1.0, rel_tol=1e-5):
            raise ValueError(
                f"The sum of view factors must be 1.0. "
                f"Current sum: {total_f} (Sky: {self.f_sky}, "
                f"Ground: {total_ground_f}, Surrounding: {total_surrounding_f})"
            )

        t_surf = surface_temperature_K
        h_r_total = 0.0

        # 1. Sky component: h_r_sky = ε·σ·F_sky·β·(T_sky + T_surf)·(T_sky² + T_surf²)
        t_sky = sky_temperature_K
        h_r_sky = (self.emissivity * STEFAN_BOLTZMANN * self.f_sky * self.beta *
                   (t_sky + t_surf) * (t_sky**2 + t_surf**2))
        h_r_total += h_r_sky

        # 2. Air component: h_r_air = ε·σ·F_sky·(1-β)·(T_air + T_surf)·(T_air² + T_surf²)
        t_air = air_temperature_K
        h_r_air = (self.emissivity * STEFAN_BOLTZMANN * self.f_sky * (1.0 - self.beta) *
                   (t_air + t_surf) * (t_air**2 + t_surf**2))
        h_r_total += h_r_air

        # 3. Ground components
        for f_gnd, t_gnd in ground_surfaces:
            if t_gnd < 0:
                raise ValueError("Ground temperatures must be non-negative Kelvin.")
            h_r_gnd = (self.emissivity * STEFAN_BOLTZMANN * f_gnd *
                       (t_gnd + t_surf) * (t_gnd**2 + t_surf**2))
            h_r_total += h_r_gnd

        # 4. Surrounding surface components
        for f_srd, t_srd in surrounding_surfaces:
            if t_srd < 0:
                raise ValueError("Surrounding surface temperatures must be non-negative Kelvin.")
            h_r_srd = (self.emissivity * STEFAN_BOLTZMANN * f_srd *
                       (t_srd + t_surf) * (t_srd**2 + t_surf**2))
            h_r_total += h_r_srd

        return h_r_total

    def calculate_separated_coefficients(
        self,
        surface_temperature_K: float,
        air_temperature_K: float,
        sky_temperature_K: float,
        ground_temperature_K: float = None
    ) -> tuple:
        """
        Calculate separated linearized radiation coefficients for sky, air, and ground.

        This follows the EnergyPlus ConvectionCoefficients.cc methodology where
        radiation is split into distinct components, each driving to its respective
        temperature. This preserves the T^4 nonlinearity better than using a single
        averaged temperature.

        The sky view factor is split using parameter β = √(F_sky):
        - Sky component: radiation exchange with actual sky at T_sky
        - Air component: atmospheric radiation exchange at T_air
        - Ground component: radiation exchange with ground at T_ground

        Args:
            surface_temperature_K: Exterior surface temperature (K)
            air_temperature_K: Outdoor air temperature (K)
            sky_temperature_K: Effective sky temperature (K)
            ground_temperature_K: Ground temperature (K), defaults to air temperature

        Returns:
            tuple: (h_r_sky, h_r_air, h_r_ground) - Linearized radiation coefficients
                   in W/(m²·K) for each component

        References:
            - EnergyPlus Engineering Reference, Section "Outside Surface Heat Balance"
            - EnergyPlus src/EnergyPlus/ConvectionCoefficients.cc lines 659-677
        """
        if ground_temperature_K is None:
            ground_temperature_K = air_temperature_K  # Default: ground at air temp

        t_surf = surface_temperature_K

        # View factors already calculated in __init__
        f_sky = self.f_sky
        f_ground = 1.0 - f_sky

        # Beta factor (EnergyPlus air-sky split)
        # β = √(F_sky) represents the fraction of sky view that is "pure" sky radiation
        # (1-β) represents the fraction that is atmospheric radiation at air temperature
        beta = self.beta

        # Linearized coefficient formula: h_r = ε·σ·F·(T_other + T_surf)·(T_other² + T_surf²)
        # This is derived from: q = ε·σ·F·(T_other⁴ - T_surf⁴)
        #                         ≈ h_r·(T_other - T_surf)

        # Sky radiation component (pure sky at T_sky)
        t_sky = sky_temperature_K
        if abs(t_sky - t_surf) < 0.1:
            # Avoid division issues when temperatures are equal
            h_r_sky = self.emissivity * STEFAN_BOLTZMANN * f_sky * beta * 4 * t_surf**3
        else:
            h_r_sky = (self.emissivity * STEFAN_BOLTZMANN * f_sky * beta *
                       (t_sky + t_surf) * (t_sky**2 + t_surf**2))

        # Air radiation component (atmospheric radiation at T_air)
        t_air = air_temperature_K
        if abs(t_air - t_surf) < 0.1:
            h_r_air = self.emissivity * STEFAN_BOLTZMANN * f_sky * (1.0 - beta) * 4 * t_surf**3
        else:
            h_r_air = (self.emissivity * STEFAN_BOLTZMANN * f_sky * (1.0 - beta) *
                       (t_air + t_surf) * (t_air**2 + t_surf**2))

        # Ground radiation component (ground at T_ground)
        t_gnd = ground_temperature_K
        if f_ground > 0:
            if abs(t_gnd - t_surf) < 0.1:
                h_r_ground = self.emissivity * STEFAN_BOLTZMANN * f_ground * 4 * t_surf**3
            else:
                h_r_ground = (self.emissivity * STEFAN_BOLTZMANN * f_ground *
                              (t_gnd + t_surf) * (t_gnd**2 + t_surf**2))
        else:
            h_r_ground = 0.0

        return h_r_sky, h_r_air, h_r_ground


if __name__ == '__main__':
    # --- Example 1: Vertical Wall, Simple Case (replaces old example) ---
    print("--- Example 1: Vertical Wall on a Clear Night (Simple Ground) ---")
    
    vertical_wall = ExternalLongwaveRadiation(surface_emissivity=0.9, surface_tilt_angle_deg=90.0)
    
    # Define temperatures
    surface_temp_K = 278.0  # 5°C
    air_temp_K = 280.0      # 7°C
    sky_temp_K = 265.0      # -8°C (clear sky)
    ground_temp_K = 279.0   # 6°C (ground is slightly warmer than surface but cooler than air)

    # For a simple case with one ground surface, the view factor is 1 - f_sky
    # A vertical wall has a view factor of 0.5 to sky and 0.5 to ground.
    simple_ground = [(0.5, ground_temp_K)] 

    net_flux = vertical_wall.calculate_flux(
        surface_temperature_K=surface_temp_K,
        air_temperature_K=air_temp_K,
        sky_temperature_K=sky_temp_K,
        ground_surfaces=simple_ground
    )
    
    print(f"Surface Properties: Emissivity={vertical_wall.emissivity}, Tilt=90.0°")
    print(f"View Factors: Sky={vertical_wall.f_sky:.3f}, Ground={sum(f for f, t in simple_ground):.3f}")
    print(f"Temperatures (K): Surface={surface_temp_K}, Air={air_temp_K}, Sky={sky_temp_K}, Ground={ground_temp_K}")
    print(f"Result: Net Longwave Radiation Flux = {net_flux:.2f} W/m^2\n")


    # --- Example 2: Complex Scenario with Multiple Surfaces ---
    print("--- Example 2: Wall Viewing Ground, Asphalt, and Another Building ---")
    
    # Consider the same vertical wall
    
    # Define a more complex environment
    # The wall sees:
    # - A patch of grass (cool)
    # - A patch of asphalt (warm from stored solar energy)
    # - The wall of a neighboring building
    
    # View factors must sum with f_sky (0.5) to 1.0. So ground + surrounding must be 0.5.
    grass_f = 0.1
    asphalt_f = 0.3
    neighbor_wall_f = 0.1
    
    # Define temperatures for these surfaces
    grass_temp_K = 282.0      # 8.85°C
    asphalt_temp_K = 290.0    # 16.85°C
    neighbor_wall_temp_K = 284.0 # 10.85°C

    # Create the lists of tuples for the method
    ground_patches = [
        (grass_f, grass_temp_K),
        (asphalt_f, asphalt_temp_K)
    ]
    surrounding_buildings = [
        (neighbor_wall_f, neighbor_wall_temp_K)
    ]

    complex_flux = vertical_wall.calculate_flux(
        surface_temperature_K=surface_temp_K,
        air_temperature_K=air_temp_K,
        sky_temperature_K=sky_temp_K,
        ground_surfaces=ground_patches,
        surrounding_surfaces=surrounding_buildings
    )

    print(f"This wall sees multiple surfaces with different temperatures.")
    total_ground_f = sum(f for f, t in ground_patches)
    total_surr_f = sum(f for f, t in surrounding_buildings)
    print(f"View Factors: Sky={vertical_wall.f_sky:.3f}, Ground={total_ground_f:.3f}, Surrounding={total_surr_f:.3f}")
    print("Ground breakdown: Grass (VF=0.1, T=282K), Asphalt (VF=0.3, T=290K)")
    print("Surrounding breakdown: Neighbor Wall (VF=0.1, T=284K)")
    print(f"Result: Net Longwave Radiation Flux = {complex_flux:.2f} W/m^2")


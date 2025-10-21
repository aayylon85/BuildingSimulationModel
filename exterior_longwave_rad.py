"""
This module calculates the exterior surface longwave radiation flux.

The model considers radiation exchange between an exterior surface and the ground,
sky, air, and surrounding surfaces. It can handle multiple ground and surrounding
surfaces, each with their own view factor and temperature.
"""

import math
from typing import List, Tuple

# Stefan-Boltzmann constant in W/(m^2.K^4)
STEFAN_BOLTZMANN = 5.670374419e-8

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


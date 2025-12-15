"""
Defines the Zone model. It manages the thermal properties of the zone's fabric
and air. The main simulation loop has been moved to main.py.

This class now provides:
- __init__(): To set up the zone solver.
- run_warmup(): To run the initial stabilization.
- run_simulation_step(): To solve a single timestep.
"""
import numpy as np
import types
from zone_solver import ZoneHeatBalanceSolver
from windows import SimpleWindow 
from air_exchange import AirExchangeManager

class Zone:
    """
    Represents a single thermal zone with surfaces, windows, and internal gains.
    Manages the solver and provides methods for simulation steps.
    """
    
    def __init__(self, zone_properties, geometry_data, constructions, dt_seconds,
                 windows_data, air_exchange_data,
                 zone_sensible_heat_capacity_multiplier=1.0,
                 max_nodes_per_layer=20, hvac_coupling_mode='auto',
                 hvac_max_power_rate_w_s=50000.0):
        """
        Initializes the zone and its thermal properties.

        Args:
            zone_properties: Dictionary of zone dimensions
            geometry_data: Dictionary of surface definitions
            constructions: Dictionary of construction objects
            dt_seconds: Timestep in seconds
            windows_data: List of window definitions
            air_exchange_data: Dictionary of infiltration/ventilation settings
            zone_sensible_heat_capacity_multiplier: Multiplier for zone thermal mass (default: 1.0)
            max_nodes_per_layer: Maximum nodes per construction layer for numerical stability (default: 20)
            hvac_coupling_mode: HVAC coupling strategy - 'auto', 'implicit', or 'explicit' (default: 'auto')
            hvac_max_power_rate_w_s: Max HVAC power rate of change for explicit coupling (default: 50000.0 W/s)
        """
        self.dimensions = zone_properties
        self.dt_sec = dt_seconds

        zone_volume = zone_properties['length'] * zone_properties['width'] * zone_properties['height']

        all_surfaces_props = {}
        for name, props in geometry_data['surface_definitions'].items():
            all_surfaces_props[name] = {**props}
            all_surfaces_props[name]['is_exterior'] = name in geometry_data['exterior_surfaces']


        self.solver = ZoneHeatBalanceSolver(
            all_surfaces_props, constructions, dt_seconds,
            zone_volume, zone_sensible_heat_capacity_multiplier,
            max_nodes_per_layer=max_nodes_per_layer,
            hvac_coupling_mode=hvac_coupling_mode,
            hvac_max_power_rate_w_s=hvac_max_power_rate_w_s
        )
        
        self.windows = {}
        # Create windows and adjust parent wall areas
        if windows_data:
            for i, win_data in enumerate(windows_data):
                parent_wall_name = win_data['wall_name']
                self.solver.reduce_surface_area(parent_wall_name, win_data['area'])
                win_name = f"Window_{i+1}_{parent_wall_name}"
                ratios = win_data.get('solar_distribution', {}) # Corrected key
                self.windows[win_name] = SimpleWindow(
                   win_data['area'], 
                   win_data['u_value'], 
                   win_data['shgc'],
                   ratios 
                )
        
        self.air_exchange_manager = None
        if air_exchange_data:
            self.air_exchange_manager = AirExchangeManager(air_exchange_data, zone_volume)


    def run_warmup(self, heating_setpoint_profile, cooling_setpoint_profile,
                   weather_profile, internal_gains_profile,
                   interior_convection_model, exterior_convection_model, hvac_system,
                   stabilization_days=3, use_adaptive_timestepping=True):
        """
        Runs a dynamic stabilization warm-up and returns the final
        zone air temperature.

        Args:
            stabilization_days (int): Number of days to cycle through for warm-up (default: 3).
            use_adaptive_timestepping (bool): Enable automatic timestep reduction on failure (default: True).
        """

        # --- DYNAMIC STABILIZATION WARM-UP --- 
        steps_per_day = int(24 * 3600 / self.dt_sec)
        
        if steps_per_day == 0 or steps_per_day > len(weather_profile):
            steps_per_day = len(weather_profile)
            if steps_per_day == 0:
                 raise ValueError("Weather profile is empty.")
        
        stab_weather_prof = weather_profile[:steps_per_day]
        stab_heating_sp_prof = heating_setpoint_profile[:steps_per_day]
        stab_cooling_sp_prof = cooling_setpoint_profile[:steps_per_day]
        stab_internal_gains_prof = internal_gains_profile[:steps_per_day]
        stab_window_open_prof = np.zeros(steps_per_day) # Keep windows closed
        
        num_stab_profile_steps = len(stab_weather_prof)
        stabilization_steps = stabilization_days * num_stab_profile_steps

        initial_air_temp_guess = (stab_heating_sp_prof[0] + stab_weather_prof[0]['air_temp_c']) / 2.0
        self.solver.set_initial_temperatures(initial_air_temp_guess)
        T_air_prev = initial_air_temp_guess
        
        for t in range(stabilization_steps):
            step_of_day = t % num_stab_profile_steps

            current_weather = stab_weather_prof[step_of_day]
            current_heating_setpoint = stab_heating_sp_prof[step_of_day]
            current_cooling_setpoint = stab_cooling_sp_prof[step_of_day]
            current_internal_gains = stab_internal_gains_prof[step_of_day]
            current_window_fraction = stab_window_open_prof[step_of_day]

            solar_gains_w_dict = {}  # No solar during warmup for simplicity

            if use_adaptive_timestepping:
                result = self._solve_step_adaptive(
                    T_air_prev, current_weather, current_internal_gains,
                    current_heating_setpoint, current_cooling_setpoint,
                    current_window_fraction, interior_convection_model,
                    exterior_convection_model, hvac_system, solar_gains_w_dict
                )
                T_air_prev = result['T_air_new']
            else:
                # Original direct call
                q_hvac = hvac_system.calculate_hvac_power(
                    T_air_prev, current_heating_setpoint, current_cooling_setpoint
                )

                all_new_temps, _, _, _, _ = self.solver.solve_step(
                    T_air_prev, current_weather, self.windows, self.air_exchange_manager,
                    interior_convection_model, exterior_convection_model,
                    current_internal_gains, solar_gains_w_dict, q_hvac, current_window_fraction,
                    hvac_system=hvac_system, heating_setpoint=current_heating_setpoint,
                    cooling_setpoint=current_cooling_setpoint
                )

                T_air_prev = all_new_temps[-1]

        # Return the final converged air temperature
        return T_air_prev

    def _solve_step_adaptive(self, T_air_prev, current_weather, current_internal_gains,
                            current_heating_setpoint, current_cooling_setpoint,
                            current_window_fraction, interior_convection_model,
                            exterior_convection_model, hvac_system,
                            solar_gains_w_dict,
                            original_dt=None, max_subdivisions=4):
        """
        Adaptive wrapper around solve_step that automatically reduces timestep on failure.

        Args:
            T_air_prev: Previous air temperature
            current_weather: Current weather conditions
            current_internal_gains: Internal heat gains
            current_heating_setpoint: Heating setpoint temperature
            current_cooling_setpoint: Cooling setpoint temperature
            current_window_fraction: Window opening fraction
            interior_convection_model: Interior convection model
            exterior_convection_model: Exterior convection model
            hvac_system: HVAC system object
            solar_gains_w_dict: Dictionary of solar gains by surface
            original_dt (float): Original timestep in seconds (defaults to self.dt_sec)
            max_subdivisions (int): Maximum number of times to halve timestep (default: 4)
                This limits minimum timestep to dt / (2^4) = dt / 16

        Returns:
            dict: Same format as run_simulation_step (T_air_new, q_hvac, q_fabric_loss, etc.)

        Raises:
            RuntimeError: If still failing after maximum subdivisions
        """
        import warnings

        if original_dt is None:
            original_dt = self.dt_sec

        current_dt = original_dt
        subdivision_level = 0

        # Calculate HVAC power using the input temperature
        q_hvac = hvac_system.calculate_hvac_power(
            T_air_prev, current_heating_setpoint, current_cooling_setpoint
        )

        while subdivision_level <= max_subdivisions:
            # Temporarily change solver timestep
            old_dt = self.solver.dt
            self.solver.dt = current_dt
            self.solver.air_capacitance_term = self.solver.air_thermal_mass / current_dt

            # Also need to update fabric solver timesteps
            for solver_data in self.solver.fabric_solvers.values():
                solver_data['solver'].dt = current_dt

            try:
                # Attempt to solve with current (possibly reduced) timestep
                # Use implicit HVAC coupling for stability with aggressive gains
                all_new_temps, q_fabric, q_window, q_air_exchange, q_hvac_actual = self.solver.solve_step(
                    T_air_prev, current_weather, self.windows, self.air_exchange_manager,
                    interior_convection_model, exterior_convection_model,
                    current_internal_gains, solar_gains_w_dict, q_hvac, current_window_fraction,
                    hvac_system=hvac_system, heating_setpoint=current_heating_setpoint,
                    cooling_setpoint=current_cooling_setpoint
                )

                # Success! Restore original timestep for next step
                self.solver.dt = old_dt
                self.solver.air_capacitance_term = self.solver.air_thermal_mass / old_dt
                for solver_data in self.solver.fabric_solvers.values():
                    solver_data['solver'].dt = old_dt

                # If we had to subdivide, warn the user
                if subdivision_level > 0:
                    warnings.warn(
                        f"Adaptive timestepping: Successfully solved with dt={current_dt}s "
                        f"({2**subdivision_level}x subdivision). Restoring original dt={original_dt}s",
                        RuntimeWarning
                    )

                return {
                    'T_air_new': all_new_temps[-1],
                    'q_hvac': q_hvac_actual,  # Use actual HVAC power from solver
                    'q_fabric_loss': q_fabric + q_window,
                    'q_air_exchange_loss': q_air_exchange,
                    'q_solar_gains': 0.0  # Already accounted in solar_gains_w_dict
                }

            except (RuntimeError, RuntimeWarning) as e:
                # Failure - need to reduce timestep
                subdivision_level += 1

                if subdivision_level > max_subdivisions:
                    # Restore timestep before failing
                    self.solver.dt = old_dt
                    self.solver.air_capacitance_term = self.solver.air_thermal_mass / old_dt
                    for solver_data in self.solver.fabric_solvers.values():
                        solver_data['solver'].dt = old_dt

                    raise RuntimeError(
                        f"Adaptive timestepping failed: Unable to solve even with "
                        f"minimum timestep dt={current_dt}s (original dt={original_dt}s, "
                        f"{2**subdivision_level}x subdivision attempted). "
                        f"Original error: {str(e)}"
                    )

                # Halve the timestep and try again
                current_dt = original_dt / (2 ** subdivision_level)
                warnings.warn(
                    f"Adaptive timestepping: solve_step failed with dt={old_dt}s. "
                    f"Retrying with dt={current_dt}s ({2**subdivision_level}x subdivision). "
                    f"Reason: {str(e)}",
                    RuntimeWarning
                )

                # Restore timestep for retry
                self.solver.dt = old_dt
                self.solver.air_capacitance_term = self.solver.air_thermal_mass / old_dt
                for solver_data in self.solver.fabric_solvers.values():
                    solver_data['solver'].dt = old_dt

    def run_simulation_step(self, T_air_prev, current_weather,
                            current_internal_gains, current_heating_setpoint,
                            current_cooling_setpoint, current_window_fraction,
                            interior_convection_model, exterior_convection_model,
                            hvac_system, use_adaptive_timestepping=True):
        """
        Runs a single simulation step and returns a dictionary of results.

        Args:
            use_adaptive_timestepping (bool): Enable automatic timestep reduction on failure (default: True)
        """
        
        # --- Calculate Solar Gains ---
        solar_gains_w_dict = {}
        q_solar_total_gain_for_plotting = 0.0
        irradiance = current_weather.get('solar_irradiance_w_m2', 0)

        for win_name, win in self.windows.items():
            # Note: T_air_prev is used for window conduction calculation
            q_cond_window, q_solar_window = win.calculate_heat_flow(
                T_air_prev, 
                current_weather['air_temp_c'], 
                irradiance
            )
            
            # This is the solar energy that enters the zone
            q_solar_total_gain_for_plotting += q_solar_window
            
            ratios = win.ratios
            if not ratios or sum(ratios.values()) == 0:
                # Default: 100% of solar gain goes to the floor
                if 'floor' not in solar_gains_w_dict:
                    solar_gains_w_dict['floor'] = 0.0
                solar_gains_w_dict['floor'] += q_solar_window
            else:
                # Distribute solar gain to surfaces
                for surface, ratio in ratios.items():
                    if ratio > 0:
                        gain_share = q_solar_window * ratio
                        if surface not in solar_gains_w_dict:
                            solar_gains_w_dict[surface] = 0.0
                        solar_gains_w_dict[surface] += gain_share
            
        # --- Use adaptive or direct solve based on flag ---
        if use_adaptive_timestepping:
            result = self._solve_step_adaptive(
                T_air_prev, current_weather, current_internal_gains,
                current_heating_setpoint, current_cooling_setpoint,
                current_window_fraction, interior_convection_model,
                exterior_convection_model, hvac_system, solar_gains_w_dict
            )
            # Update solar gains in result
            result['q_solar_gains'] = q_solar_total_gain_for_plotting
            return result
        else:
            # Original direct call (for backward compatibility / debugging)
            q_hvac = hvac_system.calculate_hvac_power(
                T_air_prev,
                current_heating_setpoint,
                current_cooling_setpoint
            )

            all_new_temps, q_fabric, q_window, q_air_exchange, q_hvac_actual = self.solver.solve_step(
                T_air_prev, current_weather, self.windows, self.air_exchange_manager,
                interior_convection_model, exterior_convection_model,
                current_internal_gains,
                solar_gains_w_dict,
                q_hvac,
                current_window_fraction,
                hvac_system=hvac_system, heating_setpoint=current_heating_setpoint,
                cooling_setpoint=current_cooling_setpoint
            )

            return {
                'T_air_new': all_new_temps[-1],
                'q_hvac': q_hvac_actual,  # Use actual HVAC power from solver
                'q_fabric_loss': q_fabric + q_window,
                'q_air_exchange_loss': q_air_exchange,
                'q_solar_gains': q_solar_total_gain_for_plotting
            }
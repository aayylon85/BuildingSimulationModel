"""
Defines the Zone model. It manages the thermal properties of the zone's fabric
and air and runs the simulation using a fully coupled solver.
"""
import numpy as np
import types
from zone_solver import ZoneHeatBalanceSolver
# constants.py is not uploaded, but I assume it exists
# from constants import AIR_DENSITY_KG_M3, AIR_SPECIFIC_HEAT_J_KG_K
from windows import SimpleWindow # <-- Uses your real windows.py file
from air_exchange import AirExchangeManager # <-- Uses your air_exchange.py file

class Zone:
    """
    Represents a single thermal zone with surfaces, windows, and internal gains.
    """
    # Note the changed signature: 'construction' (singular) is now 'constructions' (plural)
    def __init__(self, zone_properties, geometry_data, constructions, dt_seconds,
                 windows_data=None, air_exchange_data=None,
                 zone_sensible_heat_capacity_multiplier=1.0):
        """
        Initializes the zone and its thermal properties.
        """
        self.dimensions = zone_properties # Store length, width, height
        self.windows = {}
        self.air_exchange_manager = None
        
        zone_volume = zone_properties['length'] * zone_properties['width'] * zone_properties['height']
        
        # Create a deep copy of props to avoid modifying the config
        all_surfaces_props = {}
        for name, props in geometry_data['surface_definitions'].items():
            all_surfaces_props[name] = {**props} 
            all_surfaces_props[name]['is_exterior'] = name in geometry_data['exterior_surfaces']

        # The ZoneHeatBalanceSolver now manages the fabric solvers
        # It receives the DICTIONARY of constructions
        self.solver = ZoneHeatBalanceSolver(
            all_surfaces_props, constructions, dt_seconds, 
            zone_volume, zone_sensible_heat_capacity_multiplier
        )

        # Create windows and adjust parent wall areas
        if windows_data:
            for i, win_data in enumerate(windows_data):
                parent_wall_name = win_data['wall_name']
                # This logic from your original file is correct
                self.solver.reduce_surface_area(parent_wall_name, win_data['area'])
                win_name = f"Window_{i+1}_{parent_wall_name}"
                self.windows[win_name] = SimpleWindow(win_data['area'], win_data['u_value'], win_data['shgc'])
        
        if air_exchange_data:
            self.air_exchange_manager = AirExchangeManager(air_exchange_data, zone_volume)

    def run_simulation(self, heating_setpoint_profile, cooling_setpoint_profile, 
                         weather_profile, internal_gains_profile,
                         interior_convection_model, exterior_convection_model, hvac_system,
                         sim_duration_hours, window_opening_profile):
        
        dt_sec = self.solver.dt
        num_steps = int(sim_duration_hours * 3600 / dt_sec)

        # --- DYNAMIC STABILIZATION WARM-UP ---
        print("Starting dynamic stabilization warm-up...")
        stabilization_days = 3 
        steps_per_day = int(24 * 3600 / dt_sec)
        
        if steps_per_day == 0:
            steps_per_day = len(weather_profile) # Use at least one day's profile
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

            q_hvac = hvac_system.calculate_hvac_power(T_air_prev, current_heating_setpoint, current_cooling_setpoint)
            
            # --- FIX FOR WARMUP ---
            # Pass an empty dictionary for solar gains, not 0
            solar_gains_w_dict = {} 
            
            all_new_temps, _, _, _ = self.solver.solve_step(
                T_air_prev, current_weather, self.windows, self.air_exchange_manager,
                interior_convection_model, exterior_convection_model, 
                current_internal_gains, solar_gains_w_dict, q_hvac, current_window_fraction
            )
            
            T_air_prev = all_new_temps[-1]

        print("Warm-up complete. Starting main simulation.")
        # --- END WARM-UP ---
        
        # --- MAIN SIMULATION LOOP ---
        zone_air_temps = np.zeros(num_steps)
        hvac_energy = np.zeros(num_steps)
        fabric_loss = np.zeros(num_steps)
        window_loss = np.zeros(num_steps)
        air_exchange_loss = np.zeros(num_steps)
        solar_gains = np.zeros(num_steps)
        internal_gains = internal_gains_profile # Use the full profile

        zone_air_temps[0] = T_air_prev # Start from the warmed-up temperature

        for t in range(num_steps):
            if t > 0:
                T_air_prev = zone_air_temps[t-1]
            
            current_heating_setpoint = heating_setpoint_profile[t]
            current_cooling_setpoint = cooling_setpoint_profile[t]
            current_weather = weather_profile[t]
            current_internal_gains = internal_gains_profile[t]
            current_window_fraction = window_opening_profile[t]
            
            # --- SOLAR GAIN FIX (using your windows.py) ---
            solar_gains_w_dict = {}
            q_solar_total_gain_for_plotting = 0.0
            irradiance = current_weather.get('solar_irradiance_w_m2', 0)

            for win in self.windows.values():
                # Use the method from your windows.py file
                # The solver handles conduction, so we only need q_solar_gain
                _q_cond_dummy, q_solar = win.calculate_heat_flow(
                    T_air_prev, 
                    current_weather['air_temp_c'], 
                    irradiance
                )
                
                q_solar_total_gain_for_plotting += q_solar
                
                # --- DISTRIBUTE GAIN ---
                # Assume all gain goes to the "floor"
                if 'floor' not in solar_gains_w_dict:
                    solar_gains_w_dict['floor'] = 0.0
                solar_gains_w_dict['floor'] += q_solar
            # --- END SOLAR GAIN FIX ---
            
            q_hvac = hvac_system.calculate_hvac_power(T_air_prev, current_heating_setpoint, current_cooling_setpoint)

            # --- SOLVER CALL FIX ---
            # Pass the new solar_gains_w_dict
            all_new_temps, q_fabric, q_window, q_air_exchange = self.solver.solve_step(
                T_air_prev, current_weather, self.windows, self.air_exchange_manager,
                interior_convection_model, exterior_convection_model,
                current_internal_gains, 
                solar_gains_w_dict, # <-- Pass the dictionary
                q_hvac,
                current_window_fraction
            )
            
            
            zone_air_temps[t] = all_new_temps[-1]
            hvac_energy[t] = q_hvac
            fabric_loss[t] = q_fabric + q_window
            air_exchange_loss[t] = q_air_exchange
            solar_gains[t] = q_solar_total_gain_for_plotting # This is just solar

        return {
            'zone_air_temps': zone_air_temps, 'hvac_energy': hvac_energy,
            'fabric_loss': fabric_loss,
            'air_exchange_loss': air_exchange_loss, 'solar_gains': solar_gains,
            'internal_gains': internal_gains
        }
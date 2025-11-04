"""
Defines the Zone model. It manages the thermal properties of the zone's fabric
and air and runs the simulation using a fully coupled solver.
"""
import numpy as np
import types
from zone_solver import ZoneHeatBalanceSolver
from windows import SimpleWindow 
from air_exchange import AirExchangeManager

class Zone:
    """
    Represents a single thermal zone with surfaces, windows, and internal gains.
    """
    
    def __init__(self, zone_properties, geometry_data, constructions, dt_seconds,
                 windows_data, air_exchange_data,
                 zone_sensible_heat_capacity_multiplier=1.0):
        """
        Initializes the zone and its thermal properties.
        """
        self.dimensions = zone_properties
        
        zone_volume = zone_properties['length'] * zone_properties['width'] * zone_properties['height']
        
        
        all_surfaces_props = {}
        for name, props in geometry_data['surface_definitions'].items():
            all_surfaces_props[name] = {**props} 
            all_surfaces_props[name]['is_exterior'] = name in geometry_data['exterior_surfaces']

        
        self.solver = ZoneHeatBalanceSolver(
            all_surfaces_props, constructions, dt_seconds, 
            zone_volume, zone_sensible_heat_capacity_multiplier
        )
        self.windows = {}
        # Create windows and adjust parent wall areas
        if windows_data:
            for i, win_data in enumerate(windows_data):
                parent_wall_name = win_data['wall_name']
                self.solver.reduce_surface_area(parent_wall_name, win_data['area'])
                win_name = f"Window_{i+1}_{parent_wall_name}"
                ratios = win_data.get('solar_distribution_ratios', {})
                self.windows[win_name] = SimpleWindow(
                   win_data['area'], 
                   win_data['u_value'], 
                   win_data['shgc'],
                   ratios 
                )
        
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
            
            
            solar_gains_w_dict = {}
            q_solar_total_gain_for_plotting = 0.0
            irradiance = current_weather.get('solar_irradiance_w_m2', 0)

            for win_name, win in self.windows.items():
               
                q_cond_window, q_solar_window = win.calculate_heat_flow(
                    T_air_prev, 
                    current_weather['air_temp_c'], 
                    irradiance
                )
                ratios = win.ratios
                if not ratios:
                    if 'floor' not in solar_gains_w_dict:
                        solar_gains_w_dict['floor'] = 0.0
                    solar_gains_w_dict['floor'] += q_solar_window
                else:
                    for surface, ratio in ratios.items():
                        if ratio > 0:
                            gain_share = q_solar_window * ratio
                            if surface not in solar_gains_w_dict:
                                solar_gains_w_dict[surface] = 0.0
                            solar_gains_w_dict[surface] += gain_share
                
                


            
            
            q_hvac = hvac_system.calculate_hvac_power(T_air_prev, current_heating_setpoint, current_cooling_setpoint)

            
            all_new_temps, q_fabric, q_window, q_air_exchange = self.solver.solve_step(
                T_air_prev, current_weather, self.windows, self.air_exchange_manager,
                interior_convection_model, exterior_convection_model,
                current_internal_gains, 
                solar_gains_w_dict, 
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
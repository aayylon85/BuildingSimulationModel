"""
Defines the Zone model. It manages the thermal properties of the zone's fabric
and air and runs the simulation using a fully coupled solver.
"""
import numpy as np
from zone_solver import ZoneHeatBalanceSolver

class Zone:
    """
    Represents a single thermal zone with surfaces, windows, and internal gains.
    """
    def __init__(self, length, width, height, construction, dt_seconds,
                 exterior_surfaces, windows_data=None, air_exchange_data=None,
                 zone_sensible_heat_capacity_multiplier=1.0):
        """
        Initializes the zone and its thermal properties.
        """
        self.dimensions = {'length': length, 'width': width, 'height': height}
        self.windows = {}
        self.air_exchange_manager = None
        
        zone_volume = length * width * height
        
        all_surfaces_props = {
            'north_wall': {'area': width * height, 'perimeter': 2 * (width + height), 'tilt': 90, 'azimuth': 0, 'type': 'wall', 'roughness_index': 2},
            'east_wall': {'area': length * height, 'perimeter': 2 * (length + height), 'tilt': 90, 'azimuth': 90, 'type': 'wall', 'roughness_index': 2},
            'south_wall': {'area': width * height, 'perimeter': 2 * (width + height), 'tilt': 90, 'azimuth': 180, 'type': 'wall', 'roughness_index': 2},
            'west_wall': {'area': length * height, 'perimeter': 2 * (length + height), 'tilt': 90, 'azimuth': 270, 'type': 'wall', 'roughness_index': 2},
            'roof': {'area': length * width, 'perimeter': 2 * (length + width), 'tilt': 0, 'azimuth': 0, 'type': 'roof', 'roughness_index': 3},
            'floor': {'area': length * width, 'perimeter': 2 * (length + width), 'tilt': 180, 'azimuth': 0, 'type': 'floor', 'roughness_index': 3},
        }

        # Add the is_exterior flag to each surface's properties
        for name, props in all_surfaces_props.items():
            props['is_exterior'] = name in exterior_surfaces

        # The ZoneHeatBalanceSolver now manages the fabric solvers
        self.solver = ZoneHeatBalanceSolver(
            all_surfaces_props, construction, dt_seconds, 
            zone_volume, zone_sensible_heat_capacity_multiplier
        )

        # Create windows and adjust parent wall areas
        if windows_data:
            from windows import SimpleWindow
            for i, win_data in enumerate(windows_data):
                parent_wall_name = win_data['wall_name']
                self.solver.reduce_surface_area(parent_wall_name, win_data['area'])
                win_name = f"Window_{i+1}_{parent_wall_name}"
                self.windows[win_name] = SimpleWindow(win_data['area'], win_data['u_value'], win_data['shgc'])
        
        if air_exchange_data:
            from air_exchange import AirExchangeManager
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
        stabilization_steps = stabilization_days * steps_per_day

        stab_weather_prof = weather_profile[:steps_per_day]
        stab_heating_sp_prof = heating_setpoint_profile[:steps_per_day]
        stab_cooling_sp_prof = cooling_setpoint_profile[:steps_per_day]
        stab_internal_gains_prof = internal_gains_profile[:steps_per_day]
        stab_window_open_prof = np.zeros(steps_per_day)

        initial_air_temp_guess = (stab_heating_sp_prof[0] + stab_weather_prof[0]['air_temp_c']) / 2.0
        self.solver.set_initial_temperatures(initial_air_temp_guess)
        T_air_prev = initial_air_temp_guess
        
        for t in range(stabilization_steps):
            step_of_day = t % steps_per_day
            
            current_weather = stab_weather_prof[step_of_day]
            current_heating_setpoint = stab_heating_sp_prof[step_of_day]
            current_cooling_setpoint = stab_cooling_sp_prof[step_of_day]
            current_internal_gains = stab_internal_gains_prof[step_of_day]
            current_window_fraction = stab_window_open_prof[step_of_day]

            q_hvac = hvac_system.calculate_hvac_power(T_air_prev, current_heating_setpoint, current_cooling_setpoint)
            
            all_new_temps, _, _, _ = self.solver.solve_step(
                T_air_prev, current_weather, self.windows, self.air_exchange_manager,
                interior_convection_model, exterior_convection_model, 
                current_internal_gains, 0, q_hvac, current_window_fraction
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
        internal_gains = np.zeros(num_steps)

        zone_air_temps[0] = T_air_prev

        for t in range(1, num_steps):
            T_air_prev = zone_air_temps[t-1]
            current_heating_setpoint = heating_setpoint_profile[t]
            current_cooling_setpoint = cooling_setpoint_profile[t]
            current_weather = weather_profile[t]
            current_internal_gains = internal_gains_profile[t]
            current_window_fraction = window_opening_profile[t]
            
            q_solar_total_gain = sum(
                win.calculate_heat_flow(T_air_prev, current_weather['air_temp_c'], current_weather.get('solar_irradiance_w_m2', 0))[1]
                for win in self.windows.values()
            )
            
            q_hvac = hvac_system.calculate_hvac_power(T_air_prev, current_heating_setpoint, current_cooling_setpoint)

            all_new_temps, q_fabric, q_window, q_air_exchange = self.solver.solve_step(
                T_air_prev, current_weather, self.windows, self.air_exchange_manager,
                interior_convection_model, exterior_convection_model,
                current_internal_gains, q_solar_total_gain, q_hvac,
                current_window_fraction
            )
            
            zone_air_temps[t] = all_new_temps[-1]
            hvac_energy[t] = q_hvac
            fabric_loss[t] = q_fabric
            window_loss[t] = q_window
            air_exchange_loss[t] = q_air_exchange
            solar_gains[t] = q_solar_total_gain
            internal_gains[t] = current_internal_gains

        return {
            'zone_air_temps': zone_air_temps, 'hvac_energy': hvac_energy,
            'fabric_loss': fabric_loss, 'window_loss': window_loss,
            'air_exchange_loss': air_exchange_loss, 'solar_gains': solar_gains,
            'internal_gains': internal_gains_profile
        }



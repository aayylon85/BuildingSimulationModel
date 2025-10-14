"""
Defines the Zone model. It manages the thermal properties of the zone's fabric
and air but delegates HVAC calculations to a separate HVAC model.
"""
import numpy as np
from fabric_heat_transfer import CondFDSolver
from windows import SimpleWindow

# Constants for air properties at standard conditions
AIR_DENSITY_KG_M3 = 1.225
AIR_SPECIFIC_HEAT_J_KG_K = 1006

class Zone:
    """
    Represents a single thermal zone with surfaces, windows, and internal gains.
    """
    def __init__(self, length, width, height, construction, dt_seconds,
                 exterior_surfaces, windows_data=None):
        """
        Initializes the zone and its thermal properties.
        """
        self.dimensions = {'length': length, 'width': width, 'height': height}
        self.surfaces = {}
        self.windows = {}
        zone_volume = length * width * height
        self.air_thermal_mass = zone_volume * AIR_DENSITY_KG_M3 * AIR_SPECIFIC_HEAT_J_KG_K
        
        all_surfaces_props = {
            'north_wall': {'area': width * height, 'perimeter': 2 * (width + height), 'tilt': 90, 'azimuth': 0, 'type': 'wall', 'roughness_index': 2},
            'east_wall': {'area': length * height, 'perimeter': 2 * (length + height), 'tilt': 90, 'azimuth': 90, 'type': 'wall', 'roughness_index': 2},
            'south_wall': {'area': width * height, 'perimeter': 2 * (width + height), 'tilt': 90, 'azimuth': 180, 'type': 'wall', 'roughness_index': 2},
            'west_wall': {'area': length * height, 'perimeter': 2 * (length + height), 'tilt': 90, 'azimuth': 270, 'type': 'wall', 'roughness_index': 2},
            'roof': {'area': length * width, 'perimeter': 2 * (length + width), 'tilt': 0, 'azimuth': 0, 'type': 'roof', 'roughness_index': 3},
            'floor': {'area': length * width, 'perimeter': 2 * (length + width), 'tilt': 180, 'azimuth': 0, 'type': 'floor', 'roughness_index': 3},
        }
        
        for name, props in all_surfaces_props.items():
            self.surfaces[name] = props
            solver = CondFDSolver(construction, dt_seconds)
            # Set a uniform starting temperature for the fabric
            solver.set_initial_temperatures(15.0) 
            self.surfaces[name]['solver'] = solver
            self.surfaces[name]['is_exterior'] = name in exterior_surfaces

        if windows_data:
            for i, win_data in enumerate(windows_data):
                parent_wall = win_data['wall_name']
                self.surfaces[parent_wall]['area'] -= win_data['area']
                win_name = f"Window_{i+1}_{parent_wall}"
                self.windows[win_name] = SimpleWindow(win_data['area'], win_data['u_value'], win_data['shgc'])


    def run_simulation(self, T_setpoint_profile, weather_profile, internal_gains_profile,
                         interior_convection_model, exterior_convection_model, hvac_system,
                         sim_duration_hours):
        dt_sec = self.surfaces['north_wall']['solver'].dt
        num_steps = int(sim_duration_hours * 3600 / dt_sec)
        
        # --- WARM-UP AND STABILIZATION PHASE ---
        print("Starting stabilization warm-up...")
        
        # Stage 1: 24-hour stabilization at fixed conditions
        warmup_steps = int(24 * 3600 / dt_sec)
        T_air_warmup = 15.0
        warmup_weather = {
            'air_temp_c': 10.0, 'wind_speed_local_ms': 2.0, 
            'wind_speed_10m_ms': 2.0, 'wind_direction_deg': 180,
            'solar_irradiance_w_m2': 0 # No solar during warmup
        }
        for _ in range(warmup_steps):
            for surface_data in self.surfaces.values():
                self._update_surface_temperatures(surface_data, T_air_warmup, warmup_weather, 
                                                  interior_convection_model, exterior_convection_model)

        # Stage 2: 4-hour ramp-up to initial simulation conditions
        ramp_up_steps = int(4 * 3600 / dt_sec)
        T_ext_start_warmup = 10.0
        T_ext_start_sim = weather_profile[0]['air_temp_c']
        ramp_temps = np.linspace(T_ext_start_warmup, T_ext_start_sim, ramp_up_steps)
        
        for ramp_T_ext in ramp_temps:
            ramp_weather = warmup_weather.copy()
            ramp_weather['air_temp_c'] = ramp_T_ext
            for surface_data in self.surfaces.values():
                self._update_surface_temperatures(surface_data, T_air_warmup, ramp_weather, 
                                                  interior_convection_model, exterior_convection_model)
        
        print("Warm-up complete. Starting main simulation.")
        # --- END WARM-UP ---
        
        zone_air_temps = np.zeros(num_steps)
        hvac_energy = np.zeros(num_steps)
        fabric_load = np.zeros(num_steps)
        window_net_load = np.zeros(num_steps) 
        solar_gains = np.zeros(num_steps)
        zone_air_temps[0] = T_air_warmup # Start simulation at the warm-up temperature

        for t in range(1, num_steps):
            T_air_prev = zone_air_temps[t-1]
            T_setpoint = T_setpoint_profile[t]
            current_weather = weather_profile[t]
            current_internal_gains = internal_gains_profile[t]
            
            q_fabric_total_loss = 0.0
            for surface_data in self.surfaces.values():
                q_fabric_total_loss += self._update_surface_temperatures(
                    surface_data, T_air_prev, current_weather, 
                    interior_convection_model, exterior_convection_model,
                    get_flux=True
                )

            q_window_conductive_loss = 0.0
            q_solar_total_gain = 0.0
            for window in self.windows.values():
                q_cond, q_solar = window.calculate_heat_flow(
                    T_air_prev, current_weather['air_temp_c'],
                    current_weather.get('solar_irradiance_w_m2', 0)
                )
                q_window_conductive_loss += q_cond
                q_solar_total_gain += q_solar

            total_passive_loss = q_fabric_total_loss + q_window_conductive_loss
            total_passive_gain = q_solar_total_gain + current_internal_gains

            q_hvac = hvac_system.calculate_hvac_power(
                T_air_prev, T_setpoint, total_passive_loss,
                total_passive_gain, self.air_thermal_mass, dt_sec
            )

            q_net_to_air = (total_passive_gain + q_hvac) - total_passive_loss
            delta_T_air = (q_net_to_air * dt_sec) / self.air_thermal_mass
            zone_air_temps[t] = T_air_prev + delta_T_air

            fabric_load[t] = q_fabric_total_loss
            window_net_load[t] = q_window_conductive_loss 
            solar_gains[t] = q_solar_total_gain
            hvac_energy[t] = q_hvac

        return {
            'zone_air_temps': zone_air_temps, 'hvac_energy': hvac_energy,
            'fabric_load': fabric_load, 'window_net_load': window_net_load,
            'solar_gains': solar_gains, 'internal_gains': internal_gains_profile
        }

    def _update_surface_temperatures(self, surface_data, T_air, weather, 
                                     interior_convection_model, exterior_convection_model, get_flux=False):
        """
        Calculates and updates the fabric temperatures for a surface for one timestep.
        Optionally returns the inside surface heat flux.
        """
        h_inside = interior_convection_model.calculate_h_c(surface_data, T_air)
        
        if surface_data['is_exterior']:
            # Iterative solution for exterior convection coefficient
            h_outside_guess = 15.0 
            current_surf_temp = surface_data['solver'].nodes[-1]['T'] 
            
            for _ in range(3): # Iterate to find a stable surface temp and h_outside
                surface_props_for_ext_hc = {**surface_data, 'surface_temp_c': current_surf_temp}
                h_outside = exterior_convection_model.calculate_hc(surface_props_for_ext_hc, weather)
                # Calculate potential new temps without saving them yet
                T_new_nodes, _, _ = surface_data['solver'].solve_step(
                    T_air, weather['air_temp_c'], h_inside, h_outside, update_state=False
                )
                current_surf_temp = T_new_nodes[-1]
            
            # Perform the final calculation with the converged h_outside and update the state
            _, q_in_flux, _ = surface_data['solver'].solve_step(
                T_air, weather['air_temp_c'], h_inside, h_outside, update_state=True
            )
        else: # Adiabatic boundary for interior surfaces (e.g., floor)
            _, q_in_flux, _ = surface_data['solver'].solve_step(
                T_air, T_air, h_inside, 0.0, update_state=True
            )
            
        if get_flux:
            return q_in_flux * surface_data['area']
        return None


"""
Defines functions to create boundary conditions (schedules, weather)
for the thermal simulation from a configuration dictionary.
"""
import numpy as np
from bsm.weather.generators import get_weather_generator
from bsm.boundary.equipment_schedules import create_equipment_schedules

def create_boundary_conditions(config, num_steps, time_hours):
    """
    Generates time-series profiles for setpoints, gains, and weather.
    
    Args:
        config (dict): The full simulation configuration dictionary.
        num_steps (int): The total number of simulation steps.
        time_hours (np.array): A numpy array of the simulation time in hours
                               for each step.

    Returns:
        tuple: Contains the following:
            - heating_setpoint_profile (np.array)
            - cooling_setpoint_profile (np.array)
            - internal_gains_profile (np.array)
            - window_opening_profile (np.array)
            - weather_data (list[dict])
            - equipment_power_profile (np.array)
            - lighting_power_profile (np.array)
    """
    schedules = config['schedules']
    
    # FIX 1: Pass the FULL config to the generator factory, not just the weather section.
    # This allows WeatherFromFile to access 'location' and 'simulation_settings'.
    weather_gen = get_weather_generator(config)

    # FIX 2: Remove 'num_steps' from this call. 
    # The new generator classes only accept (time_hours).
    weather_data = weather_gen.generate_weather_data(time_hours)

    # Get occupancy config, with defaults if not present
    occupancy_config = config.get('occupancy', {})
    occupants = occupancy_config.get('occupants', [])
    gain_per_occupant = occupancy_config.get('heat_gain_per_occupant_w', 0.0)
    
    heating_setpoint_profile = np.zeros(num_steps)
    cooling_setpoint_profile = np.zeros(num_steps)
    internal_gains_profile = np.zeros(num_steps)
    window_opening_profile = np.zeros(num_steps)

    occ_start, occ_end = schedules['occupied_hours']

    for i, t_hr in enumerate(time_hours):
        hour_of_day = t_hr % 24
        
        # --- Setpoint and Gains Schedules ---
        if occ_start <= hour_of_day < occ_end:
            heating_setpoint_profile[i] = schedules['occupied_heating_setpoint_c']
            cooling_setpoint_profile[i] = schedules['occupied_cooling_setpoint_c']
            internal_gains_profile[i] = schedules['occupied_internal_gains_w']
        else:
            heating_setpoint_profile[i] = schedules['unoccupied_heating_setpoint_c']
            cooling_setpoint_profile[i] = schedules['unoccupied_cooling_setpoint_c']
            internal_gains_profile[i] = schedules['unoccupied_internal_gains_w']

        # --- Dynamic Internal Gains (based on specific occupants) ---
        # ADD occupant-based gains to the scheduled base gains (don't overwrite!)
        num_occupants_present = 0
        for occ_data in occupants:
            if occ_data.get('work_start_hr', 9) <= hour_of_day < occ_data.get('work_end_hr', 17):
                num_occupants_present += 1

        internal_gains_profile[i] += num_occupants_present * gain_per_occupant

    # --- Generate Equipment and Lighting Schedules ---
    # Determine mode based on whether occupants are defined
    # If occupants list is non-empty, use "simple" mode (occupant-presence-based)
    # If occupants list is empty, use "baseline" mode (fixed schedule)
    llm_enabled = config.get('llm_agents', {}).get('enabled', False)

    if llm_enabled:
        # LLM mode handles equipment/lighting separately via managers
        # Return zero profiles (runner.py will use LLM manager values)
        equipment_power_profile = np.zeros(num_steps)
        lighting_power_profile = np.zeros(num_steps)
    elif occupants:
        # Simple agents mode: occupant-presence-based with randomness
        equipment_power_profile, lighting_power_profile = create_equipment_schedules(
            config, num_steps, time_hours, mode="simple"
        )
    else:
        # Baseline mode: fixed schedules tied to occupied hours
        equipment_power_profile, lighting_power_profile = create_equipment_schedules(
            config, num_steps, time_hours, mode="baseline"
        )

    return (
        heating_setpoint_profile,
        cooling_setpoint_profile,
        internal_gains_profile,
        window_opening_profile,
        weather_data,
        equipment_power_profile,
        lighting_power_profile,
    )
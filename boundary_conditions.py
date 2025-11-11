"""
Defines functions to create boundary conditions (schedules, weather) 
for the thermal simulation from a configuration dictionary.
"""
import numpy as np
from weather import get_weather_generator

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
    """
    schedules = config['schedules']
    weather_conf = config['weather']

    # Get occupancy config, with defaults if not present
    occupancy_config = config.get('occupancy', {})
    occupants = occupancy_config.get('occupants', [])
    gain_per_occupant = occupancy_config.get('heat_gain_per_occupant_w', 0.0)
    
    heating_setpoint_profile = np.zeros(num_steps)
    cooling_setpoint_profile = np.zeros(num_steps)
    internal_gains_profile = np.zeros(num_steps)
    window_opening_profile = np.zeros(num_steps)

    
    
    weather_gen = get_weather_generator(weather_conf)
    weather_data = weather_gen.generate_weather_data(num_steps, time_hours)

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
            internal_gains_profile[i] = 0
            
        # --- Dynamic Internal Gains (based on specific occupants) ---
        num_occupants_present = 0
        for occ_data in occupants:
            if occ_data.get('work_start_hr', 9) <= hour_of_day < occ_data.get('work_end_hr', 17):
                num_occupants_present += 1
                
        internal_gains_profile[i] = num_occupants_present * gain_per_occupant

        
    return (
        heating_setpoint_profile, 
        cooling_setpoint_profile, 
        internal_gains_profile, 
        window_opening_profile, 
        weather_data
    )
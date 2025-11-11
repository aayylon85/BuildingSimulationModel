"""
Defines classes for generating weather data for the thermal simulation.
"""
import numpy as np

class SimpleSinusoidal:
    """
    Generates a simple sinusoidal weather profile based on config parameters.
    """
    def __init__(self, config):
        """
        Initializes the weather generator.
        
        Args:
            config (dict): The 'weather' section of the main config dictionary.
        """
        self.config = config

    def generate_weather_data(self, num_steps, time_hours):
        """
        Generates the time-series list of weather data.

        Args:
            num_steps (int): The total number of simulation steps.
            time_hours (np.array): A numpy array of the simulation time in hours
                                   for each step.
        
        Returns:
            list[dict]: A list of weather data dictionaries, one for each time step.
        """
        weather_data = []
        
        # Pre-calculate config values
        solar_max = self.config['solar_max_irradiance_w_m2']
        temp_base = self.config['temp_base_c']
        temp_amp = self.config['temp_amplitude_c']
        temp_phase = self.config['temp_phase_shift_hr']

        for i, t_hr in enumerate(time_hours):
            # --- Simple Sinusoidal Weather Model ---
            
            # Calculate solar radiation, ensuring it's non-negative
            solar_rad = max(0, solar_max * np.cos(2 * np.pi * (t_hr - 12) / 24))
            
            # Calculate air temperature based on a sinusoidal model
            air_temp = (temp_base + 
                        temp_amp * np.cos(2 * np.pi * (t_hr - temp_phase) / 24))
            
            weather_data.append({
                'air_temp_c': air_temp,
                'wind_speed_local_ms': 3.0, # Example fixed value
                'wind_speed_10m_ms': 2.5,   # Example fixed value
                'wind_direction_deg': 180,  # Example fixed value
                'solar_irradiance_w_m2': solar_rad
            })
            
        return weather_data
    
class WeatherFromFile:
    """
    (Placeholder) Loads weather data from an external file.
    """
    def __init__(self, config):
        """
        Initializes the weather generator.
        
        Args:
            config (dict): The 'weather' section of the main config dictionary.
                           Expected to contain a 'filepath' key.
        """
        self.config = config
        self.filepath = config.get('filepath', None)
        if not self.filepath:
            print("Warning: WeatherFromFile generator created, but no 'filepath' specified in config.")
            
    def generate_weather_data(self, num_steps, time_hours):
        """
        (Placeholder) Generates the time-series list of weather data.
        This method would typically read from self.filepath and interpolate.

        Args:
            num_steps (int): The total number of simulation steps.
            time_hours (np.array): A numpy array of the simulation time in hours
                                   for each step.
        
        Returns:
            list[dict]: A list of weather data dictionaries, one for each time step.
        """
        print(f"Notice: Using placeholder WeatherFromFile. Simulating zero weather for {num_steps} steps.")
        # Placeholder data:
        weather_data = []
        for _ in range(num_steps):
            weather_data.append({
                'air_temp_c': 20.0, # Fixed placeholder temp
                'wind_speed_local_ms': 0.0,
                'wind_speed_10m_ms': 0.0,
                'wind_direction_deg': 0,
                'solar_irradiance_w_m2': 0.0
            })
        return weather_data

# --- Weather Generator Factory ---

WEATHER_GENERATOR_MAP = {
    "simple_sinusoidal": SimpleSinusoidal,
    "file": WeatherFromFile
}

def get_weather_generator(config):
    """
    Factory function to get the appropriate weather generator instance.
    
    Args:
        config (dict): The 'weather' section of the main config dictionary.
                       Must contain a 'type' key.

    Returns:
        An instance of a weather generator class (e.g., SimpleSinusoidal).
    """
    # Default to simple_sinusoidal if 'type' key is missing
    weather_type = config.get("type", "simple_sinusoidal") 
    
    GeneratorClass = WEATHER_GENERATOR_MAP.get(weather_type)
    
    if not GeneratorClass:
        raise ValueError(f"Unknown weather generator type: '{weather_type}'. "
                         f"Available types are: {list(WEATHER_GENERATOR_MAP.keys())}")
                         
    return GeneratorClass(config)
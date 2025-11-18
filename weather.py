"""
Defines classes for generating weather data for the thermal simulation.
"""
import numpy as np
import pandas as pd
from datetime import timedelta
from weather_import import get_hourly_weather

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

    def generate_weather_data(self, time_hours):
        """
        Generates the time-series list of weather data.

        Args:
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

        # Fix: Iterate directly over values, not enumerate tuple (index, value)
        for t_hr in time_hours:
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
    Generates weather data by fetching hourly data from a file/API and interpolating
    it to the simulation time steps.
    """
    def __init__(self, config):
        """
        Initializes the weather generator.
        
        Args:
            config (dict): The FULL main simulation config dictionary.
                           Must contain 'location' and 'simulation_settings'.
        """
        self.config = config
        self.df_hourly = None
        self._fetch_hourly_data()

    def _fetch_hourly_data(self):
        """
        Fetches hourly weather data based on config location and duration.
        """
        # 1. Extract required parameters
        loc_config = self.config.get('location')
        if not loc_config:
            raise ValueError("WeatherFromFile requires a 'location' key in the config.")
            
        lat = loc_config.get('latitude')
        lon = loc_config.get('longitude')
        
        sim_settings = self.config.get('simulation_settings', {})
        start_date_str = sim_settings.get('start_date')
        duration_days = sim_settings.get('duration_days', 1)
        
        if not start_date_str:
            raise ValueError("WeatherFromFile requires 'start_date' in 'simulation_settings'.")
            
        # 2. Calculate end date
        # Note: We localize to UTC here to be consistent, though the API call uses string dates
        start_dt = pd.to_datetime(start_date_str)
        
        # Add 1 day buffer to ensure we cover the final hours
        end_dt = start_dt + timedelta(days=duration_days + 1) 
        end_date_str = end_dt.strftime('%Y-%m-%d')

        # 3. Fetch data
        print(f"Fetching weather for Lat: {lat}, Lon: {lon} from {start_date_str} to {end_date_str}...")
        df = get_hourly_weather(lat, lon, start_date_str, end_date_str)
        
        if df.empty:
            raise RuntimeError("Failed to fetch weather data.")
            
        # 4. Process DataFrame
        # Ensure 'date' is datetime objects and set as index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        self.df_hourly = df

    def generate_weather_data(self, time_hours):
        """
        Generates the interpolated time-series list of weather data.

        Args:
            time_hours (np.array): A numpy array of the simulation time in hours.
        
        Returns:
            list[dict]: A list of weather data dictionaries.
        """
        if self.df_hourly is None:
            raise RuntimeError("Weather data has not been fetched.")

        # 1. Create target DatetimeIndex from time_hours
        start_date_str = self.config['simulation_settings']['start_date']
        
        # FIX: Localize to UTC. 
        # The weather API returns UTC (timezone-aware) timestamps.
        # We must make our simulation start time UTC as well to allow merging/union.
        start_dt = pd.to_datetime(start_date_str).tz_localize('UTC')
        
        # Convert simulation hours to actual timestamps
        target_index = [start_dt + timedelta(hours=float(t)) for t in time_hours]
        target_dt_index = pd.DatetimeIndex(target_index)

        # 2. Interpolate
        # To interpolate correctly, we combine the hourly index with our target minute-level index,
        # interpolate, and then select only our target points.
        
        # Union of indices
        combined_index = self.df_hourly.index.union(target_dt_index).sort_values()
        
        # Reindex to combined, then interpolate time-wise
        # Limit direction='both' handles slight edge cases at start/end
        df_interp = self.df_hourly.reindex(combined_index).interpolate(method='time').ffill().bfill()
        
        # Select only the target simulation steps
        df_final = df_interp.loc[target_dt_index]

        # 3. Convert to list of dicts matching simulation requirements
        weather_data = []
        
        # Pre-fetch column series to avoid repeated lookups in loop
        temps = df_final.get('temperature_2m', np.zeros(len(df_final)))
        wind_speeds = df_final.get('wind_speed_10m', np.zeros(len(df_final)))
        wind_dirs = df_final.get('wind_direction_10m', np.zeros(len(df_final)))
        
        # Use shortwave radiation (GHI) if available, else defaults
        solars = df_final.get('shortwave_radiation_instant', np.zeros(len(df_final)))

        for i in range(len(df_final)):
            weather_data.append({
                'air_temp_c': float(temps.iloc[i]),
                'wind_speed_local_ms': float(wind_speeds.iloc[i]), # Using 10m speed as local proxy
                'wind_speed_10m_ms': float(wind_speeds.iloc[i]),
                'wind_direction_deg': float(wind_dirs.iloc[i]),
                'solar_irradiance_w_m2': max(0.0, float(solars.iloc[i]))
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
        config (dict): The FULL simulation configuration dictionary.
                       (Changed from just 'weather' section to support accessing 
                       sibling keys like 'location' and 'simulation_settings')

    Returns:
        An instance of a weather generator class.
    """
    # Extract weather specific config
    weather_config = config.get("weather", {})
    
    # Default to simple_sinusoidal if 'type' key is missing
    weather_type = weather_config.get("type", "simple_sinusoidal") 
    
    GeneratorClass = WEATHER_GENERATOR_MAP.get(weather_type)
    
    if not GeneratorClass:
        raise ValueError(f"Unknown weather generator type: '{weather_type}'. "
                         f"Available types are: {list(WEATHER_GENERATOR_MAP.keys())}")
    
    # Instantiate
    if weather_type == "simple_sinusoidal":
        # Simple model only needs the weather section
        return GeneratorClass(weather_config)
    elif weather_type == "file":
        # File model needs the full config to access Location and Settings
        return GeneratorClass(config)
    else:
        return GeneratorClass(config)
class Occupant:
    """
    Represents an occupant with specific working hours and
    preferences for thermal comfort.
    """
    def __init__(self, config_data):
        """
        Initializes the occupant from a configuration dictionary.
        
        Args:
            config_data (dict): A dictionary from the config file
                                containing occupant details.
        """
        self.name = config_data.get('name', 'Unnamed')
        self.work_start_hr = config_data.get('work_start_hr', 9)
        self.work_end_hr = config_data.get('work_end_hr', 17)
        
        self.window_preference = config_data.get('window_preference', 'neutral')
        self.window_temp_c = config_data.get('window_temp_c', 22.0)
        
        self.thermostat_preference = config_data.get('thermostat_preference', 'neutral')
        self.thermostat_temp_c = config_data.get('thermostat_temp_c', 21.0)
        
        self.preference_deadband_c = 1.5

    def is_present(self, hour_of_day):
        """
        Checks if the occupant is present at a given hour.
        
        Args:
            hour_of_day (float): The current hour of the day (0-24).
            
        Returns:
            bool: True if the occupant is present, False otherwise.
        """
        return self.work_start_hr <= hour_of_day < self.work_end_hr

    def get_desired_action(self, zone_temp_c, window_state, outside_temp_c):
        """
        Determines the occupant's desired action based on the current state.
        
        Args:
            zone_temp_c (float): The current indoor air temperature.
            window_state (float): The current window state (0=closed, 1=open).
            outside_temp_c (float): The current outdoor air temperature.
            
        Returns:
            tuple: (desired_window_action, desired_thermostat_action)
                   Actions can be:
                   - "open_window", "close_window", "window_neutral"
                   - "heat_up", "cool_down", "thermostat_neutral"
        """
        window_action = "window_neutral"
        thermostat_action = "thermostat_neutral"
        
        # --- Window Preference ---
        if self.window_preference == "opener":
            
            # Scenario 1: Occupant is TOO HOT
            if zone_temp_c > self.window_temp_c + self.preference_deadband_c:
                if window_state == 0.0 and outside_temp_c < zone_temp_c:
                    # Too hot, window is closed, outside is cooler. VOTE OPEN.
                    window_action = "open_window"
                elif window_state == 1.0 and outside_temp_c > zone_temp_c:
                    # Too hot, window is open, outside is hotter. VOTE CLOSE.
                    window_action = "close_window"
                    
            # Scenario 2: Occupant is TOO COLD
            elif zone_temp_c < self.window_temp_c - self.preference_deadband_c:
                if window_state == 0.0 and outside_temp_c > zone_temp_c:
                    # Too cold, window is closed, outside is warmer. VOTE OPEN.
                    window_action = "open_window"
                elif window_state == 1.0 and outside_temp_c < zone_temp_c:
                    # Too cold, window is open, outside is cooler. VOTE CLOSE.
                    window_action = "close_window"
                    
        # --- Thermostat Preference ---
        if self.thermostat_preference == "changer":
            if zone_temp_c < self.thermostat_temp_c - self.preference_deadband_c:
                thermostat_action = "heat_up" # Vote to increase setpoint
            elif zone_temp_c > self.thermostat_temp_c + self.preference_deadband_c:
                thermostat_action = "cool_down" # Vote to decrease setpoint
                
        return window_action, thermostat_action
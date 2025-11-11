class Occupant:
    """
    Represents occupants in a thermal zone, providing internal gains
    based on occupancy schedules.
    """
    def __init__(self, occupancy_schedule, heat_gain_w):
        """
        Initializes the occupant model.
        """
        self.occupancy_schedule = occupancy_schedule
        self.heat_gain_w = heat_gain_w
        

    def preferences(window_type, window_temp_pref, thermostat_type, themostat_temp_pref, temp, window_state):
        if window_type == "opener":
            if temp > window_temp_pref + 1.5:
                if window_state != "open":
                    window_state = "open"
            elif temp < window_temp_pref - 1.5:
                if window_state != "closed":
                    window_state = "closed"
        if thermostat_type == "changer":
            if temp < themostat_temp_pref - 1.5:
                themostat_temp += 1.0
            elif temp > themostat_temp_pref + 1.5:
                themostat_temp -= 1.0

        return window_state, themostat_temp

import math
"""
Defines the HVAC system model. This module is responsible for calculating
the heating or cooling power required to meet the zone's setpoints.
"""

class VerySimpleHVAC:
    """
    A simple proportional control HVAC model.
    """
    def __init__(self, heating_capacity_w, cooling_capacity_w, proportional_gain_w_k):
        """
        Initializes the HVAC system with its maximum power capacities and control gain.
        """
        self.heating_capacity_w = heating_capacity_w
        self.cooling_capacity_w = cooling_capacity_w
        self.proportional_gain_w_k = proportional_gain_w_k

    def calculate_hvac_power(self, T_air_prev, T_heating_setpoint, T_cooling_setpoint):
        """
        Calculates the required HVAC power for the current timestep using
        proportional control.

        Returns:
            float: HVAC power in Watts (positive for heating, negative for cooling).
        """
        q_hvac = 0.0

        # Heating demand with proportional control
        heating_error = T_heating_setpoint - T_air_prev
        if heating_error > 0:
            # The ideal heating power is proportional to the error
            ideal_heating_power = self.proportional_gain_w_k * heating_error
            # The actual power is capped by the system's capacity
            q_hvac = min(ideal_heating_power, self.heating_capacity_w)
        
        # Cooling demand with proportional control
        cooling_error = T_air_prev - T_cooling_setpoint
        if cooling_error > 0:
            ideal_cooling_power = self.proportional_gain_w_k * cooling_error
            q_hvac = -min(ideal_cooling_power, self.cooling_capacity_w)
        
        return q_hvac

class StatefulHVAC:
    """
    Implements a stateful, realistic HVAC model.
    
    This model includes:
    - Hysteresis (deadband) to turn on/off.
    - Ramp-up/down time to simulate system inertia.
    - Minimum run and off times to prevent short-cycling.
    """
    
    def __init__(self, heating_capacity_w, cooling_capacity_w, 
                 heating_deadband_c, cooling_deadband_c,
                 min_runtime_minutes, min_offtime_minutes,
                 ramp_up_minutes, dt_seconds):
        
        self.heating_capacity_w = heating_capacity_w
        self.cooling_capacity_w = cooling_capacity_w
        self.heating_deadband_c = heating_deadband_c
        self.cooling_deadband_c = cooling_deadband_c
        
        # Calculate step-based parameters from minute-based inputs
        self.dt_minutes = dt_seconds / 60.0
        # Ensure dt_minutes is not zero to avoid division by zero
        if self.dt_minutes == 0:
             raise ValueError("dt_seconds must be greater than zero.")
             
        self._min_runtime_steps = math.ceil(min_runtime_minutes / self.dt_minutes)
        self._min_offtime_steps = math.ceil(min_offtime_minutes / self.dt_minutes)
        self._ramp_up_steps = math.ceil(ramp_up_minutes / self.dt_minutes)
        if self._ramp_up_steps == 0:
            self._ramp_up_steps = 1 # Ramp up in one step if set too low
            
        # --- RAMP LOGIC ---
        # Calculate a fixed WATTAGE increment per step for ramp
        # Use max capacity for a symmetrical ramp increment
        max_capacity = max(self.heating_capacity_w, self.cooling_capacity_w, 1.0) # Avoid div by zero
        if max_capacity == 0:
             print("Warning: HVAC capacity is zero.")
             self.ramp_increment_w = 0
        else:
             self.ramp_increment_w = max_capacity / self._ramp_up_steps
        # --- END RAMP LOGIC ---

        # Initialize state variables
        self.hvac_state = "OFF"  # "OFF", "HEATING", "COOLING"
        self.steps_in_current_state = self._min_offtime_steps # Start as if it's been off
        self.current_output_w = 0.0

    def calculate_hvac_power(self, T_air_prev, T_heating_setpoint, T_cooling_setpoint):
        """
        Calculates the required HVAC power for the current timestep.
        """
        
        # --- 1. Determine Target State (what the thermostat *wants* to do) ---
        heating_on_temp = T_heating_setpoint - self.heating_deadband_c
        cooling_on_temp = T_cooling_setpoint + self.cooling_deadband_c
        
        target_state = "OFF"
        if self.hvac_state == "HEATING":
            # If heating, it *wants* to stay on until it hits the setpoint
            target_state = "HEATING"
            if T_air_prev > T_heating_setpoint:
                target_state = "OFF"
        elif self.hvac_state == "COOLING":
            # If cooling, it *wants* to stay on until it hits the setpoint
            target_state = "COOLING"
            if T_air_prev < T_cooling_setpoint:
                target_state = "OFF"
        else: # hvac_state is "OFF"
            # If off, it *wants* to turn on if it crosses a threshold
            if T_air_prev < heating_on_temp:
                target_state = "HEATING"
            elif T_air_prev > cooling_on_temp:
                target_state = "COOLING"

        # --- 2. Apply Anti-Short-Cycling Logic (what the system *is allowed* to do) ---
        
        new_state = self.hvac_state
        state_changed = False

        if self.hvac_state == "OFF":
            if target_state != "OFF" and self.steps_in_current_state >= self._min_offtime_steps:
                new_state = target_state # Allow turning on
                state_changed = True
        
        elif self.hvac_state == "HEATING":
            if target_state != "HEATING" and self.steps_in_current_state >= self._min_runtime_steps:
                new_state = target_state # Allow turning off or switching to cooling
                state_changed = True

        elif self.hvac_state == "COOLING":
            if target_state != "COOLING" and self.steps_in_current_state >= self._min_runtime_steps:
                new_state = target_state # Allow turning off or switching to heating
                state_changed = True
        
        # --- Update State and Counters Correctly ---
        self.hvac_state = new_state
        
        if state_changed:
            self.steps_in_current_state = 1 # Reset to 1 (first step in new state)
        else:
            self.steps_in_current_state += 1 # Increment steps in existing state
        # --- END ---

        # --- 3. Calculate Power Output (NEW ROBUST RAMP LOGIC) ---
        
        # Determine the *target* power based on the *allowed* state
        target_output_w = 0.0
        if self.hvac_state == "HEATING":
            target_output_w = self.heating_capacity_w
        elif self.hvac_state == "COOLING":
            target_output_w = -self.cooling_capacity_w
        # If state is "OFF", target_output_w remains 0.0
            
        # Calculate the difference and the direction
        delta_w = target_output_w - self.current_output_w
        
        if abs(delta_w) < 1e-6:
            # We are at the target, do nothing
            self.current_output_w = target_output_w
        elif delta_w > 0:
            # Need to ramp up (e.g., 0 -> 1000 or -4000 -> -3000)
            self.current_output_w += self.ramp_increment_w
            # Don't overshoot the target
            self.current_output_w = min(self.current_output_w, target_output_w)
        else: # delta_w < 0
            # Need to ramp down (e.g., 5000 -> 4000 or -1000 -> -2000)
            self.current_output_w -= self.ramp_increment_w
            # Don't overshoot the target
            self.current_output_w = max(self.current_output_w, target_output_w)
        
        # Final clamp to ensure we are within absolute bounds
        self.current_output_w = max(-self.cooling_capacity_w, 
                                min(self.heating_capacity_w, self.current_output_w))
                                
        return self.current_output_w


class PIDControlledHVAC:
    """
    Implements a PID (Proportional-Integral-Derivative) controller for HVAC.
    This provides smoother control than on/off or simple proportional systems,
    mimicking inverter-driven or modulating HVAC systems.
    """
    def __init__(self, heating_capacity_w, cooling_capacity_w, kp, ki, kd, dt_seconds):
        """
        Args:
            heating_capacity_w (float): Max heating power (W)
            cooling_capacity_w (float): Max cooling power (W)
            kp (float): Proportional gain
            ki (float): Integral gain
            kd (float): Derivative gain
            dt_seconds (float): Simulation timestep size
        """
        self.heating_capacity_w = heating_capacity_w
        self.cooling_capacity_w = cooling_capacity_w
        
        # PID Parameters
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt_seconds
        
        # State variables for PID
        self._integral = 0.0
        self._prev_error = 0.0
        
        # We track the previous mode to reset integral when switching modes
        # Modes: 0 = Off/Deadband, 1 = Heating, -1 = Cooling
        self._last_mode = 0 

    def calculate_hvac_power(self, T_air_prev, T_heating_setpoint, T_cooling_setpoint):
        """
        Calculates power using PID logic.
        """
        error = 0.0
        current_mode = 0 # 0: Neutral, 1: Heating, -1: Cooling
        
        # 1. Determine Error and Mode
        # Note: We treat error as positive = "needs action"
        if T_air_prev < T_heating_setpoint:
            error = T_heating_setpoint - T_air_prev
            current_mode = 1
        elif T_air_prev > T_cooling_setpoint:
            error = T_air_prev - T_cooling_setpoint
            current_mode = -1
        else:
            # Inside deadband
            error = 0.0
            current_mode = 0

        # 2. Reset logic (Anti-windup / Mode switching)
        # If we switched modes (e.g., heating to cooling) or entered deadband,
        # we should reset the integral term to prevent fighting the new direction.
        if current_mode != self._last_mode:
            self._integral = 0.0
            self._prev_error = 0.0 # Reset derivative kick
        
        self._last_mode = current_mode

        if current_mode == 0:
            return 0.0

        # 3. PID Calculation
        
        # Proportional term
        P = self.kp * error
        
        # Integral term
        self._integral += error * self.dt
        I = self.ki * self._integral
        
        # Derivative term
        derivative = (error - self._prev_error) / self.dt
        D = self.kd * derivative
        
        self._prev_error = error
        
        # Total raw PID output (unscaled)
        pid_output = P + I + D
        
        # 4. Apply Output to Capacity
        q_hvac = 0.0
        
        if current_mode == 1: # Heating
            # Clamp integral to avoid "windup" where I-term grows infinitely
            # Simple clamping strategy: If output is maxed out, stop growing integral
            if pid_output > self.heating_capacity_w:
                pid_output = self.heating_capacity_w
                # Anti-windup: Back-calculate integral to keep it at the limit
                # (Optional, but simple clamping prevents growth)
                self._integral -= error * self.dt 
            
            q_hvac = max(0.0, pid_output)
            
        elif current_mode == -1: # Cooling
            if pid_output > self.cooling_capacity_w:
                pid_output = self.cooling_capacity_w
                self._integral -= error * self.dt 

            # Cooling power is returned as negative Watts
            q_hvac = -max(0.0, pid_output)

        return q_hvac


def create_hvac_system(hvac_props: dict, dt_seconds: float):
    """
    Factory function to create an HVAC system object from config properties.
    
    Args:
        hvac_props: The 'hvac_system' dictionary from the config file.
        dt_seconds: The simulation time step in seconds.
        
    Returns:
        An instantiated HVAC object (e.g., StatefulHVAC, VerySimpleHVAC, PIDControlledHVAC).
    """
    
    # Get the model type, default to 'StatefulHVAC' if not specified
    model_type = hvac_props.get('model_type', 'StatefulHVAC')
    print(f"Creating HVAC model of type: {model_type}")

    try:
        if model_type == 'StatefulHVAC':
            return StatefulHVAC(
                heating_capacity_w=hvac_props['heating_capacity_w'],
                cooling_capacity_w=hvac_props['cooling_capacity_w'],
                heating_deadband_c=hvac_props['heating_deadband_c'],
                cooling_deadband_c=hvac_props['cooling_deadband_c'],
                min_runtime_minutes=hvac_props['min_runtime_minutes'],
                min_offtime_minutes=hvac_props['min_offtime_minutes'],
                ramp_up_minutes=hvac_props['ramp_up_minutes'],
                dt_seconds=dt_seconds
            )
            
        elif model_type == 'VerySimpleHVAC':
            return VerySimpleHVAC(
                heating_capacity_w=hvac_props['heating_capacity_w'],
                cooling_capacity_w=hvac_props['cooling_capacity_w'],
                proportional_gain_w_k=hvac_props['proportional_gain_w_k']
            )

        elif model_type == 'PIDControlledHVAC':
            return PIDControlledHVAC(
                heating_capacity_w=hvac_props['heating_capacity_w'],
                cooling_capacity_w=hvac_props['cooling_capacity_w'],
                kp=hvac_props.get('kp', 1000),   # Default if not in config
                ki=hvac_props.get('ki', 10),     # Default if not in config
                kd=hvac_props.get('kd', 0),      # Default if not in config
                dt_seconds=dt_seconds
            )
            
        else:
            raise ValueError(f"Unknown HVAC model_type in config: '{model_type}'")
            
    except KeyError as e:
        print(f"Error: Missing required HVAC parameter {e} for model '{model_type}' in config file.")
        raise
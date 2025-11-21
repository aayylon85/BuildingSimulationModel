"""
Main script to set up and run a single-zone thermal simulation from a 
JSON configuration file.

This version has been refactored to run the simulation step-by-step
in this file, allowing for dynamic occupant interactions.
"""
import json
import numpy as np
import argparse
import sys
import os
import datetime
import shutil

from materials import create_constructions_dict 
from zone import Zone
from exterior_heat_transfer import AdaptiveConvectionAlgorithm
from interior_heat_transfer import InternalAdaptiveConvection
from hvac_def import create_hvac_system
from plotting import plot_simulation_results
from boundary_conditions import create_boundary_conditions
# Import the Occupant class
from occupants import Occupant


def run_simulation_from_config(config_path):
    """
    Loads a JSON configuration, runs the simulation step-by-step,
    plots the results, and saves the results to disk.
    """
    # Capture start time for accurate file timestamping
    start_dt = datetime.datetime.now()

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{config_path}'", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON in configuration file '{config_path}'.", file=sys.stderr)
        print("Please check for syntax errors (e.g., trailing commas, unmatched brackets).", file=sys.stderr)
        sys.exit(1)

    # --- Run the simulation with error handling for config structure ---
    try:
        # --- 1. Simulation Setup from Config ---
        sim_settings = config['simulation_settings']
        dt_minutes = sim_settings['dt_minutes']
        duration_days = sim_settings['duration_days']
        duration_hours = duration_days * 24
        
        
        dt_sec = dt_minutes * 60
        num_steps = int(duration_hours * 3600 / dt_sec)
        
        # Use np.arange for precise time steps
        time_hours = np.arange(num_steps) * dt_sec / 3600.0

        # --- 2. Define Zone, Construction, and Windows from Config ---
        constructions = create_constructions_dict(config)
        zone_props = config['zone_properties']
        geometry_data = config['geometry']
        
        zone = Zone(zone_properties=zone_props,
                    geometry_data=geometry_data,
                    constructions=constructions,
                    dt_seconds=dt_sec,
                    windows_data=config['windows'],
                    air_exchange_data=config.get('air_exchange'),
                    zone_sensible_heat_capacity_multiplier=zone_props.get('zone_sensible_heat_capacity_multiplier', 1.0)
                   )

        # --- 3. Define HVAC System from Config ---
        hvac_props = config['hvac_system']
        hvac_system = create_hvac_system(hvac_props, dt_sec)

        # --- 4. Define Boundary Conditions from Config ---
        (
            heating_setpoint_profile, 
            cooling_setpoint_profile, 
            internal_gains_profile, 
            window_opening_profile, # This will be all zeros
            weather_data
        ) = create_boundary_conditions(config, num_steps, time_hours)
            
        # --- 5. Configure Convection Models from Config ---
        conv_models = config['convection_models']
        exterior_convection_model = AdaptiveConvectionAlgorithm(conv_models['exterior_hf'], conv_models['exterior_hn'])
        interior_convection_model = InternalAdaptiveConvection(conv_models['interior'])
        
        # --- 6. Create Occupants ---
        occupant_objects = []
        if 'occupancy' in config and 'occupants' in config['occupancy']:
            for occ_data in config['occupancy']['occupants']:
                occupant_objects.append(Occupant(occ_data))
        print(f"Initialized {len(occupant_objects)} occupants.")

        # --- 7. Run Warm-up ---
        print("Starting dynamic stabilization warm-up...")
        T_air_prev = zone.run_warmup(
            heating_setpoint_profile, cooling_setpoint_profile, weather_data,
            internal_gains_profile, interior_convection_model,
            exterior_convection_model, hvac_system
        )
        print("Warm-up complete. Starting main simulation.")

        # --- 8. Run Main Simulation (Step-by-Step) ---
        
        # --- Result Arrays ---
        zone_air_temps = np.zeros(num_steps)
        hvac_energy = np.zeros(num_steps)
        fabric_loss = np.zeros(num_steps)
        air_exchange_loss = np.zeros(num_steps)
        solar_gains = np.zeros(num_steps)
        window_state_profile = np.zeros(num_steps)
        
        # --- Dynamic State Variables ---
        current_window_fraction = 0.0 # Start with window closed
        # Copy setpoint profiles so they can be modified dynamically
        dynamic_heating_setpoint = np.copy(heating_setpoint_profile)
        dynamic_cooling_setpoint = np.copy(cooling_setpoint_profile)
        
        # --- Occupant Action Setup ---
        occupant_check_interval_minutes = 60
        steps_per_occupant_check = int(occupant_check_interval_minutes / dt_minutes)
        if steps_per_occupant_check == 0:
            steps_per_occupant_check = 1 # Check every step if dt > 10 min
        
        thermostat_adjustment_c = 1.0 # How much to change setpoint per vote

        zone_air_temps[0] = T_air_prev # Start from the warmed-up temperature

        for t in range(num_steps):
            if t > 0:
                T_air_prev = zone_air_temps[t-1]
            
            # --- (A) Check for Occupant Actions ---
            # Get current weather for this step *before* occupant check
            current_weather = weather_data[t]
            
            if t % steps_per_occupant_check == 0:
                hour_of_day = time_hours[t] % 24
                outside_temp_c = current_weather['air_temp_c']
                
                window_votes = 0
                thermostat_votes = 0
                
                present_occupants = []
                for occ in occupant_objects:
                    if occ.is_present(hour_of_day):
                        present_occupants.append(occ)

                if present_occupants:
                    for occ in present_occupants:
                        # Get votes based on the previous step's temperature
                        win_action, therm_action = occ.get_desired_action(
                            T_air_prev, 
                            current_window_fraction,
                            outside_temp_c  # Pass outside temperature
                        )
                        
                        if win_action == "open_window":
                            window_votes += 1
                        elif win_action == "close_window":
                            window_votes -= 1
                            
                        if therm_action == "heat_up":
                            thermostat_votes += 1
                        elif therm_action == "cool_down":
                            thermostat_votes -= 1
                
                # --- Aggregate window votes ---
                # Opposing votes cancel out. Net positive = open.
                if window_votes > 0:
                    current_window_fraction = 1.0
                elif window_votes < 0:
                    current_window_fraction = 0.0
                # if votes == 0, no change
                

                # --- Aggregate thermostat votes ---
                # Apply changes to the *rest* of the schedule
                if thermostat_votes > 0:
                    dynamic_heating_setpoint[t:] += thermostat_adjustment_c
                    dynamic_cooling_setpoint[t:] += thermostat_adjustment_c
                elif thermostat_votes < 0:
                    dynamic_heating_setpoint[t:] -= thermostat_adjustment_c
                    dynamic_cooling_setpoint[t:] -= thermostat_adjustment_c
            
            # --- (B) Get current step's boundary conditions ---
            # current_weather is already fetched above
            current_internal_gains = internal_gains_profile[t]
            current_heating_setpoint = dynamic_heating_setpoint[t]
            current_cooling_setpoint = dynamic_cooling_setpoint[t]
            
            # --- (C) Run one simulation step ---
            step_results = zone.run_simulation_step(
                T_air_prev,
                current_weather,
                current_internal_gains,
                current_heating_setpoint,
                current_cooling_setpoint,
                current_window_fraction, # Pass the dynamic value
                interior_convection_model,
                exterior_convection_model,
                hvac_system
            )
            
            # --- (D) Store results ---
            zone_air_temps[t] = step_results['T_air_new']
            hvac_energy[t] = step_results['q_hvac']
            fabric_loss[t] = step_results['q_fabric_loss']
            air_exchange_loss[t] = step_results['q_air_exchange_loss']
            solar_gains[t] = step_results['q_solar_gains']
            window_state_profile[t] = current_window_fraction

        print("Simulation complete.")

        # --- 9. Process and Display Results ---
        total_heating_kwh = np.sum(hvac_energy[hvac_energy > 0]) * dt_sec / (3600 * 1000)
        total_cooling_kwh = -np.sum(hvac_energy[hvac_energy < 0]) * dt_sec / (3600 * 1000)
        total_hvac_kwh = total_heating_kwh + total_cooling_kwh
        print(f"\nTotal Heating Energy Consumed: {total_heating_kwh:.2f} kWh")
        print(f"\nTotal Cooling Energy Consumed: {total_cooling_kwh:.2f} kWh")
        print(f"\nTotal HVAC Energy Consumed: {total_hvac_kwh:.2f} kWh")
        
        # Manually assemble the results dictionary for plotting
        results = {
            'zone_air_temps': zone_air_temps,
            'hvac_energy': hvac_energy,
            'fabric_loss': fabric_loss,
            'air_exchange_loss': air_exchange_loss,
            'solar_gains': solar_gains,
            'internal_gains': internal_gains_profile,
            'window_state': window_state_profile # Add window state to results
        }

        # --- SAVE RESULTS ---
        # Format date and time strings
        date_str = start_dt.strftime("%Y-%m-%d")
        time_str = start_dt.strftime("%H-%M-%S")
        
        # Define directory structure: results/YYYY-MM-DD/
        results_dir = os.path.join("results", date_str)
        
        # Create directory if it doesn't exist
        try:
            os.makedirs(results_dir, exist_ok=True)
        except OSError as e:
            print(f"Error creating results directory '{results_dir}': {e}", file=sys.stderr)
        
        # Define base filename
        base_filename = f"{date_str}_{time_str}_{duration_days}days"
        
        # 1. Save copy of configuration file
        config_save_path = os.path.join(results_dir, f"{base_filename}_config.json")
        try:
            shutil.copy(config_path, config_save_path)
            print(f"Saved configuration copy to: {config_save_path}")
        except Exception as e:
            print(f"Warning: Could not save configuration copy: {e}")

        # 2. Save Simulation Data to CSV
        csv_save_path = os.path.join(results_dir, f"{base_filename}_results.csv")
        try:
            # Extract outside air temps from weather data list/array
            outside_temps = np.array([w['air_temp_c'] for w in weather_data])
            
            # Prepare data columns
            # Columns: Time, Zone Temp, Outside Temp, HVAC Power, Fabric Loss, Air Ex Loss, Solar, Internal, Window State
            header = "Time (hrs),Zone Temp (C),Outside Temp (C),HVAC Power (W),Fabric Loss (W),Air Exchange Loss (W),Solar Gains (W),Internal Gains (W),Window State (0-1)"
            
            data_to_save = np.column_stack((
                time_hours,
                zone_air_temps,
                outside_temps,
                hvac_energy,
                fabric_loss,
                air_exchange_loss,
                solar_gains,
                internal_gains_profile,
                window_state_profile
            ))
            
            np.savetxt(csv_save_path, data_to_save, delimiter=",", header=header, comments="", fmt="%.4f")
            print(f"Saved simulation results to: {csv_save_path}")
        except Exception as e:
            print(f"Error saving CSV results: {e}", file=sys.stderr)
        
        
        # --- 10. Plot Results ---
        plot_simulation_results(
            time_hours,
            weather_data,
            heating_setpoint_profile, # Original setpoint
            cooling_setpoint_profile, # Original setpoint
            results,
            num_steps,
            duration_hours
        )

    except KeyError as e:
        print(f"Error: Missing or invalid key in configuration file '{config_path}'.", file=sys.stderr)
        print(f"Details: The required key {e} was not found or is nested incorrectly.", file=sys.stderr)
        print("Please ensure the config file has the correct structure.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # Catch other potential simulation errors
        print(f"An unexpected error occurred during the simulation setup or run: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc() # Print full error traceback
        sys.exit(1)


if __name__ == '__main__':
    # ... (argparse code remains unchanged) ...
    parser = argparse.ArgumentParser(
        description="Run a single-zone thermal simulation from a JSON config file."
    )
    parser.add_argument(
        'config_file', 
        type=str, 
        nargs='?', 
        default='simulation_config.json', 
        help="Path to the simulation JSON configuration file (default: simulation_config.json)"
    )
    
    args = parser.parse_args()
    
    run_simulation_from_config(args.config_file)
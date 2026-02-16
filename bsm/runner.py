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

from bsm.components.materials import create_constructions_dict
from bsm.core.zone import Zone
from bsm.heat_transfer.exterior_convection import AdaptiveConvectionAlgorithm
from bsm.heat_transfer.interior_convection import InternalAdaptiveConvection
from bsm.components.hvac import create_hvac_system
from bsm.output.plotting import plot_simulation_results
from bsm.boundary.conditions import create_boundary_conditions
from bsm.output.report_generator import generate_report_from_results
# Import the Occupant class
from bsm.boundary.occupants import Occupant

# LLM agent integration (optional)
LLM_AVAILABLE = False
LLM_IMPORT_ERROR = None
try:
    from bsm.agents.manager import LLMOccupantManager
    LLM_AVAILABLE = True
except ImportError as e:
    LLM_IMPORT_ERROR = str(e)


def get_simulation_datetime(config: dict, time_hour: float) -> datetime.datetime:
    """
    Convert simulation time_hour to actual datetime.

    Args:
        config: Simulation configuration dict
        time_hour: Time in hours from simulation start

    Returns:
        datetime object representing the simulation time
    """
    start_date_str = config.get('simulation_settings', {}).get('start_date', '2026-01-14')
    start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
    # Add timezone info (UTC)
    import pytz
    start_date = start_date.replace(tzinfo=pytz.UTC)
    return start_date + datetime.timedelta(hours=time_hour)


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
                    zone_sensible_heat_capacity_multiplier=zone_props.get('zone_sensible_heat_capacity_multiplier', 1.0),
                    max_nodes_per_layer=sim_settings.get('max_nodes_per_layer', 20),
                    hvac_coupling_mode=sim_settings.get('hvac_coupling_mode', 'auto'),
                    hvac_max_power_rate_w_s=sim_settings.get('hvac_max_power_rate_w_s', 50000.0),
                    enable_opaque_solar_absorption=sim_settings.get('enable_opaque_solar_absorption', False),
                    interior_radiation_method=sim_settings.get('interior_radiation_method', 'area_weighted')
                   )

        # --- 3. Define HVAC System from Config ---
        hvac_props = config['hvac_system']
        hvac_system = create_hvac_system(hvac_props, dt_sec)

        # --- 4. Define Boundary Conditions from Config ---
        (
            heating_setpoint_profile,
            cooling_setpoint_profile,
            internal_gains_profile,
            window_opening_profile,  # This will be all zeros
            weather_data,
            scheduled_equipment_power,  # Equipment power profile (baseline/simple modes)
            scheduled_lighting_power,   # Lighting power profile (baseline/simple modes)
        ) = create_boundary_conditions(config, num_steps, time_hours)
            
        # --- 5. Configure Convection Models from Config ---
        conv_models = config['convection_models']
        exterior_convection_model = AdaptiveConvectionAlgorithm(conv_models['exterior_hf'], conv_models['exterior_hn'])
        interior_convection_model = InternalAdaptiveConvection(conv_models['interior'])
        
        # --- 6. Create Occupants ---
        # Check if LLM agents are enabled
        llm_requested = config.get('llm_agents', {}).get('enabled', False)
        use_llm_agents = llm_requested and LLM_AVAILABLE
        llm_manager = None

        if llm_requested and not LLM_AVAILABLE:
            print(f"Warning: LLM agents requested but unavailable: {LLM_IMPORT_ERROR}", file=sys.stderr)
            print("Falling back to rule-based occupants.", file=sys.stderr)

        if use_llm_agents:
            print("Initializing LLM-based occupant agents...")
            # Create functions to provide current state to LLM manager
            # These will be called during the simulation loop
            current_zone_temp = [20.0]  # Mutable container for current temp
            current_weather_data = [{}]  # Mutable container for current weather

            def get_zone_temp():
                return current_zone_temp[0]

            def get_weather():
                return current_weather_data[0]

            # LLMOccupantManager creates unique agent directories per run automatically
            llm_manager = LLMOccupantManager(
                config=config,
                get_zone_temp=get_zone_temp,
                get_weather=get_weather,
                results_dir="results",  # Agent data goes in results/agents/YYYY-MM-DD/HH-MM-SS/
            )
            print(f"Initialized {len(llm_manager.get_agent_ids())} LLM occupant agents.")
        else:
            # Fall back to rule-based occupants
            occupant_objects = []
            if 'occupancy' in config and 'occupants' in config['occupancy']:
                for occ_data in config['occupancy']['occupants']:
                    occupant_objects.append(Occupant(occ_data))
            print(f"Initialized {len(occupant_objects)} rule-based occupants.")

        # --- 6b. Initialize UI Demo Writer (optional) ---
        ui_writer = None
        if config.get('ui_demo_output', {}).get('enabled', False):
            try:
                from bsm.output.ui_demo_writer import UIDemoWriter
                output_dir = config.get('ui_demo_output', {}).get('output_dir', 'UI_demo_outputs')
                ui_writer = UIDemoWriter(output_dir=output_dir)
                print(f"UI demo output enabled. Writing to: {output_dir}/")
            except ImportError as e:
                print(f"Warning: UI demo writer unavailable: {e}", file=sys.stderr)

        # Pass UI writer to LLM manager if both are enabled
        if ui_writer and use_llm_agents and llm_manager is not None:
            llm_manager.set_ui_writer(ui_writer)

        # --- 7. Run Warm-up ---
        stabilization_days = sim_settings.get('stabilization_days', 3)
        print(f"Starting dynamic stabilization warm-up ({stabilization_days} days)...")
        T_air_prev = zone.run_warmup(
            heating_setpoint_profile, cooling_setpoint_profile, weather_data,
            internal_gains_profile, interior_convection_model,
            exterior_convection_model, hvac_system, stabilization_days,
            use_adaptive_timestepping=sim_settings.get('use_adaptive_timestepping', True)
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
        # Get parameters from config or use defaults
        occupant_check_interval_minutes = config.get('occupancy', {}).get('check_interval_minutes', 60)
        steps_per_occupant_check = int(occupant_check_interval_minutes / dt_minutes)
        if steps_per_occupant_check == 0:
            steps_per_occupant_check = 1 # Check every step if dt > 10 min

        # How much to change setpoint per occupant vote
        thermostat_adjustment_c = config.get('occupancy', {}).get('thermostat_adjustment_c', 1.0)

        zone_air_temps[0] = T_air_prev # Start from the warmed-up temperature

        # Additional arrays for tracking equipment and lighting
        # For LLM mode: populated by LLM manager
        # For non-LLM modes: use scheduled profiles from conditions.py
        if use_llm_agents:
            equipment_power_profile = np.zeros(num_steps)
            lighting_power_profile = np.zeros(num_steps)
        else:
            # Use the scheduled profiles generated by conditions.py
            equipment_power_profile = np.copy(scheduled_equipment_power)
            lighting_power_profile = np.copy(scheduled_lighting_power)

        occupant_count_profile = np.zeros(num_steps)
        thermostat_offset_profile = np.zeros(num_steps)  # Track agent thermostat adjustments

        # Progress tracking
        progress_interval_steps = max(1, int(60 / dt_minutes))  # Report every simulated hour
        last_progress_day = -1
        llm_decision_count = 0

        for t in range(num_steps):
            if t > 0:
                T_air_prev = zone_air_temps[t-1]

            # --- Progress Reporting ---
            current_hour = time_hours[t]
            current_day = int(current_hour // 24)
            hour_of_day_progress = current_hour % 24

            # Report progress at start of each new day
            if current_day != last_progress_day:
                pct_complete = (t / num_steps) * 100
                sim_datetime_str = get_simulation_datetime(config, current_hour).strftime('%Y-%m-%d')
                if use_llm_agents:
                    print(f"[Progress] Day {current_day + 1}/{duration_days} ({sim_datetime_str}) - {pct_complete:.1f}% complete, LLM decisions: {llm_decision_count}")
                else:
                    print(f"[Progress] Day {current_day + 1}/{duration_days} ({sim_datetime_str}) - {pct_complete:.1f}% complete")
                last_progress_day = current_day

            # --- (A) Check for Occupant Actions ---
            # Get current weather for this step *before* occupant check
            current_weather = weather_data[t]

            # Update state containers for LLM manager
            if use_llm_agents:
                current_zone_temp[0] = T_air_prev
                current_weather_data[0] = current_weather

            # --- Track occupant count for non-LLM modes (every timestep) ---
            if not use_llm_agents:
                hour_of_day_tracking = time_hours[t] % 24
                count = 0
                for occ in occupant_objects:
                    if occ.is_present(hour_of_day_tracking):
                        count += 1
                occupant_count_profile[t] = count

            # --- LLM Agent Mode ---
            if use_llm_agents and llm_manager is not None:
                # Get simulation datetime
                sim_datetime = get_simulation_datetime(config, time_hours[t])

                # Call step() every timestep - let checkpoint system control when decisions happen
                # Decisions trigger at: (1) hourly checkpoints, OR (2) plan events starting/ending
                equipment_power, lighting_power, thermostat_offset, window_fraction = llm_manager.step(
                    now=sim_datetime,
                    force_all=False,  # Let DecisionCheckpointManager control decision timing
                )
                # Count only when actual decisions were made
                if llm_manager.had_decisions_this_step():
                    llm_decision_count += 1

                # Update state based on LLM decisions
                current_window_fraction = window_fraction

                # Apply thermostat offset to setpoints
                dynamic_heating_setpoint[t:] = heating_setpoint_profile[t:] + thermostat_offset
                dynamic_cooling_setpoint[t:] = cooling_setpoint_profile[t:] + thermostat_offset

                # Track power consumption, occupant count, and thermostat offset
                equipment_power_profile[t] = equipment_power
                lighting_power_profile[t] = lighting_power
                occupant_count_profile[t] = llm_manager.get_present_count()
                thermostat_offset_profile[t] = thermostat_offset

            # --- Rule-Based Occupant Mode ---
            elif t % steps_per_occupant_check == 0:
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
            # Base internal gains from schedule (occupant metabolic heat, miscellaneous)
            current_internal_gains = internal_gains_profile[t]

            # Add equipment, lighting, and metabolic heat
            if use_llm_agents and llm_manager is not None:
                # LLM mode: get_total_internal_gains_w() includes:
                # - Equipment power (from equipment_power_profile[t])
                # - Lighting power (from lighting_power_profile[t])
                # - Metabolic heat (120W per present occupant)
                current_internal_gains += llm_manager.get_total_internal_gains_w()
            else:
                # Non-LLM mode: add equipment and lighting from scheduled profiles
                # (occupant metabolic heat is already in internal_gains_profile)
                current_internal_gains += equipment_power_profile[t]
                current_internal_gains += lighting_power_profile[t]

            current_heating_setpoint = dynamic_heating_setpoint[t]
            current_cooling_setpoint = dynamic_cooling_setpoint[t]
            
            # --- (C) Run one simulation step ---
            step_results = zone.run_simulation_step(
                T_air_prev,
                current_weather,
                current_internal_gains,
                current_heating_setpoint,
                current_cooling_setpoint,
                current_window_fraction,  # Pass the dynamic value
                interior_convection_model,
                exterior_convection_model,
                hvac_system,
                use_adaptive_timestepping=sim_settings.get('use_adaptive_timestepping', True)
            )
            
            # --- (D) Store results ---
            zone_air_temps[t] = step_results['T_air_new']
            hvac_energy[t] = step_results['q_hvac']
            fabric_loss[t] = step_results['q_fabric_loss']
            air_exchange_loss[t] = step_results['q_air_exchange_loss']
            solar_gains[t] = step_results['q_solar_gains']
            window_state_profile[t] = current_window_fraction

            # --- (E) Write to UI demo output (if enabled) ---
            if ui_writer and use_llm_agents and llm_manager is not None:
                sim_datetime = get_simulation_datetime(config, time_hours[t])
                ui_writer.write_timestep({
                    "simulation_time": sim_datetime.isoformat(),
                    "external_temp_c": current_weather['air_temp_c'],
                    "building_temp_c": step_results['T_air_new'],
                    "total_power_w": abs(hvac_energy[t]) + equipment_power_profile[t] + lighting_power_profile[t],
                    "hvac_power_w": hvac_energy[t],
                    "equipment_power_w": equipment_power_profile[t],
                    "lighting_power_w": lighting_power_profile[t],
                    "equipment_state": llm_manager.equipment.get_equipment_state(),
                    "lighting_state": llm_manager.lighting.get_detailed_lighting_state(),
                    "occupant_states": llm_manager.get_all_occupant_states(),
                })

        if use_llm_agents:
            print(f"Simulation complete. Total LLM decision rounds: {llm_decision_count}")
        else:
            print("Simulation complete.")

        # Finalize UI demo output
        if ui_writer:
            ui_writer.finalize()
            stats = ui_writer.get_stats()
            print(f"UI demo output: {stats['timesteps']} timesteps, "
                  f"{stats['action_files']} agent files, {stats['conversation_files']} conversations")

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
            'window_state': window_state_profile,
            'dynamic_heating_setpoint': dynamic_heating_setpoint,
            'dynamic_cooling_setpoint': dynamic_cooling_setpoint,
            'equipment_power': equipment_power_profile,
            'lighting_power': lighting_power_profile,
            'occupant_count': occupant_count_profile,
            'thermostat_offset': thermostat_offset_profile,
        }

        # --- SAVE RESULTS ---
        # Format date and time strings
        date_str = start_dt.strftime("%Y-%m-%d")
        time_str = start_dt.strftime("%H-%M-%S")

        # Define directory structure: results/YYYY-MM-DD/HH-MM-SS/
        # Each simulation run gets its own timestamped folder to prevent overwrites
        run_dir = os.path.join("results", date_str, time_str)

        # Create directory if it doesn't exist
        try:
            os.makedirs(run_dir, exist_ok=True)
        except OSError as e:
            print(f"Error creating results directory '{run_dir}': {e}", file=sys.stderr)

        # Define base filename (simpler since directory provides uniqueness)
        base_filename = f"sim_{duration_days}days"
        
        # 1. Save copy of configuration file
        config_save_path = os.path.join(run_dir, f"{base_filename}_config.json")
        try:
            shutil.copy(config_path, config_save_path)
            print(f"Saved configuration copy to: {config_save_path}")
        except Exception as e:
            print(f"Warning: Could not save configuration copy: {e}")

        # 2. Save Simulation Data to CSV
        csv_save_path = os.path.join(run_dir, f"{base_filename}_results.csv")
        try:
            # Extract outside air temps from weather data list/array
            outside_temps = np.array([w['air_temp_c'] for w in weather_data])

            # Prepare data columns - always include equipment/lighting for all modes
            header = "Time (hrs),Zone Temp (C),Outside Temp (C),HVAC Power (W),Fabric Loss (W),Air Exchange Loss (W),Solar Gains (W),Internal Gains (W),Window State (0-1),Equipment Power (W),Lighting Power (W),Occupant Count,Thermostat Offset (C),Heating Setpoint (C),Cooling Setpoint (C)"

            data_to_save = np.column_stack((
                time_hours,
                zone_air_temps,
                outside_temps,
                hvac_energy,
                fabric_loss,
                air_exchange_loss,
                solar_gains,
                internal_gains_profile,
                window_state_profile,
                equipment_power_profile,
                lighting_power_profile,
                occupant_count_profile,
                thermostat_offset_profile,
                dynamic_heating_setpoint,
                dynamic_cooling_setpoint,
            ))

            np.savetxt(csv_save_path, data_to_save, delimiter=",", header=header, comments="", fmt="%.4f")
            print(f"Saved simulation results to: {csv_save_path}")
        except Exception as e:
            print(f"Error saving CSV results: {e}", file=sys.stderr)

        # 3. Calculate comprehensive energy metrics (for ALL modes)
        total_equipment_kwh = np.sum(equipment_power_profile) * dt_sec / (3600 * 1000)
        total_lighting_kwh = np.sum(lighting_power_profile) * dt_sec / (3600 * 1000)
        total_energy_kwh = total_hvac_kwh + total_equipment_kwh + total_lighting_kwh

        # Peak power calculations
        peak_heating_w = float(np.max(hvac_energy[hvac_energy > 0])) if np.any(hvac_energy > 0) else 0.0
        peak_cooling_w = float(np.abs(np.min(hvac_energy[hvac_energy < 0]))) if np.any(hvac_energy < 0) else 0.0
        peak_equipment_w = float(np.max(equipment_power_profile)) if np.any(equipment_power_profile > 0) else 0.0
        peak_lighting_w = float(np.max(lighting_power_profile)) if np.any(lighting_power_profile > 0) else 0.0
        peak_total_w = float(np.max(np.abs(hvac_energy) + equipment_power_profile + lighting_power_profile))

        # Temperature statistics
        mean_zone_temp = float(np.mean(zone_air_temps))
        min_zone_temp = float(np.min(zone_air_temps))
        max_zone_temp = float(np.max(zone_air_temps))

        # Comfort hours: count timesteps where temp is within setpoint bounds
        comfort_mask = (zone_air_temps >= dynamic_heating_setpoint) & (zone_air_temps <= dynamic_cooling_setpoint)
        comfort_hours_pct = float(np.sum(comfort_mask) / num_steps * 100)

        # 4. Save JSON summary
        json_save_path = os.path.join(run_dir, f"{base_filename}_summary.json")
        try:
            json_results = {
                "simulation_info": {
                    "config_file": config_path,
                    "duration_days": duration_days,
                    "timestep_minutes": dt_minutes,
                    "total_timesteps": num_steps,
                    "llm_agents_enabled": use_llm_agents,
                    "start_datetime": start_dt.isoformat(),
                },
                "energy_summary_kwh": {
                    "heating": round(total_heating_kwh, 3),
                    "cooling": round(total_cooling_kwh, 3),
                    "total_hvac": round(total_hvac_kwh, 3),
                    "equipment": round(total_equipment_kwh, 3),
                    "lighting": round(total_lighting_kwh, 3),
                    "total": round(total_energy_kwh, 3),
                },
                "peak_power_w": {
                    "heating": round(peak_heating_w, 1),
                    "cooling": round(peak_cooling_w, 1),
                    "equipment": round(peak_equipment_w, 1),
                    "lighting": round(peak_lighting_w, 1),
                    "total": round(peak_total_w, 1),
                },
                "temperature_stats_c": {
                    "mean": round(mean_zone_temp, 2),
                    "min": round(min_zone_temp, 2),
                    "max": round(max_zone_temp, 2),
                    "comfort_hours_pct": round(comfort_hours_pct, 1),
                },
            }

            # Add agent summary if LLM mode
            if use_llm_agents and llm_manager is not None:
                json_results["agent_summary"] = {
                    "num_agents": len(llm_manager._agents) if hasattr(llm_manager, '_agents') else 0,
                    "total_decisions": llm_decision_count,
                    "thermostat_adjustments": int(np.sum(np.abs(np.diff(thermostat_offset_profile)) > 0.01)),
                    "window_changes": int(np.sum(np.abs(np.diff(window_state_profile)) > 0.01)),
                }

            with open(json_save_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            print(f"Saved JSON summary to: {json_save_path}")
        except Exception as e:
            print(f"Error saving JSON summary: {e}", file=sys.stderr)


        # --- 10. Create figures directory ---
        figures_dir = os.path.join(run_dir, "figures")
        try:
            os.makedirs(figures_dir, exist_ok=True)
        except OSError as e:
            print(f"Warning: Could not create figures directory: {e}")
            figures_dir = None

        # --- 11. Plot Results (with dynamic setpoints showing agent interventions) ---
        plot_simulation_results(
            time_hours,
            weather_data,
            dynamic_heating_setpoint,  # FIXED: Use dynamic setpoints with agent interventions
            dynamic_cooling_setpoint,  # FIXED: Use dynamic setpoints with agent interventions
            results,
            num_steps,
            duration_hours,
            save_dir=figures_dir,  # NEW: Pass save directory for figures
            use_llm_agents=use_llm_agents,  # NEW: Pass mode for agent activity plot
        )

        # --- 12. Generate Markdown Report ---
        generate_report_from_results(run_dir, base_filename)

        print(f"\n=== Simulation Complete ===")
        print(f"Results saved to: {run_dir}")
        print(f"  - CSV data: {base_filename}_results.csv")
        print(f"  - JSON summary: {base_filename}_summary.json")
        print(f"  - Report: {base_filename}_report.md")
        print(f"  - Figures: figures/")

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


# ---------------------------------------------------------------------------
# Checkpoint and Resume Functions
# ---------------------------------------------------------------------------

def save_simulation_checkpoint(
    checkpoint_id: str,
    config_path: str,
    simulation_time_hour: float,
    simulation_datetime: datetime.datetime,
    zone_air_temp_c: float,
    dynamic_heating_setpoint: np.ndarray,
    dynamic_cooling_setpoint: np.ndarray,
    current_window_fraction: float,
    llm_manager,  # LLMOccupantManager or None
    current_step: int,
    total_steps: int,
    results_dir: str = "results",
) -> str:
    """
    Save a checkpoint of the current simulation state.

    This is a manual checkpoint function - call it when you want to save state
    for later resumption.

    Args:
        checkpoint_id: Unique identifier for this checkpoint
        config_path: Path to simulation config file
        simulation_time_hour: Current simulation time in hours from start
        simulation_datetime: Current simulation datetime
        zone_air_temp_c: Current zone air temperature
        dynamic_heating_setpoint: Heating setpoint array
        dynamic_cooling_setpoint: Cooling setpoint array
        current_window_fraction: Current window open fraction
        llm_manager: LLMOccupantManager instance (or None if not using LLM agents)
        current_step: Current simulation step
        total_steps: Total simulation steps
        results_dir: Directory for storing checkpoints

    Returns:
        Path to saved checkpoint directory
    """
    from bsm.state.persistence import SimulationPersistence, create_checkpoint

    # Ensure LLM agent state is saved before checkpoint
    if llm_manager is not None:
        llm_manager.save_all_agents()

    # Create checkpoint
    checkpoint = create_checkpoint(
        checkpoint_id=checkpoint_id,
        config_path=config_path,
        simulation_time_hour=simulation_time_hour,
        simulation_datetime=simulation_datetime,
        zone_air_temp_c=zone_air_temp_c,
        dynamic_heating_setpoint=list(dynamic_heating_setpoint),
        dynamic_cooling_setpoint=list(dynamic_cooling_setpoint),
        current_window_fraction=current_window_fraction,
        llm_manager=llm_manager,
        current_step=current_step,
        total_steps=total_steps,
    )

    # Save it
    persistence = SimulationPersistence(results_dir=results_dir)
    checkpoint_dir = persistence.save_checkpoint(checkpoint)

    return str(checkpoint_dir)


def list_checkpoints(results_dir: str = "results"):
    """
    List all available checkpoints.

    Args:
        results_dir: Directory containing checkpoints

    Returns:
        List of checkpoint summaries
    """
    from bsm.state.persistence import SimulationPersistence

    persistence = SimulationPersistence(results_dir=results_dir)
    checkpoints = persistence.list_checkpoints()

    if not checkpoints:
        print("No checkpoints found.")
        return []

    print(f"\nAvailable checkpoints ({len(checkpoints)}):")
    print("-" * 80)
    for cp in checkpoints:
        print(f"  ID: {cp['id']}")
        print(f"      Created: {cp['created_at']}")
        print(f"      Simulation time: {cp['simulation_datetime']}")
        print(f"      Progress: {cp['progress']}")
        if cp.get('zone_temp_c') is not None:
            print(f"      Zone temp: {cp['zone_temp_c']:.1f}C")
        print()

    return checkpoints


def resume_simulation_from_checkpoint(
    checkpoint_id: str,
    additional_hours: float = 24.0,
    results_dir: str = "results",
):
    """
    Resume a simulation from a saved checkpoint.

    Creates a NEW results directory while loading state from the checkpoint.
    The original checkpoint is preserved.

    Args:
        checkpoint_id: ID of checkpoint to resume from
        additional_hours: How many additional simulation hours to run
        results_dir: Directory containing checkpoints

    Note:
        This is a placeholder for the full resume implementation.
        Full implementation requires refactoring run_simulation_from_config
        to accept start_step and initial state parameters.
    """
    from bsm.state.persistence import SimulationPersistence

    persistence = SimulationPersistence(results_dir=results_dir)

    try:
        checkpoint = persistence.load_checkpoint(checkpoint_id)
    except ValueError as e:
        print(f"Error: {e}")
        return

    print(f"\n[RESUME] Loading checkpoint: {checkpoint_id}")
    print(f"  Simulation datetime: {checkpoint.simulation_datetime}")
    print(f"  Progress: Step {checkpoint.current_step}/{checkpoint.total_steps}")
    print(f"  Zone temperature: {checkpoint.zone_air_temp_c:.1f}C")
    print(f"  Agent run dir: {checkpoint.agent_run_dir}")
    print(f"  Calendar DB: {checkpoint.calendar_db_path}")
    print(f"  Will run for additional {additional_hours} hours")

    # Load original config
    try:
        with open(checkpoint.config_path, 'r') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading config from checkpoint: {e}")
        return

    print(f"\n[RESUME] To fully resume, the runner needs to be called with:")
    print(f"  - Start from step: {checkpoint.current_step}")
    print(f"  - Initial zone temp: {checkpoint.zone_air_temp_c}C")
    print(f"  - Agent state from: {checkpoint.agent_run_dir}")
    print(f"  - Calendar from: {checkpoint.calendar_db_path}")
    print(f"\nNote: Full resume implementation requires runner refactoring.")
    print("The checkpoint data has been loaded and validated.")

    return checkpoint


def main():
    """Main entry point for CLI usage."""
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


if __name__ == '__main__':
    main()
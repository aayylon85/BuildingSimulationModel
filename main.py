"""
Main script to set up and run a single-zone thermal simulation from a 
JSON configuration file.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

from materials import create_constructions_dict # <-- CHANGED
from zone import Zone
from exterior_heat_transfer import AdaptiveConvectionAlgorithm
from interior_heat_transfer import InternalAdaptiveConvection
from hvac_def import StatefulHVAC
# Removed unused import: from air_exchange import AirExchangeManager

def run_simulation_from_config(config_path):
    """
    Loads a JSON configuration, runs the simulation, and plots the results.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    # --- 1. Simulation Setup from Config ---
    sim_settings = config['simulation_settings']
    dt_minutes = sim_settings['dt_minutes']
    duration_hours = sim_settings['duration_hours']
    dt_sec = dt_minutes * 60
    num_steps = int(duration_hours * 3600 / dt_sec)
    
    # Use np.arange for precise time steps
    time_hours = np.arange(num_steps) * dt_sec / 3600.0

    # --- 2. Define Zone, Construction, and Windows from Config ---
    constructions = create_constructions_dict(config)
    zone_props = config['zone_properties']
    geometry_data = config['geometry'] # Load new geometry section
    
    zone = Zone(zone_properties=zone_props,
                geometry_data=geometry_data, # Pass geometry
                constructions=constructions,
                dt_seconds=dt_sec,
                windows_data=config['windows'],
                air_exchange_data=config.get('air_exchange'),
                zone_sensible_heat_capacity_multiplier=zone_props.get('zone_sensible_heat_capacity_multiplier', 1.0)
               )

    # --- 3. Define HVAC System from Config ---
    hvac_props = config['hvac_system']
    hvac_system = StatefulHVAC(
        hvac_props['heating_capacity_w'], 
        hvac_props['cooling_capacity_w'],
        hvac_props['heating_deadband_c'],
        hvac_props['cooling_deadband_c'],
        hvac_props['min_runtime_minutes'],
        hvac_props['min_offtime_minutes'],
        hvac_props['ramp_up_minutes'],
        dt_sec
    )

    # --- 4. Define Boundary Conditions from Config ---
    schedules = config['schedules']
    weather_conf = config['weather']
    
    heating_setpoint_profile = np.zeros(num_steps)
    cooling_setpoint_profile = np.zeros(num_steps)
    internal_gains_profile = np.zeros(num_steps)
    window_opening_profile = np.zeros(num_steps)
    weather_data = []

    occ_start, occ_end = schedules['occupied_hours']
    win_start, win_end = schedules.get('window_opening_hours', [0, 0])

    for i, t_hr in enumerate(time_hours):
        hour_of_day = t_hr % 24
        
        if occ_start <= hour_of_day < occ_end:
            heating_setpoint_profile[i] = schedules['occupied_heating_setpoint_c']
            cooling_setpoint_profile[i] = schedules['occupied_cooling_setpoint_c']
            internal_gains_profile[i] = schedules['occupied_internal_gains_w']
        else:
            heating_setpoint_profile[i] = schedules['unoccupied_heating_setpoint_c']
            cooling_setpoint_profile[i] = schedules['unoccupied_cooling_setpoint_c']
            internal_gains_profile[i] = 0
            
        if win_start <= hour_of_day < win_end:
            window_opening_profile[i] = 1.0 # Window is fully open
        else:
            window_opening_profile[i] = 0.0 # Window is closed

        # Simple sinusoidal weather model
        solar_rad = max(0, weather_conf['solar_max_irradiance_w_m2'] * np.cos(2 * np.pi * (t_hr - 12) / 24))
        weather_data.append({
            'air_temp_c': weather_conf['temp_base_c'] + weather_conf['temp_amplitude_c'] * np.cos(2 * np.pi * (t_hr - weather_conf['temp_phase_shift_hr']) / 24),
            'wind_speed_local_ms': 3.0,
            'wind_speed_10m_ms': 2.5,
            'wind_direction_deg': 180,
            'solar_irradiance_w_m2': solar_rad
        })
        
    # --- 5. Configure Convection Models from Config ---
    conv_models = config['convection_models']
    exterior_convection_model = AdaptiveConvectionAlgorithm(conv_models['exterior_hf'], conv_models['exterior_hn'])
    interior_convection_model = InternalAdaptiveConvection(conv_models['interior'])

    # --- 6. Run Simulation ---
    print(f"Starting simulation from config file: {config_path}")
    results = zone.run_simulation(
        heating_setpoint_profile, cooling_setpoint_profile, weather_data, 
        internal_gains_profile, interior_convection_model, 
        exterior_convection_model, hvac_system, duration_hours,
        window_opening_profile
    )
    print("Simulation complete.")

    # --- 7. Process and Display Results ---
    total_heating_kwh = np.sum(results['hvac_energy'][results['hvac_energy'] > 0]) * dt_sec / (3600 * 1000)
    print(f"\nTotal Heating Energy Consumed: {total_heating_kwh:.2f} kWh")
    
    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 15), sharex=True)
    fig.suptitle('Zone Thermal Simulation', fontsize=16)

    # --- Plot 1: Temperatures ---
    outside_air_temps = [w['air_temp_c'] for w in weather_data]
    ax1.plot(time_hours, outside_air_temps, 'c--', label='Outside Air Temp', alpha=0.8)
    ax1.plot(time_hours, heating_setpoint_profile, 'k:', label='Heating Setpoint', lw=2)
    ax1.plot(time_hours, cooling_setpoint_profile, 'b:', label='Cooling Setpoint', lw=2, alpha=0.7)
    ax1.plot(time_hours, results['zone_air_temps'], 'r-', label='Actual Zone Air Temp', lw=2)
    ax1.set_title('Zone Temperatures')
    ax1.set_ylabel('Temperature (°C)')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right')

    # --- Plot 2: Net Load and HVAC Response ---
    fabric_loss = results['fabric_loss']
    air_exchange_loss = results['air_exchange_loss']
    solar_gain = results['solar_gains'][:num_steps]
    internal_gains = results['internal_gains'][:num_steps]
    hvac_energy = results['hvac_energy']

    # Positive = loss/load, Negative = gain
    total_passive_loss = fabric_loss + air_exchange_loss
    total_gains = solar_gain + internal_gains
    net_load = np.array(total_passive_loss) - np.array(total_gains) # The load the HVAC must meet

    ax2.plot(time_hours, hvac_energy, 'r-', label='HVAC Energy Supplied', lw=2.5)
    ax2.plot(time_hours, net_load, 'k--', label='Net Zone Load (Loss-Gains)', alpha=0.8)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1.0)
    ax2.set_title('Net Load and HVAC Response')
    ax2.set_ylabel('Power (W)')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right')

    # --- Plot 3: Constituent Loads/Gains ---
    ax3.plot(time_hours, fabric_loss, 'b--', label='Fabric Loss (+)', alpha=0.8)
    ax3.plot(time_hours, air_exchange_loss, 'c--', label='Air Exchange Loss (+)', alpha=0.8)
    ax3.plot(time_hours, -solar_gain, 'y-.', label='Solar Gain (-)', lw=2)
    ax3.plot(time_hours, -internal_gains, 'g:', label='Internal Gains (-)', lw=2.5)

    ax3.axhline(0, color='black', linestyle='-', linewidth=1.0)
    ax3.set_title('Constituent Energy Flows (Positive = Loss, Negative = Gain)')
    ax3.set_ylabel('Power (W)')
    ax3.set_xlabel('Time (hours)')
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='upper right')

    plt.xlim(0, duration_hours)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == '__main__':
    run_simulation_from_config('simulation_config.json')



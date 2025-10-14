"""
Main script to set up and run a single-zone thermal simulation from a 
JSON configuration file.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

from materials import create_construction_from_config
from zone import Zone
from exterior_heat_transfer import AdaptiveConvectionAlgorithm
from interior_heat_transfer import InternalAdaptiveConvection
from hvac_def import VerySimpleHVAC

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
    time_hours = np.linspace(0, duration_hours, num_steps)

    # --- 2. Define Zone, Construction, and Windows from Config ---
    construction = create_construction_from_config(config)
    zone_props = config['zone_properties']
    
    zone = Zone(length=zone_props['length'], width=zone_props['width'], height=zone_props['height'],
                construction=construction, dt_seconds=dt_sec,
                exterior_surfaces=zone_props['exterior_surfaces'],
                windows_data=config['windows'])

    # --- 3. Define HVAC System from Config ---
    hvac_props = config['hvac_system']
    hvac_system = VerySimpleHVAC(hvac_props['heating_capacity_w'], hvac_props['cooling_capacity_w'])

    # --- 4. Define Boundary Conditions from Config ---
    schedules = config['schedules']
    weather_conf = config['weather']
    
    T_setpoint_profile = np.zeros(num_steps)
    internal_gains_profile = np.zeros(num_steps)
    weather_data = []

    for i, t_hr in enumerate(time_hours):
        hour_of_day = t_hr % 24
        start_occ, end_occ = schedules['occupied_hours']
        
        if start_occ <= hour_of_day < end_occ:
            T_setpoint_profile[i] = schedules['occupied_setpoint_c']
            internal_gains_profile[i] = schedules['occupied_internal_gains_w']
        else:
            T_setpoint_profile[i] = schedules['unoccupied_setpoint_c']
            internal_gains_profile[i] = 0

        solar_rad = max(0, weather_conf['solar_max_irradiance_w_m2'] * np.cos(2 * np.pi * (t_hr - 12) / 24))
        weather_data.append({
            'air_temp_c': weather_conf['temp_base_c'] + weather_conf['temp_amplitude_c'] * np.cos(2 * np.pi * (t_hr - weather_conf['temp_phase_shift_hr']) / 24),
            'wind_speed_local_ms': 3.0, # Simplified for now, can be expanded in config
            'wind_speed_10m_ms': 2.5,   # Simplified for now
            'wind_direction_deg': 180,  # Simplified for now
            'solar_irradiance_w_m2': solar_rad
        })
        
    # --- 5. Configure Convection Models from Config ---
    conv_models = config['convection_models']
    exterior_convection_model = AdaptiveConvectionAlgorithm(conv_models['exterior_hf'], conv_models['exterior_hn'])
    interior_convection_model = InternalAdaptiveConvection(conv_models['interior'])

    # --- 6. Run Simulation ---
    print(f"Starting simulation from config file: {config_path}")
    results = zone.run_simulation(
        T_setpoint_profile, weather_data, internal_gains_profile,
        interior_convection_model, exterior_convection_model, 
        hvac_system, duration_hours
    )
    print("Simulation complete.")

    # --- 7. Process and Display Results ---
    total_heating_kwh = np.sum(results['hvac_energy'][results['hvac_energy'] > 0]) * dt_sec / (3600 * 1000)
    print(f"\nTotal Heating Energy Consumed: {total_heating_kwh:.2f} kWh")
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    fig.suptitle('Zone Thermal Simulation from Config', fontsize=16)

    outside_air_temps = [w['air_temp_c'] for w in weather_data]
    ax1.plot(time_hours, outside_air_temps, 'c--', label='Outside Air Temp', alpha=0.8)
    ax1.plot(time_hours, T_setpoint_profile, 'k:', label='Heating Setpoint (Day/Night)', lw=2)
    ax1.plot(time_hours, results['zone_air_temps'], 'r-', label='Actual Zone Air Temp', lw=2)
    ax1.set_title('Zone Temperatures')
    ax1.set_ylabel('Temperature (°C)')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()

    fabric_loss = results['fabric_load']
    window_conductive_loss = results['window_net_load']
    solar_gain = results['solar_gains']
    internal_gains = results['internal_gains']
    hvac_heating = results['hvac_energy']

    ax2.plot(time_hours, hvac_heating, 'r-', label='HVAC Heating Supplied', lw=2.5)
    ax2.plot(time_hours, fabric_loss, 'b--', label='Fabric Loss (+)', alpha=0.8)
    ax2.plot(time_hours, window_conductive_loss, 'm--', label='Window Conduction Loss (+)', alpha=0.8)
    ax2.plot(time_hours, -solar_gain, 'y-.', label='Solar Gain (-)', lw=2)
    ax2.plot(time_hours, -internal_gains, 'g:', label='Internal Gains (-)', lw=2.5)

    ax2.axhline(0, color='black', linestyle='-', linewidth=1.0)
    ax2.set_title('Zone Energy Balance (Positive = Loss/Load, Negative = Gain)')
    ax2.set_ylabel('Power (W)')
    ax2.set_xlabel('Time (hours)')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()

    plt.xlim(0, duration_hours)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == '__main__':
    run_simulation_from_config('simulation_config.json')

"""
Contains functions for plotting simulation results.
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_simulation_results(time_hours, weather_data, heating_setpoint_profile, 
                            cooling_setpoint_profile, results, num_steps, 
                            duration_hours):
    """
    Generates and displays plots for the thermal simulation results.

    Args:
        time_hours (np.array): Array of time values in hours.
        weather_data (list): List of weather data dictionaries.
        heating_setpoint_profile (np.array): Array of heating setpoints.
        cooling_setpoint_profile (np.array): Array of cooling setpoints.
        results (dict): Dictionary containing simulation output arrays.
        num_steps (int): Number of simulation steps.
        duration_hours (int): Total simulation duration in hours.
    """
    
    # --- Plotting ---
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
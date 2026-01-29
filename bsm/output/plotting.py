"""
Contains functions for plotting simulation results.
Enhanced with automatic figure saving and agent activity visualization.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def plot_simulation_results(
    time_hours,
    weather_data,
    heating_setpoint_profile,
    cooling_setpoint_profile,
    results,
    num_steps,
    duration_hours,
    save_dir: Optional[str] = None,
    use_llm_agents: bool = False,
):
    """
    Generates, saves, and displays plots for the thermal simulation results.

    Args:
        time_hours (np.array): Array of time values in hours.
        weather_data (list): List of weather data dictionaries.
        heating_setpoint_profile (np.array): Array of heating setpoints (dynamic, includes agent interventions).
        cooling_setpoint_profile (np.array): Array of cooling setpoints (dynamic, includes agent interventions).
        results (dict): Dictionary containing simulation output arrays.
        num_steps (int): Number of simulation steps.
        duration_hours (int): Total simulation duration in hours.
        save_dir (str, optional): Directory to save figures. If None, figures are not saved.
        use_llm_agents (bool): Whether LLM agents are enabled (enables agent activity plot).
    """
    saved_figures = []

    # --- Plot 1: Temperature Profile ---
    fig1, ax1 = plt.subplots(figsize=(14, 5))
    fig1.suptitle('Zone Temperature Profile', fontsize=14)

    outside_air_temps = [w['air_temp_c'] for w in weather_data]
    ax1.plot(time_hours, outside_air_temps, 'c--', label='Outside Air Temp', alpha=0.8, linewidth=1)
    ax1.plot(time_hours, heating_setpoint_profile, 'r:', label='Heating Setpoint', linewidth=2)
    ax1.plot(time_hours, cooling_setpoint_profile, 'b:', label='Cooling Setpoint', linewidth=2, alpha=0.7)
    ax1.plot(time_hours, results['zone_air_temps'], 'k-', label='Zone Air Temp', linewidth=2)

    # Add comfort band visualization
    ax1.fill_between(time_hours, heating_setpoint_profile, cooling_setpoint_profile,
                     alpha=0.1, color='green', label='Comfort Band')

    ax1.set_ylabel('Temperature (°C)')
    ax1.set_xlabel('Time (hours)')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, duration_hours)

    plt.tight_layout()
    if save_dir:
        fig1_path = os.path.join(save_dir, 'temperatures.png')
        fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
        saved_figures.append(fig1_path)
        print(f"Saved figure: {fig1_path}")
    plt.close(fig1)

    # --- Plot 2: Energy Flows ---
    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig2.suptitle('Energy Flows', fontsize=14)

    fabric_loss = results['fabric_loss']
    air_exchange_loss = results['air_exchange_loss']
    solar_gain = results['solar_gains'][:num_steps]
    hvac_energy = results['hvac_energy']

    # In LLM mode, internal_gains from schedule is 0W because agents control equipment dynamically.
    # The actual internal gains come from: equipment + lighting + occupant metabolic heat.
    if use_llm_agents:
        equipment_power = results.get('equipment_power', np.zeros(num_steps))
        lighting_power = results.get('lighting_power', np.zeros(num_steps))
        occupant_count = results.get('occupant_count', np.zeros(num_steps))
        # Each occupant contributes ~120W of metabolic heat
        occupant_heat = np.array(occupant_count) * 120.0
        internal_gains = np.array(equipment_power) + np.array(lighting_power) + occupant_heat
    else:
        internal_gains = results['internal_gains'][:num_steps]

    # Net load calculation
    total_passive_loss = fabric_loss + air_exchange_loss
    total_gains = solar_gain + internal_gains
    net_load = np.array(total_passive_loss) - np.array(total_gains)

    # HVAC Response
    ax2a.plot(time_hours, hvac_energy, 'r-', label='HVAC Power (+ heating, - cooling)', linewidth=2)
    ax2a.plot(time_hours, net_load, 'k--', label='Net Zone Load', alpha=0.7, linewidth=1)
    ax2a.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax2a.fill_between(time_hours, 0, hvac_energy, where=np.array(hvac_energy) > 0,
                      alpha=0.3, color='red', label='Heating')
    ax2a.fill_between(time_hours, 0, hvac_energy, where=np.array(hvac_energy) < 0,
                      alpha=0.3, color='blue', label='Cooling')
    ax2a.set_ylabel('Power (W)')
    ax2a.set_title('HVAC Response')
    ax2a.grid(True, linestyle=':', alpha=0.6)
    ax2a.legend(loc='upper right')

    # Constituent Loads/Gains
    ax2b.plot(time_hours, fabric_loss, 'b-', label='Fabric Loss', alpha=0.8, linewidth=1.5)
    ax2b.plot(time_hours, air_exchange_loss, 'c-', label='Air Exchange Loss', alpha=0.8, linewidth=1.5)
    ax2b.plot(time_hours, -solar_gain, 'y-', label='Solar Gain (negative)', linewidth=1.5)
    ax2b.plot(time_hours, -internal_gains, 'g-', label='Internal Gains (negative)', linewidth=1.5)
    ax2b.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax2b.set_ylabel('Power (W)')
    ax2b.set_xlabel('Time (hours)')
    ax2b.set_title('Constituent Energy Flows (Positive = Loss, Negative = Gain)')
    ax2b.grid(True, linestyle=':', alpha=0.6)
    ax2b.legend(loc='upper right')
    ax2b.set_xlim(0, duration_hours)

    plt.tight_layout()
    if save_dir:
        fig2_path = os.path.join(save_dir, 'energy_flows.png')
        fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
        saved_figures.append(fig2_path)
        print(f"Saved figure: {fig2_path}")
    plt.close(fig2)

    # --- Plot 3: Agent Activity (LLM mode only) ---
    if use_llm_agents:
        fig3, (ax3a, ax3b, ax3c) = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        fig3.suptitle('Agent Activity', fontsize=14)

        # Thermostat offset
        thermostat_offset = results.get('thermostat_offset', np.zeros(num_steps))
        ax3a.plot(time_hours, thermostat_offset, 'r-', linewidth=2)
        ax3a.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax3a.fill_between(time_hours, 0, thermostat_offset, alpha=0.3, color='red')
        ax3a.set_ylabel('Offset (°C)')
        ax3a.set_title('Thermostat Offset (Agent Adjustments)')
        ax3a.grid(True, linestyle=':', alpha=0.6)
        ax3a.set_ylim(-3, 3)

        # Window state
        window_state = results.get('window_state', np.zeros(num_steps))
        ax3b.fill_between(time_hours, 0, window_state, alpha=0.5, color='blue', step='post')
        ax3b.set_ylabel('Window Open Fraction')
        ax3b.set_title('Window State (Agent Votes)')
        ax3b.grid(True, linestyle=':', alpha=0.6)
        ax3b.set_ylim(-0.1, 1.1)

        # Occupancy and equipment/lighting
        occupant_count = results.get('occupant_count', np.zeros(num_steps))
        equipment_power = results.get('equipment_power', np.zeros(num_steps))
        lighting_power = results.get('lighting_power', np.zeros(num_steps))

        ax3c.fill_between(time_hours, 0, occupant_count, alpha=0.3, color='green', label='Occupants', step='post')
        ax3c_twin = ax3c.twinx()
        ax3c_twin.plot(time_hours, equipment_power, 'orange', linewidth=1.5, label='Equipment (W)')
        ax3c_twin.plot(time_hours, lighting_power, 'purple', linewidth=1.5, label='Lighting (W)')
        ax3c.set_ylabel('Occupant Count')
        ax3c_twin.set_ylabel('Power (W)')
        ax3c.set_xlabel('Time (hours)')
        ax3c.set_title('Occupancy and Internal Loads')
        ax3c.grid(True, linestyle=':', alpha=0.6)
        ax3c.legend(loc='upper left')
        ax3c_twin.legend(loc='upper right')
        ax3c.set_xlim(0, duration_hours)

        plt.tight_layout()
        if save_dir:
            fig3_path = os.path.join(save_dir, 'agent_activity.png')
            fig3.savefig(fig3_path, dpi=300, bbox_inches='tight')
            saved_figures.append(fig3_path)
            print(f"Saved figure: {fig3_path}")
        plt.close(fig3)

    # --- Plot 4: Summary Dashboard ---
    fig4, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig4.suptitle('Simulation Summary Dashboard', fontsize=14)

    # Temperature summary (top-left)
    ax_temp = axes[0, 0]
    ax_temp.plot(time_hours, results['zone_air_temps'], 'k-', label='Zone Temp', linewidth=1.5)
    ax_temp.plot(time_hours, heating_setpoint_profile, 'r:', label='Heat SP', linewidth=1)
    ax_temp.plot(time_hours, cooling_setpoint_profile, 'b:', label='Cool SP', linewidth=1)
    ax_temp.fill_between(time_hours, heating_setpoint_profile, cooling_setpoint_profile,
                         alpha=0.1, color='green')
    ax_temp.set_ylabel('Temperature (°C)')
    ax_temp.set_title('Zone Temperature')
    ax_temp.legend(loc='upper right', fontsize=8)
    ax_temp.grid(True, linestyle=':', alpha=0.6)

    # HVAC power (top-right)
    ax_hvac = axes[0, 1]
    ax_hvac.fill_between(time_hours, 0, hvac_energy, where=np.array(hvac_energy) > 0,
                         alpha=0.7, color='red', label='Heating')
    ax_hvac.fill_between(time_hours, 0, hvac_energy, where=np.array(hvac_energy) < 0,
                         alpha=0.7, color='blue', label='Cooling')
    ax_hvac.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax_hvac.set_ylabel('Power (W)')
    ax_hvac.set_title('HVAC Power')
    ax_hvac.legend(loc='upper right', fontsize=8)
    ax_hvac.grid(True, linestyle=':', alpha=0.6)

    # Energy components (bottom-left)
    ax_energy = axes[1, 0]
    ax_energy.stackplot(time_hours,
                        [np.maximum(0, fabric_loss), np.maximum(0, air_exchange_loss)],
                        labels=['Fabric Loss', 'Air Exchange Loss'],
                        alpha=0.7, colors=['coral', 'cyan'])
    ax_energy.plot(time_hours, -solar_gain, 'y-', label='Solar Gain', linewidth=1.5)
    ax_energy.plot(time_hours, -internal_gains, 'g-', label='Internal Gains', linewidth=1.5)
    ax_energy.set_ylabel('Power (W)')
    ax_energy.set_xlabel('Time (hours)')
    ax_energy.set_title('Heat Flows')
    ax_energy.legend(loc='upper right', fontsize=8)
    ax_energy.grid(True, linestyle=':', alpha=0.6)

    # Outside temperature (bottom-right)
    ax_out = axes[1, 1]
    ax_out.plot(time_hours, outside_air_temps, 'c-', linewidth=1.5)
    ax_out.fill_between(time_hours, min(outside_air_temps), outside_air_temps, alpha=0.3, color='cyan')
    ax_out.set_ylabel('Temperature (°C)')
    ax_out.set_xlabel('Time (hours)')
    ax_out.set_title('Outside Air Temperature')
    ax_out.grid(True, linestyle=':', alpha=0.6)

    for ax in axes.flat:
        ax.set_xlim(0, duration_hours)

    plt.tight_layout()
    if save_dir:
        fig4_path = os.path.join(save_dir, 'summary_dashboard.png')
        fig4.savefig(fig4_path, dpi=300, bbox_inches='tight')
        saved_figures.append(fig4_path)
        print(f"Saved figure: {fig4_path}")
    plt.close(fig4)

    if saved_figures:
        print(f"\nAll figures saved to: {save_dir}")

    return saved_figures

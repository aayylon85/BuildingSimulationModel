"""
Defines the fully coupled heat balance solver for the zone.
"""
import numpy as np
from fabric_heat_transfer import CondFDSolver

# Constants for air properties at standard conditions
AIR_DENSITY_KG_M3 = 1.225
AIR_SPECIFIC_HEAT_J_KG_K = 1006

class ZoneHeatBalanceSolver:
    """
    Assembles and solves the fully coupled system of linear equations for all
    fabric nodes and the single zone air node.
    """
    def __init__(self, all_surfaces_props, construction, dt_seconds,
                 zone_volume, zone_sensible_heat_capacity_multiplier):
        """
        Initializes the solver, creates fabric solvers for each surface, and
        determines the total size of the system matrix.
        """
        self.dt = dt_seconds
        self.fabric_solvers = {}
        self.surface_props = all_surfaces_props
        self.total_fabric_nodes = 0

        for name, props in self.surface_props.items():
            solver = CondFDSolver(construction, dt_seconds)
            # Pass the full properties dictionary down to the solver wrapper
            self.fabric_solvers[name] = {'solver': solver, 'props': props}
            self.total_fabric_nodes += len(solver.nodes)
            
        self.air_thermal_mass = (zone_volume * AIR_DENSITY_KG_M3 * AIR_SPECIFIC_HEAT_J_KG_K * zone_sensible_heat_capacity_multiplier)
        self.air_capacitance_term = self.air_thermal_mass / self.dt

    def reduce_surface_area(self, surface_name, area_to_subtract):
        """Reduces a surface's area, e.g., to account for a window."""
        if surface_name in self.surface_props:
            self.surface_props[surface_name]['area'] -= area_to_subtract

    def set_initial_temperatures(self, temp_c):
        """Sets the same initial temperature for all fabric and air nodes."""
        for solver_data in self.fabric_solvers.values():
            solver_data['solver'].set_initial_temperatures(temp_c)

    def solve_step(self, T_air_prev, weather, windows, air_exchange_manager,
                     interior_convection_model, exterior_convection_model,
                     internal_gains_w, solar_gains_w, hvac_power_w,
                     window_open_fraction):
        """
        Assembles and solves the matrix for one timestep.
        """
        num_eq = self.total_fabric_nodes + 1
        A = np.zeros((num_eq, num_eq))
        B = np.zeros(num_eq)
        
        T_old_fabric = []
        node_offset = 0

        # --- Populate fabric-related rows of the matrix ---
        for name, solver_data in self.fabric_solvers.items():
            solver = solver_data['solver']
            props = solver_data['props']
            T_old_fabric.extend([node['T'] for node in solver.nodes])

            h_in = interior_convection_model.calculate_h_c(solver_data, T_air_prev)
            
            # Now correctly checks the 'is_exterior' flag from the passed props
            if props['is_exterior']:
                h_out_guess = 15.0
                current_surf_temp = solver.nodes[-1]['T']
                for _ in range(3):
                    surface_props_for_ext_hc = {**props, 'surface_temp_c': current_surf_temp}
                    h_out = exterior_convection_model.calculate_hc(surface_props_for_ext_hc, weather)
                    current_surf_temp = (current_surf_temp + weather['air_temp_c']) / 2
            else:
                h_out = 0.0

            solver.populate_matrix_equations(
                A, B, node_offset, self.total_fabric_nodes,
                h_in, h_out, weather['air_temp_c']
            )
            node_offset += len(solver.nodes)

        # --- Populate the final row for the Zone Air Heat Balance ---
        air_node_idx = self.total_fabric_nodes
        B[air_node_idx] = self.air_capacitance_term * T_air_prev + internal_gains_w + solar_gains_w + hvac_power_w
        A[air_node_idx, air_node_idx] = self.air_capacitance_term

        node_offset = 0
        for name, solver_data in self.fabric_solvers.items():
            solver = solver_data['solver']
            props = solver_data['props']
            area = props['area']
            h_in = interior_convection_model.calculate_h_c(solver_data, T_air_prev)
            
            A[air_node_idx, air_node_idx] += h_in * area
            A[air_node_idx, node_offset] -= h_in * area
            node_offset += len(solver.nodes)
            
        for window in windows.values():
            u_eff = window.u_value * window.area
            A[air_node_idx, air_node_idx] += u_eff
            B[air_node_idx] += u_eff * weather['air_temp_c']

        if air_exchange_manager:
            hvac_is_heating = hvac_power_w > 0
            air_exchange_coeff = air_exchange_manager.get_mass_flow_rate_coeff_w_k(
                T_air_prev, weather['air_temp_c'], weather['wind_speed_local_ms'],
                window_open_fraction, hvac_is_heating
            )
            A[air_node_idx, air_node_idx] += air_exchange_coeff
            B[air_node_idx] += air_exchange_coeff * weather['air_temp_c']

        T_new = np.linalg.solve(A, B)
        
        T_new_fabric = T_new[:-1]
        node_offset = 0
        for solver_data in self.fabric_solvers.values():
            solver = solver_data['solver']
            num_nodes = len(solver.nodes)
            solver.update_temperatures(T_new_fabric[node_offset : node_offset + num_nodes])
            node_offset += num_nodes

        T_air_new = T_new[-1]
        q_fabric_total = 0
        q_window_total = 0
        
        node_offset = 0
        for name, solver_data in self.fabric_solvers.items():
            props = solver_data['props']
            h_in = interior_convection_model.calculate_h_c(solver_data, T_air_new)
            q_fabric_total += h_in * props['area'] * (T_air_new - T_new[node_offset])
            node_offset += len(solver_data['solver'].nodes)

        for window in windows.values():
            q_window_total += window.u_value * window.area * (T_air_new - weather['air_temp_c'])
            
        air_exchange_load = 0
        if 'air_exchange_coeff' in locals():
             air_exchange_load = air_exchange_coeff * (T_air_new - weather['air_temp_c'])

        return T_new, q_fabric_total, q_window_total, air_exchange_load


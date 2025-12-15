"""
Defines the fully coupled heat balance solver for the zone.
"""
import numpy as np
import types
import warnings
from fabric_heat_transfer import CondFDSolver
from constants import AIR_DENSITY_KG_M3, AIR_SPECIFIC_HEAT_J_KG_K

class ZoneHeatBalanceSolver:
    """
    Assembles and solves the fully coupled system of linear equations for all
    fabric nodes and the single zone air node.
    """
    def __init__(self, all_surfaces_props, constructions, dt_seconds,
                 zone_volume, zone_sensible_heat_capacity_multiplier,
                 max_nodes_per_layer=20, hvac_coupling_mode='auto',
                 hvac_max_power_rate_w_s=50000.0):
        """
        Initializes the solver, creates fabric solvers for each surface, and
        determines the total size of the system matrix.

        Args:
            all_surfaces_props (dict): Dict of properties for each surface.
                Must include a 'construction_name' key.
            constructions (dict): A dictionary mapping 'construction_name'
                strings to CondFD-compatible construction objects.
            dt_seconds (float): Timestep in seconds
            zone_volume (float): Volume of the zone in m³
            zone_sensible_heat_capacity_multiplier (float): Multiplier for zone thermal mass
            max_nodes_per_layer (int): Maximum nodes per construction layer (default: 20)
            hvac_coupling_mode (str): HVAC coupling strategy - 'auto', 'implicit', or 'explicit' (default: 'auto')
            hvac_max_power_rate_w_s (float): Max HVAC power rate of change for explicit coupling (default: 50000.0 W/s)
        """
        self.dt = dt_seconds
        self.max_nodes_per_layer = max_nodes_per_layer
        self.hvac_coupling_mode = hvac_coupling_mode
        self.hvac_max_power_rate_w_s = hvac_max_power_rate_w_s
        self.fabric_solvers = {}
        self.surface_props = all_surfaces_props
        self.total_fabric_nodes = 0

        
        for name, props in self.surface_props.items():
            construction_name = props.get('construction_name')
            if not construction_name:
                raise ValueError(f"Surface '{name}' does not specify a 'construction_name'.")
            
            construction_obj = constructions.get(construction_name)
            if not construction_obj:
                raise ValueError(f"Construction '{construction_name}' for surface "
                                 f"'{name}' not found in constructions dictionary.")

            solver = CondFDSolver(construction_obj, dt_seconds,
                                 max_nodes_per_layer=self.max_nodes_per_layer)


            self.fabric_solvers[name] = {'solver': solver, 'props': props}
            self.total_fabric_nodes += len(solver.nodes)
            
        self.air_thermal_mass = (zone_volume * AIR_DENSITY_KG_M3 * AIR_SPECIFIC_HEAT_J_KG_K * zone_sensible_heat_capacity_multiplier)
        self.air_capacitance_term = self.air_thermal_mass / self.dt

        # Track previous HVAC power for rate limiting (explicit coupling only)
        self.q_hvac_prev = 0.0

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
                     internal_gains_w, solar_gains_w_dict, hvac_power_w,
                     window_open_fraction,
                     max_iterations=50, tolerance=0.01, temp_min_c=-50.0, temp_max_c=80.0,
                     relaxation_factor=0.7,
                     hvac_system=None, heating_setpoint=None, cooling_setpoint=None):
        """
        Assembles and solves the matrix for one timestep using an
        iterative approach for non-linear coefficients with under-relaxation.

        Args:
            max_iterations (int): Maximum number of iterations for convergence (default: 50).
            tolerance (float): Convergence tolerance in degrees C (default: 0.01).
            temp_min_c (float): Minimum physically reasonable temperature in C (default: -50).
            temp_max_c (float): Maximum physically reasonable temperature in C (default: 80).
            relaxation_factor (float): Under-relaxation factor for damping (0.5-0.9, default: 0.7).
                Higher values converge faster but may oscillate. Lower values are more stable.
            hvac_system: Optional HVAC system object for implicit coupling
            heating_setpoint: Optional heating setpoint for implicit HVAC
            cooling_setpoint: Optional cooling setpoint for implicit HVAC

        Returns:
            tuple: (T_new, q_fabric_total, q_window_total, air_exchange_load, q_hvac)

        Raises:
            RuntimeError: If the matrix is singular or temperatures are out of bounds.
        """
        
        
        num_eq = self.total_fabric_nodes + 1
        
        # Create initial guess for T_new using previous timestep's values
        T_old_fabric = []
        for solver_data in self.fabric_solvers.values():
            T_old_fabric.extend([node['T'] for node in solver_data['solver'].nodes])
        T_new = np.array(T_old_fabric + [T_air_prev])

        T_iter_guess = np.zeros(num_eq)
        converged = False
        iteration_count = 0

        for iteration_count in range(max_iterations):
            # Store the result of the previous iteration to check for convergence
            T_iter_guess = np.copy(T_new)
            T_air_guess = T_iter_guess[-1]

            A = np.zeros((num_eq, num_eq))
            B = np.zeros(num_eq)
            
            h_in_values = {} # Store h_in for use in the air balance row
            node_offset = 0

            # --- Populate fabric-related rows of the matrix ---
            for name, solver_data in self.fabric_solvers.items():
                solver = solver_data['solver']
                props = solver_data['props']
                
                # Get surface temperatures from the current iteration's guess
                num_nodes = len(solver.nodes)
                T_surf_in_guess = T_iter_guess[node_offset]
                
                # --- Calculate h_in using T_iter_guess ---
                
                mock_solver = types.SimpleNamespace(nodes=[{'T': T_surf_in_guess}])
                temp_surface_data = {
                'solver': mock_solver,
                'props': props
                }
                h_in = interior_convection_model.calculate_h_c(temp_surface_data, T_air_guess)
                h_in_values[name] = h_in 

            
                h_out = 0.0 # Default for non-exterior surfaces
                if props['is_exterior']:
                    T_surf_out_guess = T_iter_guess[node_offset + num_nodes - 1]
                    surface_props_for_ext_hc = {**props, 'surface_temp_c': T_surf_out_guess}
                    h_out = exterior_convection_model.calculate_hc(surface_props_for_ext_hc, weather)

                # --- Get solar gains for this surface ---
                surface_solar_gain_w = solar_gains_w_dict.get(name, 0.0)

                solver.populate_matrix_equations(
                    A, B, node_offset, self.total_fabric_nodes,
                    h_in, h_out, weather['air_temp_c'],
                    props['area'],           
                    surface_solar_gain_w   
                )
                node_offset += num_nodes

            # --- Populate the final row for the Zone Air Heat Balance ---
            air_node_idx = self.total_fabric_nodes

            # Determine if we should use implicit HVAC coupling (auto-detection)
            use_implicit_hvac = False
            if hvac_system is not None and heating_setpoint is not None and cooling_setpoint is not None:
                # Check config override
                if self.hvac_coupling_mode == 'explicit':
                    use_implicit_hvac = False
                elif self.hvac_coupling_mode == 'implicit':
                    use_implicit_hvac = True
                else:  # 'auto' mode
                    # Auto mode: use implicit for high-gain proportional controllers only
                    if hasattr(hvac_system, 'proportional_gain_w_k'):
                        gain = hvac_system.proportional_gain_w_k
                        # Use implicit if gain > 5000 W/K
                        use_implicit_hvac = (gain > 5000.0)
                    # For complex HVAC (StatefulHVAC, PID), use_implicit_hvac remains False

            if use_implicit_hvac:
                # Implicit HVAC: include proportional gain in matrix
                # This path is only taken for high-gain proportional controllers
                if hasattr(hvac_system, 'proportional_gain_w_k'):
                    K_hvac = hvac_system.proportional_gain_w_k
                    # Limit gain to prevent over-stiffness (max 10,000 W/K)
                    K_hvac = min(K_hvac, 10000.0)

                    # Determine which setpoint to use based on current temperature estimate
                    T_air_guess = T_iter_guess[-1]
                    if T_air_guess < heating_setpoint:
                        # Heating mode: add gain to diagonal and setpoint to RHS
                        A[air_node_idx, air_node_idx] = self.air_capacitance_term + K_hvac
                        B[air_node_idx] = self.air_capacitance_term * T_air_prev + internal_gains_w + K_hvac * heating_setpoint
                    elif T_air_guess > cooling_setpoint:
                        # Cooling mode: add gain to diagonal and setpoint to RHS
                        A[air_node_idx, air_node_idx] = self.air_capacitance_term + K_hvac
                        B[air_node_idx] = self.air_capacitance_term * T_air_prev + internal_gains_w + K_hvac * cooling_setpoint
                    else:
                        # Deadband: no HVAC
                        A[air_node_idx, air_node_idx] = self.air_capacitance_term
                        B[air_node_idx] = self.air_capacitance_term * T_air_prev + internal_gains_w
                else:
                    # Explicit HVAC (fallback) with rate limiting
                    max_power_change = self.hvac_max_power_rate_w_s * self.dt
                    power_change = hvac_power_w - self.q_hvac_prev

                    if abs(power_change) > max_power_change:
                        hvac_power_w = self.q_hvac_prev + np.sign(power_change) * max_power_change

                    self.q_hvac_prev = hvac_power_w

                    B[air_node_idx] = self.air_capacitance_term * T_air_prev + internal_gains_w + hvac_power_w
                    A[air_node_idx, air_node_idx] = self.air_capacitance_term
            else:
                # Explicit HVAC (original approach) with rate limiting
                # Limit HVAC power rate of change for stability
                max_power_change = self.hvac_max_power_rate_w_s * self.dt
                power_change = hvac_power_w - self.q_hvac_prev

                if abs(power_change) > max_power_change:
                    hvac_power_w = self.q_hvac_prev + np.sign(power_change) * max_power_change

                # Update tracked power for next timestep
                self.q_hvac_prev = hvac_power_w

                B[air_node_idx] = self.air_capacitance_term * T_air_prev + internal_gains_w + hvac_power_w
                A[air_node_idx, air_node_idx] = self.air_capacitance_term

            node_offset = 0
            for name, solver_data in self.fabric_solvers.items():
                props = solver_data['props']
                area = props['area']
                h_in = h_in_values[name] 
                
                A[air_node_idx, air_node_idx] += h_in * area
                A[air_node_idx, node_offset] -= h_in * area
                node_offset += len(solver_data['solver'].nodes)
                
            for window in windows.values():
                u_eff = window.u_value * window.area
                A[air_node_idx, air_node_idx] += u_eff
                B[air_node_idx] += u_eff * weather['air_temp_c']

            # --- Calculate air exchange using T_air_guess ---
            air_exchange_coeff = 0.0 
            if air_exchange_manager:
                
                air_exchange_coeff = air_exchange_manager.get_mass_flow_rate_coeff_w_k(
                    T_air_guess, weather['air_temp_c'],
                    weather['wind_speed_local_ms'],
                    window_open_fraction
                )
                A[air_node_idx, air_node_idx] += air_exchange_coeff
                B[air_node_idx] += air_exchange_coeff * weather['air_temp_c']

            # --- Solve the system for this iteration ---
            try:
                T_computed = np.linalg.solve(A, B)
            except np.linalg.LinAlgError as e:
                # Handle singular matrix with informative error
                raise RuntimeError(
                    f"Matrix solver failed at iteration {iteration_count + 1}: {str(e)}. "
                    "This may indicate an ill-conditioned system. "
                    "Check construction definitions, convection coefficients, or timestep size."
                )

            # --- Apply under-relaxation ---
            T_new = relaxation_factor * T_computed + (1.0 - relaxation_factor) * T_iter_guess

            # --- Check for convergence ---
            max_change = np.max(np.abs(T_new - T_iter_guess))
            if max_change < tolerance:
                converged = True
                break  # Converged

            # Log progress every 10 iterations if struggling to converge
            if (iteration_count + 1) % 10 == 0:
                warnings.warn(
                    f"Convergence progress at iteration {iteration_count + 1}: "
                    f"max temperature change = {max_change:.4f}°C (tolerance: {tolerance}°C)",
                    RuntimeWarning
                )

        # --- END of iterative loop ---

        # --- Warn if convergence not achieved ---
        if not converged:
            warnings.warn(
                f"Heat balance solver did not converge after {max_iterations} iterations. "
                f"Maximum temperature change: {np.max(np.abs(T_new - T_iter_guess)):.4f}°C. "
                f"Consider increasing max_iterations or reducing timestep.",
                RuntimeWarning
            )

        # --- Check temperature bounds ---
        if np.any(T_new < temp_min_c) or np.any(T_new > temp_max_c):
            out_of_bounds = T_new[(T_new < temp_min_c) | (T_new > temp_max_c)]
            raise RuntimeError(
                f"Temperature solution out of physical bounds: "
                f"Found temperatures {out_of_bounds}°C outside range [{temp_min_c}, {temp_max_c}]°C. "
                f"This indicates an unstable solution. Check timestep size, boundary conditions, "
                f"or material properties."
            )

        # --- Update state with final converged solution ---
        T_new_fabric = T_new[:-1]
        node_offset = 0
        for solver_data in self.fabric_solvers.values():
            solver = solver_data['solver']
            num_nodes = len(solver.nodes)
            solver.update_temperatures(T_new_fabric[node_offset : node_offset + num_nodes])
            node_offset += num_nodes

        # --- Calculate and return detailed heat flows for plotting ---
        T_air_new = T_new[-1]
        q_fabric_total = 0
        q_window_total = 0
        
        node_offset = 0
        for name, solver_data in self.fabric_solvers.items():
            props = solver_data['props']
            T_surf_in_new = T_new[node_offset] # Final converged surface temp
            
            # Recalculate final h_in for accurate reporting
            mock_solver_final = types.SimpleNamespace(nodes=[{'T': T_surf_in_new}])
            temp_surface_data = {
            'solver': mock_solver_final, # <-- This is now an object
            'props': props
            }
            h_in_final = interior_convection_model.calculate_h_c(temp_surface_data, T_air_new)
            
            # q_fabric = h * A * (T_air - T_surface)
            q_fabric_total += h_in_final * props['area'] * (T_air_new - T_surf_in_new)
            node_offset += len(solver_data['solver'].nodes)

        for window in windows.values():
            # q_window = U * A * (T_air - T_outside)
            q_window_total += window.u_value * window.area * (T_air_new - weather['air_temp_c'])
            
        # q_air_exchange = (m_dot * C_p) * (T_air - T_outside)
        air_exchange_load = air_exchange_coeff * (T_air_new - weather['air_temp_c'])

        # Calculate actual HVAC power delivered
        if use_implicit_hvac and hvac_system and hasattr(hvac_system, 'proportional_gain_w_k'):
            # Calculate actual HVAC power based on final temperature
            q_hvac_actual = hvac_system.calculate_hvac_power(T_air_new, heating_setpoint, cooling_setpoint)
        else:
            # Use the explicit hvac_power_w that was passed in
            q_hvac_actual = hvac_power_w

        return T_new, q_fabric_total, q_window_total, air_exchange_load, q_hvac_actual
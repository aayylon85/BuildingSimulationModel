"""
Defines the fully coupled heat balance solver for the zone.
"""
import numpy as np
import types
import warnings
from bsm.core.fabric_heat_transfer import CondFDSolver
from bsm.core.constants import AIR_DENSITY_KG_M3, AIR_SPECIFIC_HEAT_J_KG_K
from bsm.heat_transfer.exterior_longwave import ExternalLongwaveRadiation
from bsm.heat_transfer.interior_longwave import InteriorLongwaveRadiation
from bsm.solar.irradiance import SolarIrradianceCalculator

class ZoneHeatBalanceSolver:
    """
    Assembles and solves the fully coupled system of linear equations for all
    fabric nodes and the single zone air node.
    """
    def __init__(self, all_surfaces_props, constructions, dt_seconds,
                 zone_volume, zone_sensible_heat_capacity_multiplier,
                 max_nodes_per_layer=20, hvac_coupling_mode='auto',
                 hvac_max_power_rate_w_s=50000.0,
                 enable_opaque_solar_absorption=False,
                 interior_radiation_method='area_weighted',
                 zone_geometry=None):
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
            enable_opaque_solar_absorption (bool): Enable solar absorption on exterior opaque surfaces (default: False)
            interior_radiation_method (str): Interior radiation method - 'area_weighted', 'carroll', or 'view_factor'
                - 'area_weighted': Simple area-weighted MRT (fastest, default)
                - 'carroll': Carroll MRT preserving T^4 nonlinearity (good balance)
                - 'view_factor': Full view factor matrix (most accurate, requires zone_geometry)
            zone_geometry (dict): Zone dimensions for view factor method.
                Format: {'length': m, 'width': m, 'height': m}
                Required when interior_radiation_method='view_factor'
        """
        self.dt = dt_seconds
        self.max_nodes_per_layer = max_nodes_per_layer
        self.hvac_coupling_mode = hvac_coupling_mode
        self.hvac_max_power_rate_w_s = hvac_max_power_rate_w_s
        self.enable_opaque_solar_absorption = enable_opaque_solar_absorption
        self.interior_radiation_method = interior_radiation_method
        self.zone_geometry = zone_geometry
        self.fabric_solvers = {}
        self.surface_props = all_surfaces_props
        self.total_fabric_nodes = 0

        # Validate interior radiation method
        valid_methods = ['area_weighted', 'carroll', 'view_factor']
        if interior_radiation_method not in valid_methods:
            raise ValueError(f"interior_radiation_method must be one of {valid_methods}")
        if interior_radiation_method == 'view_factor' and zone_geometry is None:
            raise ValueError("zone_geometry is required when using 'view_factor' method")

        # Create solar irradiance calculator if opaque solar absorption is enabled
        if self.enable_opaque_solar_absorption:
            self.solar_calc = SolarIrradianceCalculator(
                diffuse_model='perez',
                ground_reflectance=0.2
            )
        else:
            self.solar_calc = None

        
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

            # Extract optical properties from exterior surface material
            # For exterior surfaces, use the properties from the first layer (exterior)
            if construction_obj and len(construction_obj) > 0:
                exterior_material = construction_obj[0]  # First layer is exterior surface
                props['solar_absorptance'] = exterior_material.solar_absorptance
                props['thermal_emissivity'] = exterior_material.thermal_emissivity

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
            ValueError: If required weather data keys are missing.
        """
        # Validate required weather data keys
        required_weather_keys = ['air_temp_c', 'wind_speed_local_ms']
        missing_keys = [k for k in required_weather_keys if k not in weather]
        if missing_keys:
            raise ValueError(f"Missing required weather data keys: {missing_keys}")

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

            # --- Pre-calculate interior radiation (MRT) for all surfaces ---
            # Collect inside surface temperatures and areas from current iteration's guess
            interior_surface_temps = {}
            interior_surface_areas = {}
            temp_node_offset = 0
            for name, solver_data in self.fabric_solvers.items():
                num_nodes = len(solver_data['solver'].nodes)
                interior_surface_temps[name] = T_iter_guess[temp_node_offset]  # Inside surface temp
                interior_surface_areas[name] = solver_data['props']['area']
                temp_node_offset += num_nodes

            # Create interior radiation model (emissivity = 0.9 per ASHRAE 140)
            # Method selected via constructor: 'area_weighted', 'carroll', or 'view_factor'
            interior_rad_model = InteriorLongwaveRadiation(
                emissivity=0.9,
                method=self.interior_radiation_method,
                zone_geometry=self.zone_geometry
            )

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
                # Separated radiation coefficients (EnergyPlus style)
                h_r_sky = 0.0
                h_r_air = 0.0
                h_r_ground = 0.0
                T_sky_c = weather['air_temp_c'] - 11.0  # Default sky depression
                T_ground_c = weather['air_temp_c']  # Default ground at air temp

                if props['is_exterior']:
                    T_surf_out_guess = T_iter_guess[node_offset + num_nodes - 1]
                    surface_props_for_ext_hc = {**props, 'surface_temp_c': T_surf_out_guess}
                    h_out = exterior_convection_model.calculate_hc(surface_props_for_ext_hc, weather)

                    # Get sky temperature from weather with fallback
                    T_air_K = weather['air_temp_c'] + 273.15
                    if 'sky_temp_k' in weather:
                        T_sky_K = weather['sky_temp_k']
                    else:
                        # Fallback: simple clear-sky depression
                        T_sky_K = T_air_K - 11.0

                    # Ground temperature from Kusuda-Achenbach model
                    # Uses seasonal variation based on annual air temperature statistics
                    # from EPW data. This is physically more accurate than using hourly
                    # air temperature, as ground has significant thermal mass.
                    if 'ground_temp_c' in weather:
                        T_ground_K = weather['ground_temp_c'] + 273.15
                    else:
                        # Fallback to EnergyPlus default (18°C)
                        T_ground_K = 273.15 + 18.0

                    # Convert to Celsius for passing to fabric solver
                    T_sky_c = T_sky_K - 273.15
                    T_ground_c = T_ground_K - 273.15

                    # Calculate separated radiation coefficients using EnergyPlus methodology
                    # This preserves T^4 nonlinearity better than single averaged temperature
                    lwr_model = ExternalLongwaveRadiation(
                        surface_emissivity=props.get('thermal_emissivity', 0.9),
                        surface_tilt_angle_deg=props.get('tilt', 90.0)
                    )

                    T_surf_K = T_surf_out_guess + 273.15

                    # Use new separated coefficients method (EnergyPlus style)
                    h_r_sky, h_r_air, h_r_ground = lwr_model.calculate_separated_coefficients(
                        surface_temperature_K=T_surf_K,
                        air_temperature_K=T_air_K,
                        sky_temperature_K=T_sky_K,
                        ground_temperature_K=T_ground_K
                    )

                # --- Exterior Surface Solar Absorption ---
                # Calculate surface-specific solar irradiance for exterior opaque surfaces
                exterior_solar_absorbed_w_m2 = 0.0

                if (self.enable_opaque_solar_absorption and
                    props['is_exterior'] and
                    self.solar_calc is not None and
                    weather.get('is_sunlit', False)):

                    # Get surface orientation
                    surface_tilt = props.get('tilt', 90.0)  # Default vertical
                    surface_azimuth = props.get('azimuth', 180.0)  # Default south

                    # Calculate surface-specific irradiance using Perez model
                    irrad_result = self.solar_calc.calculate_surface_irradiance(
                        weather_data=weather,
                        surface_tilt_deg=surface_tilt,
                        surface_azimuth_deg=surface_azimuth
                    )

                    # Apply solar absorptance to total incident irradiance
                    total_irradiance = irrad_result.get('total', 0.0)
                    solar_absorptance = props.get('solar_absorptance', 0.6)  # Default 0.6 per ASHRAE 140
                    exterior_solar_absorbed_w_m2 = total_irradiance * solar_absorptance

                # --- Get solar gains for this surface ---
                surface_solar_gain_w = solar_gains_w_dict.get(name, 0.0)

                # --- Calculate interior radiation (MRT) for this surface ---
                # This provides surface-to-surface radiation exchange via MRT simplification
                T_MRT_interior, h_r_interior = interior_rad_model.calculate_mrt_and_coefficient(
                    interior_surface_temps, interior_surface_areas, name
                )

                solver.populate_matrix_equations(
                    A, B, node_offset, self.total_fabric_nodes,
                    h_in, h_out, weather['air_temp_c'],
                    props['area'],
                    surface_solar_gain_w,
                    exterior_solar_absorbed_w_m2=exterior_solar_absorbed_w_m2,
                    # EnergyPlus-style separated radiation coefficients (exterior)
                    h_r_sky=h_r_sky,
                    h_r_air=h_r_air,
                    h_r_ground=h_r_ground,
                    T_sky_c=T_sky_c,
                    T_ground_c=T_ground_c,
                    # Interior radiation to MRT (surface-to-surface exchange)
                    h_r_interior=h_r_interior,
                    T_MRT_interior=T_MRT_interior
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
                # Explicit HVAC (original approach) with rate limiting
                # Limit HVAC power rate of change for stability
                # Note: This correctly handles heating/cooling mode switching because
                # np.sign(power_change) returns the appropriate direction (+1 for heating
                # ramp-up, -1 for cooling), limiting the rate of change in either direction.
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
                
            for win_dict in windows.values():
                window = win_dict['window']  # Extract window object from dict
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

        for win_dict in windows.values():
            window = win_dict['window']  # Extract window object from dict
            # q_window = U * A * (T_air - T_outside)
            q_window_total += window.u_value * window.area * (T_air_new - weather['air_temp_c'])
            
        # q_air_exchange = (m_dot * C_p) * (T_air - T_outside)
        air_exchange_load = air_exchange_coeff * (T_air_new - weather['air_temp_c'])

        # Calculate actual HVAC power delivered from energy balance
        # Energy balance: C_air * dT/dt = -q_fabric - q_window - q_air_exchange + q_internal + q_hvac
        # Solving for q_hvac:
        dT_air = T_air_new - T_air_prev
        q_hvac_actual = (self.air_thermal_mass * dT_air / self.dt +
                         q_fabric_total + q_window_total + air_exchange_load -
                         internal_gains_w)

        return T_new, q_fabric_total, q_window_total, air_exchange_load, q_hvac_actual
"""
Implements the Conduction Finite-Difference (CondFD) method for heat transfer
through a multi-layered construction, designed to be used with a coupled solver.
"""
import numpy as np

class CondFDSolver:
    """
    Handles the discretization of a construction and populates the relevant rows
    of a larger, coupled system of linear equations for transient heat conduction.
    """
    def __init__(self, construction, dt_seconds, space_discretization_const=3.0, max_nodes_per_layer=20):
        """
        Initializes the solver and discretizes the construction into nodes.

        Args:
            construction: List of Material namedtuples
            dt_seconds: Timestep in seconds
            space_discretization_const: Spatial discretization constant (default: 3.0)
            max_nodes_per_layer: Maximum nodes per layer to limit matrix size (default: 20)
        """
        self.dt = dt_seconds
        self.max_nodes_per_layer = max_nodes_per_layer
        self.nodes = []
        self._discretize(construction, space_discretization_const)

    def _discretize(self, construction, space_discretization_const):
        """Creates the grid of calculation nodes based on material properties."""
        import warnings
        for material in reversed(construction):
            # Validate material has positive thickness
            if material.thickness <= 0:
                raise ValueError(
                    f"Material '{material.name}' has invalid thickness: {material.thickness}. "
                    "All layers must have positive thickness."
                )
            alpha = material.conductivity / (material.density * material.specific_heat)

            # Use consistent discretization for all materials per EnergyPlus default
            # EnergyPlus Engineering Reference: C = 3.0 (inverse of grid Fourier number)
            # This provides stability (Fo < 0.5) while avoiding over-refinement
            dx = np.sqrt(space_discretization_const * alpha * self.dt)

            num_nodes_layer = int(np.ceil(material.thickness / dx))

            # Apply maximum nodes per layer limit
            if num_nodes_layer > self.max_nodes_per_layer:
                warnings.warn(
                    f"Layer '{material.name}' would have {num_nodes_layer} nodes with dx={dx:.6f}m. "
                    f"Limiting to {self.max_nodes_per_layer} nodes for numerical stability. "
                    f"New dx={material.thickness/self.max_nodes_per_layer:.6f}m",
                    RuntimeWarning
                )
                num_nodes_layer = self.max_nodes_per_layer

            # Ensure minimum of 3 nodes per layer for accuracy
            num_nodes_layer = max(3, num_nodes_layer)

            actual_dx = material.thickness / num_nodes_layer

            for _ in range(num_nodes_layer):
                self.nodes.insert(0, {
                    'T': 20.0,
                    'k': material.conductivity,
                    'rho': material.density,
                    'cp': material.specific_heat,
                    'dx': actual_dx
                })
    
    def set_initial_temperatures(self, temp_c):
        """Sets the initial temperature for all nodes."""
        for node in self.nodes:
            node['T'] = temp_c

    def update_temperatures(self, new_temperatures):
        """Updates node temperatures from the results of the coupled solver."""
        if len(new_temperatures) != len(self.nodes):
            raise ValueError("Mismatch in number of temperatures for update.")
        for i, node in enumerate(self.nodes):
            node['T'] = new_temperatures[i]

    def populate_matrix_equations(self, A, B, node_offset, air_node_idx,
                                h_inside, h_outside, T_outside_air,
                                surface_area, solar_gain_w=0.0,
                                exterior_longwave_w_m2=0.0,
                                exterior_solar_absorbed_w_m2=0.0,
                                h_exterior_longwave=0.0,
                                T_exterior_radiation=None,
                                h_r_sky=0.0, h_r_air=0.0, h_r_ground=0.0,
                                T_sky_c=None, T_ground_c=None,
                                h_r_interior=0.0, T_MRT_interior=None):
        """
        Populates the rows of the main A and B matrices corresponding to this
        fabric construction's nodes.

        Args:
            exterior_longwave_w_m2: Net exterior longwave radiation flux (W/m²) [DEPRECATED]
                                   Use separated h_r components instead
            exterior_solar_absorbed_w_m2: Solar radiation absorbed at exterior surface (W/m²)
            h_exterior_longwave: Linearized radiation heat transfer coefficient (W/m²·K)
                                [DEPRECATED - use separated components instead]
            T_exterior_radiation: Effective radiation temperature (°C) [DEPRECATED]
            h_r_sky: Linearized radiation coefficient to sky (W/m²·K)
            h_r_air: Linearized radiation coefficient to air (W/m²·K)
            h_r_ground: Linearized radiation coefficient to ground (W/m²·K)
            T_sky_c: Sky temperature (°C), defaults to T_outside_air - 11
            T_ground_c: Ground temperature (°C), defaults to T_outside_air
            h_r_interior: Linearized interior radiation coefficient (W/m²·K)
                         for radiation exchange with other interior surfaces via MRT
            T_MRT_interior: Mean Radiant Temperature (°C) of other interior surfaces

        Note:
            Uses EnergyPlus-style separated radiation components when h_r_sky/h_r_air/h_r_ground
            are provided (non-zero). Falls back to single h_exterior_longwave for backward
            compatibility if separated components are all zero.

            Interior radiation (h_r_interior) provides surface-to-surface radiation
            exchange via the MRT simplification, following EnergyPlus methodology.
        """
        # Determine if we're using separated radiation components (EnergyPlus style)
        use_separated_radiation = (h_r_sky > 0 or h_r_air > 0 or h_r_ground > 0)

        # Set default temperatures if not provided
        if T_sky_c is None:
            T_sky_c = T_outside_air - 11.0  # Default sky depression
        if T_ground_c is None:
            T_ground_c = T_outside_air  # Default ground at air temp

        # Backward compatibility: if using old single-temperature approach
        if T_exterior_radiation is None:
            T_exterior_radiation = T_outside_air
        N = len(self.nodes)
        T_old = np.array([node['T'] for node in self.nodes])

        # Node 0: Inside surface, coupled to the zone air node and interior radiation
        idx = node_offset
        node = self.nodes[0]
        rho, cp, k, dx = node['rho'], node['cp'], node['k'], node['dx']
        capacitance = rho * cp * dx / self.dt
        # Arithmetic mean for interface conductivity (standard CondFD approach)
        k_east = (self.nodes[1]['k'] + k) / 2
        dx_east = (self.nodes[1]['dx'] + dx) / 2

        # Interior radiation to MRT (if enabled)
        # Heat balance: C*dT/dt = q_conduction + q_convection + q_radiation + q_solar
        # q_radiation = h_r_interior * (T_MRT - T_surface)
        # Implicit form: h_r_interior goes to diagonal, h_r_interior * T_MRT goes to RHS
        use_interior_radiation = (h_r_interior > 0 and T_MRT_interior is not None)

        if use_interior_radiation:
            A[idx, idx] = (capacitance + h_inside + h_r_interior + k_east / dx_east) * surface_area
            # B includes: capacitance term + interior radiation driving to MRT + solar
            B[idx] = (capacitance * T_old[0] + h_r_interior * T_MRT_interior) * surface_area + solar_gain_w
        else:
            # Original formulation (convection only)
            A[idx, idx] = (capacitance + h_inside + k_east / dx_east) * surface_area
            B[idx] = capacitance * surface_area * T_old[0] + solar_gain_w

        A[idx, idx + 1] = (-k_east / dx_east) * surface_area
        A[idx, air_node_idx] = -h_inside * surface_area 

        # Node N-1: Outside surface
        idx = node_offset + N - 1
        node = self.nodes[N-1]
        rho, cp, k, dx = node['rho'], node['cp'], node['k'], node['dx']
        capacitance = rho * cp * dx / self.dt
        # Arithmetic mean for interface conductivity
        k_west = (self.nodes[N-2]['k'] + k) / 2
        dx_west = (self.nodes[N-2]['dx'] + dx) / 2

        A[idx, idx - 1] = (-k_west / dx_west) * surface_area

        # Exterior boundary condition: q_solar + q_conv + q_rad = q_conduction
        # Two modes: separated radiation (EnergyPlus style) or single h_exterior_longwave (legacy)

        if use_separated_radiation:
            # EnergyPlus-style: separate radiation coefficients to sky, air, and ground
            # Each component drives to its respective temperature, preserving T^4 nonlinearity
            # A matrix diagonal: sum of all coefficients
            h_rad_total = h_r_sky + h_r_air + h_r_ground
            A[idx, idx] = (capacitance + h_outside + h_rad_total + k_west / dx_west) * surface_area

            # B vector: each radiation component drives to its respective temperature
            # q = h_conv*(T_air - T_surf) + h_r_sky*(T_sky - T_surf) + h_r_air*(T_air - T_surf)
            #   + h_r_ground*(T_ground - T_surf) + q_solar
            # Rearranging for implicit form: A*T_surf = B
            q_exterior_w_m2 = (h_outside * T_outside_air +
                               h_r_sky * T_sky_c +
                               h_r_air * T_outside_air +
                               h_r_ground * T_ground_c +
                               exterior_solar_absorbed_w_m2)
        else:
            # Legacy mode: single h_exterior_longwave with averaged T_exterior_radiation
            # This provides backward compatibility
            A[idx, idx] = (capacitance + h_outside + h_exterior_longwave + k_west / dx_west) * surface_area

            q_exterior_w_m2 = (h_outside * T_outside_air +
                               h_exterior_longwave * T_exterior_radiation +
                               exterior_longwave_w_m2 +  # Keep for backward compatibility (will be 0)
                               exterior_solar_absorbed_w_m2)

        B[idx] = (capacitance * T_old[N-1] + q_exterior_w_m2) * surface_area
        
        # Internal nodes
        for i in range(1, N - 1):
            idx = node_offset + i
            node = self.nodes[i]
            rho, cp, k, dx = node['rho'], node['cp'], node['k'], node['dx']
            capacitance = rho * cp * dx / self.dt
            # Arithmetic mean for interface conductivity (standard CondFD approach)
            k_west = (self.nodes[i-1]['k'] + k) / 2
            dx_west = (self.nodes[i-1]['dx'] + dx) / 2
            k_east = (self.nodes[i+1]['k'] + k) / 2
            dx_east = (self.nodes[i+1]['dx'] + dx) / 2

            A[idx, idx - 1] = (-k_west / dx_west) * surface_area 
            A[idx, idx] = (capacitance + k_west / dx_west + k_east / dx_east) * surface_area 
            A[idx, idx + 1] = (-k_east / dx_east) * surface_area 
            B[idx] = capacitance * surface_area * T_old[i]


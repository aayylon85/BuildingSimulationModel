"""
Implements the Conduction Finite-Difference (CondFD) method for heat transfer
through a multi-layered construction.
"""
import numpy as np

class CondFDSolver:
    """
    Solves 1D transient heat conduction through a construction using an implicit
    finite difference method.
    """
    def __init__(self, construction, dt_seconds, space_disdiscretization_const=3.0):
        """
        Initializes the solver and discretizes the construction into nodes.
        
        Args:
            construction (list of Material): The layers of the construction.
            dt_seconds (float): The simulation time step in seconds.
            space_discretization_const (float): Constant for determining node spacing.
        """
        self.construction = construction
        self.dt = dt_seconds
        self.space_discretization_const = space_disdiscretization_const
        self.nodes = []
        self._discretize()

    def _discretize(self):
        """Creates the grid of calculation nodes based on material properties."""
        # Construction is defined from outside to inside, so we reverse for discretization
        # to ensure node 0 is the inside surface.
        for material in reversed(self.construction):
            alpha = material.conductivity / (material.density * material.specific_heat)
            # Fourier number for stability guide: (alpha * dt) / dx^2
            dx = np.sqrt(self.space_discretization_const * alpha * self.dt)
            num_nodes_layer = int(np.ceil(material.thickness / dx))
            actual_dx = material.thickness / num_nodes_layer
            
            for _ in range(num_nodes_layer):
                # We insert at the beginning to maintain inside -> outside order
                self.nodes.insert(0, {
                    'T': 20.0,  # Default initial temperature
                    'k': material.conductivity, 
                    'rho': material.density, 
                    'cp': material.specific_heat, 
                    'dx': actual_dx
                })
    
    def set_initial_temperatures(self, temp_c):
        """Sets the initial temperature for all nodes."""
        for node in self.nodes:
            node['T'] = temp_c

    def solve_step(self, T_inside_air, T_outside_air, h_inside, h_outside, update_state=True):
        """
        Solves for the new temperatures and surface fluxes for one time step.
        
        Args:
            T_inside_air (float): Inside air temperature (°C).
            T_outside_air (float): Outside air temperature (°C) or boundary temperature.
            h_inside (float): Inside surface heat transfer coefficient (W/m^2.K).
            h_outside (float): Outside surface heat transfer coefficient (W/m^2.K).
            update_state (bool): If True, updates the internal node temperatures.
            
        Returns:
            tuple: (new_temperatures, inside_flux, outside_flux)
        """
        N = len(self.nodes)
        A = np.zeros((N, N))
        B = np.zeros(N)
        T_old = np.array([node['T'] for node in self.nodes])

        # Node 0: Inside surface
        node = self.nodes[0]
        rho, cp, k, dx = node['rho'], node['cp'], node['k'], node['dx']
        capacitance = rho * cp * dx / self.dt
        k_east = (self.nodes[1]['k'] + k) / 2
        dx_east = (self.nodes[1]['dx'] + dx) / 2
        A[0, 0] = capacitance + h_inside + k_east / dx_east
        A[0, 1] = -k_east / dx_east
        B[0] = capacitance * T_old[0] + h_inside * T_inside_air

        # Node N-1: Outside surface
        node = self.nodes[N-1]
        rho, cp, k, dx = node['rho'], node['cp'], node['k'], node['dx']
        capacitance = rho * cp * dx / self.dt
        k_west = (self.nodes[N-2]['k'] + k) / 2
        dx_west = (self.nodes[N-2]['dx'] + dx) / 2
        A[N-1, N-2] = -k_west / dx_west
        A[N-1, N-1] = capacitance + h_outside + k_west / dx_west
        B[N-1] = capacitance * T_old[N-1] + h_outside * T_outside_air
        
        # Internal nodes
        for i in range(1, N - 1):
            node = self.nodes[i]
            rho, cp, k, dx = node['rho'], node['cp'], node['k'], node['dx']
            capacitance = rho * cp * dx / self.dt
            k_west = (self.nodes[i-1]['k'] + k) / 2
            dx_west = (self.nodes[i-1]['dx'] + dx) / 2
            k_east = (self.nodes[i+1]['k'] + k) / 2
            dx_east = (self.nodes[i+1]['dx'] + dx) / 2
            A[i, i-1] = -k_west / dx_west
            A[i, i] = capacitance + k_west / dx_west + k_east / dx_east
            A[i, i+1] = -k_east / dx_east
            B[i] = capacitance * T_old[i]

        T_new = np.linalg.solve(A, B)

        if update_state:
            for i, node in enumerate(self.nodes):
                node['T'] = T_new[i]
            
        inside_flux = h_inside * (T_inside_air - T_new[0])
        outside_flux = h_outside * (T_new[-1] - T_outside_air)
        
        return T_new, inside_flux, outside_flux


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
    def __init__(self, construction, dt_seconds, space_discretization_const=3.0):
        """
        Initializes the solver and discretizes the construction into nodes.
        """
        self.dt = dt_seconds
        self.nodes = []
        self._discretize(construction, space_discretization_const)

    def _discretize(self, construction, space_discretization_const):
        """Creates the grid of calculation nodes based on material properties."""
        for material in reversed(construction):
            alpha = material.conductivity / (material.density * material.specific_heat)
            dx = np.sqrt(space_discretization_const * alpha * self.dt)
            num_nodes_layer = int(np.ceil(material.thickness / dx))
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
                                h_inside, h_outside, T_outside_air):
        """
        Populates the rows of the main A and B matrices corresponding to this
        fabric construction's nodes.
        """
        N = len(self.nodes)
        T_old = np.array([node['T'] for node in self.nodes])

        # Node 0: Inside surface, coupled to the zone air node
        idx = node_offset
        node = self.nodes[0]
        rho, cp, k, dx = node['rho'], node['cp'], node['k'], node['dx']
        capacitance = rho * cp * dx / self.dt
        k_east = (self.nodes[1]['k'] + k) / 2
        dx_east = (self.nodes[1]['dx'] + dx) / 2
        
        A[idx, idx] = capacitance + h_inside + k_east / dx_east
        A[idx, idx + 1] = -k_east / dx_east
        A[idx, air_node_idx] = -h_inside # Link to the air node
        B[idx] = capacitance * T_old[0]

        # Node N-1: Outside surface
        idx = node_offset + N - 1
        node = self.nodes[N-1]
        rho, cp, k, dx = node['rho'], node['cp'], node['k'], node['dx']
        capacitance = rho * cp * dx / self.dt
        k_west = (self.nodes[N-2]['k'] + k) / 2
        dx_west = (self.nodes[N-2]['dx'] + dx) / 2
        
        A[idx, idx - 1] = -k_west / dx_west
        A[idx, idx] = capacitance + h_outside + k_west / dx_west
        B[idx] = capacitance * T_old[N-1] + h_outside * T_outside_air
        
        # Internal nodes
        for i in range(1, N - 1):
            idx = node_offset + i
            node = self.nodes[i]
            rho, cp, k, dx = node['rho'], node['cp'], node['k'], node['dx']
            capacitance = rho * cp * dx / self.dt
            k_west = (self.nodes[i-1]['k'] + k) / 2
            dx_west = (self.nodes[i-1]['dx'] + dx) / 2
            k_east = (self.nodes[i+1]['k'] + k) / 2
            dx_east = (self.nodes[i+1]['dx'] + dx) / 2

            A[idx, idx - 1] = -k_west / dx_west
            A[idx, idx] = capacitance + k_west / dx_west + k_east / dx_east
            A[idx, idx + 1] = -k_east / dx_east
            B[idx] = capacitance * T_old[i]


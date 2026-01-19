"""
Stores shared thermophysical constants for the simulation.
"""

# Constants for air properties at standard conditions
AIR_DENSITY_KG_M3 = 1.225
AIR_SPECIFIC_HEAT_J_KG_K = 1006
STEFAN_BOLTZMANN = 5.670374419e-8

# Constants: Surface Roughness Multipliers (Walton 1981)
ROUGHNESS_MULTIPLIERS = {
    1: 2.17,  # Very Rough (Stucco)
    2: 1.67,  # Rough (Brick)
    3: 1.52,  # Medium Rough (Concrete)
    4: 1.13,  # Medium Smooth (Clear pine)
    5: 1.11,  # Smooth (Smooth Plaster)
    6: 1.00,  # Very Smooth (Glass)
}
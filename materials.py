"""
Defines building materials and functions to create constructions from a config file.
"""
from collections import namedtuple

# Define a simple data structure for a material layer
Material = namedtuple("Material", ["name", "thickness", "conductivity", "density", "specific_heat"])

def create_construction_from_config(config):
    """
    Creates a list of Material objects for a construction based on the JSON config.
    """
    defined_materials = {}
    for mat_props in config['materials']:
        material = Material(
            name=mat_props['name'],
            thickness=mat_props['thickness'],
            conductivity=mat_props['conductivity'],
            density=mat_props['density'],
            specific_heat=mat_props['specific_heat']
        )
        defined_materials[mat_props['name']] = material
    
    construction_layers = []
    for layer_name in config['construction']['layers']:
        if layer_name in defined_materials:
            construction_layers.append(defined_materials[layer_name])
        else:
            raise ValueError(f"Material '{layer_name}' used in construction but not defined in materials list.")
            
    return construction_layers

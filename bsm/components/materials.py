"""
Defines building materials and functions to create constructions from a config file.
"""
from collections import namedtuple


Material = namedtuple("Material", ["name", "thickness", "conductivity", "density", "specific_heat", "solar_absorptance", "thermal_emissivity"])

def create_materials_dict(config):
    """Creates a dictionary of Material objects from the config."""
    defined_materials = {}
    for mat_props in config.get('materials', []):
        material = Material(
            name=mat_props['name'],
            thickness=mat_props['thickness'],
            conductivity=mat_props['conductivity'],
            density=mat_props['density'],
            specific_heat=mat_props['specific_heat'],
            solar_absorptance=mat_props.get('solar_absorptance', 0.6),  # Default per ASHRAE 140
            thermal_emissivity=mat_props.get('thermal_emissivity', 0.9)  # Default per ASHRAE 140
        )
        defined_materials[mat_props['name']] = material
    return defined_materials

def create_constructions_dict(config):
    """
    Creates a dictionary of construction layer lists from the config.
    The CondFDSolver expects a list of Material objects.
    """
    materials_db = create_materials_dict(config)
    constructions = {}
    
    
    for name, const_data in config.get('constructions', {}).items():
        layers = []
        for layer_name in const_data['layers']:
            if layer_name not in materials_db:
                raise ValueError(f"Material '{layer_name}' in construction '{name}' not defined in materials list.")
            layers.append(materials_db[layer_name])
        constructions[name] = layers
        
    return constructions
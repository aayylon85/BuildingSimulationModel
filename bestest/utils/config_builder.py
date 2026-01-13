"""
BESTEST Configuration Builder - Converts BESTEST case specs to simulation config format.

This module translates ASHRAE Standard 140 BESTEST test case specifications
into the simulation_config.json format used by the existing simulation engine.
"""
import json
import os
from typing import Dict, Any, Optional
from copy import deepcopy


class BESTESTConfigBuilder:
    """
    Builds simulation configuration files from BESTEST test case specifications.
    """

    def __init__(self, output_dir: str = "bestest_configs"):
        """
        Initialize the config builder.

        Args:
            output_dir: Directory to save generated config files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def build_config(self, case_data: dict, output_filename: Optional[str] = None) -> str:
        """
        Build a complete simulation configuration from BESTEST case data.

        Args:
            case_data: BESTEST case specification dictionary
            output_filename: Optional custom filename (default: case_{case_id}_config.json)

        Returns:
            Path to the generated configuration file
        """
        case_id = case_data.get('case_id', 'unknown')

        # Build simulation config dictionary
        sim_config = {}

        # 1. Simulation settings
        sim_config['simulation_settings'] = self._build_simulation_settings(case_data)

        # 2. Zone properties
        sim_config['zone_properties'] = self._build_zone_properties(case_data)

        # 3. Geometry
        sim_config['geometry'] = self._build_geometry(case_data)

        # 4. Materials
        sim_config['materials'] = self._build_materials(case_data)

        # 5. Constructions
        sim_config['constructions'] = self._build_constructions(case_data)

        # 6. Windows
        sim_config['windows'] = self._build_windows(case_data)

        # 7. HVAC system
        sim_config['hvac_system'] = self._build_hvac_system(case_data)

        # 8. Air exchange (infiltration and ventilation)
        sim_config['air_exchange'] = self._build_air_exchange(case_data)

        # 9. Schedules
        sim_config['schedules'] = self._build_schedules(case_data)

        # 10. Occupancy (disabled for BESTEST)
        sim_config['occupancy'] = self._build_occupancy(case_data)

        # 11. Weather
        sim_config['weather'] = self._build_weather(case_data)

        # 12. Convection models
        sim_config['convection_models'] = self._build_convection_models(case_data)

        # Save to file
        if output_filename is None:
            output_filename = f"case_{case_id}_config.json"

        output_path = os.path.join(self.output_dir, output_filename)

        with open(output_path, 'w') as f:
            json.dump(sim_config, f, indent=2)

        return output_path

    def _build_simulation_settings(self, case_data: dict) -> dict:
        """Build simulation settings section."""
        settings = case_data.get('simulation_settings', {})

        return {
            'dt_minutes': settings.get('dt_minutes', 10),
            'duration_days': settings.get('duration_days', 365),
            'stabilization_days': settings.get('warmup_days', 25),
            'start_date': settings.get('start_date', '2020-01-01'),
            'enable_opaque_solar_absorption': settings.get('enable_opaque_solar_absorption', True)
        }

    def _build_zone_properties(self, case_data: dict) -> dict:
        """Build zone properties section."""
        geometry = case_data.get('geometry', {})

        return {
            'length': geometry.get('length', 8.0),
            'width': geometry.get('width', 6.0),
            'height': geometry.get('height', 2.7),
            'zone_sensible_heat_capacity_multiplier': geometry.get(
                'zone_sensible_heat_capacity_multiplier', 1.0
            )
        }

    def _build_geometry(self, case_data: dict) -> dict:
        """Build geometry section with surface definitions."""
        surfaces_data = case_data.get('surfaces', {})

        geometry = {
            'exterior_surfaces': [],
            'surface_definitions': {}
        }

        for surface_name, surface_spec in surfaces_data.items():
            # Add to exterior surfaces list if exterior
            if surface_spec.get('is_exterior', True):
                geometry['exterior_surfaces'].append(surface_name)

            # Build surface definition
            geometry['surface_definitions'][surface_name] = {
                'area': surface_spec.get('area'),
                'perimeter': surface_spec.get('perimeter', 0.0),
                'tilt': surface_spec.get('tilt'),
                'azimuth': surface_spec.get('azimuth'),
                'type': surface_spec.get('type'),
                'roughness_index': surface_spec.get('roughness_index', 3),
                'construction_name': surface_spec.get('construction')
            }

        return geometry

    def _build_materials(self, case_data: dict) -> list:
        """Build materials list."""
        materials = case_data.get('materials', [])
        return deepcopy(materials)

    def _build_constructions(self, case_data: dict) -> dict:
        """Build constructions dictionary."""
        constructions_data = case_data.get('constructions', {})
        constructions = {}

        for construction_name, construction_spec in constructions_data.items():
            constructions[construction_name] = {
                'layers': construction_spec.get('layers', [])
            }

        return constructions

    def _build_windows(self, case_data: dict) -> list:
        """Build windows list."""
        windows_data = case_data.get('windows', [])
        windows = []

        for window_spec in windows_data:
            window = {
                'wall_name': window_spec.get('wall_name'),
                'area': window_spec.get('area'),
                'u_value': window_spec.get('u_value'),
                'shgc': window_spec.get('shgc')
            }

            # Glass fraction (ratio of glass to total window area, default 1.0 = no frame)
            if 'glass_fraction' in window_spec:
                window['glass_fraction'] = window_spec.get('glass_fraction')

            # Solar distribution (default all to floor if not specified)
            solar_dist = window_spec.get('solar_distribution', {})
            if not solar_dist:
                solar_dist = {'floor': 1.0}

            window['solar_distribution'] = solar_dist

            # Angular dependence (if present) - for ASHRAE 140 Table 7-12
            if 'angular_dependence' in window_spec:
                window['angular_dependence'] = window_spec['angular_dependence']

            # Shading (if present) - placeholder for future implementation
            if 'shading' in window_spec:
                window['shading'] = window_spec['shading']

            windows.append(window)

        return windows

    def _build_hvac_system(self, case_data: dict) -> dict:
        """Build HVAC system configuration."""
        hvac_data = case_data.get('hvac', {})

        hvac_config = {
            'model_type': hvac_data.get('model_type', 'IdealLoadsHVAC')
        }

        # Add capacity if specified
        if 'heating_capacity_w' in hvac_data:
            hvac_config['heating_capacity_w'] = hvac_data['heating_capacity_w']
        if 'cooling_capacity_w' in hvac_data:
            hvac_config['cooling_capacity_w'] = hvac_data['cooling_capacity_w']

        # Add setpoints (can be constant or scheduled)
        hvac_config['heating_setpoint_c'] = hvac_data.get('heating_setpoint_c', 20.0)
        hvac_config['cooling_setpoint_c'] = hvac_data.get('cooling_setpoint_c', 27.0)

        # Add schedules if present
        if 'heating_setpoint_schedule' in hvac_data:
            hvac_config['heating_setpoint_schedule'] = hvac_data['heating_setpoint_schedule']
        if 'cooling_setpoint_schedule' in hvac_data:
            hvac_config['cooling_setpoint_schedule'] = hvac_data['cooling_setpoint_schedule']

        # Model-specific parameters
        model_type = hvac_config['model_type']

        if model_type == 'IdealLoadsHVAC':
            hvac_config.setdefault('heating_capacity_w', 1e9)
            hvac_config.setdefault('cooling_capacity_w', 1e9)
            hvac_config.setdefault('proportional_gain_w_k', 100000)

        elif model_type == 'StatefulHVAC':
            hvac_config.setdefault('heating_deadband_c', hvac_data.get('heating_deadband_c', 1.0))
            hvac_config.setdefault('cooling_deadband_c', hvac_data.get('cooling_deadband_c', 1.0))
            hvac_config.setdefault('min_runtime_minutes', hvac_data.get('min_runtime_minutes', 10))
            hvac_config.setdefault('min_offtime_minutes', hvac_data.get('min_offtime_minutes', 10))
            hvac_config.setdefault('ramp_up_minutes', hvac_data.get('ramp_up_minutes', 5))

        return hvac_config

    def _build_air_exchange(self, case_data: dict) -> dict:
        """Build air exchange (infiltration and ventilation) configuration."""
        infiltration_data = case_data.get('infiltration', {})
        ventilation_data = case_data.get('ventilation', {})

        air_exchange = {
            'infiltration': {},
            'ventilation': {}
        }

        # Infiltration
        infiltration_method = infiltration_data.get('method', 'constant')

        if infiltration_method == 'constant' or 'ach_design' in infiltration_data:
            # Constant ACH infiltration for BESTEST
            air_exchange['infiltration'] = {
                'type': 'constant_ach',
                'constant_ach': infiltration_data.get('ach_design', 0.5)
            }
        else:
            # Flow coefficient model (default)
            air_exchange['infiltration'] = {
                'type': 'flow_coefficient',
                'flow_coefficient_m3_s_Pa_n': infiltration_data.get('flow_coefficient_m3_s_Pa_n', 0.001),
                'pressure_exponent_n': infiltration_data.get('pressure_exponent_n', 0.65),
                'stack_coeff_Pa_K': infiltration_data.get('stack_coeff_Pa_K', 0.1),
                'wind_coeff_Pa_s2_m2': infiltration_data.get('wind_coeff_Pa_s2_m2', 0.01),
                'shelter_factor_s': infiltration_data.get('shelter_factor_s', 0.7)
            }

        # Ventilation
        if ventilation_data.get('type') == 'scheduled':
            # Scheduled ventilation (for night cooling cases)
            air_exchange['ventilation'] = {
                'open_window_ach': 0.0,  # Will be handled by schedule
                'schedule': ventilation_data.get('schedule', {})
            }
        else:
            # No ventilation (closed windows)
            air_exchange['ventilation'] = {
                'open_window_ach': ventilation_data.get('open_window_ach', 0.0)
            }

        return air_exchange

    def _build_schedules(self, case_data: dict) -> dict:
        """Build schedules section."""
        hvac_data = case_data.get('hvac', {})
        internal_gains_data = case_data.get('internal_gains', {})

        schedules = {
            'occupied_hours': [0, 24],  # Always occupied for BESTEST
            'occupied_heating_setpoint_c': hvac_data.get('heating_setpoint_c', 20.0),
            'unoccupied_heating_setpoint_c': hvac_data.get('heating_setpoint_c', 20.0),
            'occupied_cooling_setpoint_c': hvac_data.get('cooling_setpoint_c', 27.0),
            'unoccupied_cooling_setpoint_c': hvac_data.get('cooling_setpoint_c', 27.0),
            'occupied_internal_gains_w': internal_gains_data.get('sensible_w', 200.0),
            'unoccupied_internal_gains_w': internal_gains_data.get('sensible_w', 200.0)
        }

        # Handle scheduled setpoints (e.g., thermostat setback cases)
        if 'heating_setpoint_schedule' in hvac_data:
            schedules['heating_setpoint_schedule'] = hvac_data['heating_setpoint_schedule']
        if 'cooling_setpoint_schedule' in hvac_data:
            schedules['cooling_setpoint_schedule'] = hvac_data['cooling_setpoint_schedule']

        return schedules

    def _build_occupancy(self, case_data: dict) -> dict:
        """Build occupancy section (typically disabled for BESTEST)."""
        # BESTEST cases don't include dynamic occupant behavior
        return {
            'heat_gain_per_occupant_w': 0,
            'check_interval_minutes': 60,
            'thermostat_adjustment_c': 0.0,
            'occupants': []
        }

    def _build_weather(self, case_data: dict) -> dict:
        """Build weather configuration."""
        weather_data = case_data.get('weather', {})

        weather_config = {}

        # Check if EPW file is specified
        if 'file' in weather_data:
            epw_filename = weather_data['file']
            # Construct path to EPW file in bestest/weather/epw/
            bestest_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            epw_path = os.path.join(bestest_dir, 'bestest', 'weather', 'epw', epw_filename)

            weather_config = {
                'type': 'epw',
                'epw_path': epw_path
            }
        else:
            # Fallback to simple sinusoidal (for testing)
            weather_config = {
                'type': 'simple_sinusoidal',
                'solar_max_irradiance_w_m2': 800,
                'temp_base_c': 10,
                'temp_amplitude_c': 8,
                'temp_phase_shift_hr': 14
            }

        return weather_config

    def _build_convection_models(self, case_data: dict) -> dict:
        """Build convection models configuration."""
        convection_data = case_data.get('convection', {})

        # Default BESTEST-appropriate convection models
        default_convection = {
            'exterior_hf': {
                'RoofStable': 'SparrowWindward',
                'RoofUnstable': 'SparrowWindward',
                'VerticalWallWindward': 'SparrowWindward',
                'VerticalWallLeeward': 'SparrowLeeward'
            },
            'exterior_hn': {
                'RoofStable': 'WaltonStableHorizontalOrTilt',
                'RoofUnstable': 'WaltonUnstableHorizontalOrTilt',
                'VerticalWallWindward': 'ASHRAEVerticalWall',
                'VerticalWallLeeward': 'ASHRAEVerticalWall'
            },
            'interior': {
                'VerticalWall': 'ASHRAEVerticalWall',
                'StableHorizontal': 'WaltonStableHorizontalOrTilt',
                'UnstableHorizontal': 'WaltonUnstableHorizontalOrTilt',
                'StableTilted': 'WaltonStableHorizontalOrTilt',
                'UnstableTilted': 'WaltonUnstableHorizontalOrTilt'
            }
        }

        # Override with case-specific models if provided
        convection_config = deepcopy(default_convection)

        # Handle detailed dictionary specifications
        if 'interior' in convection_data and isinstance(convection_data['interior'], dict):
            convection_config['interior'].update(convection_data['interior'])
        if 'exterior_hf' in convection_data and isinstance(convection_data['exterior_hf'], dict):
            convection_config['exterior_hf'].update(convection_data['exterior_hf'])
        if 'exterior_hn' in convection_data and isinstance(convection_data['exterior_hn'], dict):
            convection_config['exterior_hn'].update(convection_data['exterior_hn'])

        # Note: Simple string specifications like "ASHRAE_Simple" are just notes,
        # use defaults above which implement ASHRAE-type correlations

        return convection_config


def bestest_to_simulation_config(case_data: dict, output_path: str) -> str:
    """
    Convenience function to convert a BESTEST case to simulation config.

    Args:
        case_data: BESTEST case specification dictionary
        output_path: Path where the config file should be saved

    Returns:
        Path to the generated configuration file
    """
    output_dir = os.path.dirname(output_path)
    output_filename = os.path.basename(output_path)

    builder = BESTESTConfigBuilder(output_dir=output_dir)
    return builder.build_config(case_data, output_filename=output_filename)


if __name__ == '__main__':
    # Test the config builder with a minimal case
    test_case = {
        'case_id': 'test',
        'description': 'Test case',
        'geometry': {'length': 8.0, 'width': 6.0, 'height': 2.7},
        'surfaces': {
            'south_wall': {
                'area': 21.6,
                'tilt': 90,
                'azimuth': 180,
                'type': 'wall',
                'construction': 'ExteriorWall',
                'is_exterior': True
            }
        },
        'materials': [],
        'constructions': {},
        'windows': [],
        'hvac': {'model_type': 'IdealLoadsHVAC'},
        'infiltration': {'method': 'constant', 'ach_design': 0.5},
        'internal_gains': {'sensible_w': 200},
        'weather': {'file': 'Denver_CO.epw'},
        'simulation_settings': {'dt_minutes': 10, 'duration_days': 365, 'warmup_days': 25}
    }

    builder = BESTESTConfigBuilder(output_dir='test_output')
    config_path = builder.build_config(test_case)
    print(f"Generated test config: {config_path}")

# Building Simulation Model

A Python-based single-zone building thermal simulation model that solves coupled heat balance equations to predict indoor air temperature and energy consumption. This tool simulates the thermal behavior of buildings with realistic physics models including transient heat conduction, convection, solar gains, air exchange, HVAC systems, and occupant interactions.

## Features

- **Coupled Heat Balance Solver**: Fully coupled solution of fabric and air temperatures using finite-difference methods
- **Multi-Layer Constructions**: Conduction finite-difference (CondFD) solver for complex wall assemblies
- **Advanced Convection Models**: Research-grade correlations for interior and exterior surfaces
- **Multiple HVAC Control Strategies**: Proportional, hysteresis-based, and PID control
- **Real Weather Data**: Integration with Open-Meteo API for historical weather
- **Occupant Behavior**: Modeling of window opening and thermostat adjustment preferences
- **Air Exchange**: Physics-based infiltration (ASHRAE AIM-2) and ventilation modeling
- **Solar Gains**: Window heat transfer with configurable solar distribution
- **Results Export**: CSV output with timestamped results and configuration archiving

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository:
```bash
git clone <repository-url>
cd BuildingSimulationModel
```

2. (Recommended) Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

Run a simulation with the default configuration:

```bash
python main.py
```

Or specify a custom configuration file:

```bash
python main.py my_config.json
```

Results will be saved to the `results/YYYY-MM-DD/` directory with:
- CSV file with time-series data
- Copy of the configuration used
- Matplotlib plots (displayed on screen)

## Configuration

Simulations are controlled via JSON configuration files. See [simulation_config.json](simulation_config.json) for a complete example.

### Key Configuration Sections

#### 1. Simulation Settings

```json
"simulation_settings": {
  "dt_minutes": 1,              // Timestep in minutes
  "duration_days": 5,            // Simulation duration in days
  "start_date": "2020-01-01",   // Start date for weather data
  "stabilization_days": 3        // Warm-up period (optional, default: 3)
}
```

#### 2. Location

```json
"location": {
  "latitude": 50.9105,          // Degrees North
  "longitude": -1.4049,         // Degrees East
  "place_name": "Southampton",
  "time_zone": "GMT"
}
```

#### 3. Zone Properties

```json
"zone_properties": {
  "length": 10.0,                                    // meters
  "width": 8.0,                                      // meters
  "height": 2.8,                                     // meters
  "zone_sensible_heat_capacity_multiplier": 4.0     // Multiplier for internal mass
}
```

The `zone_sensible_heat_capacity_multiplier` accounts for furniture and internal thermal mass beyond just air.

#### 4. Geometry

Define surfaces with their properties:

```json
"geometry": {
  "exterior_surfaces": ["north_wall", "east_wall", ...],
  "surface_definitions": {
    "north_wall": {
      "area": 22.4,                  // m²
      "perimeter": 21.6,             // meters
      "tilt": 90,                    // degrees from horizontal (0=roof, 90=wall, 180=floor)
      "azimuth": 0,                  // degrees (0=North, 90=East, 180=South, 270=West)
      "type": "wall",                // wall, roof, or floor
      "roughness_index": 2,          // 1=very rough to 6=very smooth
      "construction_name": "ExteriorWall"
    }
  }
}
```

#### 5. Materials and Constructions

```json
"materials": [
  {
    "name": "Concrete",
    "thickness": 0.1,          // meters
    "conductivity": 1.4,       // W/(m·K)
    "density": 2300,           // kg/m³
    "specific_heat": 880       // J/(kg·K)
  }
],
"constructions": {
  "ExteriorWall": {
    "layers": ["Concrete", "Insulation", "Concrete"]  // Outside to inside
  }
}
```

#### 6. Windows

```json
"windows": [
  {
    "wall_name": "south_wall",
    "area": 6.0,              // m²
    "u_value": 1.5,           // W/(m²·K)
    "shgc": 0.5,              // Solar Heat Gain Coefficient (0-1)
    "solar_distribution": {   // Fraction of solar to each surface
      "floor": 0.6,
      "north_wall": 0.2,
      "east_wall": 0.1,
      "west_wall": 0.1
    }
  }
]
```

#### 7. Air Exchange

```json
"air_exchange": {
  "infiltration": {
    "flow_coefficient_m3_s_Pa_n": 0.00025,    // m³/(s·Pa^n)
    "pressure_exponent_n": 0.65,              // dimensionless
    "stack_coeff_Pa_K": 0.078,                // Pa/K
    "wind_coeff_Pa_s2_m2": 0.15,              // Pa·s²/m²
    "shelter_factor_s": 0.5                   // 0-1 (0=exposed, 1=sheltered)
  },
  "ventilation": {
    "open_window_ach": 5.0                     // Air changes per hour when window open
  }
}
```

#### 8. HVAC System

Three model types available: `VerySimpleHVAC`, `StatefulHVAC`, `PIDControlledHVAC`

```json
"hvac_system": {
  "model_type": "PIDControlledHVAC",
  "heating_capacity_w": 5000.0,        // Watts
  "cooling_capacity_w": 5000.0,        // Watts
  "heating_deadband_c": 1.0,           // °C (for StatefulHVAC)
  "cooling_deadband_c": 1.0,           // °C (for StatefulHVAC)
  "min_runtime_minutes": 60.0,         // minutes (for StatefulHVAC)
  "min_offtime_minutes": 10.0,         // minutes (for StatefulHVAC)
  "ramp_up_minutes": 30.0,             // minutes (for StatefulHVAC)
  "kp": 100.0,                         // Proportional gain (for PIDControlledHVAC)
  "ki": 5.0,                           // Integral gain (for PIDControlledHVAC)
  "kd": 10.0                           // Derivative gain (for PIDControlledHVAC)
}
```

#### 9. Schedules

```json
"schedules": {
  "occupied_hours": [7, 19],                   // Start and end hour
  "occupied_heating_setpoint_c": 21.0,         // °C
  "unoccupied_heating_setpoint_c": 15.0,       // °C
  "occupied_cooling_setpoint_c": 24.0,         // °C
  "unoccupied_cooling_setpoint_c": 30.0,       // °C
  "occupied_internal_gains_w": 0.0             // Watts (equipment, lighting)
}
```

#### 10. Occupancy

```json
"occupancy": {
  "heat_gain_per_occupant_w": 200.0,           // Watts
  "check_interval_minutes": 60,                // How often to poll occupants (optional, default: 60)
  "thermostat_adjustment_c": 1.0,              // °C per vote (optional, default: 1.0)
  "occupants": [
    {
      "name": "Alice",
      "work_start_hr": 8,
      "work_end_hr": 17,
      "window_preference": "opener",           // "opener", "neutral", or "closer"
      "window_temp_c": 23.0,                   // Threshold for window action
      "thermostat_preference": "changer",      // "changer" or "neutral"
      "thermostat_temp_c": 21.0                // Desired temperature
    }
  ]
}
```

#### 11. Weather

Two types available: `sinusoidal` (for testing) or `file` (real data from Open-Meteo)

```json
"weather": {
  "type": "file",                           // or "sinusoidal"
  "temp_base_c": -10,                       // Only for sinusoidal
  "temp_amplitude_c": 3,                    // Only for sinusoidal
  "temp_phase_shift_hr": 14,                // Only for sinusoidal
  "solar_max_irradiance_w_m2": 400,         // Only for sinusoidal
  "wind_Speed_ms": 20.0                     // Only for sinusoidal
}
```

#### 12. Convection Models

Specify correlations for different surface types and conditions:

```json
"convection_models": {
  "exterior_hf": {                           // Forced convection
    "RoofStable": "SparrowWindward",
    "RoofUnstable": "SparrowWindward",
    "VerticalWallWindward": "SparrowWindward",
    "VerticalWallLeeward": "SparrowLeeward"
  },
  "exterior_hn": {                           // Natural convection
    "RoofStable": "WaltonStableHorizontalOrTilt",
    "RoofUnstable": "WaltonUnstableHorizontalOrTilt",
    "VerticalWallWindward": "ASHRAEVerticalWall",
    "VerticalWallLeeward": "ASHRAEVerticalWall"
  },
  "interior": {
    "VerticalWall": "ASHRAEVerticalWall",
    "UnstableHorizontal": "WaltonUnstableHorizontalOrTilt",
    "StableHorizontal": "WaltonStableHorizontalOrTilt",
    "UnstableTilted": "WaltonUnstableHorizontalOrTilt",
    "StableTilted": "WaltonStableHorizontalOrTilt"
  }
}
```

### Unit Conventions

- **Temperature**: Degrees Celsius (°C)
- **Length**: Meters (m)
- **Area**: Square meters (m²)
- **Power**: Watts (W)
- **Energy**: Watt-hours (Wh) or kilowatt-hours (kWh)
- **Time**: Specified in key names (e.g., `_minutes`, `_days`, `_hours`)
- **Thermal Conductivity**: W/(m·K)
- **Density**: kg/m³
- **Specific Heat**: J/(kg·K)
- **Heat Transfer Coefficient**: W/(m²·K)
- **Angles**: Degrees (0-360)

## Project Structure

```
BuildingSimulationModel/
├── main.py                      # Main entry point
├── simulation_config.json       # Example configuration
├── requirements.txt             # Python dependencies
│
├── Core Simulation:
│   ├── zone.py                 # Zone model manager
│   ├── zone_solver.py          # Coupled heat balance solver
│   └── fabric_heat_transfer.py # CondFD solver for constructions
│
├── HVAC Systems:
│   └── hvac_def.py            # HVAC models and factory
│
├── Heat Transfer:
│   ├── exterior_heat_transfer.py    # Exterior convection
│   ├── interior_heat_transfer.py    # Interior convection
│   └── exterior_longwave_rad.py     # Longwave radiation (not yet integrated)
│
├── Building Components:
│   ├── windows.py              # Window model
│   ├── air_exchange.py         # Infiltration & ventilation
│   └── materials.py            # Material definitions
│
├── Boundary Conditions:
│   ├── weather.py              # Weather generators
│   ├── weather_import.py       # Open-Meteo API client
│   ├── boundary_conditions.py  # Schedule creator
│   └── occupants.py            # Occupant behavior
│
├── Utilities:
│   ├── constants.py            # Physical constants
│   └── plotting.py             # Visualization
│
└── results/                    # Output directory (auto-created)
```

## Physics Models

### Heat Balance Equation

The zone air temperature is calculated from the energy balance:

```
C_air · dT_air/dt = Q_fabric + Q_windows + Q_infiltration + Q_ventilation + Q_internal + Q_solar + Q_HVAC
```

Where:
- **Q_fabric**: Heat transfer through opaque surfaces (convection from inner surfaces)
- **Q_windows**: Conduction through windows
- **Q_infiltration**: Air leakage driven by wind and stack effect
- **Q_ventilation**: Controlled air exchange through open windows
- **Q_internal**: Internal gains from occupants, equipment, lighting
- **Q_solar**: Solar radiation through windows
- **Q_HVAC**: Heating or cooling from HVAC system

### Fabric Heat Transfer

Multi-layer constructions are solved using the Conduction Finite-Difference (CondFD) method:
- Automatic spatial discretization based on Fourier stability criterion
- Fully coupled with zone air temperature
- Non-linear convection coefficients solved iteratively

### Convection Coefficients

**Exterior**: Adaptive algorithm selecting appropriate correlations based on surface orientation and conditions (Sparrow, Blocken, Emmel, Mitchell, Walton, ASHRAE)

**Interior**: Adaptive algorithm based on surface tilt and heat flow direction (ASHRAE, Walton)

### Air Infiltration

ASHRAE AIM-2 model combining wind and stack effects:

```
ΔP_stack = C_s · |T_zone - T_ext|
ΔP_wind = C_w · (s · v_wind)²
ΔP_total = sqrt(ΔP_stack² + ΔP_wind²)
Q = C · (ΔP_total)^n
```

## Output

### CSV File Format

Results are saved with the following columns:

| Column | Description | Units |
|--------|-------------|-------|
| Time (hrs) | Elapsed time from start | hours |
| Zone Temp (C) | Indoor air temperature | °C |
| Outside Temp (C) | Outdoor air temperature | °C |
| HVAC Power (W) | Heating (+) or cooling (-) power | W |
| Fabric Loss (W) | Heat loss through opaque surfaces | W |
| Air Exchange Loss (W) | Heat loss through infiltration/ventilation | W |
| Solar Gains (W) | Solar heat gains through windows | W |
| Internal Gains (W) | Internal heat gains | W |
| Window State (0-1) | Window open fraction | - |

### File Naming Convention

```
results/YYYY-MM-DD/YYYY-MM-DD_HH-MM-SS_Xdays_results.csv
results/YYYY-MM-DD/YYYY-MM-DD_HH-MM-SS_Xdays_config.json
```

Example: `results/2025-12-01/2025-12-01_14-30-45_5days_results.csv`

## Advanced Usage

### Custom Timestep

For faster simulations with reduced accuracy:
```json
"simulation_settings": {
  "dt_minutes": 5  // Larger timestep (trade-off: speed vs accuracy)
}
```

For high accuracy (slower):
```json
"simulation_settings": {
  "dt_minutes": 0.5  // Smaller timestep
}
```

### Extending Warm-up Period

If initial conditions haven't stabilized:
```json
"simulation_settings": {
  "stabilization_days": 5  // Increase from default 3
}
```

### Debugging Convergence Issues

The solver will warn if it doesn't converge. To adjust:

```python
# In zone_solver.py, modify solve_step call:
max_iterations=20,  # Increase from default 10
tolerance=0.005     # Tighten from default 0.01
```

## Troubleshooting

### Common Issues

**1. Solver convergence warnings**
- Reduce timestep (`dt_minutes`)
- Check for unrealistic material properties
- Verify convection model assignments

**2. Temperature out of bounds error**
- Check HVAC capacities are sufficient
- Verify initial conditions and boundary conditions
- Reduce timestep

**3. Weather data download fails**
- Check internet connection
- Verify latitude/longitude are valid
- Check start_date is not in the future

**4. ImportError for dependencies**
```bash
pip install -r requirements.txt --upgrade
```

## Limitations

- Single zone only (no multi-zone capability)
- No humidity/latent loads modeling
- No ground heat transfer (floor uses adiabatic or constant temperature boundary)
- Longwave radiation model not yet integrated
- Solar position not calculated (uses direct irradiance values)
- No shading or obstruction modeling

## Contributing

Contributions are welcome! Areas for improvement:
- Multi-zone capability
- Humidity and latent loads
- Ground heat transfer modeling
- Integration of longwave radiation
- Solar position calculations
- Unit tests and validation
- GUI for configuration

## License

[Specify license here]

## References

### Convection Correlations
- Walton, G.N. (1983). "Thermal Analysis Research Program Reference Manual"
- ASHRAE Handbook of Fundamentals
- Sparrow, E.M., et al. (1979). "Forced convection heat transfer"

### Building Physics
- ASHRAE AIM-2 Model for infiltration
- ISO 13790 - Energy performance of buildings

## Contact

[Add contact information or repository links]

---

**Last Updated**: December 2025

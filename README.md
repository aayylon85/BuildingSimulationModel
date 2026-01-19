# Building Simulation Model

A Python-based single-zone building thermal simulation model that solves coupled heat balance equations to predict indoor air temperature and energy consumption. This tool simulates the thermal behavior of buildings with realistic physics models including transient heat conduction, convection, solar gains, air exchange, HVAC systems, and occupant interactions.

## Features

- **Coupled Heat Balance Solver**: Fully coupled solution of fabric and air temperatures using finite-difference methods
- **Multi-Layer Constructions**: Conduction finite-difference (CondFD) solver for complex wall assemblies with automatic node limiting
- **Advanced Convection Models**: Research-grade correlations for interior and exterior surfaces
- **Multiple HVAC Control Strategies**: Proportional (IdealLoadsHVAC), hysteresis-based (StatefulHVAC), and PID control
- **Hybrid HVAC Coupling**: Automatic selection of implicit/explicit coupling for numerical stability
- **Adaptive Timestepping**: Automatic timestep reduction for difficult convergence scenarios
- **Real Weather Data**: Integration with Open-Meteo API for historical weather
- **Occupant Behavior**: Rule-based modeling of window opening and thermostat preferences
- **LLM Occupant Agents**: Optional AI-powered occupants with memory-driven decision making
- **Air Exchange**: Physics-based infiltration (ASHRAE AIM-2) and ventilation modeling
- **Solar Gains**: Window heat transfer with surface-specific irradiance and angular-dependent transmittance
- **BESTEST Validation**: Full ASHRAE 140-2020 Section 5.2 test suite support
- **Batch Testing Framework**: Automated test rig for running multiple simulation cases
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

### Running a Single Simulation

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

### Running BESTEST Validation Suite

Run the full ASHRAE 140-2020 Section 5.2 test suite:

```bash
python test_rig.py --bestest
```

Run specific BESTEST cases:

```bash
python test_rig.py --bestest --cases 600 620 900 920
```

Results will be saved to `results/bestest_suite/` with:
- Individual simulation results for each case
- `bestest_summary.csv` with aggregated metrics
- `bestest_validation_report.md` with detailed analysis

**Currently Supported Cases**: 600, 620, 900, 920 (additional cases require features under development)

## Configuration

Simulations are controlled via JSON configuration files. See [simulation_config.json](simulation_config.json) for a complete example.

### Key Configuration Sections

#### 1. Simulation Settings

```json
"simulation_settings": {
  "dt_minutes": 1,                          // Timestep in minutes
  "duration_days": 5,                       // Simulation duration in days
  "start_date": "2020-01-01",              // Start date for weather data
  "stabilization_days": 3,                 // Warm-up period (optional, default: 3)

  // Advanced solver options (all optional):
  "max_nodes_per_layer": 20,               // Limit nodes per layer for stability (default: 20)
  "use_adaptive_timestepping": true,       // Auto-reduce timestep on failure (default: true)
  "hvac_coupling_mode": "auto",            // "auto", "implicit", or "explicit" (default: "auto")
  "hvac_max_power_rate_w_s": 50000.0       // Max HVAC power change rate (default: 50000 W/s)
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

**Simple Window** (constant SHGC, backward compatible):
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

**Angular-Dependent Window** (ASHRAE 140 compliant):
```json
"windows": [
  {
    "wall_name": "south_wall",
    "area": 12.0,
    "u_value": 3.0,
    "shgc": 0.789,                          // SHGC at normal incidence
    "glass_fraction": 1.0,                  // Fraction that is glazing (vs frame)
    "angular_dependence": {
      "enabled": true,
      "hemispherical_avg_transmittance": 0.686,  // Optional: auto-calculated if omitted
      "solar_lost_fraction": 0.035               // Per ASHRAE 140 Table 7-13
    },
    "solar_distribution": {
      "floor": 1.0
    }
  }
]
```

When `angular_dependence.enabled` is true, the window uses the `AngularDependentWindow` class which applies different SHGC values based on solar incidence angle per ASHRAE 140 Table 7-12. Direct beam radiation uses angle-dependent transmittance while diffuse radiation uses hemispherical-averaged SHGC.

#### 7. Air Exchange

Two infiltration model types are available:

**Flow Coefficient Model** (ASHRAE AIM-2 for realistic buildings):
```json
"air_exchange": {
  "infiltration": {
    "type": "flow_coefficient",
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

**Constant ACH Model** (for BESTEST validation):
```json
"air_exchange": {
  "infiltration": {
    "type": "constant_ach",
    "constant_ach": 0.5                        // Air changes per hour (constant)
  },
  "ventilation": {
    "open_window_ach": 5.0
  }
}
```

The `constant_ach` model provides a fixed infiltration rate independent of temperature difference or wind speed, as specified in ASHRAE Standard 140 BESTEST cases.

#### 8. HVAC System

Four model types available: `IdealLoadsHVAC`, `VerySimpleHVAC`, `StatefulHVAC`, `PIDControlledHVAC`

**IdealLoadsHVAC** (for BESTEST and validation):
```json
"hvac_system": {
  "model_type": "IdealLoadsHVAC",
  "heating_capacity_w": 1000000000,       // Watts (effectively unlimited)
  "cooling_capacity_w": 1000000000,       // Watts (effectively unlimited)
  "heating_setpoint_c": 20.0,            // °C
  "cooling_setpoint_c": 27.0,            // °C
  "proportional_gain_w_k": 100000        // W/K (high gain for tight control)
}
```

**PIDControlledHVAC** (realistic control):
```json
"hvac_system": {
  "model_type": "PIDControlledHVAC",
  "heating_capacity_w": 5000.0,          // Watts
  "cooling_capacity_w": 5000.0,          // Watts
  "kp": 100.0,                           // Proportional gain
  "ki": 5.0,                             // Integral gain
  "kd": 10.0                             // Derivative gain
}
```

**StatefulHVAC** (with hysteresis and ramp-up):
```json
"hvac_system": {
  "model_type": "StatefulHVAC",
  "heating_capacity_w": 5000.0,          // Watts
  "cooling_capacity_w": 5000.0,          // Watts
  "heating_deadband_c": 1.0,             // °C
  "cooling_deadband_c": 1.0,             // °C
  "min_runtime_minutes": 60.0,           // minutes
  "min_offtime_minutes": 10.0,           // minutes
  "ramp_up_minutes": 30.0                // minutes
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

The codebase is organized as a Python package (`bsm/`) with logical subpackages:

```
BuildingSimulationModel/
├── main.py                      # CLI entry point (thin wrapper)
├── test_rig.py                  # BESTEST CLI entry point (thin wrapper)
├── simulation_config.json       # Example configuration
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Modern Python packaging
│
├── bsm/                         # Main simulation package
│   ├── __init__.py              # Package exports
│   │
│   ├── core/                    # Core simulation engine
│   │   ├── zone.py              # Zone model manager
│   │   ├── zone_solver.py       # Coupled heat balance solver
│   │   ├── fabric_heat_transfer.py  # CondFD solver
│   │   └── constants.py         # Physical constants
│   │
│   ├── heat_transfer/           # Convection and radiation
│   │   ├── convection.py        # Shared convection correlations
│   │   ├── exterior_convection.py   # Exterior adaptive algorithm
│   │   ├── interior_convection.py   # Interior adaptive algorithm
│   │   ├── exterior_longwave.py     # Exterior longwave radiation
│   │   └── interior_longwave.py     # Interior longwave radiation
│   │
│   ├── solar/                   # Solar calculations
│   │   ├── position.py          # Sun position (zenith, azimuth)
│   │   ├── irradiance.py        # Surface-specific irradiance
│   │   ├── diffuse.py           # Perez anisotropic sky model
│   │   ├── incident_angle.py    # Angle of incidence
│   │   └── angular_transmittance.py  # Window angular SHGC
│   │
│   ├── weather/                 # Weather data handling
│   │   ├── generators.py        # EPW and API weather loaders
│   │   ├── api_client.py        # Open-Meteo API client
│   │   ├── sky_temperature.py   # Sky temperature models
│   │   └── ground_temperature.py    # Kusuda ground model
│   │
│   ├── components/              # Building components
│   │   ├── windows.py           # Window models
│   │   ├── air_exchange.py      # Infiltration & ventilation
│   │   ├── materials.py         # Material definitions
│   │   └── hvac.py              # HVAC system models
│   │
│   ├── boundary/                # Boundary conditions
│   │   ├── conditions.py        # Schedule creator
│   │   └── occupants.py         # Rule-based occupant behavior
│   │
│   ├── output/                  # Output and reporting
│   │   ├── plotting.py          # Visualization
│   │   └── report_generator.py  # Markdown report generation
│   │
│   ├── agents/                  # LLM occupant agents (optional)
│   │   ├── manager.py           # Main orchestrator
│   │   ├── generative_agent.py  # Memory-driven agent
│   │   ├── skeleton.py          # Agent SDK types
│   │   ├── simulation_adapter.py    # Zone state bridge
│   │   ├── equipment_manager.py     # Equipment tracking
│   │   ├── lighting_manager.py      # Lighting tracking
│   │   ├── desk_manager.py          # Desk assignments
│   │   ├── memory/              # Memory subsystem
│   │   │   └── stream.py        # Semantic memory storage
│   │   └── cognition/           # Cognitive modules
│   │       ├── modules.py       # Perceive/Retrieve/Reflect
│   │       └── conversation.py  # Multi-agent dialogue
│   │
│   ├── runner.py                # Main simulation logic
│   └── test_rig.py              # BESTEST test framework
│
├── agent_base_types/            # Pre-configured agent personalities
│   ├── alice_office_worker/
│   ├── bob_office_worker/
│   └── charlie_office_worker/
│
├── bestest/                     # BESTEST validation suite
│   ├── cases/                   # Test case definitions
│   ├── utils/                   # Config generation utilities
│   └── weather/                 # EPW weather files
│
├── bestest_configs/             # Generated BESTEST configs
├── test_configs/                # Test configurations
├── Documentation/               # Reference materials
│
└── results/                     # Output directory (auto-created)
    ├── YYYY-MM-DD/              # Individual simulation runs
    ├── bestest_suite/           # BESTEST validation results
    └── agents/                  # Agent output (when enabled)
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
- Node limiting to prevent matrix ill-conditioning (max 20 nodes per layer by default)
- Fully coupled with zone air temperature
- Non-linear convection coefficients solved iteratively with under-relaxation
- Increased iteration limit (50 iterations) with convergence monitoring

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

### Numerical Stability Features

**Adaptive Timestepping**: Automatically reduces timestep (up to 16x subdivision) when solver encounters convergence difficulties, then restores original timestep when stable.

**Hybrid HVAC Coupling**:
- **Implicit coupling** for high-gain proportional controllers (>5000 W/K) - includes HVAC gain in system matrix
- **Explicit coupling** with rate limiting for complex HVAC systems (PID, stateful) - limits power change to 50,000 W/s
- **Auto mode** (default) - automatically selects appropriate method

**Under-relaxation**: Damping factor (0.7 default) prevents oscillations in non-linear iterations

**Physical bounds checking**: Temperature range validation (-50°C to 80°C default) triggers adaptive response

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

The solver includes automatic handling for most convergence issues via adaptive timestepping. If you still encounter problems:

**Option 1: Adjust simulation settings in JSON config**:
```json
"simulation_settings": {
  "dt_minutes": 0.5,                    // Reduce timestep
  "max_nodes_per_layer": 15,            // Reduce discretization
  "hvac_coupling_mode": "implicit",     // Force implicit coupling
  "use_adaptive_timestepping": true     // Ensure adaptive is enabled
}
```

**Option 2: Check for physical issues**:
- Verify material properties are realistic
- Check HVAC capacities are sufficient
- Ensure construction layers are correctly ordered (outside to inside)
- Validate convection model assignments

**Option 3: Monitor convergence**:
The solver logs warnings every 10 iterations if convergence is slow. Watch for:
- "Convergence progress at iteration X" - indicates slow but progressing convergence
- "Adaptive timestepping" - automatic timestep reduction is working
- "Temperature solution out of physical bounds" - indicates unrealistic conditions

## Troubleshooting

### Common Issues

**1. Solver convergence warnings**
- **Solution**: Adaptive timestepping should handle this automatically
- If persists: reduce `dt_minutes` or `max_nodes_per_layer` in config
- Check for unrealistic material properties (e.g., very low conductivity with large thickness)

**2. Temperature out of bounds error**
- **Solution**: Adaptive timestepping will attempt automatic recovery
- Check HVAC capacities are sufficient for loads
- Verify initial conditions (warm-up may need more `stabilization_days`)
- For BESTEST cases: ensure `hvac_coupling_mode` is "auto" or "implicit"

**3. Matrix solver failed / singular matrix**
- Check construction definitions have all required layers
- Verify no surfaces have zero area
- Ensure all referenced construction names exist in config
- Try increasing `max_nodes_per_layer` if very low (<10)

**4. Weather data download fails**
- Check internet connection
- Verify latitude/longitude are valid (-90 to 90, -180 to 180)
- Check `start_date` is not in the future
- Open-Meteo API has rate limits - wait a moment and retry

**5. BESTEST cases fail or give unrealistic results**
- Ensure using `IdealLoadsHVAC` model type
- Check `proportional_gain_w_k` is high (100,000 W/K recommended)
- Verify `hvac_coupling_mode` is "auto" (will select implicit for high gain)
- Review case definition in `bestest/cases/section_5_2_cases.json`

**6. ImportError for dependencies**
```bash
pip install -r requirements.txt --upgrade
```

## LLM Occupant Agents (Optional)

The simulation includes an optional LLM-powered occupant agent system that simulates realistic human behavior in buildings using large language models.

### Features

- **Generative Agents**: Memory-driven agents with persistent personality and thermal preferences
- **Cognitive Loop**: Perceive → Retrieve → Reflect → Act cycle based on Stanford Generative Agents
- **Multi-Agent Consultation**: Agents consult each other for shared-space decisions (thermostat, windows)
- **Calendar System**: Meeting scheduling with RSVP support
- **Equipment & Lighting**: Agents control desk equipment and lighting based on presence
- **Semantic Memory**: Three-factor retrieval scoring (recency, relevance, importance)

### Quick Start

1. **Set your OpenAI API key**:
```bash
export OPENAI_API_KEY="your-key-here"
```

2. **Enable LLM agents in your config**:
```json
"llm_agents": {
  "enabled": true,
  "agent_model": "gpt-4o",
  "embed_model": "text-embedding-3-large",
  "decision_interval_minutes": 60,
  "agents": [
    {"id": "alice_001", "base_type": "alice_office_worker"},
    {"id": "bob_002", "base_type": "bob_office_worker"}
  ]
}
```

3. **Run the simulation**:
```bash
python main.py simulation_config.json
```

### Agent Base Types

Pre-configured agent personalities are stored in `agent_base_types/`:

```
agent_base_types/alice_office_worker/
├── scratch.json       # Working memory, thermal preferences
├── persona.md         # Personality traits
├── background.md      # Work experience
├── work_style.md      # Schedule preferences
└── relationships.md   # Colleague relationships
```

Each agent inherits from a base type but develops unique memories and behaviors during simulation.

### Configuration Options

```json
"llm_agents": {
  "enabled": true,                          // Enable/disable LLM agents
  "agent_model": "gpt-4o",                  // Model for decision making
  "embed_model": "text-embedding-3-large",  // Model for memory embeddings
  "decision_interval_minutes": 60,          // How often agents make decisions
  "reflection_threshold": 5,                // Events before triggering reflection
  "agents": [
    {
      "id": "alice_001",                    // Unique agent ID
      "base_type": "alice_office_worker",   // Personality template
      "desk": "Desk_A"                      // Optional: assigned desk
    }
  ]
}
```

### Agent Decision Process

At each decision interval, agents:

1. **Perceive**: Observe current zone temperature, time, and context
2. **Retrieve**: Query semantic memory for relevant past experiences
3. **Reflect**: Generate high-level insights if threshold reached
4. **Consult**: Discuss with colleagues for shared decisions (optional)
5. **Decide**: Choose actions (window, thermostat, equipment, lighting)
6. **Record**: Store decision and reasoning in memory

### Agent Output

When LLM agents are enabled, additional output is generated:

```
results/agents/YYYY-MM-DD/HH-MM-SS/
├── alice_001/
│   ├── scratch.json       # Final agent state
│   ├── nodes.json         # All memory events
│   └── embeddings.json    # Memory embeddings
└── bob_002/
    └── ...
```

### Disabling LLM Agents

To run simulations without LLM agents (default behavior):

```json
"llm_agents": {
  "enabled": false
}
```

Or simply omit the `llm_agents` section entirely.

---

## BESTEST Validation

### About ASHRAE Standard 140 (BESTEST)

ASHRAE Standard 140, also known as BESTEST (Building Energy Simulation Test), provides a standardized method for testing building energy simulation software. The standard includes diagnostic test cases that isolate specific building physics phenomena to verify simulation accuracy.

**Section 5.2 Cases** cover:
- **Base building cases** (600, 900): Low-mass and high-mass construction with south-facing windows
- **Window orientation variants** (620, 920): East/west glazing to test orientation handling
- **Shading cases** (610, 630, 910, 930): Overhangs and fins (requires geometric shading)
- **Setback cases** (640, 940): Thermostat scheduling (requires time-based setpoints)
- **Ventilation cases** (650, 950): Night ventilation strategies

Results are compared against reference ranges from multiple validated simulation programs including EnergyPlus, ESP-r, BLAST, DOE-2, SRES/SUN, and TRNSYS.

### Supported Test Cases

| Case | Description | Status |
|------|-------------|--------|
| 600 | Base case - low mass, south windows | ✅ Supported |
| 620 | East/west windows | ✅ Supported |
| 900 | High mass, south windows | ✅ Supported |
| 920 | High mass, east/west windows | ✅ Supported |

### Cases Under Development

| Case | Description | Feature Required |
|------|-------------|------------------|
| 610, 630, 910, 930 | Shading (overhang/fins) | Geometric shading model |
| 640, 940 | Night setback schedules | Time-dependent setpoint schedules |
| 650, 950 | Night ventilation | Time-dependent ventilation schedules |

**Note**: Cases 610/630/910/930 are partially implemented using reduced SHGC (0.55 vs 0.789) to approximate shading effects.

### Running Validation Tests

```bash
# Run all supported cases
python test_rig.py --bestest

# Run specific cases
python test_rig.py --bestest --cases 600 900

# Results location
results/bestest_suite/bestest_validation_report.md
```

## Limitations

### Current Limitations
- Single zone only (no multi-zone capability)
- No humidity/latent loads modeling
- No ground heat transfer (floor uses adiabatic or constant temperature boundary)
- Time-dependent schedules limited to occupied/unoccupied periods (no arbitrary time-based controls)
- Exterior surface solar absorption intentionally disabled for numerical stability

### Solar Radiation Status

**Implemented Features:**
- Surface-specific irradiance calculations based on window orientation ([bsm/solar/irradiance.py](bsm/solar/irradiance.py))
- Sun position tracking with solar zenith/azimuth ([bsm/solar/position.py](bsm/solar/position.py))
- Angle of incidence for tilted surfaces ([bsm/solar/incident_angle.py](bsm/solar/incident_angle.py))
- Perez anisotropic diffuse sky model ([bsm/solar/diffuse.py](bsm/solar/diffuse.py))
- Angular-dependent window transmittance per ASHRAE 140 Table 7-12 ([bsm/solar/angular_transmittance.py](bsm/solar/angular_transmittance.py))
- Sky temperature calculations for longwave radiation ([bsm/weather/sky_temperature.py](bsm/weather/sky_temperature.py))
- Exterior longwave radiation exchange ([bsm/heat_transfer/exterior_longwave.py](bsm/heat_transfer/exterior_longwave.py))

**Current Integration:**
- Window solar gains: Fully integrated with surface-specific irradiance and angular SHGC
- Longwave radiation: Integrated with sky temperature calculations
- Exterior surface solar absorption: Disabled (see [bsm/core/zone_solver.py:203-224](bsm/core/zone_solver.py#L203-L224) for rationale)

**Remaining Limitations:**
1. No geometric shading (overhangs, fins, obstructions)
2. Exterior opaque surface solar absorption disabled
3. Ground reflection uses fixed albedo (0.2)


## License

[Specify license here]

## References


## Contact

[Add contact information or repository links]

---

**Last Updated**: January 2026

### Changelog

**January 2026 (v1.0.0) - Major Restructuring:**
- **Codebase Restructure**: Reorganized 37+ root-level Python files into logical `bsm/` package hierarchy
- **LLM Agent System**: Added comprehensive documentation for the LLM-powered occupant agent system
- **Code Deduplication**: Consolidated shared Walton convection correlations into `bsm/heat_transfer/convection.py`
- **Modern Packaging**: Added `pyproject.toml` for modern Python packaging
- **Cleanup**: Removed empty `bestest_results/` directory

**Earlier January 2026:**
- Removed empty placeholder files: `devices.py`, `lighting.py`
- Added comprehensive input validation for window areas, weather data, and construction layers
- Made internal gains convective/radiative split configurable (default: 40/60 per ASHRAE)
- Updated documentation to reflect implemented solar calculation modules


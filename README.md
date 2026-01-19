# Building Simulation Model (BSM)

A physics-based single-zone building thermal simulation with LLM-powered occupant agents. Solves coupled heat balance equations to predict indoor temperature and energy consumption with realistic heat transfer physics.

## Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

```bash
git clone <repository-url>
cd BuildingSimulationModel
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run Your First Simulation

```bash
python main.py                              # Uses simulation_config.json
python main.py my_config.json               # Custom config
```

Results saved to `results/YYYY-MM-DD/` with CSV data, config copy, and plots.

### Run BESTEST Validation

```bash
python test_rig.py --bestest                # All supported cases
python test_rig.py --bestest --cases 600 900  # Specific cases
```

## Key Features

- **Coupled Heat Balance Solver** - Fully coupled fabric and air temperatures using finite-difference methods
- **Multi-Layer Constructions** - CondFD solver with automatic node limiting for stability
- **Advanced Convection Models** - Adaptive algorithms (DOE2, Sparrow, Walton, ASHRAE)
- **Multiple HVAC Systems** - IdealLoads, Stateful, PID control options
- **Solar Radiation** - Angular-dependent window transmittance per ASHRAE 140
- **Air Exchange** - ASHRAE AIM-2 infiltration model with wind and stack effects
- **Rule-Based Occupants** - Window opening and thermostat preferences
- **LLM Occupant Agents** - Memory-driven AI agents with cognitive decision loop
- **BESTEST Validation** - ASHRAE 140-2020 Section 5.2 test suite support
- **Real Weather Data** - Open-Meteo API integration for historical data

---

# Configuration Reference

Simulations are controlled via JSON configuration files. See [simulation_config.json](simulation_config.json) for a complete example.

## Simulation Settings

```json
"simulation_settings": {
  "dt_minutes": 1,
  "duration_days": 5,
  "start_date": "2020-01-01",
  "stabilization_days": 3,
  "hvac_coupling_mode": "auto",
  "hvac_max_power_rate_w_s": 50000.0,
  "enable_opaque_solar_absorption": true,
  "interior_radiation_method": "area_weighted",
  "use_adaptive_timestepping": true
}
```

| Setting | Description | Default |
|---------|-------------|---------|
| `dt_minutes` | Timestep in minutes | 1 |
| `duration_days` | Simulation duration | 5 |
| `start_date` | Start date for weather data | - |
| `stabilization_days` | Warm-up period before data collection | 3 |
| `hvac_coupling_mode` | "auto", "implicit", or "explicit" | "auto" |
| `enable_opaque_solar_absorption` | Enable solar absorption on exterior surfaces | true |
| `interior_radiation_method` | "area_weighted", "carroll", or "view_factor" | "area_weighted" |
| `use_adaptive_timestepping` | Auto-reduce timestep on convergence failure | true |

## Location

```json
"location": {
  "latitude": 50.9105,
  "longitude": -1.4049,
  "place_name": "Southampton",
  "time_zone": "GMT"
}
```

## Zone Properties

```json
"zone_properties": {
  "name": "Office_Main",
  "length": 10.0,
  "width": 8.0,
  "height": 2.8,
  "zone_sensible_heat_capacity_multiplier": 4.0
}
```

The multiplier accounts for furniture and internal thermal mass beyond air.

## Geometry & Surfaces

```json
"geometry": {
  "exterior_surfaces": ["north_wall", "south_wall", "roof"],
  "surface_definitions": {
    "north_wall": {
      "area": 22.4,
      "perimeter": 21.6,
      "tilt": 90,
      "azimuth": 0,
      "type": "wall",
      "roughness_index": 2,
      "construction_name": "ExteriorWall"
    }
  }
}
```

- **tilt**: 0=roof, 90=wall, 180=floor
- **azimuth**: 0=North, 90=East, 180=South, 270=West

## Materials & Constructions

```json
"materials": [
  {
    "name": "Concrete",
    "thickness": 0.1,
    "conductivity": 1.4,
    "density": 2300,
    "specific_heat": 880
  }
],
"constructions": {
  "ExteriorWall": {
    "layers": ["Concrete", "Insulation", "Concrete"]
  }
}
```

Layers ordered outside to inside.

## Windows

**Simple Window** (constant SHGC):
```json
"windows": [
  {
    "wall_name": "south_wall",
    "area": 6.0,
    "u_value": 1.5,
    "shgc": 0.5,
    "solar_distribution": {
      "floor": 0.6,
      "north_wall": 0.2
    }
  }
]
```

**Angular-Dependent Window** (ASHRAE 140 compliant):
```json
{
  "wall_name": "south_wall",
  "area": 12.0,
  "u_value": 3.0,
  "shgc": 0.789,
  "glass_fraction": 1.0,
  "angular_dependence": {
    "enabled": true,
    "hemispherical_avg_transmittance": 0.686,
    "solar_lost_fraction": 0.035
  }
}
```

## Air Exchange

**Flow Coefficient Model** (ASHRAE AIM-2):
```json
"air_exchange": {
  "infiltration": {
    "type": "flow_coefficient",
    "flow_coefficient_m3_s_Pa_n": 0.00025,
    "pressure_exponent_n": 0.65,
    "stack_coeff_Pa_K": 0.078,
    "wind_coeff_Pa_s2_m2": 0.15,
    "shelter_factor_s": 0.5
  },
  "ventilation": {
    "open_window_ach": 5.0
  }
}
```

**Constant ACH Model** (for BESTEST):
```json
"infiltration": {
  "type": "constant_ach",
  "constant_ach": 0.5
}
```

## HVAC System

Four model types: `IdealLoadsHVAC`, `VerySimpleHVAC`, `StatefulHVAC`, `PIDControlledHVAC`

**IdealLoadsHVAC** (for BESTEST validation):
```json
"hvac_system": {
  "model_type": "IdealLoadsHVAC",
  "heating_capacity_w": 1000000000,
  "cooling_capacity_w": 1000000000,
  "heating_setpoint_c": 20.0,
  "cooling_setpoint_c": 27.0,
  "proportional_gain_w_k": 100000
}
```

**PIDControlledHVAC** (realistic control):
```json
"hvac_system": {
  "model_type": "PIDControlledHVAC",
  "heating_capacity_w": 5000.0,
  "cooling_capacity_w": 5000.0,
  "kp": 100.0,
  "ki": 5.0,
  "kd": 10.0
}
```

## Schedules

```json
"schedules": {
  "occupied_hours": [7, 19],
  "occupied_heating_setpoint_c": 21.0,
  "unoccupied_heating_setpoint_c": 15.0,
  "occupied_cooling_setpoint_c": 24.0,
  "unoccupied_cooling_setpoint_c": 30.0,
  "occupied_internal_gains_w": 0.0,
  "unoccupied_internal_gains_w": 0.0
}
```

## Occupancy (Rule-Based)

```json
"occupancy": {
  "heat_gain_per_occupant_w": 120.0,
  "check_interval_minutes": 60,
  "thermostat_adjustment_c": 1.0,
  "occupants": [
    {
      "name": "Alice",
      "work_start_hr": 8,
      "work_end_hr": 17,
      "window_preference": "opener",
      "window_temp_c": 23.0,
      "thermostat_preference": "changer",
      "thermostat_temp_c": 21.0
    }
  ]
}
```

Window preferences: `"opener"`, `"neutral"`, `"closer"`

## LLM Occupant Agents

LLM-powered agents with memory-driven behavior. Requires OpenAI API key.

```bash
export OPENAI_API_KEY="your-key-here"
```

```json
"llm_agents": {
  "enabled": true,
  "api_timeout_seconds": 30,
  "agent_model": "gpt-4o",
  "embed_model": "text-embedding-3-large",
  "decision_interval_minutes": 60,
  "daily_planning_hour": 0,
  "agent_source": {
    "mode": "fresh",
    "base_types_folder": "agent_base_types",
    "continue_from_run": null
  },
  "agents": [
    {"id": "alice_001", "base_type": "alice_office_worker"},
    {"id": "bob_002", "base_type": "bob_office_worker"}
  ]
}
```

### Agent Base Types

Agents are initialized from folders in `agent_base_types/`:

```
agent_base_types/alice_office_worker/
├── scratch.json       # Working memory, preferences, relationship models
├── persona.md         # Core personality traits
├── background.md      # Work experience and history
├── work_style.md      # Schedule preferences
├── relationships.md   # Colleague relationships
└── memory_stream/     # Semantic memory storage
    ├── nodes.json
    └── embeddings.json
```

### Agent Source Modes

- **`fresh`**: Copy from `base_types_folder` to new run directory
- **`continue`**: Resume from previous simulation run (set `continue_from_run`)

### Cognitive Loop

At each decision interval, agents execute: **Perceive** → **Retrieve** → **Reflect** → **Consult** → **Act**

## Desks, Equipment & Lighting

For LLM agents, define the office layout:

```json
"desks": {
  "Desk_A": {
    "equipment": ["laptop_A", "monitor_A"],
    "light": "desk_light_A"
  }
},

"equipment": {
  "default_power_w": {
    "laptop": 50,
    "monitor": 30,
    "photocopier_standby": 20,
    "photocopier_active": 400
  },
  "shared_equipment": ["photocopier"],
  "items": {
    "laptop_A": {"type": "laptop", "location": "Desk_A"},
    "photocopier": {"type": "photocopier", "location": "shared"}
  }
},

"lighting": {
  "zones": {
    "desk_light_A": {"power_w": 15, "location": "Desk_A"},
    "zone_main": {"power_w": 200, "location": "shared"}
  }
},

"meeting_room": {
  "capacity": 6,
  "equipment": ["projector", "conference_phone"],
  "light": "meeting_room"
}
```

## Weather

**File-based** (fetches from Open-Meteo API):
```json
"weather": {
  "type": "file"
}
```

**Sinusoidal** (for testing):
```json
"weather": {
  "type": "sinusoidal",
  "temp_base_c": -10,
  "temp_amplitude_c": 3,
  "temp_phase_shift_hr": 14,
  "solar_max_irradiance_w_m2": 400,
  "wind_Speed_ms": 20.0
}
```

## Convection Models

```json
"convection_models": {
  "exterior_hf": {
    "RoofStable": "DOE2Windward",
    "VerticalWallWindward": "DOE2Windward",
    "VerticalWallLeeward": "DOE2Leeward"
  },
  "exterior_hn": {
    "RoofStable": "WaltonStableHorizontalOrTilt",
    "VerticalWallWindward": "ASHRAEVerticalWall"
  },
  "interior": {
    "VerticalWall": "ASHRAEVerticalWall",
    "UnstableHorizontal": "WaltonUnstableHorizontalOrTilt",
    "StableHorizontal": "WaltonStableHorizontalOrTilt"
  }
}
```

Available models: `DOE2Windward`, `DOE2Leeward`, `SparrowWindward`, `SparrowLeeward`, `ASHRAEVerticalWall`, `WaltonStableHorizontalOrTilt`, `WaltonUnstableHorizontalOrTilt`

---

# Project Structure

```
BuildingSimulationModel/
├── main.py                      # CLI entry point
├── test_rig.py                  # BESTEST CLI entry point
├── simulation_config.json       # Example configuration
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Modern Python packaging
│
├── bsm/                         # Main simulation package
│   ├── __init__.py
│   ├── runner.py                # Simulation execution logic
│   ├── test_rig.py              # BESTEST test framework
│   │
│   ├── core/                    # Core simulation engine
│   │   ├── zone.py              # Zone model manager
│   │   ├── zone_solver.py       # Coupled heat balance solver
│   │   ├── fabric_heat_transfer.py  # CondFD solver
│   │   └── constants.py
│   │
│   ├── heat_transfer/           # Convection and radiation
│   │   ├── convection.py        # Shared correlations
│   │   ├── exterior_convection.py
│   │   ├── interior_convection.py
│   │   ├── exterior_longwave.py
│   │   └── interior_longwave.py
│   │
│   ├── solar/                   # Solar calculations
│   │   ├── position.py          # Sun position
│   │   ├── irradiance.py        # Surface irradiance
│   │   ├── diffuse.py           # Perez sky model
│   │   └── angular_transmittance.py
│   │
│   ├── weather/                 # Weather data
│   │   ├── generators.py        # EPW/API weather loaders
│   │   ├── api_client.py        # Open-Meteo client
│   │   ├── sky_temperature.py
│   │   └── ground_temperature.py
│   │
│   ├── components/              # Building components
│   │   ├── windows.py
│   │   ├── hvac.py
│   │   ├── air_exchange.py
│   │   └── materials.py
│   │
│   ├── boundary/                # Boundary conditions
│   │   ├── conditions.py        # Schedules
│   │   └── occupants.py         # Rule-based occupants
│   │
│   ├── agents/                  # LLM occupant agents
│   │   ├── manager.py           # Main orchestrator
│   │   ├── generative_agent.py  # Memory-driven agent
│   │   ├── skeleton.py          # SDK types and tools
│   │   ├── simulation_adapter.py
│   │   ├── equipment_manager.py
│   │   ├── lighting_manager.py
│   │   ├── desk_manager.py
│   │   ├── memory/              # Memory subsystem
│   │   │   └── stream.py
│   │   └── cognition/           # Cognitive modules
│   │       ├── modules.py
│   │       └── conversation.py
│   │
│   └── output/                  # Output and reporting
│       ├── plotting.py
│       └── report_generator.py
│
├── agent_base_types/            # Pre-configured agent personalities
│   ├── alice_office_worker/
│   ├── bob_office_worker/
│   └── charlie_office_worker/
│
├── bestest/                     # BESTEST validation suite
│   ├── cases/                   # Test case definitions
│   ├── utils/                   # Config generation
│   └── weather/                 # EPW files
│
├── test_configs/                # Test configurations
├── Documentation/               # Reference materials
│
└── results/                     # Output directory (auto-created)
    ├── YYYY-MM-DD/              # Individual simulation runs
    ├── bestest_suite/           # BESTEST results
    └── agents/                  # LLM agent output
```

---

# Physics Models

## Heat Balance Equation

```
C_air · dT_air/dt = Q_fabric + Q_windows + Q_infiltration + Q_ventilation + Q_internal + Q_solar + Q_HVAC
```

## Fabric Heat Transfer

Multi-layer constructions solved using Conduction Finite-Difference (CondFD):
- Automatic spatial discretization (Fourier stability criterion)
- Node limiting (default 20 per layer)
- Fully coupled with zone air temperature

## Air Infiltration (ASHRAE AIM-2)

```
ΔP_stack = C_s · |T_zone - T_ext|
ΔP_wind = C_w · (s · v_wind)²
Q = C · (ΔP_total)^n
```

## Numerical Stability

- **Adaptive Timestepping**: Auto-reduces timestep (up to 16x) on convergence failure
- **Hybrid HVAC Coupling**: Implicit for high-gain, explicit with rate limiting for complex systems
- **Under-relaxation**: Damping factor (0.7) prevents oscillations

---

# BESTEST Validation

ASHRAE Standard 140-2020 Section 5.2 test suite.

| Case | Description | Status |
|------|-------------|--------|
| 600 | Low mass, south windows | Supported |
| 620 | East/west windows | Supported |
| 900 | High mass, south windows | Supported |
| 920 | High mass, east/west windows | Supported |

Results compared against EnergyPlus, ESP-r, BLAST, DOE-2, TRNSYS reference ranges.

---

# Output Format

## CSV Columns

| Column | Units |
|--------|-------|
| Time (hrs) | hours |
| Zone Temp (C) | °C |
| Outside Temp (C) | °C |
| HVAC Power (W) | Watts |
| Fabric Loss (W) | Watts |
| Air Exchange Loss (W) | Watts |
| Solar Gains (W) | Watts |
| Internal Gains (W) | Watts |
| Window State (0-1) | - |

## File Naming

```
results/YYYY-MM-DD/YYYY-MM-DD_HH-MM-SS_Xdays_results.csv
results/YYYY-MM-DD/YYYY-MM-DD_HH-MM-SS_Xdays_config.json
```

---

# Troubleshooting

## Solver Convergence Warnings
Adaptive timestepping handles most cases automatically. If persists:
- Reduce `dt_minutes`
- Check material properties are realistic
- Verify construction layers ordered correctly

## Temperature Out of Bounds
- Check HVAC capacities sufficient for loads
- Increase `stabilization_days`
- For BESTEST: use `hvac_coupling_mode: "auto"` or `"implicit"`

## Weather Data Download Fails
- Check internet connection
- Verify latitude/longitude valid
- Check `start_date` not in future

## LLM Agent Errors
- Set `OPENAI_API_KEY` environment variable
- Verify `base_type` folders exist in `agent_base_types/`
- Check API rate limits

---

# Limitations

- Single zone only (no multi-zone)
- No humidity/latent loads
- No ground heat transfer modeling
- Time-dependent schedules limited to occupied/unoccupied
- Geometric shading (overhangs, fins) under development

---

# Unit Conventions

| Property | Unit |
|----------|------|
| Temperature | °C |
| Length | meters |
| Area | m² |
| Power | Watts |
| Conductivity | W/(m·K) |
| Density | kg/m³ |
| Specific Heat | J/(kg·K) |
| Angles | degrees |

---

# License

Apache License

---

**Last Updated**: January 2026

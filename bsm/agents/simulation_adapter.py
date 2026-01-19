"""
simulation_adapter.py

Production implementation of BuildingSimulationAdapter that connects LLM agents
to the actual building simulation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from bsm.agents.skeleton import BuildingSimulationAdapter, OccupantStepDecision, OccupantAction
from bsm.agents.equipment_manager import EquipmentManager
from bsm.agents.lighting_manager import LightingManager
from bsm.agents.desk_manager import DeskManager


class ZoneStateProvider:
    """
    Provides real-time zone state to the simulation adapter.

    Bridges the gap between zone_solver internal state and
    the JSON-friendly format needed by agents.
    """

    def __init__(
        self,
        get_zone_temp: Callable[[], float],
        get_weather: Callable[[], Dict[str, Any]],
        zone_name: str = "Office_Main",
    ):
        """
        Initialize with callables that return current state.

        Args:
            get_zone_temp: Function returning current zone air temperature (C)
            get_weather: Function returning current weather dict
            zone_name: Name of the zone
        """
        self._get_zone_temp = get_zone_temp
        self._get_weather = get_weather
        self.zone_name = zone_name

    def get_thermal_state(self) -> Dict[str, Any]:
        """Get current thermal state."""
        weather = self._get_weather()
        return {
            "indoor_temp_c": round(self._get_zone_temp(), 1),
            "outdoor_temp_c": round(weather.get("air_temp_c", 10.0), 1),
            "wind_speed_ms": round(weather.get("wind_speed_local_ms", 0.0), 1),
            "solar_irradiance_w_m2": round(weather.get("solar_irradiance_w_m2", 0.0), 0),
            "is_sunny": weather.get("is_sunlit", False),
        }

    def get_weather_description(self) -> str:
        """Get human-readable weather description."""
        weather = self._get_weather()
        temp = weather.get("air_temp_c", 10.0)
        wind = weather.get("wind_speed_local_ms", 0.0)
        solar = weather.get("solar_irradiance_w_m2", 0.0)

        desc_parts = []
        if temp < 5:
            desc_parts.append("cold")
        elif temp < 15:
            desc_parts.append("cool")
        elif temp < 25:
            desc_parts.append("mild")
        else:
            desc_parts.append("warm")

        if solar > 500:
            desc_parts.append("sunny")
        elif solar > 100:
            desc_parts.append("partly cloudy")
        else:
            desc_parts.append("overcast")

        if wind > 10:
            desc_parts.append("windy")
        elif wind > 5:
            desc_parts.append("breezy")

        return ", ".join(desc_parts)


class ProductionSimulationAdapter(BuildingSimulationAdapter):
    """
    Production implementation of BuildingSimulationAdapter.

    Connects LLM agent decisions to actual simulation state.
    Handles multi-agent coordination through voting for shared controls.
    """

    def __init__(
        self,
        zone_state_provider: ZoneStateProvider,
        equipment_manager: EquipmentManager,
        lighting_manager: LightingManager,
        desk_manager: DeskManager,
        base_heating_setpoint_c: float = 21.0,
        base_cooling_setpoint_c: float = 24.0,
        room_layout: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the adapter.

        Args:
            zone_state_provider: Provides current zone thermal state
            equipment_manager: Manages equipment state
            lighting_manager: Manages lighting state
            desk_manager: Manages desk assignments
            base_heating_setpoint_c: Base heating setpoint
            base_cooling_setpoint_c: Base cooling setpoint
            room_layout: Optional room layout configuration for agent context
        """
        self.zone_state = zone_state_provider
        self.equipment = equipment_manager
        self.lighting = lighting_manager
        self.desks = desk_manager
        self._room_layout = room_layout or {}

        # Current control states
        self._base_heating_setpoint_c = base_heating_setpoint_c
        self._base_cooling_setpoint_c = base_cooling_setpoint_c
        self._thermostat_offset_c = 0.0
        self._window_open_fraction = 0.0

        # Voting accumulators for shared controls
        self._thermostat_votes: List[Tuple[str, float]] = []  # (occupant_id, desired_setpoint)
        self._window_votes: List[Tuple[str, bool]] = []  # (occupant_id, wants_open)

        # Track occupant presence
        self._present_occupants: Dict[str, bool] = {}  # occupant_id -> is_present

    def _build_room_context(
        self,
        current_desk: Optional[str],
        available_desks: List[str]
    ) -> Dict[str, Any]:
        """Build room context dict from room_layout config."""
        desk_descriptions = {}
        desks_config = self._room_layout.get("desks", {})

        for desk_name, desk_info in desks_config.items():
            position = desk_info.get("position", "")
            characteristics = desk_info.get("characteristics", [])
            char_str = ", ".join(characteristics[:2]) if characteristics else ""
            desk_descriptions[desk_name] = f"{position}" + (f" ({char_str})" if char_str else "")

        current_desk_info = {}
        if current_desk and current_desk in desks_config:
            current_desk_info = desks_config[current_desk]

        return {
            "room_description": self._room_layout.get("description", "Office space"),
            "available_desks": available_desks,
            "desk_descriptions": desk_descriptions,
            "your_desk_position": current_desk_info.get("position", "") if current_desk_info else None,
            "your_desk_characteristics": current_desk_info.get("characteristics", []) if current_desk_info else [],
        }

    def get_state(self, occupant_id: str, now: datetime) -> Dict[str, Any]:
        """
        Build complete state dict for agent prompt.

        Returns a comprehensive state including thermal conditions,
        desk/equipment status, and information about other occupants.
        """
        # Get thermal state
        thermal = self.zone_state.get_thermal_state()

        # Get desk/occupancy state for this occupant
        occupancy = self.desks.get_occupancy_for_agent(occupant_id)
        current_desk = occupancy.get("current_desk")

        # Get equipment state for this occupant
        equipment = self.equipment.get_equipment_for_occupant(occupant_id, current_desk)

        # Get lighting state for this occupant
        lighting = self.lighting.get_lighting_for_occupant(current_desk)

        # Calculate lighting conditions based on weather and time
        hour = now.hour
        is_sunny = thermal.get("is_sunny", False)

        # Estimate illuminance based on weather and time of day
        base_illuminance = 300 if is_sunny else 100
        if 6 <= hour <= 18:
            # Sunrise to sunset factor
            if hour <= 12:
                outdoor_factor = (hour - 6) / 6  # 0 at 6am, 1 at noon
            else:
                outdoor_factor = (18 - hour) / 6  # 1 at noon, 0 at 6pm
        else:
            outdoor_factor = 0
        estimated_illuminance_lux = int(base_illuminance * max(0.1, outdoor_factor))

        # Determine natural light level category
        if estimated_illuminance_lux > 300:
            natural_light_level = "bright"
        elif estimated_illuminance_lux > 150:
            natural_light_level = "moderate"
        elif estimated_illuminance_lux > 50:
            natural_light_level = "dim"
        else:
            natural_light_level = "dark"

        # Get desk light state
        desk_light_on = lighting.get("desk_light", {}).get("is_on", False) if lighting.get("desk_light") else False

        lighting_conditions = {
            "estimated_illuminance_lux": estimated_illuminance_lux,
            "natural_light_level": natural_light_level,
            "desk_light_on": desk_light_on,
        }

        # Build equipment status for agent context
        desk_eq = equipment.get("desk_equipment", {})
        equipment_states = {name: info.get("is_on", False) for name, info in desk_eq.items()}

        equipment_status = {
            "items": equipment_states,
            "all_on": all(equipment_states.values()) if equipment_states else True,
            "all_off": not any(equipment_states.values()) if equipment_states else True,
        }

        # Build other occupants list
        other_occupants = []
        for occ in occupancy.get("other_occupants", []):
            if self._present_occupants.get(occ["id"], False):
                other_occupants.append(occ["id"])

        # Day of week information
        day_of_week = now.strftime("%A")  # "Monday", "Tuesday", etc.
        is_weekend = now.weekday() >= 5   # Saturday=5, Sunday=6

        return {
            # Time and identity
            "datetime": now.isoformat(),
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "is_workday": not is_weekend,
            "occupant_id": occupant_id,
            "zone": self.zone_state.zone_name,

            # Thermal state
            "indoor_temp_c": thermal["indoor_temp_c"],
            "outdoor_temp_c": thermal["outdoor_temp_c"],
            "weather_description": self.zone_state.get_weather_description(),
            "is_sunny": thermal["is_sunny"],

            # Control states
            "thermostat_setpoint_c": self._base_heating_setpoint_c + self._thermostat_offset_c,
            "window_open": self._window_open_fraction > 0.5,
            "window_open_fraction": self._window_open_fraction,

            # Desk and location
            "desk_options": occupancy.get("desk_options", []),
            "current_desk": current_desk,
            "available_desks": occupancy.get("available_desks", []),
            "desk_occupancy": occupancy.get("desk_occupancy", {}),
            "in_meeting_room": occupancy.get("in_meeting_room", False),
            "meeting_room_occupants": occupancy.get("meeting_room_occupants", []),

            # Equipment
            "equipment_available": equipment.get("available_equipment", []),
            "desk_equipment": equipment.get("desk_equipment", {}),
            "shared_equipment": equipment.get("shared_equipment", {}),

            # Lighting
            "desk_light": lighting.get("desk_light"),
            "zone_lights": lighting.get("zone_lights", {}),

            # Other occupants
            "other_occupants_present": other_occupants,
            "total_occupants_present": len(other_occupants) + (1 if self._present_occupants.get(occupant_id, False) else 0),

            # Lighting conditions (for agent decision-making)
            "lighting_conditions": lighting_conditions,

            # Equipment status (for agent decision-making)
            "equipment_status": equipment_status,

            # Room context (layout and desk information)
            "room_context": self._build_room_context(current_desk, occupancy.get("available_desks", [])),
        }

    def apply_decision(self, decision: OccupantStepDecision) -> None:
        """
        Apply agent decision to simulation state.

        Handles all action types and collects votes for shared controls.
        """
        occupant_id = decision.occupant_id
        current_desk = self.desks.get_desk_for_occupant(occupant_id)

        for action in decision.actions:
            self._apply_action(action, occupant_id, current_desk)

    def _apply_action(
        self,
        action: OccupantAction,
        occupant_id: str,
        current_desk: Optional[str]
    ) -> None:
        """Apply a single action."""
        action_type = action.action_type
        params = action.parameters

        if action_type == "no_op":
            pass

        elif action_type == "thermostat_adjust":
            # Record vote for thermostat
            setpoint = params.get("setpoint_c")
            if setpoint is not None:
                self._thermostat_votes.append((occupant_id, float(setpoint)))

        elif action_type == "window_set":
            # Record vote for window
            wants_open = params.get("open", False)
            self._window_votes.append((occupant_id, wants_open))

        elif action_type == "lights_set":
            self.lighting.apply_lighting_action(params, occupant_id, current_desk)

        elif action_type == "choose_desk":
            desk_name = params.get("desk_name")
            if desk_name:
                # Just assign desk - agent decides when to turn on equipment
                self.desks.assign_desk(occupant_id, desk_name)

        elif action_type == "equipment_set":
            self.equipment.apply_equipment_action(params, occupant_id)

        elif action_type == "attend_meeting":
            self.desks.enter_meeting_room(occupant_id)
            # Turn on meeting room lights and equipment if first person
            if self.desks.get_meeting_room().get_occupant_count() == 1:
                self.lighting.turn_on_light("meeting_room")
                self.equipment.turn_on_equipment("projector", occupant_id)
                self.equipment.turn_on_equipment("conference_phone", occupant_id)

        elif action_type == "leave_meeting":
            self.desks.leave_meeting_room(occupant_id)
            # Turn off meeting room if empty
            if self.desks.get_meeting_room().get_occupant_count() == 0:
                self.lighting.turn_off_light("meeting_room")
                self.equipment.turn_off_equipment("projector")
                self.equipment.turn_off_equipment("conference_phone")

        elif action_type == "use_photocopier":
            is_using = params.get("using", True)
            if is_using:
                self.equipment.turn_on_equipment("photocopier", occupant_id)
            else:
                self.equipment.turn_off_equipment("photocopier")

        elif action_type == "arrive":
            self._present_occupants[occupant_id] = True
            # Assign to default desk if available
            # NOTE: Equipment and lights are NOT auto-turned on.
            # The agent decides to turn them on via their cognitive loop.
            desk_name = params.get("desk_name")
            if desk_name and desk_name in self.desks.get_available_desks():
                self.desks.assign_desk(occupant_id, desk_name)

        elif action_type == "depart":
            self._handle_occupant_departure(occupant_id)

    def _handle_occupant_departure(self, occupant_id: str) -> None:
        """Handle an occupant leaving the building."""
        self._present_occupants[occupant_id] = False

        # Get current desk before releasing
        current_desk = self.desks.get_desk_for_occupant(occupant_id)
        if current_desk:
            # Turn off equipment at desk
            self.equipment.turn_off_desk_equipment(current_desk)
            self.lighting.turn_off_desk_light(current_desk)

        # Release desk and leave meeting room
        self.desks.occupant_leaves_building(occupant_id)

        # Release any shared equipment
        self.equipment.release_occupant_equipment(occupant_id)

    def resolve_votes(self) -> Tuple[float, float]:
        """
        Resolve all pending votes for thermostat and windows.

        Returns:
            (thermostat_offset_c, window_open_fraction)
        """
        # Resolve thermostat votes
        if self._thermostat_votes:
            # Average of all desired setpoints
            avg_setpoint = sum(v[1] for v in self._thermostat_votes) / len(self._thermostat_votes)
            # Calculate offset from base
            raw_offset = avg_setpoint - self._base_heating_setpoint_c
            # Cap at +/- 2C
            self._thermostat_offset_c = max(-2.0, min(2.0, raw_offset))
            self._thermostat_votes.clear()

        # Resolve window votes
        if self._window_votes:
            open_votes = sum(1 for _, wants_open in self._window_votes if wants_open)
            close_votes = len(self._window_votes) - open_votes

            if open_votes > close_votes:
                self._window_open_fraction = 1.0
            elif close_votes > open_votes:
                self._window_open_fraction = 0.0
            # Tie = no change

            self._window_votes.clear()

        return self._thermostat_offset_c, self._window_open_fraction

    def get_thermostat_offset(self) -> float:
        """Get current thermostat offset from base setpoint."""
        return self._thermostat_offset_c

    def get_window_open_fraction(self) -> float:
        """Get current window open fraction (0-1)."""
        return self._window_open_fraction

    def set_occupant_present(self, occupant_id: str, is_present: bool) -> None:
        """Set occupant presence status."""
        self._present_occupants[occupant_id] = is_present
        if not is_present:
            self._handle_occupant_departure(occupant_id)

    def is_occupant_present(self, occupant_id: str) -> bool:
        """Check if occupant is present."""
        return self._present_occupants.get(occupant_id, False)

    def get_present_occupant_ids(self) -> List[str]:
        """Get list of present occupant IDs."""
        return [oid for oid, present in self._present_occupants.items() if present]

    def get_total_equipment_power_w(self) -> float:
        """Get total equipment power consumption."""
        return self.equipment.calculate_total_power_w()

    def get_total_equipment_heat_gain_w(self) -> float:
        """Get total equipment heat gain to zone (power * heat_gain_fraction)."""
        return self.equipment.calculate_total_heat_gain_w()

    def get_total_lighting_power_w(self) -> float:
        """Get total lighting power consumption."""
        return self.lighting.calculate_total_power_w()

    def get_total_lighting_heat_gain_w(self) -> float:
        """Get total lighting heat gain to zone (power * heat_gain_fraction)."""
        return self.lighting.calculate_total_heat_gain_w()

    def get_total_occupant_gains_w(self, metabolic_w_per_person: float = 120.0) -> float:
        """
        Get total internal gains from occupants (heat delivered to zone).

        Uses heat_gain_fraction for equipment and lighting to account for
        energy that doesn't become zone heat (visible light through windows,
        exhaust air, etc.).

        Args:
            metabolic_w_per_person: Metabolic heat per person (default 120W sensible)

        Returns:
            Total equipment_heat_gain + lighting_heat_gain + metabolic heat (W)
        """
        num_present = len(self.get_present_occupant_ids())
        metabolic = num_present * metabolic_w_per_person
        equipment_heat = self.get_total_equipment_heat_gain_w()
        lighting_heat = self.get_total_lighting_heat_gain_w()
        return metabolic + equipment_heat + lighting_heat

    def reset_day(self) -> None:
        """Reset state at start of new day."""
        self._thermostat_offset_c = 0.0
        self._window_open_fraction = 0.0
        self._thermostat_votes.clear()
        self._window_votes.clear()
        self._present_occupants.clear()
        self.desks.reset_all()
        self.equipment.reset_all()
        self.lighting.reset_all()

    def start_of_day_setup(self) -> None:
        """Set up for start of occupied period."""
        # Turn on main zone lighting
        self.lighting.set_zone_lighting(True)

    def end_of_day_cleanup(self) -> None:
        """Clean up at end of occupied period."""
        # Turn off all lights and equipment
        self.lighting.reset_all()
        self.equipment.reset_all()
        self._present_occupants.clear()

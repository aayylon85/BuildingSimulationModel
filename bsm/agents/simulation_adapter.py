"""
simulation_adapter.py

Production implementation of BuildingSimulationAdapter that connects LLM agents
to the actual building simulation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

# Setup module logger
logger = logging.getLogger(__name__)

from bsm.agents.skeleton import (
    BuildingSimulationAdapter,
    OccupantStepDecision,
    OccupantAction,
    StepDecisions,
)
from bsm.agents.equipment_manager import EquipmentManager
from bsm.agents.lighting_manager import LightingManager
from bsm.agents.desk_manager import DeskManager


# ---------------------------------------------------------------------------
# StepDecisions to OccupantAction Converter
# ---------------------------------------------------------------------------

# Kitchen appliances that use use_appliance action type
KITCHEN_APPLIANCES = {"kettle", "coffee_machine", "microwave"}


def convert_step_decisions_to_actions(decision: StepDecisions) -> List[OccupantAction]:
    """
    Convert structured StepDecisions to OccupantAction list for simulation.

    This bridges the gap between:
    - Structured output (mandatory category decisions with reasoning)
    - Simulation adapter (OccupantAction list)

    Args:
        decision: StepDecisions from agent

    Returns:
        List of OccupantAction objects for simulation adapter
    """
    actions: List[OccupantAction] = []

    # Thermostat
    if decision.thermostat.action == "adjust":
        direction = decision.thermostat.adjustment_direction
        amount = decision.thermostat.adjustment_amount or 1.0
        # Convert direction + amount to setpoint change
        if direction == "warmer":
            setpoint_delta = amount
        elif direction == "cooler":
            setpoint_delta = -amount
        else:
            setpoint_delta = 0
        if setpoint_delta != 0:
            actions.append(OccupantAction(
                action_type="thermostat_adjust",
                parameters={
                    "direction": direction,
                    "amount": amount,
                    "setpoint_delta_c": setpoint_delta,
                },
                confidence=0.8,
            ))

    # Lighting
    if decision.lighting.action in ("turn_on", "turn_off", "adjust_brightness"):
        light_on = decision.lighting.action in ("turn_on", "adjust_brightness")
        actions.append(OccupantAction(
            action_type="lights_set",
            parameters={
                "action": decision.lighting.action,
                "target": decision.lighting.target_device or "desk_light",
                "on": light_on,
                "brightness": decision.lighting.brightness_level,
            },
            confidence=0.8,
        ))

    # Equipment - process each decision separately
    for eq_decision in decision.equipment_decisions:
        if eq_decision.action == "turn_on":
            equipment_name = eq_decision.equipment_name
            if equipment_name.lower() in KITCHEN_APPLIANCES:
                actions.append(OccupantAction(
                    action_type="use_appliance",
                    parameters={"appliance_name": equipment_name},
                    confidence=0.8,
                ))
            else:
                actions.append(OccupantAction(
                    action_type="equipment_set",
                    parameters={"equipment_name": equipment_name, "on": True},
                    confidence=0.8,
                ))
        elif eq_decision.action == "turn_off":
            actions.append(OccupantAction(
                action_type="equipment_set",
                parameters={"equipment_name": eq_decision.equipment_name, "on": False},
                confidence=0.8,
            ))

    # Location
    if decision.location.action == "move" and decision.location.destination:
        actions.append(OccupantAction(
            action_type="move_to",
            parameters={"destination": decision.location.destination},
            confidence=0.8,
        ))

    # Conversation
    if decision.conversation.action == "initiate":
        actions.append(OccupantAction(
            action_type="initiate_conversation",
            parameters={
                "target_agent": decision.conversation.target_agent,  # Match schema field name
                "topic": decision.conversation.topic,
            },
            confidence=0.8,
        ))

    # Plan update
    if decision.plan_update.action == "update" and decision.plan_update.updates:
        actions.append(OccupantAction(
            action_type="update_daily_plan",
            parameters={
                "updates": decision.plan_update.updates,
                "reason": decision.plan_update.reasoning,
            },
            confidence=0.8,
        ))

    # Break decision (if present)
    if decision.break_decision:
        if decision.break_decision.action == "take_break":
            actions.append(OccupantAction(
                action_type="take_break",
                parameters={
                    "location": decision.break_decision.break_type,
                    "activity": decision.break_decision.activity,
                },
                confidence=0.8,
            ))
        elif decision.break_decision.action == "go_out_for_break":
            actions.append(OccupantAction(
                action_type="go_out_for_break",
                parameters={
                    "activity": decision.break_decision.activity or "fresh_air",
                },
                confidence=0.8,
            ))
        elif decision.break_decision.action == "return_from_lunch":
            actions.append(OccupantAction(
                action_type="return_from_lunch",
                parameters={},
                confidence=0.8,
            ))
        elif decision.break_decision.action == "return_from_break":
            actions.append(OccupantAction(
                action_type="return_from_break",
                parameters={},
                confidence=0.8,
            ))

    # If no actions, add explicit no_op
    if not actions:
        actions.append(OccupantAction(
            action_type="no_op",
            parameters={},
            confidence=1.0,
        ))

    return actions


# ---------------------------------------------------------------------------
# Office Location System
# ---------------------------------------------------------------------------

# Default office locations - can be extended via config
DEFAULT_OFFICE_LOCATIONS = {
    "desk_area": {
        "name": "Desk Area",
        "description": "Main open plan workspace with desks",
        "equipment_access": ["desk_equipment"],
    },
    "meeting_room": {
        "name": "Meeting Room",
        "description": "Enclosed meeting room with projector and conference phone",
        "capacity": 6,
        "equipment_access": ["projector", "conference_phone", "meeting_room_lights"],
    },
    "break_area": {
        "name": "Break Area / Kitchen",
        "description": "Kitchen facilities with seating for breaks and lunch",
        "appliances": ["coffee_machine", "kettle", "microwave", "fridge"],
    },
    "shared_area": {
        "name": "Shared Area",
        "description": "Common area near Desk_B with photocopier",
        "equipment_access": ["photocopier"],
    },
    "entrance": {
        "name": "Entrance",
        "description": "Main entrance, near Desk_A",
    },
}


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

        # Track agent status (at_desk, at_lunch, on_break, out_of_office)
        # Default status when present: at_desk=True, others=False
        self._agent_status: Dict[str, Dict[str, bool]] = {}  # occupant_id -> status dict

        # Track agent locations (simple location-based navigation)
        self._agent_locations: Dict[str, str] = {}  # occupant_id -> location_name
        self._office_locations = DEFAULT_OFFICE_LOCATIONS.copy()

        # Track whether agent has chosen desk for the day (prevents mid-day desk changes)
        self._desk_chosen_today: Dict[str, bool] = {}  # occupant_id -> has_chosen_desk

        # Extend with config-provided locations if available
        config_locations = self._room_layout.get("locations", {})
        self._office_locations.update(config_locations)

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
        shared_eq = equipment.get("shared_equipment", {})

        # Desk equipment items (simple name: on/off)
        desk_equipment_items = {name: info.get("is_on", False) for name, info in desk_eq.items()}

        # Shared equipment with location context
        shared_equipment_items = {}
        for name, info in shared_eq.items():
            shared_equipment_items[name] = {
                "is_on": info.get("is_on", False),
                "in_use_by": info.get("in_use_by"),
            }

        equipment_status = {
            "items": desk_equipment_items,  # Desk equipment for backward compatibility
            "desk_equipment": desk_equipment_items,
            "shared_equipment": shared_equipment_items,  # Meeting room, shared area equipment
        }

        # Build other occupants list with status
        other_occupants = []
        other_occupants_at_lunch = []
        other_occupants_on_break = []
        for occ in occupancy.get("other_occupants", []):
            occ_id = occ["id"]
            if self._present_occupants.get(occ_id, False):
                status = self.get_agent_status(occ_id)
                other_occupants.append(occ_id)
                if status.get("at_lunch", False):
                    other_occupants_at_lunch.append(occ_id)
                if status.get("on_break", False):
                    other_occupants_on_break.append(occ_id)

        # Get this agent's current status
        agent_status = self.get_agent_status(occupant_id)

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
            "thermostat": {
                "heating_setpoint_c": self._base_heating_setpoint_c + self._thermostat_offset_c,
                "cooling_setpoint_c": self._base_cooling_setpoint_c + self._thermostat_offset_c,
                "offset_c": self._thermostat_offset_c,
            },
            # Keep for backward compatibility
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
            "other_occupants_at_lunch": other_occupants_at_lunch,
            "other_occupants_on_break": other_occupants_on_break,
            "total_occupants_present": len(other_occupants) + (1 if self._present_occupants.get(occupant_id, False) else 0),

            # Agent's current status (at_desk, at_lunch, on_break, out_of_office)
            "agent_status": agent_status,

            # Lighting conditions (for agent decision-making)
            "lighting_conditions": lighting_conditions,

            # Equipment status (for agent decision-making)
            "equipment_status": equipment_status,

            # Room context (layout and desk information)
            "room_context": self._build_room_context(current_desk, occupancy.get("available_desks", [])),

            # Office occupancy (for end-of-day awareness)
            "office_occupancy": {
                "agents_present": [occupant_id] + other_occupants,
                "total_count": len(other_occupants) + 1,
                "you_would_be_last_to_leave": len(other_occupants) == 0,
            },

            # Location information (for navigation)
            "current_location": self.get_agent_location(occupant_id),
            "available_locations": self.get_available_locations(),
            "location_info": self.get_all_locations_info(),
            "agents_by_location": {
                loc: self.get_agents_at_location(loc)
                for loc in self._office_locations.keys()
            },

            # Location-specific equipment (what's available where you are)
            "location_equipment": self.get_equipment_at_location(
                self.get_agent_location(occupant_id), agent_id=occupant_id
            ),
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
            # Support both direct setpoint and direction-based adjustment
            setpoint = params.get("setpoint_c")
            direction = params.get("direction")
            amount = params.get("amount", 1.0)

            if setpoint is not None:
                # Direct setpoint provided
                self._thermostat_votes.append((occupant_id, float(setpoint)))
            elif direction:
                # Direction-based adjustment
                # Calculate desired setpoint based on direction and current base
                if direction == "warmer":
                    # Agent feels cold, wants higher heating setpoint
                    desired_setpoint = self._base_heating_setpoint_c + float(amount)
                    logger.info(f"[CTRL] {occupant_id}: Adjusting thermostat warmer (+{amount}C) -> target {desired_setpoint:.1f}C")
                elif direction == "cooler":
                    # Agent feels hot, wants lower cooling setpoint (represented as lower heating equivalent)
                    desired_setpoint = self._base_heating_setpoint_c - float(amount)
                    logger.info(f"[CTRL] {occupant_id}: Adjusting thermostat cooler (-{amount}C) -> target {desired_setpoint:.1f}C")
                else:
                    desired_setpoint = self._base_heating_setpoint_c
                self._thermostat_votes.append((occupant_id, desired_setpoint))

        elif action_type == "window_set":
            # Record vote for window
            wants_open = params.get("open", False)
            self._window_votes.append((occupant_id, wants_open))

        elif action_type == "lights_set":
            self.lighting.apply_lighting_action(params, occupant_id, current_desk)

        elif action_type == "choose_desk":
            desk_name = params.get("desk_name")
            if desk_name:
                # Enforce desk choice only on arrival - cannot change desks mid-day
                if self._desk_chosen_today.get(occupant_id, False):
                    # Agent already chose a desk today - ignore the request
                    current_desk = self.desks.get_desk_for_occupant(occupant_id)
                    print(f"[DESK] {occupant_id} tried to change desk but already has {current_desk} for today")
                    return
                # Assign desk and mark as chosen for today
                self.desks.assign_desk(occupant_id, desk_name)
                self._desk_chosen_today[occupant_id] = True

        elif action_type == "equipment_set":
            # Validate equipment is at agent's location or their desk
            equipment_name = params.get("equipment_name")
            if equipment_name:
                current_location = self.get_agent_location(occupant_id)
                # Pass agent_id to resolve desk_equipment reference
                location_equipment = self.get_equipment_at_location(
                    current_location, agent_id=occupant_id
                )
                valid_names = [eq["name"] for eq in location_equipment]

                # Also include desk equipment directly if not already included
                agent_desk = self.desks.get_desk_for_occupant(occupant_id) if current_desk else None
                if agent_desk:
                    desk_equipment = self.get_equipment_at_location(agent_desk)
                    for eq in desk_equipment:
                        if eq["name"] not in valid_names:
                            valid_names.append(eq["name"])

                # Check if equipment is valid for this agent
                if equipment_name not in valid_names:
                    logger.warning(
                        f"[ACTION] {occupant_id}: Cannot control '{equipment_name}' from {current_location} "
                        f"(available: {valid_names[:5]}...)"
                    )
                    return

            self.equipment.apply_equipment_action(params, occupant_id)

        elif action_type == "attend_meeting":
            self.desks.enter_meeting_room(occupant_id)
            self.set_agent_location(occupant_id, "meeting_room")
            self.set_agent_status(occupant_id, "at_desk", False)
            # Turn on meeting room lights and equipment if first person
            if self.desks.get_meeting_room().get_occupant_count() == 1:
                self.lighting.turn_on_light("meeting_room")
                self.equipment.turn_on_equipment("projector", occupant_id)
                self.equipment.turn_on_equipment("conference_phone", occupant_id)

        elif action_type == "leave_meeting":
            self.desks.leave_meeting_room(occupant_id)
            self.set_agent_location(occupant_id, "desk_area")
            self.set_agent_status(occupant_id, "at_desk", True)
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

        elif action_type == "move_to":
            # Move agent to a different location
            location = params.get("destination")  # Matches LocationDecision.destination field
            if location:
                self.set_agent_location(occupant_id, location)
                # Auto-clear lunch/break status if returning to desk area
                # This handles the case where agent moves to desk without explicit return action
                if location == "desk_area":
                    status = self.get_agent_status(occupant_id)
                    if status.get("at_lunch"):
                        self.set_agent_status(occupant_id, "at_lunch", False)
                        self.set_agent_status(occupant_id, "at_desk", True)
                    if status.get("on_break"):
                        self.set_agent_status(occupant_id, "on_break", False)
                        self.set_agent_status(occupant_id, "at_desk", True)

        # Handle break/lunch actions with location tracking
        elif action_type == "go_to_lunch":
            self.set_agent_status(occupant_id, "at_lunch", True)
            self.set_agent_status(occupant_id, "at_desk", False)
            self.set_agent_location(occupant_id, "break_area")

        elif action_type == "go_out_for_lunch":
            self.set_agent_status(occupant_id, "at_lunch", True)
            self.set_agent_status(occupant_id, "at_desk", False)
            self.set_agent_status(occupant_id, "out_of_office", True)
            self._agent_locations[occupant_id] = "outside"  # Outside the building

        elif action_type == "return_from_lunch":
            self.set_agent_status(occupant_id, "at_lunch", False)
            self.set_agent_status(occupant_id, "at_desk", True)
            self.set_agent_status(occupant_id, "out_of_office", False)
            self.set_agent_location(occupant_id, "desk_area")

        elif action_type == "take_break":
            self.set_agent_status(occupant_id, "on_break", True)
            self.set_agent_status(occupant_id, "at_desk", False)
            self.set_agent_location(occupant_id, "break_area")

        elif action_type == "go_out_for_break":
            # Leave building for a short break (walk, coffee run, fresh air)
            self.set_agent_status(occupant_id, "on_break", True)
            self.set_agent_status(occupant_id, "at_desk", False)
            self.set_agent_status(occupant_id, "out_of_office", True)
            self.set_agent_location(occupant_id, "outside")

        elif action_type == "return_from_break":
            self.set_agent_status(occupant_id, "on_break", False)
            self.set_agent_status(occupant_id, "at_desk", True)
            self.set_agent_status(occupant_id, "out_of_office", False)  # Clear in case was outside
            self.set_agent_location(occupant_id, "desk_area")

        elif action_type == "use_appliance":
            # Use an appliance at current location
            appliance_name = params.get("appliance_name")
            if appliance_name:
                # Verify agent is at a location with this appliance
                current_location = self.get_agent_location(occupant_id)
                location_equipment = self.get_equipment_at_location(current_location)
                equipment_names = [eq["name"] for eq in location_equipment]

                if appliance_name in equipment_names:
                    # Turn on the appliance (will auto-off after duration or next timestep)
                    self.equipment.turn_on_equipment(appliance_name, occupant_id)

        elif action_type == "initiate_conversation":
            # Record conversation intent - actual conversation handled by manager
            # This allows tracking of who wants to talk to whom about what
            target_agent = params.get("target_agent")  # Matches schema field name
            topic = params.get("topic", "general")
            if target_agent:
                # Store the conversation request for manager to process
                if not hasattr(self, '_pending_conversations'):
                    self._pending_conversations = []
                self._pending_conversations.append({
                    "initiator": occupant_id,
                    "target": target_agent,
                    "topic": topic,
                })

        elif action_type == "update_daily_plan":
            # Record plan update intent - actual update handled by manager
            # Manager has access to GenerativeAgent.update_daily_plan()
            updates = params.get("updates", {})
            reason = params.get("reason", "Agent requested plan update")
            if updates:
                if not hasattr(self, '_pending_plan_updates'):
                    self._pending_plan_updates = []
                self._pending_plan_updates.append({
                    "occupant_id": occupant_id,
                    "updates": updates,
                    "reason": reason,
                })

        # Log action application (except no_op to reduce noise)
        if action_type != "no_op":
            location = self.get_agent_location(occupant_id)
            status = self.get_agent_status(occupant_id)
            status_flags = [k for k, v in status.items() if v]
            logger.debug(
                f"[ACTION] {occupant_id}: {action_type} | "
                f"location={location} | status={status_flags} | params={params}"
            )

    def get_pending_conversations(self) -> list:
        """Get and clear pending conversation requests."""
        if not hasattr(self, '_pending_conversations'):
            return []
        conversations = self._pending_conversations
        self._pending_conversations = []
        return conversations

    def get_pending_plan_updates(self) -> list:
        """Get and clear pending plan update requests."""
        if not hasattr(self, '_pending_plan_updates'):
            return []
        updates = self._pending_plan_updates
        self._pending_plan_updates = []
        return updates

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

        # Clear location
        if occupant_id in self._agent_locations:
            del self._agent_locations[occupant_id]

        # Clear agent status on departure (safety net for at_lunch, on_break, etc.)
        if occupant_id in self._agent_status:
            del self._agent_status[occupant_id]

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

    def _get_default_agent_status(self) -> Dict[str, bool]:
        """Get default agent status (at desk, not on break/lunch)."""
        return {
            "at_desk": True,
            "at_lunch": False,
            "on_break": False,
            "out_of_office": False,
        }

    def get_agent_status(self, occupant_id: str) -> Dict[str, bool]:
        """Get agent's current status (at_desk, at_lunch, on_break, out_of_office)."""
        if occupant_id not in self._agent_status:
            self._agent_status[occupant_id] = self._get_default_agent_status()
        return self._agent_status[occupant_id].copy()

    def set_agent_status(self, occupant_id: str, status_key: str, value: bool) -> None:
        """
        Set a specific agent status flag.

        Args:
            occupant_id: Agent ID
            status_key: One of "at_desk", "at_lunch", "on_break", "out_of_office"
            value: True/False
        """
        if occupant_id not in self._agent_status:
            self._agent_status[occupant_id] = self._get_default_agent_status()
        if status_key in self._agent_status[occupant_id]:
            self._agent_status[occupant_id][status_key] = value

    # ---------------------------------------------------------------------------
    # Location Tracking Methods
    # ---------------------------------------------------------------------------

    def get_agent_location(self, occupant_id: str) -> str:
        """Get agent's current location. Defaults to 'desk_area' if not set."""
        return self._agent_locations.get(occupant_id, "desk_area")

    def set_agent_location(self, occupant_id: str, location: str) -> bool:
        """
        Set agent's current location.

        Note: This only updates location, NOT status flags. Status should be
        updated by explicit action handlers to keep behavior memory-driven
        rather than automatically prescribed.

        Args:
            occupant_id: Agent ID
            location: Location name (must be in office_locations)

        Returns:
            True if location was valid and set, False otherwise
        """
        if location not in self._office_locations:
            return False

        self._agent_locations[occupant_id] = location
        return True

    def get_available_locations(self) -> List[str]:
        """Get list of all available location names."""
        return list(self._office_locations.keys())

    def get_location_info(self, location: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific location."""
        return self._office_locations.get(location)

    def get_all_locations_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all locations."""
        return self._office_locations.copy()

    def get_agents_at_location(self, location: str) -> List[str]:
        """Get list of agent IDs at a specific location."""
        return [
            agent_id for agent_id, loc in self._agent_locations.items()
            if loc == location and self._present_occupants.get(agent_id, False)
        ]

    def get_equipment_at_location(
        self, location: str, agent_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get equipment available at a specific location.

        Args:
            location: Location name (desk_area, break_area, meeting_room, etc.)
            agent_id: Optional agent ID to resolve desk equipment for

        Returns:
            List of equipment dicts with name, type, is_on, in_use_by
        """
        location_info = self._office_locations.get(location, {})
        result = []

        # Get appliances defined in location config
        appliances = location_info.get("appliances", [])
        for appliance_name in appliances:
            eq_info = self.equipment.get_equipment_state_by_name(appliance_name)
            if eq_info:
                result.append({
                    "name": appliance_name,
                    "type": eq_info.get("type", appliance_name),
                    "is_on": eq_info.get("is_on", False),
                    "in_use_by": eq_info.get("assigned_to"),
                })

        # Get equipment_access defined in location config
        equipment_access = location_info.get("equipment_access", [])
        for eq_ref in equipment_access:
            if eq_ref == "desk_equipment":
                # Resolve desk_equipment to actual equipment for this agent's desk
                if agent_id:
                    agent_desk_name = self.desks.get_desk_for_occupant(agent_id)
                    if agent_desk_name:
                        desk_obj = self.desks.get_desk(agent_desk_name)
                        desk_equipment = desk_obj.equipment_ids if desk_obj else []
                        for eq_name in desk_equipment:
                            eq_info = self.equipment.get_equipment_state_by_name(eq_name)
                            if eq_info:
                                result.append({
                                    "name": eq_name,
                                    "type": eq_info.get("type", eq_name),
                                    "is_on": eq_info.get("is_on", False),
                                    "in_use_by": eq_info.get("assigned_to"),
                                })
                continue
            eq_info = self.equipment.get_equipment_state_by_name(eq_ref)
            if eq_info:
                result.append({
                    "name": eq_ref,
                    "type": eq_info.get("type", eq_ref),
                    "is_on": eq_info.get("is_on", False),
                    "in_use_by": eq_info.get("assigned_to"),
                })

        return result

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
        self._agent_status.clear()  # Reset all agent status
        self._agent_locations.clear()  # Reset all agent locations
        self._desk_chosen_today.clear()  # Allow desk choice on new day
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

    def save_locations(self, filepath: str) -> None:
        """Save agent locations to JSON file for persistence."""
        import json
        with open(filepath, 'w') as f:
            json.dump({
                "locations": self._agent_locations.copy(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)

    def load_locations(self, filepath: str) -> None:
        """Load agent locations from JSON file."""
        import json
        from pathlib import Path
        if Path(filepath).exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                self._agent_locations = data.get("locations", {})

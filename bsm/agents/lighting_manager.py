"""
lighting_manager.py

Manages per-desk and zone lighting state and power tracking for the building simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# Default power consumption values (Watts)
DEFAULT_LIGHTING_POWER_W = {
    "desk_light": 15,       # LED desk lamp
    "zone_main": 200,       # Main zone overhead lighting
    "meeting_room": 100,    # Meeting room lighting
}

# Default heat gain fractions for lighting
# Fraction of electrical power that becomes sensible heat to the zone
# For interior lights, ~15-20% of LED power exits as visible light through windows
# Overhead lights may have some return air fraction in commercial buildings
DEFAULT_LIGHTING_HEAT_GAIN_FRACTION = 0.85  # Default: 85% becomes zone heat


@dataclass
class LightingZone:
    """A controllable lighting zone (desk lamp or room lighting)."""

    name: str
    power_w: float
    location: str  # "Desk_A", "shared", "meeting_room", etc.
    is_on: bool = False
    controller: Optional[str] = None  # occupant_id who can control (None = anyone)
    heat_gain_fraction: float = DEFAULT_LIGHTING_HEAT_GAIN_FRACTION  # Fraction of power that becomes zone heat

    def turn_on(self) -> None:
        """Turn the light on."""
        self.is_on = True

    def turn_off(self) -> None:
        """Turn the light off."""
        self.is_on = False

    def toggle(self) -> bool:
        """Toggle light state. Returns new state."""
        self.is_on = not self.is_on
        return self.is_on

    def get_current_power(self) -> float:
        """Return current power consumption."""
        return self.power_w if self.is_on else 0.0

    def get_current_heat_gain(self) -> float:
        """Return current heat gain to zone (power * heat_gain_fraction)."""
        return self.get_current_power() * self.heat_gain_fraction

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "power_w": self.power_w,
            "heat_gain_fraction": self.heat_gain_fraction,
            "location": self.location,
            "is_on": self.is_on,
            "controller": self.controller,
            "current_power_w": self.get_current_power(),
            "current_heat_gain_w": self.get_current_heat_gain(),
        }


class LightingManager:
    """
    Manages zone lighting.

    Responsibilities:
    - Track individual desk light states
    - Track zone/meeting room lighting
    - Calculate total lighting power
    - Apply agent decisions to lighting
    """

    def __init__(self, config: dict):
        """
        Initialize from config['lighting'] section.

        Expected config structure:
        {
            "lighting": {
                "desk_light_power_w": 15,
                "zone_light_power_w": 200,
                "meeting_room_light_power_w": 100,
                "heat_gain_fraction": 0.85,
                "zones": {
                    "desk_light_A": {"power_w": 15, "location": "Desk_A"},
                    "zone_main": {"power_w": 200, "location": "shared"},
                    ...
                }
            }
        }
        """
        self._lights: Dict[str, LightingZone] = {}

        lighting_config = config.get("lighting", {})

        # Get power defaults
        default_desk_power = lighting_config.get("desk_light_power_w", DEFAULT_LIGHTING_POWER_W["desk_light"])
        default_zone_power = lighting_config.get("zone_light_power_w", DEFAULT_LIGHTING_POWER_W["zone_main"])
        default_meeting_power = lighting_config.get("meeting_room_light_power_w", DEFAULT_LIGHTING_POWER_W["meeting_room"])

        # Get heat gain fraction (default 85% - some light exits through windows)
        default_heat_gain_fraction = lighting_config.get("heat_gain_fraction", DEFAULT_LIGHTING_HEAT_GAIN_FRACTION)

        # Create lights from zones config
        zones = lighting_config.get("zones", {})
        for name, zone_config in zones.items():
            location = zone_config.get("location", "shared")

            # Determine power based on location type
            if "power_w" in zone_config:
                power = zone_config["power_w"]
            elif "desk" in name.lower():
                power = default_desk_power
            elif "meeting" in name.lower():
                power = default_meeting_power
            else:
                power = default_zone_power

            # Determine heat gain fraction: item-specific or default
            heat_gain_frac = zone_config.get("heat_gain_fraction", default_heat_gain_fraction)

            self._lights[name] = LightingZone(
                name=name,
                power_w=power,
                location=location,
                is_on=False,
                controller=zone_config.get("controller"),
                heat_gain_fraction=heat_gain_frac,
            )

        # If no zones configured, create default lighting setup
        if not zones:
            self._create_default_lighting(default_desk_power, default_zone_power, default_meeting_power, default_heat_gain_fraction)

    def _create_default_lighting(
        self,
        desk_power: float,
        zone_power: float,
        meeting_power: float,
        heat_gain_fraction: float = DEFAULT_LIGHTING_HEAT_GAIN_FRACTION
    ) -> None:
        """Create default lighting setup for 3 desks + zones."""
        desk_ids = ["Desk_A", "Desk_B", "Desk_C"]

        # Desk lights
        for desk_id in desk_ids:
            suffix = desk_id.split("_")[1]
            light_name = f"desk_light_{suffix}"
            self._lights[light_name] = LightingZone(
                name=light_name,
                power_w=desk_power,
                location=desk_id,
                heat_gain_fraction=heat_gain_fraction,
            )

        # Main zone lighting
        self._lights["zone_main"] = LightingZone(
            name="zone_main",
            power_w=zone_power,
            location="shared",
            heat_gain_fraction=heat_gain_fraction,
        )

        # Meeting room lighting
        self._lights["meeting_room"] = LightingZone(
            name="meeting_room",
            power_w=meeting_power,
            location="meeting_room",
            heat_gain_fraction=heat_gain_fraction,
        )

    def get_light(self, name: str) -> Optional[LightingZone]:
        """Get light by name."""
        return self._lights.get(name)

    def get_light_at_location(self, location: str) -> Optional[LightingZone]:
        """Get light at a specific location (desk or room)."""
        for light in self._lights.values():
            if light.location == location:
                return light
        return None

    def get_lights_at_location(self, location: str) -> List[LightingZone]:
        """Get all lights at a specific location."""
        return [light for light in self._lights.values() if light.location == location]

    def get_lighting_state(self) -> Dict[str, bool]:
        """Return current light states as a simple dict."""
        return {name: light.is_on for name, light in self._lights.items()}

    def get_detailed_lighting_state(self) -> Dict[str, Any]:
        """Return detailed lighting state for agent context."""
        return {
            "lights": {name: light.to_dict() for name, light in self._lights.items()},
            "total_power_w": self.calculate_total_power_w(),
            "total_heat_gain_w": self.calculate_total_heat_gain_w(),
        }

    def get_lighting_for_occupant(self, desk_location: Optional[str]) -> Dict[str, Any]:
        """
        Get lighting state relevant to a specific occupant.

        Returns their desk light state plus zone lighting.
        """
        result = {
            "desk_light": None,
            "zone_lights": {},
            "controllable_lights": [],
        }

        for name, light in self._lights.items():
            if desk_location and light.location == desk_location:
                result["desk_light"] = {
                    "name": name,
                    "is_on": light.is_on,
                    "power_w": light.power_w,
                }
                result["controllable_lights"].append(name)
            elif light.location in ("shared", "meeting_room"):
                result["zone_lights"][name] = {
                    "is_on": light.is_on,
                    "location": light.location,
                    "power_w": light.power_w,
                }
                result["controllable_lights"].append(name)

        return result

    def turn_on_light(self, name: str) -> bool:
        """
        Turn on light by name.

        Returns True if successful, False if light doesn't exist.
        """
        light = self._lights.get(name)
        if light:
            light.turn_on()
            return True
        return False

    def turn_off_light(self, name: str) -> bool:
        """
        Turn off light by name.

        Returns True if successful, False if light doesn't exist.
        """
        light = self._lights.get(name)
        if light:
            light.turn_off()
            return True
        return False

    def set_light_state(self, name: str, is_on: bool) -> bool:
        """Set light state directly."""
        light = self._lights.get(name)
        if light:
            light.is_on = is_on
            return True
        return False

    def toggle_light(self, name: str) -> Optional[bool]:
        """Toggle light state. Returns new state or None if not found."""
        light = self._lights.get(name)
        if light:
            return light.toggle()
        return None

    def apply_lighting_action(
        self,
        action_params: Dict[str, Any],
        occupant_id: str,
        occupant_desk: Optional[str] = None
    ) -> bool:
        """
        Apply a lights_set action from an agent decision.

        Expected action_params:
        {
            "light_name": "desk_light_A",  # specific light
            "on": true/false
        }
        or
        {
            "target": "zone_main",  # light name from LightingDecision.target_device
            "on": true/false
        }
        or
        {
            "location": "Desk_A",  # turn on/off lights at location
            "on": true/false
        }
        or
        {
            "desk_light": true/false  # control occupant's own desk light
        }
        """
        # Handle "target" parameter (from LightingDecision.target_device)
        target = action_params.get("target")
        if target:
            on_state = action_params.get("on", False)
            # Try to find the light by name
            if target in self._lights:
                return self.set_light_state(target, on_state)
            # Try desk light pattern (e.g., "desk_light" → "desk_light_A")
            if target == "desk_light" and occupant_desk:
                desk_suffix = occupant_desk.split("_")[-1] if "_" in occupant_desk else occupant_desk
                desk_light_name = f"desk_light_{desk_suffix}"
                if desk_light_name in self._lights:
                    return self.set_light_state(desk_light_name, on_state)
            return False

        # Specific light by name
        light_name = action_params.get("light_name")
        if light_name:
            return self.set_light_state(light_name, action_params.get("on", False))

        # By location
        location = action_params.get("location")
        if location:
            success = False
            for light in self._lights.values():
                if light.location == location:
                    light.is_on = action_params.get("on", False)
                    success = True
            return success

        # Own desk light
        if "desk_light" in action_params and occupant_desk:
            for light in self._lights.values():
                if light.location == occupant_desk:
                    light.is_on = action_params.get("desk_light", False)
                    return True

        # General on/off for own desk
        if "on" in action_params and occupant_desk:
            for light in self._lights.values():
                if light.location == occupant_desk:
                    light.is_on = action_params.get("on", False)
                    return True

        return False

    def turn_on_desk_light(self, desk_location: str) -> bool:
        """Turn on the light at a specific desk."""
        for light in self._lights.values():
            if light.location == desk_location:
                light.turn_on()
                return True
        return False

    def turn_off_desk_light(self, desk_location: str) -> bool:
        """Turn off the light at a specific desk."""
        for light in self._lights.values():
            if light.location == desk_location:
                light.turn_off()
                return True
        return False

    def calculate_total_power_w(self) -> float:
        """Sum of all active lighting power consumption."""
        return sum(light.get_current_power() for light in self._lights.values())

    def calculate_total_heat_gain_w(self) -> float:
        """Sum of all active lighting heat gain to zone (power * heat_gain_fraction)."""
        return sum(light.get_current_heat_gain() for light in self._lights.values())

    def calculate_power_at_location(self, location: str) -> float:
        """Calculate total lighting power at a specific location."""
        return sum(
            light.get_current_power()
            for light in self._lights.values()
            if light.location == location
        )

    def get_active_lights_list(self) -> List[str]:
        """Return list of names of currently on lights."""
        return [light.name for light in self._lights.values() if light.is_on]

    def set_zone_lighting(self, is_on: bool) -> None:
        """Turn on/off main zone lighting (typically at start/end of day)."""
        zone_light = self._lights.get("zone_main")
        if zone_light:
            zone_light.is_on = is_on

    def reset_all(self) -> None:
        """Turn off all lights (e.g., at end of day)."""
        for light in self._lights.values():
            light.turn_off()

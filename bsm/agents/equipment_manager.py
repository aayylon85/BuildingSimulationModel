"""
equipment_manager.py

Manages equipment state (on/off) and power consumption for all occupant-controllable
equipment in the building simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Default power consumption values (Watts)
DEFAULT_POWER_W = {
    "laptop": 50,
    "monitor": 30,
    "desktop_computer": 150,
    "photocopier_standby": 20,
    "photocopier_active": 400,
    "projector": 200,
    "conference_phone": 10,
}

# Default heat gain fractions by equipment type
# Fraction of electrical power that becomes sensible heat to the zone
# Based on EnergyPlus standards - nearly all equipment heat stays in zone
# Note: Photocopier has some exhaust, projector has fan exhaust
DEFAULT_HEAT_GAIN_FRACTIONS = {
    "laptop": 1.0,              # All heat stays in zone
    "monitor": 1.0,             # All heat stays in zone
    "desktop_computer": 1.0,    # All heat stays in zone
    "photocopier": 0.92,        # ~8% exhausted
    "projector": 0.95,          # ~5% fan exhaust
    "conference_phone": 1.0,    # All heat stays in zone
}


@dataclass
class Equipment:
    """Single piece of equipment with power consumption tracking."""

    name: str
    equipment_type: str
    power_w: float
    location: str = "shared"  # desk location (e.g., "Desk_A") or "shared"
    is_on: bool = False
    assigned_to: Optional[str] = None  # occupant_id or None
    heat_gain_fraction: float = 1.0  # Fraction of power that becomes zone heat

    def get_current_power(self) -> float:
        """Return current power consumption based on state."""
        if not self.is_on:
            # Photocopiers have standby power
            if self.equipment_type == "photocopier":
                return DEFAULT_POWER_W.get("photocopier_standby", 20)
            return 0.0
        return self.power_w

    def get_current_heat_gain(self) -> float:
        """Return current heat gain to zone (power * heat_gain_fraction)."""
        return self.get_current_power() * self.heat_gain_fraction

    def turn_on(self, occupant_id: Optional[str] = None) -> None:
        """Turn equipment on."""
        self.is_on = True
        if occupant_id:
            self.assigned_to = occupant_id

    def turn_off(self) -> None:
        """Turn equipment off."""
        self.is_on = False
        self.assigned_to = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "type": self.equipment_type,
            "power_w": self.power_w,
            "heat_gain_fraction": self.heat_gain_fraction,
            "location": self.location,
            "is_on": self.is_on,
            "assigned_to": self.assigned_to,
            "current_power_w": self.get_current_power(),
            "current_heat_gain_w": self.get_current_heat_gain(),
        }


class EquipmentManager:
    """
    Manages all equipment in the zone.

    Responsibilities:
    - Track equipment state (on/off/assigned)
    - Calculate total equipment power consumption
    - Apply agent decisions to equipment
    - Provide equipment state for agent context
    """

    def __init__(self, config: dict):
        """
        Initialize from config['equipment'] section.

        Expected config structure:
        {
            "equipment": {
                "default_power_w": {"laptop": 50, "monitor": 30, ...},
                "heat_gain_fractions": {"laptop": 1.0, "photocopier": 0.92, ...},
                "shared_equipment": ["photocopier"],
                "items": {
                    "laptop_A": {"type": "laptop", "location": "Desk_A"},
                    ...
                }
            }
        }
        """
        self._equipment: Dict[str, Equipment] = {}
        self._shared_equipment: List[str] = []

        equipment_config = config.get("equipment", {})

        # Get power defaults, merging with global defaults
        power_defaults = {**DEFAULT_POWER_W}
        power_defaults.update(equipment_config.get("default_power_w", {}))

        # Get heat gain fraction defaults, merging with global defaults
        heat_gain_defaults = {**DEFAULT_HEAT_GAIN_FRACTIONS}
        heat_gain_defaults.update(equipment_config.get("heat_gain_fractions", {}))

        # Track shared equipment names
        self._shared_equipment = equipment_config.get("shared_equipment", [])

        # Create equipment from items config
        items = equipment_config.get("items", {})
        for name, item_config in items.items():
            eq_type = item_config.get("type", "unknown")
            location = item_config.get("location", "shared")

            # Determine power: use item-specific, then type default
            if "power_w" in item_config:
                power = item_config["power_w"]
            elif eq_type == "photocopier":
                power = power_defaults.get("photocopier_active", 400)
            else:
                power = power_defaults.get(eq_type, 50)

            # Determine heat gain fraction: use item-specific, then type default
            if "heat_gain_fraction" in item_config:
                heat_gain_frac = item_config["heat_gain_fraction"]
            else:
                heat_gain_frac = heat_gain_defaults.get(eq_type, 1.0)

            self._equipment[name] = Equipment(
                name=name,
                equipment_type=eq_type,
                power_w=power,
                location=location,
                is_on=False,
                heat_gain_fraction=heat_gain_frac,
            )

        # If no items configured, create default equipment for 3 desks
        if not items:
            self._create_default_equipment(power_defaults, heat_gain_defaults)

    def _create_default_equipment(
        self,
        power_defaults: Dict[str, float],
        heat_gain_defaults: Dict[str, float]
    ) -> None:
        """Create default equipment setup for 3 desks + shared."""
        desk_ids = ["Desk_A", "Desk_B", "Desk_C"]

        for desk_id in desk_ids:
            suffix = desk_id.split("_")[1]

            # Laptop at each desk
            laptop_name = f"laptop_{suffix}"
            self._equipment[laptop_name] = Equipment(
                name=laptop_name,
                equipment_type="laptop",
                power_w=power_defaults.get("laptop", 50),
                location=desk_id,
                heat_gain_fraction=heat_gain_defaults.get("laptop", 1.0),
            )

            # Monitor at each desk
            monitor_name = f"monitor_{suffix}"
            self._equipment[monitor_name] = Equipment(
                name=monitor_name,
                equipment_type="monitor",
                power_w=power_defaults.get("monitor", 30),
                location=desk_id,
                heat_gain_fraction=heat_gain_defaults.get("monitor", 1.0),
            )

        # Shared photocopier
        self._equipment["photocopier"] = Equipment(
            name="photocopier",
            equipment_type="photocopier",
            power_w=power_defaults.get("photocopier_active", 400),
            location="shared",
            heat_gain_fraction=heat_gain_defaults.get("photocopier", 0.92),
        )
        self._shared_equipment = ["photocopier"]

        # Meeting room equipment
        self._equipment["projector"] = Equipment(
            name="projector",
            equipment_type="projector",
            power_w=power_defaults.get("projector", 200),
            location="meeting_room",
            heat_gain_fraction=heat_gain_defaults.get("projector", 0.95),
        )
        self._equipment["conference_phone"] = Equipment(
            name="conference_phone",
            equipment_type="conference_phone",
            power_w=power_defaults.get("conference_phone", 10),
            location="meeting_room",
            heat_gain_fraction=heat_gain_defaults.get("conference_phone", 1.0),
        )

    def get_equipment(self, name: str) -> Optional[Equipment]:
        """Get equipment by name."""
        return self._equipment.get(name)

    def get_equipment_at_location(self, location: str) -> List[Equipment]:
        """Get all equipment at a specific location."""
        return [eq for eq in self._equipment.values() if eq.location == location]

    def get_equipment_state(self) -> Dict[str, Any]:
        """Return current equipment states for agent context."""
        return {
            "items": {name: eq.to_dict() for name, eq in self._equipment.items()},
            "shared_equipment": self._shared_equipment,
            "total_power_w": self.calculate_total_power_w(),
            "total_heat_gain_w": self.calculate_total_heat_gain_w(),
        }

    def get_equipment_for_occupant(self, occupant_id: str, desk_location: Optional[str]) -> Dict[str, Any]:
        """
        Get equipment state relevant to a specific occupant.

        Returns equipment at their desk plus shared equipment.
        """
        result = {
            "desk_equipment": {},
            "shared_equipment": {},
            "available_equipment": [],
        }

        for name, eq in self._equipment.items():
            # Equipment at occupant's desk
            if desk_location and eq.location == desk_location:
                result["desk_equipment"][name] = {
                    "type": eq.equipment_type,
                    "is_on": eq.is_on,
                    "power_w": eq.get_current_power(),
                }
            # Shared equipment
            elif eq.location in ("shared", "meeting_room"):
                result["shared_equipment"][name] = {
                    "type": eq.equipment_type,
                    "is_on": eq.is_on,
                    "in_use_by": eq.assigned_to,
                    "power_w": eq.get_current_power(),
                }

        # List of available equipment names
        result["available_equipment"] = list(result["desk_equipment"].keys()) + list(result["shared_equipment"].keys())

        return result

    def turn_on_equipment(self, name: str, occupant_id: Optional[str] = None) -> bool:
        """
        Turn on equipment by name.

        Returns True if successful, False if equipment doesn't exist.
        """
        eq = self._equipment.get(name)
        if eq:
            eq.turn_on(occupant_id)
            return True
        return False

    def turn_off_equipment(self, name: str) -> bool:
        """
        Turn off equipment by name.

        Returns True if successful, False if equipment doesn't exist.
        """
        eq = self._equipment.get(name)
        if eq:
            eq.turn_off()
            return True
        return False

    def set_equipment_state(self, name: str, is_on: bool, occupant_id: Optional[str] = None) -> bool:
        """Set equipment state directly."""
        eq = self._equipment.get(name)
        if eq:
            if is_on:
                eq.turn_on(occupant_id)
            else:
                eq.turn_off()
            return True
        return False

    def apply_equipment_action(self, action_params: Dict[str, Any], occupant_id: str) -> bool:
        """
        Apply an equipment_set action from an agent decision.

        Expected action_params:
        {
            "equipment_name": "laptop_A",
            "on": true/false
        }
        or
        {
            "equipment_type": "laptop",  # turns on/off all at desk
            "on": true/false
        }
        """
        equipment_name = action_params.get("equipment_name")
        equipment_type = action_params.get("equipment_type")
        is_on = action_params.get("on", False)

        if equipment_name:
            return self.set_equipment_state(equipment_name, is_on, occupant_id if is_on else None)

        if equipment_type:
            # Turn on/off all equipment of this type
            success = False
            for eq in self._equipment.values():
                if eq.equipment_type == equipment_type:
                    if is_on:
                        eq.turn_on(occupant_id)
                    else:
                        eq.turn_off()
                    success = True
            return success

        return False

    def turn_on_desk_equipment(self, desk_location: str, occupant_id: str) -> None:
        """Turn on all equipment at a specific desk."""
        for eq in self._equipment.values():
            if eq.location == desk_location:
                eq.turn_on(occupant_id)

    def turn_off_desk_equipment(self, desk_location: str) -> None:
        """Turn off all equipment at a specific desk."""
        for eq in self._equipment.values():
            if eq.location == desk_location:
                eq.turn_off()

    def release_occupant_equipment(self, occupant_id: str) -> None:
        """Turn off all equipment assigned to an occupant."""
        for eq in self._equipment.values():
            if eq.assigned_to == occupant_id:
                eq.turn_off()

    def calculate_total_power_w(self) -> float:
        """Sum of all active equipment power consumption."""
        return sum(eq.get_current_power() for eq in self._equipment.values())

    def calculate_total_heat_gain_w(self) -> float:
        """Sum of all active equipment heat gain to zone (power * heat_gain_fraction)."""
        return sum(eq.get_current_heat_gain() for eq in self._equipment.values())

    def calculate_power_by_location(self, location: str) -> float:
        """Calculate total power consumption at a specific location."""
        return sum(
            eq.get_current_power()
            for eq in self._equipment.values()
            if eq.location == location
        )

    def get_active_equipment_list(self) -> List[str]:
        """Return list of names of currently active equipment."""
        return [eq.name for eq in self._equipment.values() if eq.is_on]

    def reset_all(self) -> None:
        """Turn off all equipment (e.g., at end of day)."""
        for eq in self._equipment.values():
            eq.turn_off()

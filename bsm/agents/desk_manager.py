"""
desk_manager.py

Tracks desk assignments and occupancy for the building simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Desk:
    """A workstation desk that can be occupied by one occupant."""

    name: str
    equipment_ids: List[str] = field(default_factory=list)
    light_id: Optional[str] = None
    current_occupant: Optional[str] = None  # occupant_id or None

    def is_available(self) -> bool:
        """Check if desk is available for assignment."""
        return self.current_occupant is None

    def assign(self, occupant_id: str) -> bool:
        """
        Assign an occupant to this desk.

        Returns True if successful, False if already occupied.
        """
        if self.current_occupant is not None:
            return False
        self.current_occupant = occupant_id
        return True

    def release(self) -> Optional[str]:
        """
        Release the desk. Returns the previous occupant_id or None.
        """
        prev = self.current_occupant
        self.current_occupant = None
        return prev

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "equipment_ids": self.equipment_ids,
            "light_id": self.light_id,
            "current_occupant": self.current_occupant,
            "is_available": self.is_available(),
        }


@dataclass
class MeetingRoom:
    """A meeting room that can hold multiple occupants."""

    name: str
    capacity: int
    equipment_ids: List[str] = field(default_factory=list)
    light_id: Optional[str] = None
    current_occupants: List[str] = field(default_factory=list)

    def is_available(self) -> bool:
        """Check if meeting room has capacity."""
        return len(self.current_occupants) < self.capacity

    def get_occupant_count(self) -> int:
        """Get current number of occupants."""
        return len(self.current_occupants)

    def add_occupant(self, occupant_id: str) -> bool:
        """
        Add an occupant to the meeting room.

        Returns True if successful, False if at capacity or already present.
        """
        if occupant_id in self.current_occupants:
            return True  # Already in room
        if len(self.current_occupants) >= self.capacity:
            return False
        self.current_occupants.append(occupant_id)
        return True

    def remove_occupant(self, occupant_id: str) -> bool:
        """
        Remove an occupant from the meeting room.

        Returns True if was present and removed, False otherwise.
        """
        if occupant_id in self.current_occupants:
            self.current_occupants.remove(occupant_id)
            return True
        return False

    def clear(self) -> None:
        """Remove all occupants from the meeting room."""
        self.current_occupants.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "capacity": self.capacity,
            "equipment_ids": self.equipment_ids,
            "light_id": self.light_id,
            "current_occupants": self.current_occupants.copy(),
            "occupant_count": self.get_occupant_count(),
            "is_available": self.is_available(),
        }


class DeskManager:
    """
    Manages desk assignments and meeting room occupancy tracking.

    Responsibilities:
    - Track desk assignments (one occupant per desk)
    - Track meeting room occupancy (multiple occupants)
    - Provide occupancy information for agent context
    - Handle desk selection and release
    """

    def __init__(self, config: dict):
        """
        Initialize from config['desks'] and config['meeting_room'] sections.

        Expected config structure:
        {
            "desks": {
                "Desk_A": {"equipment": ["laptop_A", "monitor_A"], "light": "desk_light_A"},
                "Desk_B": {...},
                "Desk_C": {...}
            },
            "meeting_room": {
                "capacity": 6,
                "equipment": ["projector", "conference_phone"],
                "light": "meeting_room"
            }
        }
        """
        self._desks: Dict[str, Desk] = {}
        self._meeting_room: Optional[MeetingRoom] = None
        self._occupant_locations: Dict[str, str] = {}  # occupant_id -> location

        # Initialize desks
        desks_config = config.get("desks", {})
        for name, desk_config in desks_config.items():
            equipment = desk_config.get("equipment", [])
            light = desk_config.get("light")
            self._desks[name] = Desk(
                name=name,
                equipment_ids=equipment,
                light_id=light,
            )

        # If no desks configured, create defaults
        if not desks_config:
            self._create_default_desks()

        # Initialize meeting room
        meeting_config = config.get("meeting_room", {})
        if meeting_config:
            self._meeting_room = MeetingRoom(
                name="meeting_room",
                capacity=meeting_config.get("capacity", 6),
                equipment_ids=meeting_config.get("equipment", []),
                light_id=meeting_config.get("light", "meeting_room"),
            )
        else:
            self._meeting_room = MeetingRoom(
                name="meeting_room",
                capacity=6,
                equipment_ids=["projector", "conference_phone"],
                light_id="meeting_room",
            )

    def _create_default_desks(self) -> None:
        """Create default desk setup."""
        desk_configs = {
            "Desk_A": {"equipment": ["laptop_A", "monitor_A"], "light": "desk_light_A"},
            "Desk_B": {"equipment": ["laptop_B", "monitor_B"], "light": "desk_light_B"},
            "Desk_C": {"equipment": ["laptop_C", "monitor_C"], "light": "desk_light_C"},
        }
        for name, desk_config in desk_configs.items():
            self._desks[name] = Desk(
                name=name,
                equipment_ids=desk_config["equipment"],
                light_id=desk_config["light"],
            )

    def get_desk(self, name: str) -> Optional[Desk]:
        """Get desk by name."""
        return self._desks.get(name)

    def get_meeting_room(self) -> Optional[MeetingRoom]:
        """Get the meeting room."""
        return self._meeting_room

    def assign_desk(self, occupant_id: str, desk_name: str) -> bool:
        """
        Assign occupant to desk.

        Returns False if desk is occupied or doesn't exist.
        """
        desk = self._desks.get(desk_name)
        if not desk:
            return False

        # Release any current desk assignment
        current_desk = self.get_desk_for_occupant(occupant_id)
        if current_desk:
            self.release_desk(occupant_id)

        if desk.assign(occupant_id):
            self._occupant_locations[occupant_id] = desk_name
            return True
        return False

    def release_desk(self, occupant_id: str) -> Optional[str]:
        """
        Release occupant's current desk assignment.

        Returns the desk name that was released, or None.
        """
        for desk in self._desks.values():
            if desk.current_occupant == occupant_id:
                desk.release()
                if occupant_id in self._occupant_locations:
                    del self._occupant_locations[occupant_id]
                return desk.name
        return None

    def get_available_desks(self) -> List[str]:
        """List of unoccupied desk names."""
        return [name for name, desk in self._desks.items() if desk.is_available()]

    def get_occupied_desks(self) -> Dict[str, str]:
        """Map of desk_name -> occupant_id for occupied desks."""
        return {
            name: desk.current_occupant
            for name, desk in self._desks.items()
            if desk.current_occupant is not None
        }

    def get_desk_for_occupant(self, occupant_id: str) -> Optional[str]:
        """Get desk name for occupant, or None if not assigned."""
        for desk in self._desks.values():
            if desk.current_occupant == occupant_id:
                return desk.name
        return None

    def get_desk_occupancy(self) -> Dict[str, Optional[str]]:
        """Map of desk_name -> occupant_id (or None if empty)."""
        return {name: desk.current_occupant for name, desk in self._desks.items()}

    def get_all_assignments(self) -> Dict[str, str]:
        """Map of occupant_id -> location (desk name or 'meeting_room')."""
        return self._occupant_locations.copy()

    def enter_meeting_room(self, occupant_id: str) -> bool:
        """
        Add occupant to meeting room.

        Returns False if at capacity.
        """
        if not self._meeting_room:
            return False
        if self._meeting_room.add_occupant(occupant_id):
            # Update location tracking (could be in meeting room AND have a desk)
            return True
        return False

    def leave_meeting_room(self, occupant_id: str) -> bool:
        """Remove occupant from meeting room."""
        if not self._meeting_room:
            return False
        return self._meeting_room.remove_occupant(occupant_id)

    def is_in_meeting_room(self, occupant_id: str) -> bool:
        """Check if occupant is in meeting room."""
        if not self._meeting_room:
            return False
        return occupant_id in self._meeting_room.current_occupants

    def get_meeting_room_occupants(self) -> List[str]:
        """Get list of occupants currently in meeting room."""
        if not self._meeting_room:
            return []
        return self._meeting_room.current_occupants.copy()

    def get_state(self) -> Dict[str, Any]:
        """Return full state for agent context."""
        return {
            "desks": {name: desk.to_dict() for name, desk in self._desks.items()},
            "meeting_room": self._meeting_room.to_dict() if self._meeting_room else None,
            "available_desks": self.get_available_desks(),
            "desk_occupancy": self.get_desk_occupancy(),
        }

    def get_occupancy_for_agent(self, occupant_id: str) -> Dict[str, Any]:
        """
        Get occupancy state relevant to a specific occupant.

        Returns desk info, meeting room status, and other occupants.
        """
        current_desk = self.get_desk_for_occupant(occupant_id)
        in_meeting = self.is_in_meeting_room(occupant_id)

        # Get other occupants' locations
        other_occupants = []
        for desk in self._desks.values():
            if desk.current_occupant and desk.current_occupant != occupant_id:
                other_occupants.append({
                    "id": desk.current_occupant,
                    "location": desk.name,
                    "type": "desk",
                })

        if self._meeting_room:
            for occ_id in self._meeting_room.current_occupants:
                if occ_id != occupant_id:
                    # Check if already listed (they may have a desk too)
                    existing = next(
                        (o for o in other_occupants if o["id"] == occ_id), None
                    )
                    if existing:
                        existing["in_meeting"] = True
                    else:
                        other_occupants.append({
                            "id": occ_id,
                            "location": "meeting_room",
                            "type": "meeting_room",
                        })

        return {
            "current_desk": current_desk,
            "in_meeting_room": in_meeting,
            "desk_options": list(self._desks.keys()),
            "available_desks": self.get_available_desks(),
            "desk_occupancy": self.get_desk_occupancy(),
            "meeting_room_capacity": self._meeting_room.capacity if self._meeting_room else 0,
            "meeting_room_occupants": self.get_meeting_room_occupants(),
            "other_occupants": other_occupants,
        }

    def apply_desk_action(self, action_params: Dict[str, Any], occupant_id: str) -> bool:
        """
        Apply a choose_desk action from an agent decision.

        Expected action_params:
        {
            "desk_name": "Desk_A"
        }
        """
        desk_name = action_params.get("desk_name")
        if not desk_name:
            return False
        return self.assign_desk(occupant_id, desk_name)

    def occupant_leaves_building(self, occupant_id: str) -> None:
        """
        Handle occupant leaving the building.

        Releases desk and removes from meeting room.
        """
        self.release_desk(occupant_id)
        self.leave_meeting_room(occupant_id)
        if occupant_id in self._occupant_locations:
            del self._occupant_locations[occupant_id]

    def reset_all(self) -> None:
        """Reset all assignments (e.g., at start of new day)."""
        for desk in self._desks.values():
            desk.release()
        if self._meeting_room:
            self._meeting_room.clear()
        self._occupant_locations.clear()

    def get_total_present_count(self) -> int:
        """Get total number of occupants present (at desks or in meeting room only)."""
        desk_occupants = set(
            desk.current_occupant
            for desk in self._desks.values()
            if desk.current_occupant
        )
        meeting_occupants = set(
            self._meeting_room.current_occupants if self._meeting_room else []
        )
        return len(desk_occupants | meeting_occupants)

    def get_present_occupants(self) -> List[str]:
        """Get list of all present occupant IDs."""
        desk_occupants = set(
            desk.current_occupant
            for desk in self._desks.values()
            if desk.current_occupant
        )
        meeting_occupants = set(
            self._meeting_room.current_occupants if self._meeting_room else []
        )
        return list(desk_occupants | meeting_occupants)

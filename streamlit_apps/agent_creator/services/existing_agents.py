"""Service for loading existing agent data."""

import json
from pathlib import Path
from typing import Dict, Any, List, Set


def load_existing_agents(base_path: str = "agent_base_types") -> Dict[str, Dict[str, Any]]:
    """Load all existing agents from agent_base_types folder.

    Returns:
        Dict mapping agent_id to agent info:
        {
            "alice_001": {
                "name": "Alice",
                "first_name": "Alice",
                "agent_id": "alice_001",
                "folder_name": "alice_office_worker",
                "thermal_preference": "slightly_warm",
                ...
            }
        }
    """
    base = Path(base_path)
    agents = {}

    if not base.exists():
        return agents

    for folder in base.iterdir():
        # Skip template and hidden folders
        if not folder.is_dir() or folder.name.startswith("_") or folder.name.startswith("."):
            continue

        scratch_path = folder / "scratch.json"
        if not scratch_path.exists():
            continue

        try:
            with open(scratch_path) as f:
                data = json.load(f)

            agent_id = data.get("agent_id", "")
            if agent_id:
                agents[agent_id] = {
                    "agent_id": agent_id,
                    "name": data.get("name", ""),
                    "first_name": data.get("first_name", ""),
                    "folder_name": folder.name,
                    "thermal_preference": data.get("thermal_preference", "neutral"),
                    "runs_warm": data.get("runs_warm", False),
                    "typical_arrival": data.get("typical_arrival", ""),
                    "typical_departure": data.get("typical_departure", ""),
                    "preferred_desk": data.get("preferred_desk", ""),
                }
        except (json.JSONDecodeError, IOError):
            continue

    return agents


def get_existing_agent_ids(base_path: str = "agent_base_types") -> Set[str]:
    """Get set of all existing agent IDs.

    Returns:
        Set of agent IDs (e.g., {"alice_001", "bob_002"})
    """
    agents = load_existing_agents(base_path)
    return set(agents.keys())


def validate_agent_id_unique(agent_id: str, base_path: str = "agent_base_types") -> tuple:
    """Validate that agent_id is unique.

    Returns:
        Tuple of (is_valid, error_message)
    """
    import re

    # Check format
    if not re.match(r"^[a-z]+_\d{3}$", agent_id):
        return False, "Agent ID must be lowercase letters followed by underscore and 3 digits (e.g., 'david_004')"

    # Check uniqueness
    existing_ids = get_existing_agent_ids(base_path)
    if agent_id in existing_ids:
        return False, f"Agent ID '{agent_id}' already exists"

    return True, ""


def get_suggested_agent_id(name: str, base_path: str = "agent_base_types") -> str:
    """Suggest an available agent ID based on name.

    Args:
        name: Agent's first name (e.g., "David")

    Returns:
        Suggested agent ID (e.g., "david_004")
    """
    existing_ids = get_existing_agent_ids(base_path)
    base_name = name.lower().strip()

    # Find next available number
    for i in range(1, 1000):
        candidate = f"{base_name}_{i:03d}"
        if candidate not in existing_ids:
            return candidate

    return f"{base_name}_999"


def get_agent_summary(agent_id: str, base_path: str = "agent_base_types") -> str:
    """Get a brief summary of an agent for display.

    Returns:
        String like "Alice (alice_001) - slightly warm preference, arrives 08:30-09:00"
    """
    agents = load_existing_agents(base_path)
    agent = agents.get(agent_id)

    if not agent:
        return f"Unknown agent: {agent_id}"

    return (
        f"{agent['name']} ({agent['agent_id']}) - "
        f"{agent['thermal_preference']} preference, "
        f"arrives {agent['typical_arrival']}"
    )

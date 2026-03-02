"""
ui_demo_writer.py

Writes simulation state to UI_demo_outputs/ for visualization.
Creates a SQLite database updated at each timestep, plus JSON files
for agent actions and conversations.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# Actions that change simulation state and should be recorded for UI visualization
RECORDABLE_ACTIONS = {
    # Arrival/departure
    "arrive",            # Agent arriving at office
    "leave_building",    # Agent departing
    "depart",            # Agent departing (alternative)
    # Movement (all breaks/lunch use move_to now)
    "move_to",           # Changes agent location
    "go_out_for_break",  # Legacy: going outside for break
    # Environment controls
    "lights_set",        # Changes lighting state
    "equipment_set",     # Changes device state (including kitchen appliances)
    "equipment_auto_off",  # System auto-off event (kitchen appliances timer)
    "thermostat_adjust", # Changes temperature
    "window_set",        # Changes window state
    # Social interactions
    "initiate_conversation",  # Social interaction
    "accept_conversation",    # Social interaction response
}


class UIDemoWriter:
    """
    Writes simulation state to UI_demo_outputs/ for visualization.

    Creates:
    - simulation_state.db: SQLite database with timestep, device, light, and occupant state
    - agent_actions/{agent_id}_actions.json: Per-agent action logs
    - conversations/{conv_id}.json: Per-conversation logs
    """

    def __init__(self, output_dir: str = "UI_demo_outputs"):
        """
        Initialize the UI demo writer.

        Args:
            output_dir: Directory to write outputs to (will be cleared on init)
        """
        self.output_dir = Path(output_dir)
        self.db_path = self.output_dir / "simulation_state.db"
        self.actions_dir = self.output_dir / "agent_actions"
        self.conversations_dir = self.output_dir / "conversations"

        # Clear and recreate output directory
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

        self._init_directories()
        self._init_database()

        # Cache for agent action files (to avoid reading/writing full file each time)
        self._agent_actions_cache: Dict[str, Dict[str, Any]] = {}

    def _init_directories(self) -> None:
        """Create output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.actions_dir.mkdir(parents=True, exist_ok=True)
        self.conversations_dir.mkdir(parents=True, exist_ok=True)

    def _init_database(self) -> None:
        """Create SQLite tables for simulation state."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Main timestep state table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS timestep_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                simulation_time TEXT NOT NULL,
                external_temp_c REAL,
                building_temp_c REAL,
                total_power_w REAL,
                hvac_power_w REAL,
                equipment_power_w REAL,
                lighting_power_w REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Device state per timestep - pivoted schema with one row per simulation_time
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS device_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                simulation_time TEXT NOT NULL,
                laptop_A_on BOOLEAN, laptop_A_power REAL,
                laptop_B_on BOOLEAN, laptop_B_power REAL,
                laptop_C_on BOOLEAN, laptop_C_power REAL,
                monitor_A_on BOOLEAN, monitor_A_power REAL,
                monitor_B_on BOOLEAN, monitor_B_power REAL,
                monitor_C_on BOOLEAN, monitor_C_power REAL,
                photocopier_on BOOLEAN, photocopier_power REAL,
                projector_on BOOLEAN, projector_power REAL,
                conference_phone_on BOOLEAN, conference_phone_power REAL,
                coffee_machine_on BOOLEAN, coffee_machine_power REAL,
                kettle_on BOOLEAN, kettle_power REAL,
                microwave_on BOOLEAN, microwave_power REAL,
                fridge_on BOOLEAN, fridge_power REAL
            )
        """)

        # Light state per timestep - pivoted schema with one row per simulation_time
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS light_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                simulation_time TEXT NOT NULL,
                desk_light_A_on BOOLEAN, desk_light_A_power REAL,
                desk_light_B_on BOOLEAN, desk_light_B_power REAL,
                desk_light_C_on BOOLEAN, desk_light_C_power REAL,
                zone_main_on BOOLEAN, zone_main_power REAL,
                meeting_room_on BOOLEAN, meeting_room_power REAL
            )
        """)

        # Occupant state per timestep
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS occupant_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestep_id INTEGER REFERENCES timestep_state(id),
                agent_id TEXT NOT NULL,
                agent_name TEXT,
                is_in_office BOOLEAN,
                location TEXT,
                current_desk TEXT,
                at_desk BOOLEAN,
                on_break BOOLEAN,
                at_lunch BOOLEAN
            )
        """)

        # Create indices for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestep_time ON timestep_state(simulation_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_device_time ON device_state(simulation_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_light_time ON light_state(simulation_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_occupant_timestep ON occupant_state(timestep_id)")

        conn.commit()
        conn.close()

    def write_timestep(self, timestep_data: Dict[str, Any]) -> None:
        """
        Write current simulation state to database.

        Args:
            timestep_data: Dict containing:
                - simulation_time: ISO format datetime string
                - external_temp_c: External temperature (C)
                - building_temp_c: Building/zone temperature (C)
                - total_power_w: Total power consumption (W)
                - hvac_power_w: HVAC power (W, positive=heating, negative=cooling)
                - equipment_power_w: Equipment power (W)
                - lighting_power_w: Lighting power (W)
                - equipment_state: Dict from EquipmentManager.get_equipment_state()
                - lighting_state: Dict from LightingManager.get_detailed_lighting_state()
                - occupant_states: List of occupant state dicts
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Insert main timestep record
            cursor.execute("""
                INSERT INTO timestep_state (
                    simulation_time, external_temp_c, building_temp_c,
                    total_power_w, hvac_power_w, equipment_power_w, lighting_power_w
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                timestep_data.get("simulation_time"),
                timestep_data.get("external_temp_c"),
                timestep_data.get("building_temp_c"),
                timestep_data.get("total_power_w"),
                timestep_data.get("hvac_power_w"),
                timestep_data.get("equipment_power_w"),
                timestep_data.get("lighting_power_w"),
            ))

            timestep_id = cursor.lastrowid
            simulation_time = timestep_data.get("simulation_time")

            # Insert device state - single row with all devices as columns
            equipment_state = timestep_data.get("equipment_state", {})
            items = equipment_state.get("items", {})

            def get_device(name):
                """Helper to get device info safely."""
                return items.get(name, {})

            cursor.execute("""
                INSERT INTO device_state (
                    simulation_time,
                    laptop_A_on, laptop_A_power,
                    laptop_B_on, laptop_B_power,
                    laptop_C_on, laptop_C_power,
                    monitor_A_on, monitor_A_power,
                    monitor_B_on, monitor_B_power,
                    monitor_C_on, monitor_C_power,
                    photocopier_on, photocopier_power,
                    projector_on, projector_power,
                    conference_phone_on, conference_phone_power,
                    coffee_machine_on, coffee_machine_power,
                    kettle_on, kettle_power,
                    microwave_on, microwave_power,
                    fridge_on, fridge_power
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                simulation_time,
                get_device("laptop_A").get("is_on"), get_device("laptop_A").get("current_power_w", 0),
                get_device("laptop_B").get("is_on"), get_device("laptop_B").get("current_power_w", 0),
                get_device("laptop_C").get("is_on"), get_device("laptop_C").get("current_power_w", 0),
                get_device("monitor_A").get("is_on"), get_device("monitor_A").get("current_power_w", 0),
                get_device("monitor_B").get("is_on"), get_device("monitor_B").get("current_power_w", 0),
                get_device("monitor_C").get("is_on"), get_device("monitor_C").get("current_power_w", 0),
                get_device("photocopier").get("is_on"), get_device("photocopier").get("current_power_w", 0),
                get_device("projector").get("is_on"), get_device("projector").get("current_power_w", 0),
                get_device("conference_phone").get("is_on"), get_device("conference_phone").get("current_power_w", 0),
                get_device("coffee_machine").get("is_on"), get_device("coffee_machine").get("current_power_w", 0),
                get_device("kettle").get("is_on"), get_device("kettle").get("current_power_w", 0),
                get_device("microwave").get("is_on"), get_device("microwave").get("current_power_w", 0),
                get_device("fridge").get("is_on"), get_device("fridge").get("current_power_w", 0),
            ))

            # Insert light state - single row with all lights as columns
            lighting_state = timestep_data.get("lighting_state", {})
            lights = lighting_state.get("lights", {})

            def get_light(name):
                """Helper to get light info safely."""
                return lights.get(name, {})

            cursor.execute("""
                INSERT INTO light_state (
                    simulation_time,
                    desk_light_A_on, desk_light_A_power,
                    desk_light_B_on, desk_light_B_power,
                    desk_light_C_on, desk_light_C_power,
                    zone_main_on, zone_main_power,
                    meeting_room_on, meeting_room_power
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                simulation_time,
                get_light("desk_light_A").get("is_on"), get_light("desk_light_A").get("current_power_w", 0),
                get_light("desk_light_B").get("is_on"), get_light("desk_light_B").get("current_power_w", 0),
                get_light("desk_light_C").get("is_on"), get_light("desk_light_C").get("current_power_w", 0),
                get_light("zone_main").get("is_on"), get_light("zone_main").get("current_power_w", 0),
                get_light("meeting_room").get("is_on"), get_light("meeting_room").get("current_power_w", 0),
            ))

            # Insert occupant states
            for occupant in timestep_data.get("occupant_states", []):
                cursor.execute("""
                    INSERT INTO occupant_state (
                        timestep_id, agent_id, agent_name, is_in_office,
                        location, current_desk, at_desk, on_break, at_lunch
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestep_id,
                    occupant.get("agent_id"),
                    occupant.get("agent_name"),
                    occupant.get("is_in_office"),
                    occupant.get("location"),
                    occupant.get("current_desk"),
                    occupant.get("at_desk"),
                    occupant.get("on_break"),
                    occupant.get("at_lunch"),
                ))

            conn.commit()

        except Exception as e:
            conn.rollback()
            print(f"Warning: Failed to write timestep to UI demo database: {e}")

        finally:
            conn.close()

    def write_agent_action(
        self,
        agent_id: str,
        action_data: Dict[str, Any],
        agent_name: Optional[str] = None,
    ) -> None:
        """
        Append action to agent's JSON file.

        Only records actions that change simulation state (in RECORDABLE_ACTIONS).
        Skips internal actions like take_break, update_daily_plan.

        Args:
            agent_id: The agent ID (e.g., "alice_001")
            action_data: Dict containing:
                - timestamp: ISO format datetime string
                - action_type: Type of action (e.g., "thermostat_adjust")
                - details: Dict with action-specific parameters
                - reason: Brief explanation for the action
            agent_name: Optional human-readable name
        """
        # Filter to only record actions that change simulation state
        action_type = action_data.get("action_type", "")
        if action_type not in RECORDABLE_ACTIONS:
            return  # Skip non-recordable actions

        # Get or create cache entry for this agent
        if agent_id not in self._agent_actions_cache:
            action_file = self.actions_dir / f"{agent_id}_actions.json"
            if action_file.exists():
                try:
                    with open(action_file, "r") as f:
                        self._agent_actions_cache[agent_id] = json.load(f)
                except (json.JSONDecodeError, IOError):
                    self._agent_actions_cache[agent_id] = {
                        "agent_id": agent_id,
                        "agent_name": agent_name or agent_id.split("_")[0].capitalize(),
                        "actions": [],
                    }
            else:
                self._agent_actions_cache[agent_id] = {
                    "agent_id": agent_id,
                    "agent_name": agent_name or agent_id.split("_")[0].capitalize(),
                    "actions": [],
                }

        # Append action
        self._agent_actions_cache[agent_id]["actions"].append(action_data)

        # Write to file
        action_file = self.actions_dir / f"{agent_id}_actions.json"
        try:
            with open(action_file, "w") as f:
                json.dump(self._agent_actions_cache[agent_id], f, indent=2)
        except IOError as e:
            print(f"Warning: Failed to write agent action for {agent_id}: {e}")

    def write_conversation(
        self,
        conversation_id: str,
        conversation_data: Dict[str, Any],
    ) -> None:
        """
        Write conversation to its own JSON file.

        Args:
            conversation_id: Unique ID for the conversation
            conversation_data: Dict containing:
                - start_time: ISO format datetime string
                - participants: List of agent IDs
                - initiated_by: Agent ID who started the conversation
                - topics: List of topics discussed
                - dialogue: List of utterance dicts with speaker, speaker_name, utterance
                - duration_minutes: Duration of conversation
                - summary: Summary of the conversation
        """
        conversation_file = self.conversations_dir / f"{conversation_id}.json"

        # Add conversation_id to the data
        full_data = {"conversation_id": conversation_id, **conversation_data}

        try:
            with open(conversation_file, "w") as f:
                json.dump(full_data, f, indent=2)
            # Update manifest for UI discovery
            self._update_conversations_manifest(conversation_id)
        except IOError as e:
            print(f"Warning: Failed to write conversation {conversation_id}: {e}")

    def _update_conversations_manifest(self, conversation_id: str) -> None:
        """
        Maintain a manifest of all conversation IDs for UI discovery.

        The manifest file allows the UI to discover all conversations without
        relying on time-based filename guessing, which can miss conversations
        if the UI connects after the conversation timestamp window passes.
        """
        manifest_path = self.conversations_dir / "manifest.json"

        # Read existing manifest or create new
        if manifest_path.exists():
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
            except (IOError, json.JSONDecodeError):
                manifest = {"conversations": []}
        else:
            manifest = {"conversations": []}

        # Add new conversation ID if not already present
        if conversation_id not in manifest["conversations"]:
            manifest["conversations"].append(conversation_id)
            try:
                with open(manifest_path, "w") as f:
                    json.dump(manifest, f, indent=2)
            except IOError as e:
                print(f"Warning: Failed to update conversations manifest: {e}")

    def finalize(self) -> None:
        """
        Finalize the UI demo output.

        Call this at the end of the simulation to ensure all data is written.
        """
        # Flush any remaining cached agent actions
        for agent_id, data in self._agent_actions_cache.items():
            action_file = self.actions_dir / f"{agent_id}_actions.json"
            try:
                with open(action_file, "w") as f:
                    json.dump(data, f, indent=2)
            except IOError as e:
                print(f"Warning: Failed to finalize agent actions for {agent_id}: {e}")

        # Vacuum the database to optimize storage
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("VACUUM")
            conn.close()
        except Exception as e:
            print(f"Warning: Failed to vacuum UI demo database: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the UI demo output.

        Returns:
            Dict with counts of timesteps, devices, lights, occupants, actions, conversations
        """
        stats = {
            "timesteps": 0,
            "device_records": 0,
            "light_records": 0,
            "occupant_records": 0,
            "action_files": 0,
            "conversation_files": 0,
        }

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM timestep_state")
            stats["timesteps"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM device_state")
            stats["device_records"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM light_state")
            stats["light_records"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM occupant_state")
            stats["occupant_records"] = cursor.fetchone()[0]

            conn.close()
        except Exception:
            pass

        stats["action_files"] = len(list(self.actions_dir.glob("*_actions.json")))
        stats["conversation_files"] = len(list(self.conversations_dir.glob("*.json")))

        return stats

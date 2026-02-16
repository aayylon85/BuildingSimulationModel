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
                occupant_count INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Device state per timestep
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS device_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestep_id INTEGER REFERENCES timestep_state(id),
                device_name TEXT NOT NULL,
                device_type TEXT,
                is_on BOOLEAN,
                power_w REAL,
                location TEXT,
                assigned_to TEXT
            )
        """)

        # Light state per timestep
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS light_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestep_id INTEGER REFERENCES timestep_state(id),
                light_name TEXT NOT NULL,
                is_on BOOLEAN,
                power_w REAL,
                location TEXT
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
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_device_timestep ON device_state(timestep_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_light_timestep ON light_state(timestep_id)")
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
                    total_power_w, hvac_power_w, equipment_power_w, lighting_power_w,
                    occupant_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestep_data.get("simulation_time"),
                timestep_data.get("external_temp_c"),
                timestep_data.get("building_temp_c"),
                timestep_data.get("total_power_w"),
                timestep_data.get("hvac_power_w"),
                timestep_data.get("equipment_power_w"),
                timestep_data.get("lighting_power_w"),
                len([o for o in timestep_data.get("occupant_states", []) if o.get("is_in_office")]),
            ))

            timestep_id = cursor.lastrowid

            # Insert device states
            equipment_state = timestep_data.get("equipment_state", {})
            items = equipment_state.get("items", {})
            for device_name, device_info in items.items():
                cursor.execute("""
                    INSERT INTO device_state (
                        timestep_id, device_name, device_type, is_on,
                        power_w, location, assigned_to
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestep_id,
                    device_name,
                    device_info.get("equipment_type"),
                    device_info.get("is_on"),
                    device_info.get("current_power_w", 0),
                    device_info.get("location"),
                    device_info.get("assigned_to"),
                ))

            # Insert light states
            lighting_state = timestep_data.get("lighting_state", {})
            lights = lighting_state.get("lights", {})
            for light_name, light_info in lights.items():
                cursor.execute("""
                    INSERT INTO light_state (
                        timestep_id, light_name, is_on, power_w, location
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    timestep_id,
                    light_name,
                    light_info.get("is_on"),
                    light_info.get("current_power_w", 0),
                    light_info.get("location"),
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

        Args:
            agent_id: The agent ID (e.g., "alice_001")
            action_data: Dict containing:
                - timestamp: ISO format datetime string
                - action_type: Type of action (e.g., "thermostat_adjust")
                - details: Dict with action-specific parameters
                - reason: Brief explanation for the action
            agent_name: Optional human-readable name
        """
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
        except IOError as e:
            print(f"Warning: Failed to write conversation {conversation_id}: {e}")

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

"""
Simulation state persistence and resumption for building simulations.

Enables saving simulation state at any point and resuming later.
Checkpoints are manual-only (not automatic).
"""

import json
import shutil
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SimulationCheckpoint:
    """
    Complete simulation state at a point in time.

    Stores everything needed to resume a simulation.
    """
    # Identification
    checkpoint_id: str
    created_at: str  # Real-world time when checkpoint was created (ISO)
    simulation_time_hour: float  # Simulation time in hours from start
    simulation_datetime: str  # Simulation datetime (ISO)

    # Configuration
    config_path: str

    # Thermal state
    zone_air_temp_c: float
    dynamic_heating_setpoint: List[float]
    dynamic_cooling_setpoint: List[float]
    current_window_fraction: float

    # Agent state (references to agent folders)
    agent_run_dir: str  # Path to agents/{date}/{time}/ directory
    agent_ids: List[str]
    calendar_db_path: str  # Path to calendar.sqlite

    # Progress
    current_step: int = 0
    total_steps: int = 0

    # Optional accumulated results (for analysis)
    results_so_far: Optional[Dict[str, Any]] = None


class SimulationPersistence:
    """
    Manages saving and loading simulation checkpoints.

    Checkpoints include:
    - Thermal state (zone temperature, setpoints, window state)
    - Agent state (memories, scratch, core memories)
    - Calendar state (meetings, invitations, RSVPs)
    - Progress info (current step, total steps)
    """

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.checkpoints_dir = self.results_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        checkpoint: SimulationCheckpoint,
        include_full_state: bool = True,
    ) -> Path:
        """
        Save a simulation checkpoint.

        Args:
            checkpoint: The checkpoint to save
            include_full_state: If True, copy agent state files and calendar into checkpoint

        Returns:
            Path to saved checkpoint directory
        """
        checkpoint_dir = self.checkpoints_dir / checkpoint.checkpoint_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save checkpoint metadata
        meta_path = checkpoint_dir / "checkpoint.json"
        with open(meta_path, 'w') as f:
            json.dump(asdict(checkpoint), f, indent=2, default=str)

        if include_full_state:
            # Copy agent state
            agent_src = Path(checkpoint.agent_run_dir)
            agent_dest = checkpoint_dir / "agents"
            if agent_src.exists():
                if agent_dest.exists():
                    shutil.rmtree(agent_dest)
                shutil.copytree(agent_src, agent_dest)

            # Copy calendar database
            cal_src = Path(checkpoint.calendar_db_path)
            if cal_src.exists():
                cal_dest = checkpoint_dir / "calendar.sqlite"
                shutil.copy2(cal_src, cal_dest)

        print(f"[CHECKPOINT] Saved: {checkpoint_dir}")
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_id: str) -> SimulationCheckpoint:
        """
        Load a simulation checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to load

        Returns:
            SimulationCheckpoint with loaded state
        """
        checkpoint_dir = self.checkpoints_dir / checkpoint_id
        meta_path = checkpoint_dir / "checkpoint.json"

        if not meta_path.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")

        with open(meta_path, 'r') as f:
            data = json.load(f)

        # Update paths to point to checkpoint copies if they exist
        checkpoint_agents = checkpoint_dir / "agents"
        if checkpoint_agents.exists():
            data["agent_run_dir"] = str(checkpoint_agents)

        checkpoint_cal = checkpoint_dir / "calendar.sqlite"
        if checkpoint_cal.exists():
            data["calendar_db_path"] = str(checkpoint_cal)

        return SimulationCheckpoint(**data)

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints with summary info.

        Returns:
            List of checkpoint summaries sorted by creation time (newest first)
        """
        checkpoints = []

        for cp_dir in self.checkpoints_dir.iterdir():
            if cp_dir.is_dir():
                meta_path = cp_dir / "checkpoint.json"
                if meta_path.exists():
                    try:
                        with open(meta_path, 'r') as f:
                            data = json.load(f)
                        checkpoints.append({
                            "id": data["checkpoint_id"],
                            "created_at": data["created_at"],
                            "simulation_datetime": data["simulation_datetime"],
                            "simulation_time_hours": data.get("simulation_time_hour", 0),
                            "progress": f"{data.get('current_step', 0)}/{data.get('total_steps', '?')}",
                            "zone_temp_c": data.get("zone_air_temp_c"),
                        })
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Warning: Could not read checkpoint {cp_dir}: {e}")

        return sorted(checkpoints, key=lambda x: x["created_at"], reverse=True)

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to delete

        Returns:
            True if deleted, False if not found
        """
        checkpoint_dir = self.checkpoints_dir / checkpoint_id
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            print(f"[CHECKPOINT] Deleted: {checkpoint_id}")
            return True
        return False

    def get_checkpoint_path(self, checkpoint_id: str) -> Optional[Path]:
        """Get the path to a checkpoint directory."""
        checkpoint_dir = self.checkpoints_dir / checkpoint_id
        return checkpoint_dir if checkpoint_dir.exists() else None


def create_checkpoint(
    checkpoint_id: str,
    config_path: str,
    simulation_time_hour: float,
    simulation_datetime: datetime,
    zone_air_temp_c: float,
    dynamic_heating_setpoint: List[float],
    dynamic_cooling_setpoint: List[float],
    current_window_fraction: float,
    llm_manager,  # LLMOccupantManager
    current_step: int,
    total_steps: int,
    results_so_far: Optional[Dict[str, Any]] = None,
) -> SimulationCheckpoint:
    """
    Create a checkpoint from current simulation state.

    Should be called from runner.py during simulation loop.

    Args:
        checkpoint_id: Unique identifier for this checkpoint
        config_path: Path to simulation config file
        simulation_time_hour: Current simulation time in hours from start
        simulation_datetime: Current simulation datetime
        zone_air_temp_c: Current zone air temperature
        dynamic_heating_setpoint: Current heating setpoint array
        dynamic_cooling_setpoint: Current cooling setpoint array
        current_window_fraction: Current window open fraction
        llm_manager: LLMOccupantManager instance (to get agent paths)
        current_step: Current simulation step
        total_steps: Total simulation steps
        results_so_far: Optional accumulated results dict

    Returns:
        SimulationCheckpoint ready to be saved
    """
    # Get agent paths from manager
    agent_run_dir = ""
    calendar_db_path = ""
    agent_ids = []

    if llm_manager is not None:
        agent_paths = getattr(llm_manager, '_agent_paths', {})
        agent_run_dir = agent_paths.get("run_dir", "")
        shared_paths = agent_paths.get("shared", {})
        calendar_db_path = shared_paths.get("calendar_db", "")
        agent_ids = llm_manager.get_agent_ids() if hasattr(llm_manager, 'get_agent_ids') else []

    return SimulationCheckpoint(
        checkpoint_id=checkpoint_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        simulation_time_hour=simulation_time_hour,
        simulation_datetime=simulation_datetime.isoformat(),
        config_path=config_path,
        zone_air_temp_c=zone_air_temp_c,
        dynamic_heating_setpoint=list(dynamic_heating_setpoint),
        dynamic_cooling_setpoint=list(dynamic_cooling_setpoint),
        current_window_fraction=current_window_fraction,
        agent_run_dir=agent_run_dir,
        agent_ids=agent_ids,
        calendar_db_path=calendar_db_path,
        current_step=current_step,
        total_steps=total_steps,
        results_so_far=results_so_far,
    )

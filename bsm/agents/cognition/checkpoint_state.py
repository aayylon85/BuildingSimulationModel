"""
checkpoint_state.py

Checkpoint state tracking for the 6-step decision flow.

The CheckpointState dataclass maintains state across all 6 steps of a checkpoint,
including:
- Location tracking (initial and current after move)
- Context (simulation state, memories, daily plan)
- Step outputs (populated as each step completes)
- Accumulated actions (for final execution)

This allows each step to reference decisions from previous steps and ensures
proper action ordering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from bsm.agents.skeleton import (
    OccupantAction,
    PlanReviewDecision,
    CurrentLocationDecision,
    StepConversationDecision,
    MoveDecision,
    NewLocationDecision,
    DailyPlan,
    LOCATION_TRANSITIONS,
)

if TYPE_CHECKING:
    from bsm.agents.cognition.sync_manager import CommitmentSyncManager
    from bsm.agents.generative_agent import GenerativeAgent
    from bsm.agents.memory.stream import MemoryNode

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    """
    Tracks state across the 6-step checkpoint flow.

    This dataclass is created at the start of each checkpoint and passed
    through all 6 steps, accumulating decisions and actions as it goes.

    Attributes:
        agent_id: The agent making decisions
        now: Current simulation datetime
        checkpoint_reason: Why this checkpoint was triggered
        initial_location: Where the agent started
        current_location: Where the agent is now (updates after move)
        sim_state: Full simulation state dict
        memories: Retrieved memories by focal point
        daily_plan: Agent's plan for the day
        sync_manager: Reference to commitment sync manager
        agent: Reference to the GenerativeAgent
        step1-step6: Outputs from each step (populated as steps complete)
        actions: Accumulated list of actions to execute
    """

    # Core identifiers
    agent_id: str
    now: datetime
    checkpoint_reason: str

    # Location tracking
    initial_location: str
    current_location: str  # Updates after move in step 4

    # Context (set once at checkpoint start)
    sim_state: Dict[str, Any]
    memories: Dict[str, List["MemoryNode"]]
    daily_plan: Optional[DailyPlan]
    sync_manager: "CommitmentSyncManager"
    agent: "GenerativeAgent"

    # Step outputs (populated as we progress through steps)
    step1: Optional[PlanReviewDecision] = None
    step2: Optional[CurrentLocationDecision] = None
    step3: Optional[StepConversationDecision] = None
    step4: Optional[MoveDecision] = None
    step5: Optional[NewLocationDecision] = None
    step6: Optional[StepConversationDecision] = None

    # Accumulated actions for execution
    actions: List[OccupantAction] = field(default_factory=list)

    def add_action(self, action_type: str, **params) -> None:
        """
        Add an action to the accumulated list.

        Args:
            action_type: The type of action (e.g., 'move_to', 'equipment_set')
            **params: Action parameters
        """
        action = OccupantAction(action_type=action_type, parameters=params)
        self.actions.append(action)
        logger.debug(f"{self.agent_id}: Added action {action_type} with params {params}")

    def moved(self) -> bool:
        """Check if the agent decided to move in step 4."""
        return self.step4 is not None and self.step4.action == "move"

    def get_destination(self) -> Optional[str]:
        """Get the move destination if agent decided to move."""
        if self.moved() and self.step4.destination:
            return self.step4.destination
        return None

    def has_colleagues_at(self, location: str) -> bool:
        """
        Check if there are colleagues at a given location.

        Args:
            location: The location to check

        Returns:
            True if at least one colleague is there
        """
        agents_by_location = self.sim_state.get("agents_by_location", {})
        agents_at_loc = agents_by_location.get(location, [])
        # Filter out self to get only colleagues
        colleagues = [a for a in agents_at_loc if a != self.agent_id]
        return len(colleagues) > 0

    def get_colleagues_at_current_location(self) -> List[str]:
        """Get list of colleague IDs at current location."""
        agents_by_location = self.sim_state.get("agents_by_location", {})
        agents_at_loc = agents_by_location.get(self.current_location, [])
        # Filter out self to get only colleagues
        return [a for a in agents_at_loc if a != self.agent_id]

    def can_move_to(self, destination: str) -> bool:
        """
        Check if a move to the destination is valid.

        Uses LOCATION_TRANSITIONS to enforce valid moves.

        Args:
            destination: Target location

        Returns:
            True if the move is allowed
        """
        valid_destinations = LOCATION_TRANSITIONS.get(self.current_location, [])
        return destination in valid_destinations

    def get_valid_destinations(self) -> List[str]:
        """Get list of locations the agent can move to from current location."""
        return LOCATION_TRANSITIONS.get(self.current_location, [])

    def get_equipment_at_location(self, location: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get equipment available at a location.

        Args:
            location: Location to check (defaults to current_location)

        Returns:
            List of equipment dicts
        """
        if location is None:
            location = self.current_location

        equipment_by_location = self.sim_state.get("equipment_at_location", {})
        return equipment_by_location.get(location, [])

    def get_commitment_status(self) -> Optional[str]:
        """Get status of any active commitment from step 1."""
        if self.step1 and self.step1.commitment_status:
            return self.step1.commitment_status
        return None

    def get_priorities(self) -> List[str]:
        """Get priorities from step 1."""
        if self.step1:
            return self.step1.priorities
        return []

    def summary(self) -> str:
        """
        Generate a summary of the checkpoint state for logging.

        Returns:
            Human-readable summary string
        """
        lines = [
            f"Checkpoint: {self.checkpoint_reason}",
            f"Agent: {self.agent_id}",
            f"Time: {self.now.strftime('%H:%M')}",
            f"Location: {self.initial_location}",
        ]

        if self.step1:
            lines.append(f"Priorities: {', '.join(self.step1.priorities)}")
            if self.step1.commitment_status:
                lines.append(f"Commitment: {self.step1.commitment_status}")

        if self.moved():
            lines.append(f"Moving to: {self.step4.destination}")

        lines.append(f"Actions: {len(self.actions)}")

        return " | ".join(lines)

    def retrieve_for_step(self, step_name: str) -> Dict[str, List["MemoryNode"]]:
        """
        Retrieve memories relevant to the current decision step.

        Each step has different focal points to retrieve contextually
        appropriate memories. This replaces the single retrieval at
        checkpoint start with per-step memory access.

        Args:
            step_name: One of "plan_review", "current_location_actions",
                       "current_location_conversation", "move_decision",
                       "new_location_actions", "new_location_conversation"

        Returns:
            Dict mapping focal points to retrieved MemoryNode lists
        """
        from bsm.agents.cognition.retrieval import retrieve

        step_focal_points = {
            "plan_review": [
                "my plan for today",
                "my priorities",
                "commitments I made",
            ],
            "current_location_actions": [
                "my comfort preferences",
                f"what I do at {self.current_location}",
            ],
            "current_location_conversation": [
                "my relationship with colleagues here",
                "recent conversations",
            ],
            "move_decision": [
                "where I need to be",
                "upcoming commitments",
                "location preferences",
            ],
            "new_location_actions": [
                f"what I usually do at {self.current_location}",
                "equipment I need here",
            ],
            "new_location_conversation": [
                "relationship with colleagues here",
                "topics to discuss",
            ],
        }

        focal_points = step_focal_points.get(step_name, [])
        if not focal_points:
            logger.warning(f"Unknown step name: {step_name}, no focal points defined")
            return {}

        return retrieve(self.agent, focal_points, self.now)

    async def refresh_perception_at_new_location(self, adapter) -> List["MemoryNode"]:
        """
        Perceive new environment after moving to a different location.

        When an agent moves during a checkpoint, they should perceive their
        new environment before making decisions there. This method:
        1. Gets fresh simulation state from the adapter
        2. Runs perception to observe new surroundings
        3. Returns perceived observations as MemoryNodes

        Args:
            adapter: SimulationAdapter instance to get fresh state

        Returns:
            List of MemoryNode observations from perception
        """
        from bsm.agents.cognition.perception import perceive

        # Get fresh simulation state at new location
        self.sim_state = adapter.get_state(self.agent_id, self.now)

        # Perceive new environment
        perceived = await perceive(self.agent, self.sim_state, self.now)

        logger.debug(
            f"{self.agent_id}: Perceived {len(perceived)} observations "
            f"at new location {self.current_location}"
        )

        return perceived


def create_checkpoint_state(
    agent_id: str,
    now: datetime,
    checkpoint_reason: str,
    sim_state: Dict[str, Any],
    memories: Dict[str, List["MemoryNode"]],
    daily_plan: Optional[DailyPlan],
    sync_manager: "CommitmentSyncManager",
    agent: "GenerativeAgent",
) -> CheckpointState:
    """
    Factory function to create a CheckpointState.

    This is the primary way to create a CheckpointState at the start
    of a checkpoint.

    Args:
        agent_id: The agent making decisions
        now: Current simulation datetime
        checkpoint_reason: Why this checkpoint was triggered
        sim_state: Full simulation state dict
        memories: Retrieved memories by focal point
        daily_plan: Agent's plan for the day
        sync_manager: Reference to commitment sync manager
        agent: Reference to the GenerativeAgent

    Returns:
        Initialized CheckpointState
    """
    current_location = sim_state.get("current_location", "desk_area")

    state = CheckpointState(
        agent_id=agent_id,
        now=now,
        checkpoint_reason=checkpoint_reason,
        initial_location=current_location,
        current_location=current_location,
        sim_state=sim_state,
        memories=memories,
        daily_plan=daily_plan,
        sync_manager=sync_manager,
        agent=agent,
    )

    logger.debug(f"{agent_id}: Created checkpoint state for '{checkpoint_reason}'")
    return state

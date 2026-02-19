"""
sync_manager.py

Commitment synchronization manager for coordinating agents with shared commitments.

When two agents agree to meet (e.g., "coffee at 15:00"), the CommitmentSyncManager
tracks their locations and coordination status so they can actually meet.

This solves the problem where agents would agree to meet but then:
- Go to different locations
- Not know if the other person is ready
- Keep waiting indefinitely without feedback

The sync manager is integrated into Step 1 (plan review) and Step 4 (move decision)
of the 6-step checkpoint flow to provide partner status information.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Literal, Optional, Tuple

logger = logging.getLogger(__name__)


AgentStatus = Literal["not_ready", "approaching", "at_location", "waiting", "fulfilled"]


@dataclass
class CommitmentSync:
    """Tracking state for a shared commitment between agents."""
    commitment_id: str
    agents: List[str]
    location: str
    time: datetime
    activity: str
    agent_status: Dict[str, AgentStatus] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    fulfilled_at: Optional[datetime] = None

    def __post_init__(self):
        # Initialize all agents as not_ready
        if not self.agent_status:
            self.agent_status = {agent: "not_ready" for agent in self.agents}


class CommitmentSyncManager:
    """
    Coordinates agents for shared commitments.

    When a commitment is registered (typically after conversation extraction),
    the sync manager tracks each agent's progress toward fulfilling it:

    - "not_ready": Agent hasn't started moving toward commitment
    - "approaching": Agent is moving toward the commitment location
    - "at_location": Agent is at the commitment location
    - "waiting": Agent is at location and waiting for partner
    - "fulfilled": Commitment has been fulfilled

    The sync manager provides partner status information that agents can use
    in their decision-making (shown in Step 1 plan review and Step 4 move decision).
    """

    def __init__(self):
        # commitment_id -> CommitmentSync
        self._syncs: Dict[str, CommitmentSync] = {}
        # agent_id -> current location (updated by simulation adapter)
        self._agent_locations: Dict[str, str] = {}
        # agent_id -> when they started waiting (for timeout calculation)
        self._waiting_since: Dict[str, datetime] = {}

    def register(
        self,
        commitment_id: str,
        agents: List[str],
        location: str,
        time: datetime,
        activity: str,
        now: Optional[datetime] = None,
    ) -> None:
        """
        Register a shared commitment for coordination.

        Called after commitment extraction from conversation.

        Args:
            commitment_id: Unique ID (typically f"{activity}_{time}")
            agents: List of agent IDs involved
            location: Where they're meeting
            time: When they're meeting
            activity: What they're doing (coffee, walk, etc.)
            now: Current simulation time
        """
        if commitment_id in self._syncs:
            logger.debug(f"Commitment {commitment_id} already registered, updating")

        self._syncs[commitment_id] = CommitmentSync(
            commitment_id=commitment_id,
            agents=agents,
            location=location,
            time=time,
            activity=activity,
            created_at=now,
        )

        logger.info(
            f"Registered commitment sync: {commitment_id} "
            f"(agents={agents}, location={location}, time={time.strftime('%H:%M')})"
        )

    def update_agent_location(self, agent_id: str, location: str) -> None:
        """
        Update an agent's current location.

        Called by simulation adapter when agent moves.
        Automatically updates status in any relevant commitments.

        Args:
            agent_id: The agent who moved
            location: Their new location
        """
        old_location = self._agent_locations.get(agent_id)
        self._agent_locations[agent_id] = location

        if old_location != location:
            logger.debug(f"{agent_id}: Location updated {old_location} -> {location}")

        # Update status in all commitments this agent is part of
        for sync in self._syncs.values():
            if agent_id in sync.agents:
                if location == sync.location:
                    # Agent arrived at commitment location
                    sync.agent_status[agent_id] = "at_location"
                    logger.debug(
                        f"{agent_id}: Arrived at commitment location "
                        f"for {sync.commitment_id}"
                    )
                elif sync.agent_status.get(agent_id) == "at_location":
                    # Agent left commitment location
                    sync.agent_status[agent_id] = "not_ready"

    def notify_agent_approaching(
        self,
        commitment_id: str,
        agent_id: str,
    ) -> None:
        """
        Mark an agent as approaching a commitment.

        Called when agent decides to move toward commitment location.

        Args:
            commitment_id: The commitment being approached
            agent_id: The agent moving
        """
        sync = self._syncs.get(commitment_id)
        if not sync:
            return

        if agent_id in sync.agents:
            sync.agent_status[agent_id] = "approaching"
            logger.debug(f"{agent_id}: Approaching commitment {commitment_id}")

    def notify_agent_waiting(
        self,
        commitment_id: str,
        agent_id: str,
        now: datetime,
    ) -> None:
        """
        Mark an agent as waiting for partner(s).

        Called when agent is at location but partner hasn't arrived.

        Args:
            commitment_id: The commitment being waited on
            agent_id: The agent waiting
            now: Current simulation time
        """
        sync = self._syncs.get(commitment_id)
        if not sync:
            return

        if agent_id in sync.agents:
            sync.agent_status[agent_id] = "waiting"
            self._waiting_since[agent_id] = now
            logger.debug(f"{agent_id}: Waiting for partner at {commitment_id}")

    def get_partner_status(
        self,
        agent_id: str,
        commitment_id: str,
    ) -> Optional[str]:
        """
        Get human-readable status of partner(s) for a commitment.

        Used in Step 1 (plan review) and Step 4 (move decision) prompts.

        Args:
            agent_id: The agent asking
            commitment_id: The commitment to check

        Returns:
            String like "bob_002: at_location" or None if not found
        """
        sync = self._syncs.get(commitment_id)
        if not sync:
            return None

        partners = [a for a in sync.agents if a != agent_id]
        if not partners:
            return None

        statuses = []
        for partner in partners:
            status = sync.agent_status.get(partner, "not_ready")
            location = self._agent_locations.get(partner, "unknown")
            statuses.append(f"{partner}: {status} (at {location})")

        return "; ".join(statuses)

    def all_at_location(self, commitment_id: str) -> bool:
        """
        Check if all agents are at the commitment location.

        Args:
            commitment_id: The commitment to check

        Returns:
            True if all agents are at_location or waiting
        """
        sync = self._syncs.get(commitment_id)
        if not sync:
            return False

        for agent_id in sync.agents:
            status = sync.agent_status.get(agent_id, "not_ready")
            if status not in ("at_location", "waiting"):
                return False

        return True

    def check_timeout(
        self,
        commitment_id: str,
        agent_id: str,
        now: datetime,
        max_wait_minutes: int = 10,
    ) -> Tuple[bool, int]:
        """
        Check if an agent has been waiting too long.

        Args:
            commitment_id: The commitment being waited on
            agent_id: The waiting agent
            now: Current simulation time
            max_wait_minutes: Maximum wait time before timeout

        Returns:
            Tuple of (is_timed_out, minutes_waited)
        """
        sync = self._syncs.get(commitment_id)
        if not sync:
            return False, 0

        if sync.agent_status.get(agent_id) != "waiting":
            return False, 0

        waiting_since = self._waiting_since.get(agent_id)
        if not waiting_since:
            return False, 0

        minutes_waited = int((now - waiting_since).total_seconds() / 60)
        is_timed_out = minutes_waited >= max_wait_minutes

        return is_timed_out, minutes_waited

    def mark_fulfilled(
        self,
        commitment_id: str,
        now: datetime,
    ) -> None:
        """
        Mark a commitment as fulfilled.

        Called when agents successfully complete the commitment.

        Args:
            commitment_id: The commitment that was fulfilled
            now: Current simulation time
        """
        sync = self._syncs.get(commitment_id)
        if not sync:
            return

        sync.fulfilled_at = now
        for agent_id in sync.agents:
            sync.agent_status[agent_id] = "fulfilled"
            # Clear waiting state
            self._waiting_since.pop(agent_id, None)

        logger.info(f"Commitment {commitment_id} fulfilled at {now.strftime('%H:%M')}")

    def get_active_commitments_for_agent(
        self,
        agent_id: str,
        now: datetime,
        time_window_minutes: int = 30,
    ) -> List[CommitmentSync]:
        """
        Get active (unfulfilled) commitments for an agent within time window.

        Args:
            agent_id: The agent to check
            now: Current simulation time
            time_window_minutes: How far ahead to look

        Returns:
            List of CommitmentSync objects
        """
        active = []
        window_start = now - timedelta(minutes=15)  # Include slightly past
        window_end = now + timedelta(minutes=time_window_minutes)

        for sync in self._syncs.values():
            if agent_id not in sync.agents:
                continue
            if sync.fulfilled_at is not None:
                continue
            if window_start <= sync.time <= window_end:
                active.append(sync)

        return active

    def cleanup_old_commitments(
        self,
        now: datetime,
        max_age_hours: int = 24,
    ) -> int:
        """
        Remove old commitments to prevent memory growth.

        Args:
            now: Current simulation time
            max_age_hours: Remove commitments older than this

        Returns:
            Number of commitments removed
        """
        cutoff = now - timedelta(hours=max_age_hours)
        to_remove = []

        for commitment_id, sync in self._syncs.items():
            if sync.created_at and sync.created_at < cutoff:
                to_remove.append(commitment_id)

        for commitment_id in to_remove:
            del self._syncs[commitment_id]

        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} old commitments")

        return len(to_remove)

    def clear_all(self) -> None:
        """Clear all sync state (e.g., start of new simulation)."""
        self._syncs.clear()
        self._agent_locations.clear()
        self._waiting_since.clear()
        logger.debug("Cleared all commitment sync state")

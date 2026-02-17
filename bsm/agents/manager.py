"""
llm_integration.py

Main integration orchestrator that ties together LLM-based occupant agents
with the building simulation.
"""

from __future__ import annotations

import asyncio
import os
import random
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import OpenAI, APIError, APIConnectionError, RateLimitError
from agents import Agent, Runner

from bsm.agents.skeleton import (
    CalendarStore,
    ClothingChoice,
    EmbeddingClient,
    OccupantStepDecision,
    DailyPlan,
    LunchPlan,
    BreakPlan,
    StepDecisions,
    DeskSelectionDecision,
    SimContext,
    SocialCommitment,
    build_step_agent,
    build_arrival_agent,
    build_day_planner_agent,
    DEFAULT_AGENT_MODEL,
    DEFAULT_EMBED_MODEL,
)
from bsm.agents.generative_agent import GenerativeAgent, initialize_agents_for_simulation
from bsm.agents.cognition.modules import (
    perceive,
    retrieve,
    reflect,
    get_decision_focal_points,
    get_planning_focal_points,
    get_meeting_context,
    format_colleague_context,
    format_step_prompt,
    format_planning_prompt,
    record_decision_to_memory,
    record_plan_to_memory,
    get_importance,
)
from bsm.agents.cognition.conversation import (
    consultation_conversation,
    agent_conversation,
    record_agreement_to_memory,
    record_conversation_to_memory,
    ConsultationOutcome,
    ConversationResult,
)
from bsm.agents.memory.stream import MemoryStream, MemoryNode
from bsm.agents.equipment_manager import EquipmentManager
from bsm.agents.lighting_manager import LightingManager
from bsm.agents.desk_manager import DeskManager
from bsm.agents.simulation_adapter import (
    ZoneStateProvider,
    ProductionSimulationAdapter,
    convert_step_decisions_to_actions,
)


# ---------------------------------------------------------------------------
# Async Retry Helper for OpenAI API Calls
# ---------------------------------------------------------------------------

async def run_with_retry(
    agent: Agent,
    prompt: str,
    context: Any = None,
    max_turns: int = 5,
    max_retries: int = 3,
    base_delay: float = 1.0,
):
    """
    Run an agent with exponential backoff retry for transient API failures.

    Handles:
    - RateLimitError (429): Rate limit exceeded
    - APIError (5xx): Server errors
    - APIConnectionError: Network issues

    Args:
        agent: The OpenAI Agent to run
        prompt: The prompt to send
        context: Optional SimContext
        max_turns: Maximum agent turns
        max_retries: Maximum retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)

    Returns:
        Runner result on success

    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    for attempt in range(max_retries):
        try:
            if context:
                return await Runner.run(agent, prompt, context=context, max_turns=max_turns)
            else:
                return await Runner.run(agent, prompt, max_turns=max_turns)
        except (RateLimitError, APIError, APIConnectionError) as e:
            last_exception = e
            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                print(f"[RETRY] API call failed (attempt {attempt + 1}/{max_retries}): {type(e).__name__}. "
                      f"Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
    raise last_exception


# ---------------------------------------------------------------------------
# Decision Checkpoint Manager
# ---------------------------------------------------------------------------

class DecisionCheckpointManager:
    """
    Manages when agents should make decisions based on:
    - Hourly checkpoints (every 60 minutes)
    - Event-triggered checkpoints (meeting start/end, lunch, breaks)
    """

    def __init__(self):
        # Track last hourly checkpoint per agent
        self._last_hourly_check: Dict[str, datetime] = {}
        # Track which events have been processed per agent
        self._processed_events: Dict[str, set] = {}
        # Track when agents started lunch/break for return checkpoints
        self._lunch_started: Dict[str, datetime] = {}
        self._break_started: Dict[str, datetime] = {}
        # F.4: Track when meeting_end checkpoints fire (actual meeting end times)
        self._meeting_ended_at: Dict[str, Dict[str, datetime]] = {}  # agent_id -> {meeting_id -> datetime}

    def should_checkpoint(
        self,
        agent_id: str,
        current_time: datetime,
        daily_plan: Optional[DailyPlan],
        agent_status: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """
        Determine if agent should make decisions now.

        Args:
            agent_id: The agent ID
            current_time: Current simulation datetime
            daily_plan: The agent's daily plan (if available)
            agent_status: Current agent status dict with location, in_meeting, etc.
                         Used to protect meetings from non-essential checkpoints.

        Returns:
            Tuple of (should_decide, reason) where reason is one of:
            - "hourly": Regular hourly checkpoint
            - "meeting_start:{title}": Meeting is starting
            - "meeting_end:{title}": Meeting is ending
            - "lunch_time": Time for planned lunch
            - "return_from_lunch": Time to return from lunch (30-60 min after start)
            - "take_break:morning": Time for morning break
            - "take_break:afternoon": Time for afternoon break
            - "return_from_break": Time to return from break (10-20 min after start)
            - "commitment:{activity}": Time for a social commitment
            - "": No checkpoint needed
        """
        if agent_id not in self._processed_events:
            self._processed_events[agent_id] = set()

        # F.1: Date prefix for event IDs - prevents accumulation across days
        date_prefix = current_time.date().isoformat()

        # Check hourly checkpoint first
        last_check = self._last_hourly_check.get(agent_id)
        if last_check is None or (current_time - last_check).total_seconds() >= 3600:
            self._last_hourly_check[agent_id] = current_time
            return True, "hourly"

        # Check lunch return checkpoint (30-60 min after start)
        if agent_id in self._lunch_started:
            minutes = (current_time - self._lunch_started[agent_id]).total_seconds() / 60
            if 30 <= minutes <= 60:
                del self._lunch_started[agent_id]
                return True, "return_from_lunch"

        # Check break return checkpoint (10-20 min after start)
        if agent_id in self._break_started:
            minutes = (current_time - self._break_started[agent_id]).total_seconds() / 60
            if 10 <= minutes <= 20:
                del self._break_started[agent_id]
                return True, "return_from_break"

        # If no daily plan, only use hourly checkpoints
        if not daily_plan:
            return False, ""

        current_time_str = current_time.strftime("%H:%M")
        current_minutes = current_time.hour * 60 + current_time.minute

        # ================================================================
        # MEETING PROTECTION: Determine if agent is currently in a meeting
        # During meetings, we block non-essential checkpoints (breaks, lunch,
        # commitments, work activities) to prevent agents from leaving meetings.
        # ================================================================
        in_meeting = False

        # Check if current time falls within any meeting's time window
        for meeting in daily_plan.meetings:
            try:
                start_dt = datetime.fromisoformat(meeting.start_datetime_iso)
                end_dt = datetime.fromisoformat(meeting.end_datetime_iso)
                start_minutes = start_dt.hour * 60 + start_dt.minute
                end_minutes = end_dt.hour * 60 + end_dt.minute

                # F.4: Check if this meeting has actually ended (meeting_end checkpoint fired)
                meeting_id = f"{meeting.title}_{meeting.start_datetime_iso}"
                agent_meetings = self._meeting_ended_at.get(agent_id, {})
                if meeting_id in agent_meetings:
                    # Meeting has ended - use actual end time
                    actual_end = agent_meetings[meeting_id]
                    if current_time >= actual_end:
                        continue  # Meeting is over, don't count as in_meeting

                # If current time is between meeting start and (scheduled) end, agent is "in meeting"
                if start_minutes <= current_minutes < end_minutes:
                    in_meeting = True
                    break
            except (ValueError, TypeError):
                continue

        # Also check agent_status if provided (location-based detection)
        if agent_status:
            if agent_status.get("in_meeting", False):
                in_meeting = True
            # Being in meeting_room during a meeting time strongly suggests in meeting
            elif agent_status.get("location") == "meeting_room" and in_meeting:
                pass  # Already set by time check

        # Check meeting checkpoints
        for meeting in daily_plan.meetings:
            meeting_id = f"{meeting.title}_{meeting.start_datetime_iso}"

            # Parse meeting times
            try:
                start_dt = datetime.fromisoformat(meeting.start_datetime_iso)
                end_dt = datetime.fromisoformat(meeting.end_datetime_iso)
                start_minutes = start_dt.hour * 60 + start_dt.minute
                end_minutes = end_dt.hour * 60 + end_dt.minute
            except (ValueError, TypeError):
                continue

            # Skip meetings that have already ended (prevents trying to join ended meetings)
            time_since_end = current_minutes - end_minutes
            if time_since_end > 0:
                # Meeting has ended - don't trigger any checkpoints for it
                continue

            # Calculate time relative to meeting
            time_to_start = start_minutes - current_minutes
            time_to_end = end_minutes - current_minutes

            # Meeting prep checkpoint (6-10 minutes BEFORE start, non-overlapping with start)
            prep_event_id = f"meeting_prep:{meeting_id}"
            if (6 <= time_to_start <= 10 and
                    prep_event_id not in self._processed_events[agent_id]):
                self._processed_events[agent_id].add(prep_event_id)
                return True, f"meeting_prep:{meeting.title}"

            # Meeting start checkpoint (0-5 minutes BEFORE start only, not after)
            # This is non-overlapping: prep=6-10min, start=0-5min before
            start_event_id = f"meeting_start:{meeting_id}"
            if (0 <= time_to_start <= 5 and
                    start_event_id not in self._processed_events[agent_id]):
                self._processed_events[agent_id].add(start_event_id)
                return True, f"meeting_start:{meeting.title}"

            # Meeting end checkpoint (0-5 minutes BEFORE end only, not after)
            # This gives agent time to wrap up meeting activities
            end_event_id = f"meeting_end:{meeting_id}"
            if (0 <= time_to_end <= 5 and
                    end_event_id not in self._processed_events[agent_id]):
                self._processed_events[agent_id].add(end_event_id)
                # F.4: Record when meeting actually ends (for smarter protection)
                if agent_id not in self._meeting_ended_at:
                    self._meeting_ended_at[agent_id] = {}
                self._meeting_ended_at[agent_id][meeting_id] = current_time
                return True, f"meeting_end:{meeting.title}"

        # ================================================================
        # NON-ESSENTIAL CHECKPOINTS: Only trigger when NOT in a meeting
        # These checkpoints (lunch, breaks, work activities, commitments)
        # should not interrupt active meetings.
        # ================================================================
        if not in_meeting:
            # Check lunch checkpoint
            if daily_plan.lunch_plan:
                lunch_event_id = f"{date_prefix}:lunch:{daily_plan.lunch_plan.time}"
                try:
                    lunch_hour, lunch_minute = map(int, daily_plan.lunch_plan.time.split(":"))
                    lunch_minutes = lunch_hour * 60 + lunch_minute
                    if (abs(current_minutes - lunch_minutes) <= 5 and
                            lunch_event_id not in self._processed_events[agent_id]):
                        self._processed_events[agent_id].add(lunch_event_id)
                        return True, "lunch_time"
                except (ValueError, TypeError):
                    pass

            # Check morning break checkpoint
            if daily_plan.morning_break:
                break_event_id = f"{date_prefix}:take_break:morning:{daily_plan.morning_break.preferred_time}"
                try:
                    break_hour, break_minute = map(int, daily_plan.morning_break.preferred_time.split(":"))
                    break_minutes = break_hour * 60 + break_minute
                    if (abs(current_minutes - break_minutes) <= 5 and
                            break_event_id not in self._processed_events[agent_id]):
                        self._processed_events[agent_id].add(break_event_id)
                        return True, "take_break:morning"
                except (ValueError, TypeError):
                    pass

            # Check afternoon break checkpoint
            if daily_plan.afternoon_break:
                break_event_id = f"{date_prefix}:take_break:afternoon:{daily_plan.afternoon_break.preferred_time}"
                try:
                    break_hour, break_minute = map(int, daily_plan.afternoon_break.preferred_time.split(":"))
                    break_minutes = break_hour * 60 + break_minute
                    if (abs(current_minutes - break_minutes) <= 5 and
                            break_event_id not in self._processed_events[agent_id]):
                        self._processed_events[agent_id].add(break_event_id)
                        return True, "take_break:afternoon"
                except (ValueError, TypeError):
                    pass

            # Check work activity checkpoints
            if hasattr(daily_plan, 'work_activities') and daily_plan.work_activities:
                for activity in daily_plan.work_activities:
                    activity_event_id = f"{date_prefix}:work_activity:{activity.activity}_{activity.planned_time}"
                    try:
                        act_hour, act_minute = map(int, activity.planned_time.split(":"))
                        act_minutes = act_hour * 60 + act_minute
                        if (abs(current_minutes - act_minutes) <= 5 and
                                activity_event_id not in self._processed_events[agent_id]):
                            self._processed_events[agent_id].add(activity_event_id)
                            return True, f"work_activity:{activity.activity}"
                    except (ValueError, TypeError):
                        pass

            # Check social commitment checkpoints (from conversations)
            # Group commitments by (time, agents) to trigger single checkpoint for related commitments
            if hasattr(daily_plan, 'social_commitments') and daily_plan.social_commitments:
                from collections import defaultdict
                commitment_groups: dict[tuple, list] = defaultdict(list)

                for commitment in daily_plan.social_commitments:
                    if commitment.fulfilled:
                        continue
                    # Skip commitments without specific times
                    if commitment.time in ("unspecified", "needs_confirmation", ""):
                        continue
                    try:
                        # Validate time format
                        comm_hour, comm_minute = map(int, commitment.time.split(":"))
                        if not (0 <= comm_hour <= 23 and 0 <= comm_minute <= 59):
                            continue
                        # Group by (time, agents) - sort agents for consistent key
                        agents_key = tuple(sorted(commitment.with_agents)) if commitment.with_agents else ()
                        key = (commitment.time, agents_key)
                        commitment_groups[key].append(commitment)
                    except (ValueError, TypeError):
                        pass

                # Check each group as a single checkpoint
                for (time_str, agents_key), group in commitment_groups.items():
                    try:
                        comm_hour, comm_minute = map(int, time_str.split(":"))
                        comm_minutes = comm_hour * 60 + comm_minute
                        # Create combined event ID for the whole group (F.1: with date prefix)
                        activities = sorted([c.activity for c in group])
                        combined_id = f"{date_prefix}:commitment:{'+'.join(activities)}_{time_str}"

                        # F.2: Check for commitment prep (10-15 min before commitment time)
                        minutes_until = comm_minutes - current_minutes
                        if 10 <= minutes_until <= 15:
                            prep_id = f"{date_prefix}:commitment_prep:{'+'.join(activities)}_{time_str}"
                            if prep_id not in self._processed_events[agent_id]:
                                self._processed_events[agent_id].add(prep_id)
                                return True, f"commitment_prep:{'+'.join(activities)}"

                        # Check if within 5 minutes of commitment time
                        if (abs(current_minutes - comm_minutes) <= 5 and
                                combined_id not in self._processed_events[agent_id]):
                            self._processed_events[agent_id].add(combined_id)
                            # Return combined reason with all activities
                            return True, f"commitment:{'+'.join(activities)}"
                    except (ValueError, TypeError):
                        pass

                # M.3: Check for "commitment_waiting" - agent is waiting for partner
                # This fires if: commitment time passed, agent at location, partner not there
                for commitment in daily_plan.social_commitments:
                    if commitment.fulfilled:
                        continue
                    if commitment.time in ("unspecified", "needs_confirmation", ""):
                        continue
                    try:
                        comm_hour, comm_minute = map(int, commitment.time.split(":"))
                        comm_minutes = comm_hour * 60 + comm_minute
                        minutes_past = current_minutes - comm_minutes

                        # Trigger 5-10 minutes after commitment time
                        if 5 <= minutes_past <= 10:
                            waiting_id = f"{date_prefix}:commitment_waiting:{commitment.activity}_{commitment.time}"
                            if waiting_id in self._processed_events[agent_id]:
                                continue

                            # Check if agent is at the commitment location
                            commitment_location = commitment.location or "break_area"
                            if agent_status:
                                agent_location = agent_status.get("location", "desk_area")
                                if agent_location != commitment_location:
                                    continue

                            # Check if any partner is NOT at the location
                            partner_missing = False
                            missing_partner = None
                            for partner_id in commitment.with_agents:
                                if partner_id == agent_id:
                                    continue
                                # Get partner location from adapter if available
                                # This requires access to the adapter, which we don't have here
                                # So we'll mark all waiting situations and let the prompt handle it
                                partner_missing = True
                                missing_partner = partner_id
                                break

                            if partner_missing:
                                self._processed_events[agent_id].add(waiting_id)
                                return True, f"commitment_waiting:{commitment.activity}:{missing_partner}"
                    except (ValueError, TypeError):
                        pass

        # Check departure preparation checkpoint (10-20 min before departure)
        if daily_plan.actual_departure_time:
            try:
                dep_hour, dep_minute = map(int, daily_plan.actual_departure_time.split(":"))
                dep_minutes = dep_hour * 60 + dep_minute
                departure_event_id = f"{date_prefix}:departure_prep:{daily_plan.actual_departure_time}"
                time_to_departure = dep_minutes - current_minutes
                if (10 <= time_to_departure <= 20 and
                        departure_event_id not in self._processed_events[agent_id]):
                    self._processed_events[agent_id].add(departure_event_id)
                    return True, "departure_prep"
            except (ValueError, TypeError):
                pass

        return False, ""

    def reset_for_new_day(self, agent_id: str) -> None:
        """Reset checkpoint tracking for a new day."""
        self._processed_events[agent_id] = set()
        # Clear lunch/break tracking
        if agent_id in self._lunch_started:
            del self._lunch_started[agent_id]
        if agent_id in self._break_started:
            del self._break_started[agent_id]
        # Keep hourly check - it will naturally reset when >1 hour passes


def _validate_break_times(plan: DailyPlan) -> DailyPlan:
    """
    Validate and adjust break times to not conflict with meetings.

    If a break time overlaps with a meeting, clear that break from the plan.
    The agent can reschedule during their decision checkpoints.

    Args:
        plan: The daily plan from the planner agent

    Returns:
        Adjusted plan with conflicting breaks removed
    """
    if not plan.meetings:
        return plan

    def parse_time(time_str: str) -> Optional[datetime]:
        """Parse HH:MM time string to datetime for comparison."""
        try:
            return datetime.strptime(time_str, "%H:%M")
        except (ValueError, TypeError):
            return None

    def times_conflict(break_time_str: str, meeting_start: str, meeting_end: str) -> bool:
        """Check if break time falls within meeting time range."""
        break_time = parse_time(break_time_str)
        meeting_start_time = parse_time(meeting_start)
        meeting_end_time = parse_time(meeting_end)

        if not all([break_time, meeting_start_time, meeting_end_time]):
            return False

        # Add 15-minute buffer around meetings
        buffer = timedelta(minutes=15)
        meeting_start_with_buffer = meeting_start_time - buffer
        meeting_end_with_buffer = meeting_end_time + buffer

        return meeting_start_with_buffer <= break_time <= meeting_end_with_buffer

    # Check morning break
    if plan.morning_break and plan.morning_break.preferred_time:
        for meeting in plan.meetings:
            start_iso = meeting.start_datetime_iso
            end_iso = meeting.end_datetime_iso
            # Extract HH:MM from ISO datetime
            if start_iso and end_iso:
                meeting_start = start_iso.split('T')[1][:5] if 'T' in start_iso else start_iso
                meeting_end = end_iso.split('T')[1][:5] if 'T' in end_iso else end_iso
                if times_conflict(plan.morning_break.preferred_time, meeting_start, meeting_end):
                    print(f"  [VALIDATION] Morning break at {plan.morning_break.preferred_time} conflicts with meeting, removing")
                    plan.morning_break = None
                    break

    # Check afternoon break
    if plan.afternoon_break and plan.afternoon_break.preferred_time:
        for meeting in plan.meetings:
            start_iso = meeting.start_datetime_iso
            end_iso = meeting.end_datetime_iso
            if start_iso and end_iso:
                meeting_start = start_iso.split('T')[1][:5] if 'T' in start_iso else start_iso
                meeting_end = end_iso.split('T')[1][:5] if 'T' in end_iso else end_iso
                if times_conflict(plan.afternoon_break.preferred_time, meeting_start, meeting_end):
                    print(f"  [VALIDATION] Afternoon break at {plan.afternoon_break.preferred_time} conflicts with meeting, removing")
                    plan.afternoon_break = None
                    break

    # Check lunch
    if plan.lunch_plan and plan.lunch_plan.time:
        for meeting in plan.meetings:
            start_iso = meeting.start_datetime_iso
            end_iso = meeting.end_datetime_iso
            if start_iso and end_iso:
                meeting_start = start_iso.split('T')[1][:5] if 'T' in start_iso else start_iso
                meeting_end = end_iso.split('T')[1][:5] if 'T' in end_iso else end_iso
                if times_conflict(plan.lunch_plan.time, meeting_start, meeting_end):
                    print(f"  [VALIDATION] Lunch at {plan.lunch_plan.time} conflicts with meeting, adjusting")
                    # Try to find a non-conflicting lunch time (12:00-14:00 range)
                    for hour in [12, 13, 14, 11]:
                        candidate = f"{hour:02d}:00"
                        has_conflict = False
                        for m in plan.meetings:
                            s = m.start_datetime_iso
                            e = m.end_datetime_iso
                            if s and e:
                                ms = s.split('T')[1][:5] if 'T' in s else s
                                me = e.split('T')[1][:5] if 'T' in e else e
                                if times_conflict(candidate, ms, me):
                                    has_conflict = True
                                    break
                        if not has_conflict:
                            plan.lunch_plan.time = candidate
                            print(f"  [VALIDATION] Adjusted lunch time to {candidate}")
                            break
                    break

    return plan


def _create_agent_directories(
    base_dir: str,
    agent_ids: List[str],
    run_start: datetime,
) -> Dict[str, Dict[str, str]]:
    """
    Create the agent directory structure:
    agents/YYYY-MM-DD/HH-MM-SS/{agent_id}/
        - scratch.json (working memory)
        - core_memories.json (permanent personality)
        - memory_stream/ (events, thoughts, chats in JSON)
        - decisions.log
    agents/YYYY-MM-DD/HH-MM-SS/shared/
        - calendar.sqlite
        - sessions.sqlite
        - conversations.log
        - actions.log

    Args:
        base_dir: Base directory for output (usually results_dir)
        agent_ids: List of agent IDs to create directories for
        run_start: Timestamp for directory naming

    Returns:
        Dict with paths:
        {
            "shared": {"calendar_db": path, "sessions_db": path, "conversation_log": path, "action_log": path},
            "agent_id": {"core_memory_db": path, "action_memory_db": path, "decision_log": path},
            ...
        }
    """
    # Create directory structure
    date_str = run_start.strftime("%Y-%m-%d")
    time_str = run_start.strftime("%H-%M-%S")
    run_dir = Path(base_dir) / "agents" / date_str / time_str

    # Create shared directory
    shared_dir = run_dir / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "shared": {
            "calendar_db": str(shared_dir / "calendar.sqlite"),
            "sessions_db": str(shared_dir / "sessions.sqlite"),
            "conversation_log": str(shared_dir / "conversations.log"),
            "action_log": str(shared_dir / "actions.log"),
        },
        "run_dir": str(run_dir),
    }

    # Create per-agent directories
    for agent_id in agent_ids:
        agent_dir = run_dir / agent_id
        agent_dir.mkdir(parents=True, exist_ok=True)

        paths[agent_id] = {
            "decision_log": str(agent_dir / "decisions.log"),
            "agent_folder": str(agent_dir),  # Path to agent folder for JSON storage
        }

    return paths


class LLMOccupantManager:
    """
    Main integration point for LLM-based occupants.

    This class:
    1. Initializes all LLM components (agents, memory, calendar)
    2. Seeds agent memories from config
    3. Coordinates daily planning and timestep decisions
    4. Aggregates equipment/lighting power for thermal calculations
    """

    def __init__(
        self,
        config: dict,
        get_zone_temp: Callable[[], float],
        get_weather: Callable[[], Dict[str, Any]],
        results_dir: Optional[str] = None,
    ):
        """
        Initialize the LLM occupant management system.

        Args:
            config: Full simulation config dict
            get_zone_temp: Function returning current zone air temperature (C)
            get_weather: Function returning current weather dict
            results_dir: Directory for storing databases (defaults to current dir)

        Raises:
            RuntimeError: If OpenAI API is unavailable
        """
        self.config = config
        llm_config = config.get("llm_agents", {})

        # Store agent model from config (used for all LLM agent builders)
        self._agent_model = llm_config.get("agent_model", DEFAULT_AGENT_MODEL)

        # Store API timeout from config (default 30 seconds per OpenAI best practices)
        self._api_timeout = llm_config.get("api_timeout_seconds", 30.0)

        # Enable prompt logging for debugging (logs full prompts and retrieved memories)
        self._log_prompts = llm_config.get("log_prompts", False)

        # Validate API connection (fail early, no fallback)
        self._validate_api_connection()

        # Get agent configuration first (needed for directory creation)
        self._agents_config = llm_config.get("agents", [])
        self._agent_ids = [a["id"] for a in self._agents_config]

        # Create agent directory structure
        base_dir = results_dir or "."
        run_start = datetime.now(timezone.utc)
        self._agent_paths = _create_agent_directories(
            base_dir=base_dir,
            agent_ids=self._agent_ids,
            run_start=run_start,
        )

        # Extract shared paths
        calendar_db = self._agent_paths["shared"]["calendar_db"]
        sessions_db = self._agent_paths["shared"]["sessions_db"]

        # Initialize per-agent decision log files
        self._decision_log_paths: Dict[str, str] = {}
        self._prompt_log_paths: Dict[str, str] = {}
        for agent_id in self._agent_ids:
            log_path = self._agent_paths[agent_id]["decision_log"]
            self._decision_log_paths[agent_id] = log_path
            with open(log_path, 'w') as f:
                f.write(f"# LLM Agent Decision Log - {agent_id}\n")
                f.write(f"# Run: {run_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("# Format: timestamp | action_type | parameters | rationale\n")
                f.write("-" * 80 + "\n")

            # Initialize prompt log paths (always create path, write only if enabled)
            agent_dir = Path(self._agent_paths[agent_id]["decision_log"]).parent
            prompt_log_path = str(agent_dir / "prompts.log")
            self._prompt_log_paths[agent_id] = prompt_log_path
            if self._log_prompts:
                with open(prompt_log_path, 'w') as f:
                    f.write(f"# LLM Agent Prompt Log - {agent_id}\n")
                    f.write(f"# Run: {run_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("# Contains full prompts and retrieved memories for debugging\n")
                    f.write("=" * 80 + "\n")

        # Initialize sub-managers
        self.equipment = EquipmentManager(config)
        self.lighting = LightingManager(config)
        self.desks = DeskManager(config)

        # Get base setpoints from schedules
        schedules = config.get("schedules", {})
        base_heating = schedules.get("occupied_heating_setpoint_c", 21.0)
        base_cooling = schedules.get("occupied_cooling_setpoint_c", 24.0)

        # Initialize state provider
        zone_name = config.get("zone_properties", {}).get("name", "Office_Main")
        self._zone_state = ZoneStateProvider(
            get_zone_temp=get_zone_temp,
            get_weather=get_weather,
            zone_name=zone_name,
        )

        # Create production adapter
        room_layout = config.get("room_layout", {})
        self.adapter = ProductionSimulationAdapter(
            zone_state_provider=self._zone_state,
            equipment_manager=self.equipment,
            lighting_manager=self.lighting,
            desk_manager=self.desks,
            base_heating_setpoint_c=base_heating,
            base_cooling_setpoint_c=base_cooling,
            room_layout=room_layout,
        )

        # Initialize embedding client (shared across agents)
        self.embedder = EmbeddingClient(
            model=llm_config.get("embed_model", DEFAULT_EMBED_MODEL),
            timeout=self._api_timeout,
        )

        # Initialize calendar store (shared across agents)
        self.calendar = CalendarStore(calendar_db)

        # Initialize GenerativeAgent instances from agent_base_types
        self._agents: Dict[str, GenerativeAgent] = initialize_agents_for_simulation(
            config=config,
            results_dir=base_dir,
            embedder=self.embedder,
            run_start=run_start,
        )

        # Build OpenAI Agents for LLM calls (one step agent and one planner per agent)
        self._step_agents: Dict[str, Agent] = {}
        self._planner_agents: Dict[str, Agent] = {}
        for agent_id in self._agent_ids:
            self._step_agents[agent_id] = build_step_agent(agent_id, model=self._agent_model)
            self._planner_agents[agent_id] = build_day_planner_agent(agent_id, model=self._agent_model)

        # Decision interval
        self._decision_interval_minutes = llm_config.get("decision_interval_minutes", 15)
        self._daily_planning_hour = llm_config.get("daily_planning_hour", 0)

        # Track daily plans
        self._daily_plans: Dict[str, DailyPlan] = {}
        self._last_plan_date: Optional[date] = None

        # Track last decision time per agent
        self._last_decision_time: Dict[str, datetime] = {}

        # Track if decisions were made during current step (for progress reporting)
        self._decisions_made_this_step: int = 0

        # Initialize checkpoint manager for event-triggered decisions
        self._checkpoint_manager = DecisionCheckpointManager()

        # Weather history buffer for conversation context
        # Stores recent weather data (up to 72 hours / ~3 days)
        self._weather_history: List[Dict[str, Any]] = []
        self._weather_history_max_hours: int = 72

        # Note: Core memories are now seeded from markdown files in agent_base_types/
        # via GenerativeAgent._seed_core_memories_from_markdown()

        run_dir = self._agent_paths.get("run_dir", base_dir)
        self._actions_log_path = self._agent_paths.get("shared", {}).get("action_log")

        # UI demo writer (optional, set via set_ui_writer())
        self._ui_writer = None

        print(f"[LLM] Initialized {len(self._agent_ids)} LLM occupant agents")
        print(f"[LLM] Agent data directory: {run_dir}")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_agent_run_dir: str,
        checkpoint_calendar_db: str,
        config: dict,
        get_zone_temp: Callable[[], float],
        get_weather: Callable[[], Dict[str, Any]],
        results_dir: Optional[str] = None,
    ) -> "LLMOccupantManager":
        """
        Create LLMOccupantManager from a checkpoint.

        Loads agents from checkpoint directory instead of creating fresh.
        Creates a NEW results directory while loading state from checkpoint.

        Args:
            checkpoint_agent_run_dir: Path to checkpoint's agent directory
            checkpoint_calendar_db: Path to checkpoint's calendar.sqlite
            config: Simulation configuration dict
            get_zone_temp: Function returning current zone temperature
            get_weather: Function returning current weather dict
            results_dir: Directory for new results (defaults to current dir)

        Returns:
            LLMOccupantManager initialized from checkpoint state
        """
        from pathlib import Path
        import shutil

        # Override config to continue from checkpoint
        config = dict(config)  # Don't modify original
        config['llm_agents'] = dict(config.get('llm_agents', {}))
        config['llm_agents']['agent_source'] = {
            'mode': 'continue',
            'continue_from_run': checkpoint_agent_run_dir,
        }

        # Create the manager (this creates new directories)
        manager = cls(
            config=config,
            get_zone_temp=get_zone_temp,
            get_weather=get_weather,
            results_dir=results_dir,
        )

        # Copy calendar from checkpoint to new results directory
        new_calendar_path = manager._agent_paths["shared"]["calendar_db"]
        if Path(checkpoint_calendar_db).exists():
            shutil.copy2(checkpoint_calendar_db, new_calendar_path)
            # Reinitialize calendar with copied database
            manager.calendar = CalendarStore(new_calendar_path)
            print(f"[RESUME] Loaded calendar from checkpoint: {checkpoint_calendar_db}")

        print(f"[RESUME] Manager created from checkpoint. New run dir: {manager._agent_paths.get('run_dir')}")
        return manager

    def _validate_api_connection(self) -> None:
        """
        Check OpenAI API is available.

        Raises:
            RuntimeError: If API key is missing or connection fails
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable not set. "
                "LLM agents require a valid OpenAI API key."
            )

        try:
            client = OpenAI()
            # Simple test call to verify connection
            client.models.list()
            print("[LLM] OpenAI API connection verified")
        except APIConnectionError as e:
            raise RuntimeError(f"Cannot connect to OpenAI API: {e}")
        except APIError as e:
            raise RuntimeError(f"OpenAI API error: {e}")

    def _persist_meeting_to_calendar(
        self,
        agent_id: str,
        meeting,  # MeetingPlan
        now: datetime
    ) -> None:
        """
        Persist a meeting from DailyPlan to the calendar database.

        Creates a calendar event and sends invitations to invitees.
        Checks for duplicates to prevent race condition issues from parallel meeting creation.
        """
        try:
            # Only create if there are invitees or it's a real meeting
            if not meeting.title:
                return

            # Parse meeting times
            start = datetime.fromisoformat(meeting.start_datetime_iso)
            end = datetime.fromisoformat(meeting.end_datetime_iso)

            # Check for duplicate meetings before creating
            # This prevents issues from parallel meeting creation in Phase 1
            day_start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = start.replace(hour=23, minute=59, second=59, microsecond=999999)
            existing_events = self.calendar.list_events("shared", day_start, day_end)
            for existing in existing_events:
                if (existing.get("title") == meeting.title and
                    existing.get("start_datetime_iso") == meeting.start_datetime_iso):
                    # Meeting already exists - skip creating duplicate
                    print(f"    Skipping duplicate meeting '{meeting.title}' at {meeting.start_datetime_iso}")
                    return

            # Create the event
            event_id = self.calendar.create_event(
                calendar_id="shared",
                created_by=agent_id,
                title=meeting.title,
                start=start,
                end=end,
                now=now,
                location=meeting.location if meeting.location else "meeting_room",
            )

            # Send invitations to invitees
            if meeting.invitees:
                for invitee_id in meeting.invitees:
                    if invitee_id != agent_id:  # Don't invite self
                        self.calendar.add_invitation(
                            event_id=event_id,
                            inviter_id=agent_id,
                            invitee_id=invitee_id,
                            now=now,
                        )
                print(f"    Created meeting '{meeting.title}' with {len(meeting.invitees)} invitees")
        except Exception as e:
            print(f"    Warning: Failed to persist meeting '{meeting.title}': {e}")

    def get_agent_ids(self) -> List[str]:
        """Get list of configured agent IDs."""
        return self._agent_ids.copy()

    def set_ui_writer(self, ui_writer) -> None:
        """
        Set the UI demo writer for logging actions and conversations.

        Args:
            ui_writer: UIDemoWriter instance from bsm.output.ui_demo_writer
        """
        self._ui_writer = ui_writer

    def get_all_occupant_states(self) -> List[Dict[str, Any]]:
        """
        Get current state of all occupants for UI output.

        Returns:
            List of dicts with agent_id, agent_name, is_in_office, location,
            current_desk, at_desk, on_break, at_lunch for each agent.
        """
        states = []
        for agent_id in self._agent_ids:
            agent = self._agents.get(agent_id)
            is_present = self.adapter.is_occupant_present(agent_id)
            location = self.adapter.get_agent_location(agent_id) if is_present else "outside"
            status = self.adapter.get_agent_status(agent_id)

            # Get specific desk assignment from desk manager
            current_desk = self.desks.get_desk_for_occupant(agent_id) if is_present else None

            states.append({
                "agent_id": agent_id,
                "agent_name": agent.first_name if agent else agent_id.split("_")[0].capitalize(),
                "is_in_office": is_present,
                "location": location,
                "current_desk": current_desk,
                "at_desk": status.get("at_desk", False),
                "on_break": status.get("on_break", False),
                "at_lunch": status.get("at_lunch", False),
            })
        return states

    def get_agent_config(self, agent_id: str) -> Optional[dict]:
        """Get configuration for a specific agent."""
        for config in self._agents_config:
            if config["id"] == agent_id:
                return config
        return None

    async def _plan_agent_day_async(
        self,
        agent_id: str,
        day: date,
        now: datetime,
    ) -> Optional[DailyPlan]:
        """
        Plan a single agent's day using cognitive planning (async).

        Uses memory retrieval and LLM-based planning.
        """
        agent = self._agents.get(agent_id)
        if not agent:
            print(f"  {agent_id}: Agent not found")
            return None

        # Check if weekend
        is_weekend = day.weekday() >= 5
        # Read works_weekends from agent's scratch (loaded from agent_base_types)
        works_weekends = agent.scratch.get("works_weekends", False)

        if is_weekend and not works_weekends:
            print(f"  {agent_id}: Skipping (weekend)")
            return None

        # RETRIEVE: Get planning-relevant memories
        focal_points = get_planning_focal_points()
        retrieved = retrieve(agent, focal_points, now)

        # Get calendar events and invitations
        # Convert date to datetime for calendar query (start of day to end of day)
        day_start = datetime.combine(day, datetime.min.time()).replace(tzinfo=timezone.utc)
        day_end = datetime.combine(day, datetime.max.time()).replace(tzinfo=timezone.utc)
        calendar_events = self.calendar.list_events("shared", day_start, day_end)
        pending_invitations = self.calendar.get_pending_invitations(agent_id)

        # Get meetings agent has already created or accepted
        my_meetings = self.calendar.get_agent_meetings(agent_id, day_start, day_end)

        # Add meetings to agent memory for natural retrieval
        if my_meetings and agent.memory_stream:
            for meeting in my_meetings:
                title = meeting.get("title", "Untitled meeting")
                start_iso = meeting.get("start_iso", "")
                end_iso = meeting.get("end_iso", "")
                location = meeting.get("location", "meeting_room")
                attendees = meeting.get("attendees", [])

                # Format time nicely
                try:
                    start_dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
                    end_dt = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
                    time_str = f"{start_dt.strftime('%H:%M')} to {end_dt.strftime('%H:%M')}"
                except (ValueError, AttributeError):
                    time_str = "scheduled time"

                meeting_desc = f"I have a meeting '{title}' from {time_str} in {location}"
                if attendees:
                    other_attendees = [a for a in attendees if a != agent_id]
                    if other_attendees:
                        meeting_desc += f" with {', '.join(other_attendees)}"

                # Add to memory with meeting-related keywords
                # Use LLM to assess importance (meetings vary in importance)
                meeting_importance = await get_importance(agent, meeting_desc, 5.0, use_llm=True)
                agent.memory_stream.add_event(
                    description=meeting_desc,
                    subject="I",
                    predicate="have meeting",
                    obj=title,
                    now=now,
                    importance=meeting_importance,
                )
                print(f"[SCHEDULE] {agent_id}: Meeting recorded - '{title}' at {time_str}")

        # Build planning prompt with cognitive context
        prompt = format_planning_prompt(
            agent=agent,
            day=day.isoformat(),
            day_of_week=day.strftime("%A"),
            retrieved_memories=retrieved,
            calendar_events=calendar_events,
            pending_invitations=pending_invitations,
            my_meetings=my_meetings,
        )

        # Call LLM via async Runner.run (SDK best practice)
        # max_turns=15 allows for calendar queries and invitation handling
        planner_agent = self._planner_agents[agent_id]
        context = SimContext(
            occupant_id=agent_id,
            now=now,
            simulation=self.adapter,
            calendar=self.calendar,
            configured_agent_ids=list(self._agents.keys()),
        )
        result = await run_with_retry(planner_agent, prompt, context=context, max_turns=15)
        plan = result.final_output

        # Validate break times don't conflict with meetings
        plan = _validate_break_times(plan)

        # Record plan to memory
        await record_plan_to_memory(agent, plan, now)

        # Store plan
        self._daily_plans[agent_id] = plan
        agent.set_daily_plan(plan.model_dump())

        # Store clothing choice for today (resets daily)
        if plan.clothing:
            agent.set_todays_clothing(plan.clothing.model_dump())
            # Add clothing as a perception for the day
            if agent.memory_stream:
                clothing_memory = f"I'm wearing {plan.clothing.description} today - {plan.clothing.warmth_level} warmth"
                # Use LLM to assess importance
                importance = await get_importance(agent, clothing_memory, 2.0, use_llm=True)
                agent.memory_stream.add_event(
                    description=clothing_memory,
                    subject=agent.scratch.get("first_name", agent_id),
                    predicate="is wearing",
                    obj=plan.clothing.description,
                    importance=importance,
                    now=now,
                )

        # Save agent state
        agent.save()

        return plan

    def _maybe_update_plan(
        self,
        agent_id: str,
        event: str,
        now: datetime,
    ) -> None:
        """
        Check if an agent's daily plan needs updating based on significant events.

        Triggered by events like meeting cancellations, urgent requests, or schedule conflicts.

        Args:
            agent_id: The agent whose plan to check
            event: Description of the event that occurred
            now: Current simulation datetime
        """
        significant_events = [
            "meeting_cancelled",
            "urgent_request",
            "schedule_conflict",
            "extended_conversation",
            "emergency",
        ]

        # Check if this event is significant enough to warrant a plan update
        if not any(e in event.lower() for e in significant_events):
            return

        agent = self._agents.get(agent_id)
        if not agent:
            return

        # Update the plan with a note about the change
        updates = {
            "plan_modified": True,
            "modification_event": event,
        }
        result = agent.update_daily_plan(updates, reason=event, now=now)
        if result.get("updated"):
            print(f"[SCHEDULE] {agent_id}: Plan adjusted - {event}")
            if result.get("skipped_past_events"):
                print(f"[SCHEDULE] {agent_id}: (skipped past events: {result['skipped_past_events']})")

    def _check_plan_update_triggers(
        self,
        agent_id: str,
        sim_state: Dict[str, Any],
        now: datetime,
    ) -> List[str]:
        """
        Check for conditions that should trigger a plan update.

        Returns list of trigger reasons (empty if no update needed).

        Triggers checked:
        - Meeting cancellations
        - Urgent invitation (meeting within 1 hour)
        - Schedule deviations (significantly behind/ahead of plan)
        """
        triggers = []
        agent = self._agents.get(agent_id)
        if not agent:
            return triggers

        # Check for urgent pending invitations (within 1 hour)
        pending_invitations = self.calendar.get_pending_invitations(agent_id)
        for invite in pending_invitations:
            event_start_iso = invite.get("event_start_iso", "")
            if event_start_iso:
                try:
                    from datetime import timezone
                    event_start = datetime.fromisoformat(event_start_iso)
                    if event_start.tzinfo is None:
                        event_start = event_start.replace(tzinfo=timezone.utc)
                    now_utc = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
                    minutes_until = (event_start - now_utc).total_seconds() / 60
                    if 0 < minutes_until <= 60:
                        triggers.append(f"urgent_request: Meeting invitation from {invite.get('inviter_id')} in {int(minutes_until)} minutes")
                except (ValueError, TypeError):
                    pass

        # Check if currently in meeting room but no active meeting
        # (indicates meeting may have been cancelled)
        current_location = sim_state.get("current_location", "desk_area")
        if current_location == "meeting_room":
            meeting_context = get_meeting_context(agent_id, self.calendar, now)
            if not meeting_context.get("current_meeting"):
                triggers.append("location_check: Your meeting has ended - consider returning to your desk")

        return triggers

    def plan_day(self, day: date, now: datetime) -> Dict[str, DailyPlan]:
        """
        Call daily planning for all agents using cognitive planning.

        Uses THREE-PHASE approach for proper calendar coordination:
        - Phase 1: All agents create meetings (parallel)
        - Phase 2: All agents respond to invitations (coordinated)
        - Phase 3: All agents finalize plans with confirmed meetings

        Should be called once at the start of each simulated day.

        Args:
            day: The date to plan for
            now: Current simulation datetime

        Returns:
            Dict mapping agent_id to their DailyPlan
        """
        print(f"\n[LLM] Running THREE-PHASE daily planning for {len(self._agent_ids)} agents on {day.isoformat()}")

        self._daily_plans.clear()
        self._last_plan_date = day

        # Reset adapter state for new day
        self.adapter.reset_day()
        self.adapter.start_of_day_setup()

        # Filter agents who work today
        working_agents = []
        for agent_id in self._agent_ids:
            agent = self._agents.get(agent_id)
            if not agent:
                continue
            is_weekend = day.weekday() >= 5
            works_weekends = agent.scratch.get("works_weekends", False)
            if is_weekend and not works_weekends:
                print(f"  {agent_id}: Skipping (weekend)")
                continue
            working_agents.append(agent_id)

        if not working_agents:
            print(f"  No agents working today")
            return {}

        # Run three-phase planning
        async def run_three_phase_planning():
            # === PHASE 1: Meeting Creation ===
            print(f"\n  [Phase 1] Meeting creation for {len(working_agents)} agents...")
            await self._phase1_create_meetings(working_agents, day, now)

            # === PHASE 2: RSVP Coordination ===
            print(f"\n  [Phase 2] RSVP coordination for {len(working_agents)} agents...")
            await self._phase2_rsvp_coordination(working_agents, day, now)

            # === PHASE 3: Plan Finalization ===
            print(f"\n  [Phase 3] Plan finalization for {len(working_agents)} agents...")
            results = await self._phase3_finalize_plans(working_agents, day, now)
            return results

        results = asyncio.run(run_three_phase_planning())

        # Process results
        for agent_id, plan, error in results:
            if error:
                print(f"  {agent_id}: FAILED - {error}")
                raise RuntimeError(f"Daily planning failed for {agent_id}: {error}")
            elif plan:
                print(f"  {agent_id}: arrive {plan.actual_arrival_time}, "
                      f"depart {plan.actual_departure_time}, "
                      f"{len(plan.meetings)} meetings")

                # Persist meetings to calendar database
                for meeting in plan.meetings:
                    self._persist_meeting_to_calendar(agent_id, meeting, now)

                # Log daily plan to agent's log file
                log_path = self._decision_log_paths.get(agent_id)
                if log_path:
                    with open(log_path, 'a') as f:
                        f.write(f"\n=== DAILY PLAN: {day.isoformat()} ===\n")
                        f.write(f"Arrival: {plan.actual_arrival_time}\n")
                        f.write(f"Departure: {plan.actual_departure_time}\n")
                        f.write(f"Meetings: {len(plan.meetings)}\n")
                        for meeting in plan.meetings:
                            f.write(f"  - {meeting.title} ({meeting.start_datetime_iso} - {meeting.end_datetime_iso})\n")

        return self._daily_plans.copy()

    async def _phase1_create_meetings(
        self,
        agent_ids: List[str],
        day: date,
        now: datetime,
    ) -> None:
        """
        Phase 1: All agents propose meetings (parallel).

        Agents can create meetings and send invitations, but don't finalize plans yet.
        """
        day_start = datetime.combine(day, datetime.min.time()).replace(tzinfo=timezone.utc)
        day_end = datetime.combine(day, datetime.max.time()).replace(tzinfo=timezone.utc)

        async def create_meetings_for_agent(agent_id: str):
            agent = self._agents.get(agent_id)
            if not agent:
                return

            # Get current calendar state
            calendar_events = self.calendar.list_events("shared", day_start, day_end)
            my_meetings = self.calendar.get_agent_meetings(agent_id, day_start, day_end)

            # Build a focused prompt for meeting creation only
            from bsm.agents.cognition.modules import get_planning_focal_points, retrieve
            focal_points = get_planning_focal_points()
            retrieved = retrieve(agent, focal_points, now)

            prompt = f"""
Today is {day.isoformat()} ({day.strftime('%A')}).

PHASE 1: MEETING CREATION
Your task is to propose any meetings you want to schedule for today.

Current shared calendar for today:
{self._format_calendar_events(calendar_events)}

Meetings you've already scheduled:
{self._format_my_meetings(my_meetings)}

Your colleagues: {', '.join([aid for aid in agent_ids if aid != agent_id])}

INSTRUCTIONS:
1. If you want to schedule a meeting, use create_shared_event() to create it
2. After creating, use invite_agents_to_meeting() to invite colleagues
3. You can create multiple meetings if needed
4. Do NOT respond to invitations yet - that happens in Phase 2
5. When done creating meetings, output a simple confirmation

Output format: {{"meetings_created": true/false, "count": N}}
"""

            # Use the planner agent but with limited scope
            planner_agent = self._planner_agents[agent_id]
            context = SimContext(
                occupant_id=agent_id,
                now=now,
                simulation=self.adapter,
                calendar=self.calendar,
                configured_agent_ids=list(self._agents.keys()),
            )
            try:
                # Run with limited turns for just meeting creation
                # Note: Validation error is expected since output isn't DailyPlan
                # The important work (tool calls) has already been done
                await Runner.run(planner_agent, prompt, context=context, max_turns=8)
            except Exception as e:
                # Expected: output validation fails since we're not returning DailyPlan
                # Only log unexpected errors
                if "validation error" not in str(e).lower() and "invalid json" not in str(e).lower():
                    print(f"    {agent_id}: Phase 1 error - {e}")

        # Run all agents in parallel
        await asyncio.gather(*[create_meetings_for_agent(aid) for aid in agent_ids])

    async def _phase2_rsvp_coordination(
        self,
        agent_ids: List[str],
        day: date,
        now: datetime,
    ) -> None:
        """
        Phase 2: All agents respond to ALL pending invitations (parallel).

        After Phase 1, all meetings exist. Now everyone can see all invitations
        and respond to them.
        """
        day_start = datetime.combine(day, datetime.min.time()).replace(tzinfo=timezone.utc)
        day_end = datetime.combine(day, datetime.max.time()).replace(tzinfo=timezone.utc)

        async def process_invitations_for_agent(agent_id: str):
            agent = self._agents.get(agent_id)
            if not agent:
                return

            # Get ALL pending invitations for this agent
            pending_invitations = self.calendar.get_pending_invitations(agent_id)
            if not pending_invitations:
                print(f"    {agent_id}: No pending invitations")
                return

            # Get meetings agent has already accepted/created
            my_meetings = self.calendar.get_agent_meetings(agent_id, day_start, day_end)

            prompt = f"""
Today is {day.isoformat()} ({day.strftime('%A')}).

PHASE 2: RSVP COORDINATION
Your task is to respond to meeting invitations.

Your current confirmed meetings:
{self._format_my_meetings(my_meetings)}

PENDING INVITATIONS requiring your response:
{self._format_pending_invitations(pending_invitations)}

INSTRUCTIONS:
1. For EACH pending invitation, decide: accept (true) or decline (false)
2. Use respond_to_invitation(event_id, accept=true/false) for each
3. Consider: time conflicts, meeting relevance, your schedule
4. Accept meetings that are relevant to your work and don't conflict

When done, output: {{"invitations_processed": N}}
"""

            planner_agent = self._planner_agents[agent_id]
            context = SimContext(
                occupant_id=agent_id,
                now=now,
                simulation=self.adapter,
                calendar=self.calendar,
                configured_agent_ids=list(self._agents.keys()),
            )
            try:
                # Note: Validation error is expected since output isn't DailyPlan
                # The important work (tool calls) has already been done
                await Runner.run(planner_agent, prompt, context=context, max_turns=10)
                print(f"    {agent_id}: Processed {len(pending_invitations)} invitations")
            except Exception as e:
                # Expected: output validation fails since we're not returning DailyPlan
                # Only log unexpected errors, but still report success for invitations
                if "validation error" not in str(e).lower() and "invalid json" not in str(e).lower():
                    print(f"    {agent_id}: Phase 2 error - {e}")
                else:
                    # Tools were called successfully even though output validation failed
                    print(f"    {agent_id}: Processed {len(pending_invitations)} invitations")

        # Run all agents in parallel
        await asyncio.gather(*[process_invitations_for_agent(aid) for aid in agent_ids])

    async def _phase3_finalize_plans(
        self,
        agent_ids: List[str],
        day: date,
        now: datetime,
    ) -> List[Tuple[str, Optional[DailyPlan], Optional[Exception]]]:
        """
        Phase 3: All agents finalize their daily plans with confirmed meetings.

        After Phases 1 and 2, all RSVPs are done. Now create final DailyPlan
        with only confirmed meetings (created by agent + accepted invitations).
        """
        results = []

        for agent_id in agent_ids:
            try:
                plan = await self._finalize_agent_plan(agent_id, day, now)
                results.append((agent_id, plan, None))
            except Exception as e:
                results.append((agent_id, None, e))

        return results

    async def _finalize_agent_plan(
        self,
        agent_id: str,
        day: date,
        now: datetime,
    ) -> Optional[DailyPlan]:
        """Finalize a single agent's daily plan."""
        agent = self._agents.get(agent_id)
        if not agent:
            return None

        day_start = datetime.combine(day, datetime.min.time()).replace(tzinfo=timezone.utc)
        day_end = datetime.combine(day, datetime.max.time()).replace(tzinfo=timezone.utc)

        # Get CONFIRMED meetings (created + accepted)
        confirmed_meetings = self.calendar.get_agent_meetings(agent_id, day_start, day_end)

        # RETRIEVE: Get planning-relevant memories
        from bsm.agents.cognition.modules import (
            get_planning_focal_points, retrieve, format_planning_prompt, record_plan_to_memory
        )
        focal_points = get_planning_focal_points()
        retrieved = retrieve(agent, focal_points, now)

        # Build final planning prompt
        prompt = format_planning_prompt(
            agent=agent,
            day=day.isoformat(),
            day_of_week=day.strftime("%A"),
            retrieved_memories=retrieved,
            calendar_events=[],  # Not needed - we have confirmed meetings
            pending_invitations=[],  # All processed in Phase 2
            my_meetings=confirmed_meetings,
        )

        # Add explicit instruction about confirmed meetings
        prompt += f"""

IMPORTANT: Your meetings for today have been CONFIRMED through the RSVP process.
You have {len(confirmed_meetings)} confirmed meeting(s):
{self._format_my_meetings(confirmed_meetings)}

Include ALL these meetings in your DailyPlan.meetings list.
Do NOT use calendar tools - just output your final DailyPlan now.
"""

        # Call LLM for final plan
        planner_agent = self._planner_agents[agent_id]
        context = SimContext(
            occupant_id=agent_id,
            now=now,
            simulation=self.adapter,
            calendar=self.calendar,
            configured_agent_ids=list(self._agents.keys()),
        )
        result = await run_with_retry(planner_agent, prompt, context=context, max_turns=5)
        plan = result.final_output

        # Validate break times don't conflict with meetings
        plan = _validate_break_times(plan)

        # Record plan to memory
        await record_plan_to_memory(agent, plan, now)

        # Store plan
        self._daily_plans[agent_id] = plan
        agent.set_daily_plan(plan.model_dump())

        # Store clothing choice for today
        if plan.clothing:
            agent.set_todays_clothing(plan.clothing.model_dump())
            if agent.memory_stream:
                clothing_memory = f"I'm wearing {plan.clothing.description} today - {plan.clothing.warmth_level} warmth"
                # Use LLM to assess importance
                importance = await get_importance(agent, clothing_memory, 2.0, use_llm=True)
                agent.memory_stream.add_event(
                    description=clothing_memory,
                    subject=agent.scratch.get("first_name", agent_id),
                    predicate="is wearing",
                    obj=plan.clothing.description,
                    importance=importance,
                    now=now,
                )

        agent.save()
        return plan

    def _format_calendar_events(self, events: List[Dict[str, Any]]) -> str:
        """Format calendar events for prompt."""
        if not events:
            return "  (No meetings scheduled yet)"
        lines = []
        for e in events:
            start = e.get("start_datetime_iso", "?")
            title = e.get("title", "Untitled")
            creator = e.get("created_by", "unknown")
            lines.append(f"  - {start}: {title} (by {creator})")
        return "\n".join(lines)

    def _format_my_meetings(self, meetings: List[Dict[str, Any]]) -> str:
        """Format agent's meetings for prompt."""
        if not meetings:
            return "  (None)"
        lines = []
        for m in meetings:
            start = m.get("start_datetime_iso", "?")
            title = m.get("title", "Untitled")
            role = m.get("role", "attendee")
            lines.append(f"  - {start}: {title} (as {role})")
        return "\n".join(lines)

    def _format_pending_invitations(self, invitations: List[Dict[str, Any]]) -> str:
        """Format pending invitations for prompt."""
        if not invitations:
            return "  (None)"
        lines = []
        for inv in invitations:
            event_id = inv.get("event_id", "?")
            title = inv.get("event_title", "Untitled")
            inviter = inv.get("inviter_id", "unknown")
            start = inv.get("event_start_iso", "?")
            lines.append(f"  - event_id: {event_id}")
            lines.append(f"    {start}: {title} (from {inviter})")
        return "\n".join(lines)

    def end_of_day_processing(self, day: date, now: datetime) -> Dict[str, int]:
        """
        Process end of day for all agents (memory consolidation).

        Should be called at the end of each simulated day to consolidate
        significant memories into core memory.

        Args:
            day: The day that just ended
            now: Current simulation datetime

        Returns:
            Dict mapping agent_id to number of memories consolidated
        """
        print(f"\n[END OF DAY] Processing memory consolidation for {day}")

        async def consolidate_all():
            results = {}
            for agent_id, agent in self._agents.items():
                try:
                    count = await agent.consolidate_memories_with_llm(day, now)
                    results[agent_id] = count
                except Exception as e:
                    print(f"  {agent_id}: Consolidation failed - {e}")
                    results[agent_id] = 0
            return results

        results = asyncio.run(consolidate_all())

        # Print summary
        total = sum(results.values())
        if total > 0:
            print(f"[END OF DAY] Total memories consolidated: {total}")
            for agent_id, count in results.items():
                if count > 0:
                    print(f"  {agent_id}: {count} memories")

        return results

    def should_run_daily_planning(self, now: datetime) -> bool:
        """Check if daily planning should run based on current time."""
        current_date = now.date()

        # Never planned or different day
        if self._last_plan_date is None or self._last_plan_date != current_date:
            # Check if it's the planning hour
            if now.hour == self._daily_planning_hour:
                return True
            # Or if this is the first timestep of the day
            if self._last_plan_date != current_date:
                return True

        return False

    def should_make_decision(self, agent_id: str, now: datetime) -> Tuple[bool, str]:
        """
        Check if an agent should make a decision at this time.

        Uses checkpoint-based logic:
        - Hourly checkpoints (every 60 minutes)
        - Event-triggered checkpoints (meeting start/end, lunch, breaks)
        - Social commitment checkpoints (from conversations)

        Also falls back to decision_interval_minutes if no checkpoint triggered
        but enough time has passed.

        Returns:
            Tuple of (should_decide, reason) where reason indicates why
        """
        # Get agent's daily plan if available
        daily_plan = self._daily_plans.get(agent_id)

        # Sync social commitments from agent's stored dict (conversations add them there)
        # The checkpoint manager needs these to trigger commitment checkpoints
        if daily_plan:
            agent = self._agents.get(agent_id)
            if agent:
                agent_daily_plan = agent.get_daily_plan()
                if agent_daily_plan and isinstance(agent_daily_plan, dict):
                    stored_commitments = agent_daily_plan.get("social_commitments", [])
                    if stored_commitments:
                        # Convert dicts to SocialCommitment objects and merge
                        for c_dict in stored_commitments:
                            try:
                                commitment = SocialCommitment(**c_dict)
                                # Check if this commitment already exists
                                existing = any(
                                    sc.activity == commitment.activity and
                                    sc.time == commitment.time
                                    for sc in daily_plan.social_commitments
                                )
                                if not existing:
                                    daily_plan.social_commitments.append(commitment)
                            except Exception:
                                pass  # Skip invalid commitment dicts

        # Check for checkpoint triggers (hourly + events)
        should_checkpoint, reason = self._checkpoint_manager.should_checkpoint(
            agent_id=agent_id,
            current_time=now,
            daily_plan=daily_plan,
        )

        if should_checkpoint:
            return True, reason

        # Fallback: use decision_interval_minutes as minimum interval
        last_time = self._last_decision_time.get(agent_id)
        if last_time is None:
            return True, "first_decision"

        elapsed = (now - last_time).total_seconds() / 60.0
        if elapsed >= self._decision_interval_minutes:
            return True, "interval"

        return False, ""

    def is_agent_working(self, agent_id: str, now: datetime) -> bool:
        """
        Check if an agent should be working at this time based on their daily plan.

        Returns True if:
        - It's not a weekend (or agent works_weekends is True)
        - Current time is between arrival and departure from daily plan
        """
        # Check if weekend
        is_weekend = now.weekday() >= 5  # Saturday=5, Sunday=6

        # Read works_weekends from agent's scratch (loaded from agent_base_types)
        agent = self._agents.get(agent_id)
        works_weekends = agent.scratch.get("works_weekends", False) if agent else False

        # If weekend and agent doesn't work weekends, return False
        if is_weekend and not works_weekends:
            return False

        plan = self._daily_plans.get(agent_id)
        if not plan:
            return False

        try:
            # Parse arrival and departure times using datetime for robustness
            arrival_dt = datetime.strptime(plan.actual_arrival_time, "%H:%M")
            departure_dt = datetime.strptime(plan.actual_departure_time, "%H:%M")

            arrival_hour, arrival_min = arrival_dt.hour, arrival_dt.minute
            departure_hour, departure_min = departure_dt.hour, departure_dt.minute

            current_minutes = now.hour * 60 + now.minute
            arrival_minutes = arrival_hour * 60 + arrival_min
            departure_minutes = departure_hour * 60 + departure_min

            return arrival_minutes <= current_minutes < departure_minutes
        except (ValueError, AttributeError):
            # If parsing fails, assume working during normal hours
            return 8 <= now.hour < 18

    def step(
        self,
        now: datetime,
        force_all: bool = False,
    ) -> Tuple[float, float, float, float]:
        """
        Execute one timestep for all agents.

        Queries agents who need to make decisions, applies their actions,
        and resolves any conflicts.

        Args:
            now: Current simulation datetime
            force_all: If True, force all agents to make decisions

        Returns:
            Tuple of (equipment_power_w, lighting_power_w, thermostat_offset_c, window_fraction)
        """
        # Reset decision counter for this step (used for progress reporting)
        self._decisions_made_this_step = 0

        # Check for daily planning
        if self.should_run_daily_planning(now):
            self.plan_day(now.date(), now)

        # Process each agent
        for agent_id in self._agent_ids:
            # Check if agent should be working
            is_working = self.is_agent_working(agent_id, now)

            # Handle arrival/departure transitions
            was_present = self.adapter.is_occupant_present(agent_id)

            if is_working and not was_present:
                # Only handle initial morning arrival, not returning from outside break/lunch
                # Use _desk_chosen_today as indicator that agent has already arrived
                has_arrived_today = self.adapter._desk_chosen_today.get(agent_id, False)
                if not has_arrived_today:
                    self._handle_arrival(agent_id, now)
            elif not is_working and was_present:
                # Agent is departing
                self._handle_departure(agent_id, now)

            # Make decisions for present agents
            if is_working:
                should_decide, checkpoint_reason = self.should_make_decision(agent_id, now)
                if force_all or should_decide:
                    self._make_agent_decision(agent_id, now, checkpoint_reason=checkpoint_reason or "forced")
                    self._decisions_made_this_step += 1

        # Check for equipment auto-off (kitchen appliances like kettle, coffee machine)
        auto_off_list = self.equipment.check_all_auto_off(now)
        if auto_off_list:
            print(f"[EQUIPMENT] Auto-off: {', '.join(auto_off_list)}")

        # Resolve any pending votes
        thermostat_offset, window_fraction = self.adapter.resolve_votes()

        # Calculate total power
        equipment_power = self.adapter.get_total_equipment_power_w()
        lighting_power = self.adapter.get_total_lighting_power_w()

        return equipment_power, lighting_power, thermostat_offset, window_fraction

    def update_weather_history(self, weather_data: Dict[str, Any], timestep_hours: float = 1.0) -> None:
        """
        Update the weather history buffer with new weather data.

        Called by the simulation runner each timestep to maintain a rolling
        history of recent weather conditions for conversation context.

        Args:
            weather_data: Current timestep's weather dict from Open Meteo
            timestep_hours: Duration of each timestep in hours (for calculating max entries)
        """
        if weather_data:
            self._weather_history.append(weather_data.copy())

            # Calculate max entries based on timestep size
            # For example: 72 hours with 15-min timesteps = 288 entries
            max_entries = int(self._weather_history_max_hours / timestep_hours)
            max_entries = max(max_entries, 10)  # Keep at least 10 entries

            # Trim to max size
            if len(self._weather_history) > max_entries:
                self._weather_history = self._weather_history[-max_entries:]

    def get_weather_context(self) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Get current weather and recent history for conversation context.

        Returns:
            Tuple of (current_weather, weather_history) where:
            - current_weather: Most recent weather dict or None
            - weather_history: List of recent weather dicts (up to 72 hours)
        """
        current = self._zone_state._get_weather() if self._zone_state else None
        return current, self._weather_history.copy()

    def _handle_arrival(self, agent_id: str, now: datetime) -> None:
        """Handle an agent arriving at work."""
        print(f"[LLM] {agent_id} arriving at {now.strftime('%H:%M')}")

        # Mark as present and set initial location
        self.adapter.set_occupant_present(agent_id, True)
        self.adapter.set_agent_location(agent_id, "desk_area")

        agent = self._agents.get(agent_id)
        if agent:
            # Set flag for perceive() to create arrival-related memories
            agent.set_just_arrived(True)

            # Make desk selection decision via dedicated arrival agent
            desk_decision = self._make_arrival_decision(agent_id, now)
            if desk_decision:
                # Apply desk selection
                selected_desk = desk_decision.selected_desk
                self.adapter.desks.assign_desk(agent_id, selected_desk)
                self.adapter._desk_chosen_today[agent_id] = True
                print(f"  {agent_id} selected {selected_desk}: {desk_decision.reasoning[:50]}...")

                # Turn on equipment specified by the agent
                for equipment_name in desk_decision.equipment_to_turn_on:
                    # Map generic names to desk-specific equipment
                    desk_suffix = selected_desk.split('_')[-1] if '_' in selected_desk else 'A'
                    full_equipment_name = f"{equipment_name}_{desk_suffix}"
                    self.adapter.equipment.set_equipment_state(full_equipment_name, True, agent_id)
                    print(f"    Turned on {full_equipment_name}")
            else:
                # Fallback: assign first available desk
                available = self.adapter.desks.get_available_desks()
                if available:
                    fallback_desk = available[0]
                    self.adapter.desks.assign_desk(agent_id, fallback_desk)
                    self.adapter._desk_chosen_today[agent_id] = True
                    print(f"  {agent_id} assigned to {fallback_desk} (fallback)")

    def _make_arrival_decision(self, agent_id: str, now: datetime) -> Optional[DeskSelectionDecision]:
        """
        Make desk selection decision for arriving agent.

        This uses a dedicated arrival agent to select the desk for the day.
        The decision is made once and locked for the entire day.
        """
        try:
            decision = asyncio.run(
                self._make_arrival_decision_async(agent_id, now)
            )
            return decision
        except Exception as e:
            print(f"[LLM] {agent_id} arrival decision failed: {e}")
            return None

    async def _make_arrival_decision_async(
        self,
        agent_id: str,
        now: datetime,
    ) -> Optional[DeskSelectionDecision]:
        """
        Execute arrival agent to select desk (async).

        Args:
            agent_id: The agent arriving at work
            now: Current simulation datetime

        Returns:
            DeskSelectionDecision with selected desk and equipment to turn on
        """
        agent = self._agents.get(agent_id)
        if not agent:
            print(f"[LLM] {agent_id}: Agent not found for arrival decision")
            return None

        # Get available desks and their features
        available_desks = self.adapter.desks.get_available_desks()
        desk_occupancy = self.adapter.desks.get_desk_occupancy()

        # Build arrival context prompt
        desk_info_lines = []
        for desk_name in available_desks:
            desk_info_lines.append(f"  - {desk_name}: Available")
        for desk_name, occupant in desk_occupancy.items():
            if occupant:
                desk_info_lines.append(f"  - {desk_name}: Occupied by {occupant}")

        desk_info = "\n".join(desk_info_lines) if desk_info_lines else "  No desk information available"

        # Get agent's preferred desk from memories/traits if available
        preferred_desk = agent.scratch.get("preferred_desk", "No specific preference")

        prompt = f"""
=== ARRIVAL AT WORK ===
Time: {now.strftime('%H:%M')} on {now.strftime('%A, %Y-%m-%d')}

=== WHO YOU ARE ===
{agent.get_identity_stable_set()}

=== YOUR DESK PREFERENCE ===
{preferred_desk}

=== DESK AVAILABILITY ===
{desk_info}

=== EQUIPMENT AT EACH DESK ===
Each desk has: laptop, monitor, desk light

Select your desk for today and specify which equipment to turn on.
""".strip()

        # Build and run arrival agent
        arrival_agent = build_arrival_agent(agent_id, model=self._agent_model)

        context = SimContext(
            occupant_id=agent_id,
            now=now,
            simulation=self.adapter,
            calendar=self.calendar,
        )

        try:
            result = await run_with_retry(
                arrival_agent,
                prompt,
                context=context,
                max_turns=1,  # Simple decision, no tool calls needed
            )

            desk_decision: DeskSelectionDecision = result.final_output

            # Validate selected desk is available
            if desk_decision.selected_desk not in available_desks:
                print(f"[LLM] {agent_id} selected unavailable desk {desk_decision.selected_desk}, using first available")
                if available_desks:
                    desk_decision.selected_desk = available_desks[0]
                else:
                    return None

            return desk_decision

        except Exception as e:
            print(f"[LLM] {agent_id} arrival agent error: {e}")
            return None

    def _handle_departure(self, agent_id: str, now: datetime) -> None:
        """Handle an agent departing from work."""
        print(f"[LLM] {agent_id} departing at {now.strftime('%H:%M')}")

        # Clean up checkpoint timers to prevent stale timers accumulating
        if agent_id in self._checkpoint_manager._lunch_started:
            del self._checkpoint_manager._lunch_started[agent_id]
        if agent_id in self._checkpoint_manager._break_started:
            del self._checkpoint_manager._break_started[agent_id]

        # Handle departure (turns off equipment, releases desk)
        self.adapter.set_occupant_present(agent_id, False)

    async def _make_agent_decision_async(
        self,
        agent_id: str,
        now: datetime,
        checkpoint_reason: str = "interval",
    ) -> Optional[StepDecisions]:
        """
        Execute cognitive loop for agent decision (async - SDK best practice).

        Perceive -> CheckInteraction -> Retrieve -> Reflect -> Act

        Args:
            agent_id: The agent making the decision
            now: Current simulation datetime
            checkpoint_reason: Why decision is being made (hourly, meeting_start, lunch_time, etc.)

        Returns:
            StepDecisions object with structured decisions for all categories
        """
        agent = self._agents.get(agent_id)
        if not agent:
            print(f"[LLM] {agent_id}: Agent not found")
            return None

        # Decrement chat buffers at start of each decision cycle
        agent.decrement_all_chat_buffers()

        # 1. Get simulation state
        sim_state = self.adapter.get_state(agent_id, now)

        # 1.5 Check for plan update triggers
        triggers = self._check_plan_update_triggers(agent_id, sim_state, now)
        for trigger in triggers:
            self._maybe_update_plan(agent_id, trigger, now)
            # Record trigger to memory stream
            if agent.memory_stream:
                description = f"My schedule may need adjusting: {trigger}"
                # Use LLM to assess importance
                importance = await get_importance(agent, description, 5.0, use_llm=True)
                agent.memory_stream.add_event(
                    description=description,
                    subject="I",
                    predicate="notice",
                    obj="schedule adjustment needed",
                    now=now,
                    importance=importance,
                )

        # 2. PERCEIVE: Convert state to memory events (LLM-assessed importance)
        perceived_events = await perceive(agent, sim_state, now)

        # 3. Accumulate importance for reflection trigger
        for event in perceived_events:
            agent.decrement_importance_trigger(event.importance)

        # 4. Get other occupants for consultation (general conversations disabled)
        other_occupants = sim_state.get("other_occupants_present", [])

        # 5. REFLECT: Generate insights if threshold reached
        await reflect(agent, now)

        # NOTE: Post-conversation reflections removed - commitments are now
        # extracted directly in record_conversation_to_memory() and added to
        # daily_plan. The goal is actionable plan changes, not memory thoughts.

        # 6. Get meeting context
        meeting_context = get_meeting_context(agent_id, self.calendar, now)
        pending_invitations = self.calendar.get_pending_invitations(agent_id)

        # Add focal point for meeting attendance only if meeting is imminent (within 10 min)
        focal_points = get_decision_focal_points(sim_state, checkpoint_reason)
        minutes_to_next = meeting_context.get("minutes_to_next_meeting")
        has_current_meeting = meeting_context.get("current_meeting") is not None
        if has_current_meeting or (minutes_to_next is not None and minutes_to_next <= 10):
            focal_points.append("upcoming meeting")

        # 7. RETRIEVE: Get relevant memories for decision
        retrieved = retrieve(agent, focal_points, now)

        # 8. Get colleague context
        colleague_context = format_colleague_context(agent, other_occupants)

        # 9. Build prompt with cognitive and meeting context
        prompt = format_step_prompt(
            agent=agent,
            sim_state=sim_state,
            retrieved_memories=retrieved,
            now=now,
            meeting_context=meeting_context,
            pending_invitations=pending_invitations,
            colleague_context=colleague_context,
            checkpoint_reason=checkpoint_reason,
        )

        # 9.5. Log prompt and memories for debugging (if enabled)
        self._log_prompt(agent_id, now, prompt, checkpoint_reason, retrieved)

        # 10. Call LLM via async Runner.run with retry (OpenAI best practice)
        # max_turns=5 is sufficient for step decisions (minimal tool use)
        step_agent = self._step_agents[agent_id]
        context = SimContext(
            occupant_id=agent_id,
            now=now,
            simulation=self.adapter,
            calendar=self.calendar,
            configured_agent_ids=list(self._agents.keys()),
        )
        result = await run_with_retry(step_agent, prompt, context=context, max_turns=5)
        step_decision: StepDecisions = result.final_output

        # 10.5. Convert StepDecisions to OccupantAction list
        actions = convert_step_decisions_to_actions(step_decision)

        # 10.6. Create OccupantStepDecision wrapper for backward compatibility with adapter
        decision = OccupantStepDecision(
            occupant_id=step_decision.occupant_id,
            datetime_iso=step_decision.timestamp,
            location_zone=sim_state.get("current_location", "desk_area"),
            current_desk=sim_state.get("current_desk"),
            is_present=True,
            actions=actions,
            brief_rationale=f"Checkpoint: {checkpoint_reason}",
        )

        # 11. Handle invitation responses (calendar access needed)
        for action in decision.actions:
            if action.action_type == "respond_to_invitation":
                event_id = action.parameters.get("event_id")
                accept = action.parameters.get("accept", False)
                if event_id:
                    success = self.calendar.respond_to_invitation(
                        event_id=event_id,
                        agent_id=agent_id,
                        accept=accept,
                        now=now,
                    )
                    if success:
                        response_str = "accepted" if accept else "declined"
                        print(f"[LLM] {agent_id} {response_str} invitation for event {event_id[:8]}...")

        # 11.1. Log lunch/break actions (status flags handled by simulation_adapter.py)
        for action in decision.actions:
            if action.action_type in ("go_to_lunch", "go_out_for_lunch"):
                out_of_building = action.action_type == "go_out_for_lunch"
                location = "outside" if out_of_building else "break_area"
                print(f"[LLM] {agent_id} going to lunch ({location})")
            elif action.action_type == "return_from_lunch":
                print(f"[LLM] {agent_id} returning from lunch")
            elif action.action_type == "take_break":
                print(f"[LLM] {agent_id} taking a break")
            elif action.action_type == "return_from_break":
                print(f"[LLM] {agent_id} returning from break")

        # 11.5. CONSULTATION: Before shared-space actions, consult with other occupants
        decision = await self._maybe_consult_on_shared_action(
            agent=agent,
            decision=decision,
            other_occupants=other_occupants,
            sim_state=sim_state,
            now=now,
        )

        # 12. Apply decision to simulation
        # Capture location before applying decision for memory recording
        previous_location = self.adapter.get_agent_location(agent_id)

        self.adapter.apply_decision(decision)

        # 12.1. Record location changes to agent memory
        new_location = self.adapter.get_agent_location(agent_id)
        if new_location != previous_location:
            # Determine what kind of move this was
            location_actions = [a for a in decision.actions if a.action_type in (
                "move_to", "go_to_lunch", "go_out_for_lunch", "return_from_lunch",
                "take_break", "return_from_break", "attend_meeting", "leave_meeting"
            )]
            move_reason = ""
            if location_actions:
                move_action = location_actions[0]
                if move_action.action_type == "go_to_lunch":
                    move_reason = " to have lunch"
                elif move_action.action_type == "go_out_for_lunch":
                    move_reason = " to have lunch outside"
                elif move_action.action_type == "return_from_lunch":
                    move_reason = " after lunch"
                elif move_action.action_type == "take_break":
                    activity = move_action.parameters.get("activity", "break")
                    move_reason = f" for a {activity}"
                elif move_action.action_type == "return_from_break":
                    move_reason = " after break"
                elif move_action.action_type == "attend_meeting":
                    move_reason = " for a meeting"
                elif move_action.action_type == "leave_meeting":
                    move_reason = " after the meeting ended"

            # Record location change to agent memory
            location_desc = f"I moved from {previous_location or 'unknown'} to {new_location}{move_reason}"
            # Use LLM to assess importance
            location_importance = await get_importance(agent, location_desc, 3.0, use_llm=True)
            agent.memory_stream.add_event(
                description=location_desc,
                subject=agent_id,
                predicate="moved to",
                obj=new_location,
                now=now,
                importance=location_importance,
            )
            print(f"[LOCATION] {agent_id}: {location_desc}")

            # Special memory for returning from outside the building
            if previous_location == "outside" and new_location != "outside":
                # Determine what activity they were doing
                return_action = next(
                    (a for a in decision.actions if a.action_type in ("return_from_lunch", "return_from_break")),
                    None
                )
                if return_action:
                    activity = return_action.parameters.get("activity", "break")
                    if return_action.action_type == "return_from_lunch":
                        outside_desc = f"I went outside for lunch. It was refreshing to step out of the office."
                    else:
                        outside_desc = f"I went outside for a {activity}. The fresh air was nice."

                    # Use LLM to assess importance
                    outside_importance = await get_importance(agent, outside_desc, 4.0, use_llm=True)
                    agent.memory_stream.add_event(
                        description=outside_desc,
                        subject=agent_id,
                        predicate="returned from",
                        obj="outside activity",
                        now=now,
                        importance=outside_importance,
                    )
                    print(f"[OUTSIDE] {agent_id}: {outside_desc}")

        # 12.25. Process pending conversations
        pending_convs = self.adapter.get_pending_conversations()
        for conv in pending_convs:
            await self._process_conversation(
                conv["initiator"], conv["target"], conv["topic"], now
            )

        # 12.3. Track lunch/break starts for return checkpoints
        for action in decision.actions:
            if action.action_type in ("go_to_lunch", "go_out_for_lunch"):
                self._checkpoint_manager._lunch_started[agent_id] = now
                # Check for commitment fulfillment
                self._check_commitment_fulfillment(agent_id, "lunch", now)
            elif action.action_type in ("take_break", "go_out_for_break"):
                self._checkpoint_manager._break_started[agent_id] = now
                # Check for commitment fulfillment
                activity = action.parameters.get("activity", "break")
                self._check_commitment_fulfillment(agent_id, activity, now)

        # 12.5. Process plan updates via action path (from adapter's pending list)
        pending_plan_updates = self.adapter.get_pending_plan_updates()
        for plan_update in pending_plan_updates:
            target_agent_id = plan_update.get("occupant_id", agent_id)
            target_agent = self._agents.get(target_agent_id)
            if target_agent:
                updates = plan_update.get("updates", {})
                reason = plan_update.get("reason", "Plan adjustment")
                result = target_agent.update_daily_plan(updates, reason=reason, now=now)
                if result.get("updated"):
                    print(f"[SCHEDULE] {target_agent_id}: Plan adjusted - {reason}")
                    if result.get("skipped_past_events"):
                        print(f"[SCHEDULE] {target_agent_id}: (skipped past events: {result['skipped_past_events']})")

        # 13. Record decision to memory (using StepDecisions for structured reasoning)
        await record_decision_to_memory(agent, step_decision, now)

        # 13.5. Handle commitment outcomes (M.2 - relationship consequences)
        await self._handle_commitment_outcome(agent_id, step_decision, checkpoint_reason, now)

        # 14. Update tracking
        self._last_decision_time[agent_id] = now

        # 15. Log to file with memory context (using StepDecisions for structured output)
        self._log_decision(agent_id, now, step_decision, retrieved, sim_state, checkpoint_reason)

        # 16. Log to shared action log (using actions list)
        self._log_action(agent_id, decision, now)

        # 17. Save agent state
        agent.save()

        return step_decision

    async def _maybe_consult_on_shared_action(
        self,
        agent: GenerativeAgent,
        decision: OccupantStepDecision,
        other_occupants: List[str],
        sim_state: Dict[str, Any],
        now: datetime,
    ) -> OccupantStepDecision:
        """
        Check if decision includes shared-space changes and consult with others.

        If the agent wants to adjust thermostat or windows while others are present,
        this triggers a consultation conversation to reach consensus.

        Args:
            agent: The agent making the decision
            decision: The original decision from the LLM
            other_occupants: List of other occupant IDs present
            sim_state: Current simulation state
            now: Current datetime

        Returns:
            Modified decision (may have updated setpoint or cancelled action)
        """
        # Check for shared-space actions
        shared_space_action = None
        proposed_setpoint = None

        for action in decision.actions:
            if action.action_type == "thermostat_adjust":
                shared_space_action = action
                # Calculate proposed setpoint from delta
                # Actions use setpoint_delta_c (relative change), not setpoint_c (absolute)
                delta = action.parameters.get("setpoint_delta_c", 0)
                current_setpoint = sim_state.get("thermostat_setpoint_c", 21.0)
                proposed_setpoint = current_setpoint + delta if delta else None
                # Store the calculated setpoint back for later use
                if proposed_setpoint is not None:
                    action.parameters["_proposed_setpoint_c"] = proposed_setpoint
                break
            elif action.action_type == "window_set":
                shared_space_action = action
                break

        # No shared-space action, return original decision
        if not shared_space_action:
            return decision

        # For thermostat: skip consultation if proposed setpoint equals current setpoint
        if shared_space_action.action_type == "thermostat_adjust":
            current_setpoint = sim_state.get("thermostat_setpoint_c")
            if current_setpoint is not None and proposed_setpoint is not None:
                # Check if they're effectively the same (within 0.1 degree tolerance)
                if abs(proposed_setpoint - current_setpoint) < 0.1:
                    # No actual change - skip consultation
                    return decision

        # No other occupants, no consultation needed
        if not other_occupants:
            return decision

        # Get other agents who are present
        present_agents = [agent]  # Include initiator
        for other_id in other_occupants:
            other_agent = self._agents.get(other_id)
            if other_agent:
                present_agents.append(other_agent)

        # If only the initiating agent (shouldn't happen but safety check)
        if len(present_agents) <= 1:
            return decision

        # ORGANIC CONVERSATIONS: Only consult if there's potential conflict
        # Check if other agents have significantly different thermal preferences
        if shared_space_action.action_type == "thermostat_adjust" and proposed_setpoint is not None:
            initiator_pref = agent.scratch.get("thermal_comfort_c", 21.0)
            has_potential_conflict = False

            for other_agent in present_agents[1:]:  # Skip initiator
                other_pref = other_agent.scratch.get("thermal_comfort_c", 21.0)
                # Check if preferences differ significantly (> 1.5°C apart)
                # or if proposed change moves away from other's preference
                pref_diff = abs(initiator_pref - other_pref)
                proposal_moves_away = (
                    (proposed_setpoint > other_pref + 1.0 and initiator_pref > other_pref) or
                    (proposed_setpoint < other_pref - 1.0 and initiator_pref < other_pref)
                )
                if pref_diff > 1.5 or proposal_moves_away:
                    has_potential_conflict = True
                    break

            if not has_potential_conflict:
                # No significant preference difference - allow action without consultation
                # Record a brief notification to other agents' memories instead
                for other_agent in present_agents[1:]:
                    if other_agent.memory_stream:
                        notification = f"{agent.first_name} adjusted the thermostat to {proposed_setpoint:.1f}°C"
                        # Use LLM to assess importance
                        notify_importance = await get_importance(other_agent, notification, 2.0, use_llm=True)
                        other_agent.memory_stream.add_event(
                            description=notification,
                            subject=agent.first_name,
                            predicate="adjusted",
                            obj="thermostat",
                            now=now,
                            importance=notify_importance,
                        )
                print(f"[CTRL] {agent.agent_id}: Adjusted thermostat (colleagues have similar preferences - no consultation needed)")
                return decision

        # Build proposed action description
        if shared_space_action.action_type == "thermostat_adjust":
            direction = shared_space_action.parameters.get("direction", "adjust")
            delta = shared_space_action.parameters.get("setpoint_delta_c", 0)
            current_setpoint = sim_state.get("thermostat_setpoint_c", 21.0)
            if proposed_setpoint is not None:
                proposed_action = f"adjust thermostat {direction} to {proposed_setpoint:.1f}°C (currently {current_setpoint:.1f}°C)"
            else:
                proposed_action = f"adjust thermostat {direction} by {abs(delta):.1f}°C"
        else:
            window_state = "open" if shared_space_action.parameters.get("open", False) else "close"
            proposed_action = f"{window_state} the window"

        current_temp = sim_state.get("zone_air_temp_c", 21.0)

        # Run consultation
        try:
            outcome = await consultation_conversation(
                initiator=agent,
                proposed_action=proposed_action,
                proposed_setpoint_c=proposed_setpoint,
                current_temp_c=current_temp,
                present_agents=present_agents,
                now=now,
                max_turns=6,
                calendar=self.calendar,
            )

            # Record agreement to all participants' memories
            await record_agreement_to_memory(present_agents, outcome, now)

            # Log consultation to file
            self._log_consultation(outcome, now)

            # Modify decision based on outcome
            if not outcome.consensus_reached:
                # No agreement - cancel the shared-space action
                print(f"[CONSULTATION] No consensus reached - cancelling {shared_space_action.action_type}")
                new_actions = [
                    a for a in decision.actions
                    if a.action_type != shared_space_action.action_type
                ]
                # Add no_op if no actions remain
                if not new_actions:
                    from bsm.agents.skeleton import OccupantAction
                    new_actions = [OccupantAction(action_type="no_op", parameters={})]

                return OccupantStepDecision(
                    occupant_id=decision.occupant_id,
                    datetime_iso=decision.datetime_iso,
                    location_zone=decision.location_zone,
                    current_desk=decision.current_desk,
                    is_present=decision.is_present,
                    actions=new_actions,
                    brief_rationale=f"Consultation: {outcome.summary}",
                )

            elif outcome.final_setpoint_c and outcome.final_setpoint_c != proposed_setpoint:
                # Consensus reached with different setpoint
                print(f"[CONSULTATION] Agreed on {outcome.final_setpoint_c}°C (was {proposed_setpoint}°C)")

                # Update the action with agreed setpoint
                new_actions = []
                for action in decision.actions:
                    if action.action_type == "thermostat_adjust":
                        from bsm.agents.skeleton import OccupantAction
                        new_actions.append(OccupantAction(
                            action_type="thermostat_adjust",
                            parameters={"setpoint_c": outcome.final_setpoint_c},
                        ))
                    else:
                        new_actions.append(action)

                return OccupantStepDecision(
                    occupant_id=decision.occupant_id,
                    datetime_iso=decision.datetime_iso,
                    location_zone=decision.location_zone,
                    current_desk=decision.current_desk,
                    is_present=decision.is_present,
                    actions=new_actions,
                    brief_rationale=f"Consultation: {outcome.summary}",
                )

            else:
                # Consensus reached with original proposal
                print(f"[CONSULTATION] Original proposal agreed")
                return decision

        except Exception as e:
            print(f"[CONSULTATION] Failed: {e}")
            return decision  # Fall back to original decision on error

    def _log_consultation(
        self,
        outcome: ConsultationOutcome,
        now: datetime,
    ) -> None:
        """Log a consultation outcome to the shared actions log."""
        try:
            log_path = self._actions_log_path
            if not log_path or not os.path.exists(log_path):
                return

            timestamp = now.strftime('%Y-%m-%d %H:%M')
            participants = ", ".join(outcome.participants)

            with open(log_path, 'a') as f:
                f.write(f"\n[{timestamp}] CONSULTATION\n")
                f.write(f"  Participants: {participants}\n")
                f.write(f"  Proposed: {outcome.proposed_action}\n")
                f.write(f"  Consensus: {'Yes' if outcome.consensus_reached else 'No'}\n")
                if outcome.agreed_action:
                    f.write(f"  Agreed: {outcome.agreed_action}\n")
                f.write(f"  Summary: {outcome.summary}\n")

        except IOError as e:
            print(f"Warning: Failed to log consultation: {e}")

    async def _process_conversation(
        self,
        initiator_id: str,
        target_id: str,
        topic: str,
        now: datetime,
    ) -> Optional[ConversationResult]:
        """
        Process a social conversation between agents.

        Checks that both agents are at the same location before initiating.

        Args:
            initiator_id: Agent who started the conversation
            target_id: Agent being talked to
            topic: Conversation topic
            now: Current simulation datetime

        Returns:
            ConversationResult if successful, None otherwise
        """
        initiator = self._agents.get(initiator_id)
        target = self._agents.get(target_id)

        if not initiator or not target:
            print(f"[CONV] Cannot find agents: {initiator_id} or {target_id}")
            return None

        # Check same location
        initiator_loc = self.adapter.get_agent_location(initiator_id)
        target_loc = self.adapter.get_agent_location(target_id)
        if initiator_loc != target_loc:
            print(f"[CONV] {initiator_id} can't talk to {target_id} - different locations ({initiator_loc} vs {target_loc})")
            return None

        # Block conversations when either agent is outside the building
        # We don't simulate what happens outside, so no conversations there
        if initiator_loc == "outside" or target_loc == "outside":
            print(f"[CONV] {initiator_id} can't converse - one or both agents outside building")
            return None

        try:
            # Sync agent status from adapter to agent.scratch before conversation
            # This ensures _check_target_willingness() has accurate availability info
            target_status = self.adapter.get_agent_status(target_id)
            target_location = self.adapter.get_agent_location(target_id)
            target.scratch["current_status"] = {
                "in_meeting": target_location == "meeting_room",  # in_meeting if in meeting_room
                "on_break": target_status.get("on_break", False),
                "at_lunch": target_status.get("at_lunch", False),
                "out_of_office": target_status.get("out_of_office", False),
            }

            # Get list of valid colleagues in the office for anti-hallucination
            valid_colleagues = list(self._agents.keys())

            # Get weather context for natural conversation about weather
            current_weather, weather_history = self.get_weather_context()

            result = await agent_conversation(
                init_agent=initiator,
                target_agent=target,
                now=now,
                calendar=self.calendar,
                valid_colleagues=valid_colleagues,
                topic=topic,
                current_weather=current_weather,
                weather_history=weather_history,
            )

            # Handle declined conversations differently
            if result and result.declined:
                # Log the declined conversation but don't record to memory as a full conversation
                print(f"[CONV] {target_id} declined conversation with {initiator_id}: {result.decline_reason}")
                self._log_conversation(initiator_id, target_id, topic, result, now)
                return result

            # Record successful conversation to memory for BOTH parties
            # This ensures both agents remember the conversation, get relationship updates,
            # and have any commitments extracted and added to their daily plans
            if result:
                await record_conversation_to_memory(
                    agent=initiator,
                    conversation=result,
                    other_agent_id=target_id,
                    now=now,
                )
                await record_conversation_to_memory(
                    agent=target,
                    conversation=result,
                    other_agent_id=initiator_id,
                    now=now,
                )
                # CRITICAL: Save both agents to persist memory changes and social_commitments
                # Without this, only the current agent in _occupant_step would be saved
                initiator.save()
                target.save()
                print(f"[CONV-DEBUG] Saved both agents after conversation: {initiator_id}, {target_id}")

            self._log_conversation(initiator_id, target_id, topic, result, now)
            self._log_conversation_to_shared_file(result, now)
            return result
        except Exception as e:
            print(f"[CONV] Conversation failed: {e}")
            return None

    def _log_conversation(
        self,
        initiator_id: str,
        target_id: str,
        topic: str,
        result: ConversationResult,
        now: datetime,
    ) -> None:
        """Log a conversation to the actions log."""
        try:
            log_path = self._actions_log_path
            if not log_path or not os.path.exists(log_path):
                return

            timestamp = now.strftime('%Y-%m-%d %H:%M')

            with open(log_path, 'a') as f:
                f.write(f"\n[{timestamp}] CONVERSATION\n")
                f.write(f"  Initiator: {initiator_id}\n")
                f.write(f"  Target: {target_id}\n")
                f.write(f"  Topic: {topic}\n")
                f.write(f"  Turns: {len(result.utterances)}\n")
                if result.summary:
                    f.write(f"  Summary: {result.summary}\n")
                f.write("  ---\n")
                for utt in result.utterances:
                    speaker_name = utt.speaker_id.split('_')[0].capitalize() if '_' in utt.speaker_id else utt.speaker_id
                    f.write(f"  {speaker_name}: {utt.utterance[:100]}{'...' if len(utt.utterance) > 100 else ''}\n")

        except IOError as e:
            print(f"Warning: Failed to log conversation: {e}")

    def _make_agent_decision(
        self,
        agent_id: str,
        now: datetime,
        checkpoint_reason: str = "interval",
    ) -> Optional[OccupantStepDecision]:
        """
        Sync wrapper for cognitive loop (for compatibility with main.py).

        Args:
            agent_id: The agent making the decision
            now: Current simulation datetime
            checkpoint_reason: Why decision is being made (hourly, meeting_start, lunch_time, etc.)

        Returns the decision or None if failed.
        """
        try:
            decision = asyncio.run(
                self._make_agent_decision_async(agent_id, now, checkpoint_reason=checkpoint_reason)
            )

            # Log non-trivial decisions to console
            if decision:
                non_trivial = []
                # Check each category for non-trivial actions
                if decision.thermostat.action != "maintain_current":
                    non_trivial.append(f"thermostat:{decision.thermostat.action}")
                if decision.lighting.action != "keep_current":
                    non_trivial.append(f"lighting:{decision.lighting.action}")
                # Check equipment_decisions list for any non-trivial actions
                for eq_dec in decision.equipment_decisions:
                    if eq_dec.action != "keep_current":
                        non_trivial.append(f"equipment:{eq_dec.equipment_name}:{eq_dec.action}")
                if decision.location.action != "stay":
                    non_trivial.append(f"location:{decision.location.action}")
                if decision.conversation.action != "none":
                    non_trivial.append(f"conversation:{decision.conversation.action}")
                if decision.break_decision and decision.break_decision.action != "continue_working":
                    non_trivial.append(f"break:{decision.break_decision.action}")
                if decision.meeting_equipment and decision.meeting_equipment.action != "accept_current":
                    non_trivial.append(f"meeting_equipment:{decision.meeting_equipment.action}")
                if decision.plan_update.action != "keep_current":
                    non_trivial.append(f"plan_update:{decision.plan_update.action}")

                if non_trivial:
                    actions_str = ", ".join(non_trivial)
                    print(f"[LLM] {agent_id}: {actions_str}")

            return decision

        except Exception as e:
            print(f"[LLM] {agent_id} decision failed: {e}")
            raise RuntimeError(f"Agent decision failed for {agent_id}: {e}")

    def _log_prompt(
        self,
        agent_id: str,
        now: datetime,
        prompt: str,
        checkpoint_reason: str,
        retrieved_memories: Optional[Dict[str, List[MemoryNode]]] = None,
    ) -> None:
        """
        Log full prompt and retrieved memories for debugging.

        Saves to: results/agents/{date}/{time}/{agent_id}/prompts.log

        Args:
            agent_id: Agent ID
            now: Current simulation datetime
            prompt: Full prompt sent to LLM
            checkpoint_reason: Why decision was triggered
            retrieved_memories: Dict of focal_point -> list of MemoryNode
        """
        if not self._log_prompts:
            return

        log_path = self._prompt_log_paths.get(agent_id)
        if not log_path:
            return

        try:
            with open(log_path, 'a') as f:
                f.write(f"\n{'=' * 80}\n")
                f.write(f"[{now.strftime('%Y-%m-%d %H:%M')}] checkpoint: {checkpoint_reason}\n")
                f.write(f"{'=' * 80}\n")

                # Log retrieved memories
                if retrieved_memories:
                    f.write("\n--- RETRIEVED MEMORIES ---\n")
                    for focal_pt, memories in retrieved_memories.items():
                        f.write(f"\nFocal point: '{focal_pt}'\n")
                        for mem in memories[:5]:  # Limit to top 5 per focal point
                            if hasattr(mem, 'description'):
                                f.write(f"  - {mem.description[:200]}\n")
                            else:
                                f.write(f"  - {str(mem)[:200]}\n")

                # Log full prompt
                f.write("\n--- FULL PROMPT ---\n")
                f.write(prompt)
                f.write("\n--- END PROMPT ---\n\n")

        except IOError as e:
            print(f"Warning: Failed to log prompt for {agent_id}: {e}")

    def _log_decision(
        self,
        agent_id: str,
        now: datetime,
        decision: StepDecisions,
        retrieved_memories: Optional[Dict[str, List[MemoryNode]]] = None,
        sim_state: Optional[Dict[str, Any]] = None,
        checkpoint_reason: str = "interval",
    ) -> None:
        """Log a structured decision to the agent's decision log file with full context."""
        try:
            log_path = self._decision_log_paths.get(agent_id)
            if not log_path:
                return
            with open(log_path, 'a') as f:
                timestamp = now.strftime('%Y-%m-%d %H:%M')

                # Header with checkpoint info
                f.write(f"\n{'='*70}\n")
                f.write(f"[{timestamp}] CHECKPOINT: {checkpoint_reason}\n")
                f.write(f"{'='*70}\n")

                # State before decision
                if sim_state:
                    f.write(f"\n--- STATE BEFORE ---\n")
                    f.write(f"Location: {sim_state.get('current_location', 'N/A')}\n")
                    f.write(f"Desk: {sim_state.get('current_desk', 'N/A')}\n")
                    f.write(f"Indoor Temp: {sim_state.get('indoor_temp_c', 'N/A')}C\n")
                    equipment = sim_state.get('equipment_status', {}).get('items', {})
                    if equipment:
                        f.write(f"Equipment:\n")
                        for eq, is_on in equipment.items():
                            f.write(f"  {eq}: {'ON' if is_on else 'OFF'}\n")

                # All category decisions with reasoning
                f.write(f"\n--- DECISIONS ---\n")
                f.write(f"Thermostat: {decision.thermostat.action}\n")
                f.write(f"  Reason: {decision.thermostat.reasoning}\n")
                if decision.thermostat.action == "adjust":
                    f.write(f"  Direction: {decision.thermostat.adjustment_direction}, Amount: {decision.thermostat.adjustment_amount}\n")

                f.write(f"Lighting: {decision.lighting.action}\n")
                f.write(f"  Reason: {decision.lighting.reasoning}\n")

                # Equipment decisions (list format)
                f.write(f"Equipment Decisions: {len(decision.equipment_decisions)} items\n")
                for eq_dec in decision.equipment_decisions:
                    f.write(f"  - {eq_dec.equipment_name}: {eq_dec.action}\n")
                    f.write(f"    Reason: {eq_dec.reasoning}\n")

                f.write(f"Location: {decision.location.action}\n")
                f.write(f"  Reason: {decision.location.reasoning}\n")
                if decision.location.destination:
                    f.write(f"  Destination: {decision.location.destination}\n")

                f.write(f"Conversation: {decision.conversation.action}\n")
                f.write(f"  Reason: {decision.conversation.reasoning}\n")

                f.write(f"Plan Update: {decision.plan_update.action}\n")
                f.write(f"  Reason: {decision.plan_update.reasoning}\n")

                # Optional decisions
                if decision.break_decision:
                    f.write(f"Break: {decision.break_decision.action}\n")
                    f.write(f"  Reason: {decision.break_decision.reasoning}\n")

                # Log retrieved memories (cognitive context)
                if retrieved_memories:
                    f.write(f"\n--- RETRIEVED MEMORIES ---\n")
                    for focal_pt, memories in retrieved_memories.items():
                        f.write(f"  '{focal_pt}': {len(memories)} memories\n")
                        for mem in memories[:2]:
                            desc = mem.description[:80] + "..." if len(mem.description) > 80 else mem.description
                            f.write(f"    - {desc}\n")

        except IOError as e:
            print(f"Warning: Failed to log decision for {agent_id}: {e}")

    def _log_conversation_to_shared_file(
        self,
        conversation: ConversationResult,
        now: datetime,
    ) -> None:
        """Log conversation to shared conversations.log file and UI demo output."""
        try:
            log_path = self._agent_paths.get("shared", {}).get("conversation_log")
            if not log_path:
                return

            with open(log_path, "a") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"TIME: {now.strftime('%Y-%m-%d %H:%M')}\n")
                f.write(f"PARTICIPANTS: {', '.join(conversation.participants)}\n")
                f.write(f"INITIATED BY: {conversation.initiated_by}\n")
                f.write(f"TOPICS: {', '.join(conversation.topics_discussed)}\n")
                f.write(f"\n--- DIALOGUE ---\n")
                for utt in conversation.utterances:
                    f.write(f"{utt.speaker_id}: \"{utt.utterance}\"\n")
                f.write(f"\nSUMMARY: {conversation.summary}\n")
                f.write(f"DURATION: {conversation.duration_minutes} minutes\n")
        except IOError as e:
            print(f"Warning: Failed to log conversation: {e}")

        # Write to UI demo output if enabled
        if self._ui_writer:
            # Create conversation ID from timestamp and participants
            p1 = conversation.participants[0] if conversation.participants else "unknown"
            p2 = conversation.participants[1] if len(conversation.participants) > 1 else "unknown"
            conv_id = f"conv_{now.strftime('%Y%m%d_%H%M%S')}_{p1}_{p2}"

            self._ui_writer.write_conversation(
                conversation_id=conv_id,
                conversation_data={
                    "start_time": now.isoformat(),
                    "participants": conversation.participants,
                    "initiated_by": conversation.initiated_by,
                    "topics": conversation.topics_discussed,
                    "dialogue": [
                        {
                            "speaker": utt.speaker_id,
                            "speaker_name": utt.speaker_id.split("_")[0].capitalize(),
                            "utterance": utt.utterance,
                        }
                        for utt in conversation.utterances
                    ],
                    "duration_minutes": conversation.duration_minutes,
                    "summary": conversation.summary,
                }
            )

    def _check_commitment_fulfillment(
        self,
        agent_id: str,
        activity: str,
        now: datetime,
    ) -> None:
        """
        Check if this break/lunch action fulfills a social commitment.

        Looks for matching unfulfilled commitments and checks if the other
        party is also at the same location. If so, marks the commitment
        as fulfilled for both agents.
        """
        agent = self._agents.get(agent_id)
        if not agent:
            return

        daily_plan = agent.get_daily_plan()
        if not daily_plan:
            return

        if not hasattr(daily_plan, 'social_commitments') or not daily_plan.social_commitments:
            return

        current_location = self.adapter.get_agent_location(agent_id)
        current_minutes = now.hour * 60 + now.minute

        for commitment in daily_plan.social_commitments:
            if commitment.fulfilled:
                continue

            # Check if this activity matches the commitment
            activity_lower = activity.lower()
            commitment_activity = commitment.activity.lower()
            if not (activity_lower in commitment_activity or commitment_activity in activity_lower):
                continue

            # Check if time is within 15 minutes of commitment time
            if commitment.time != "unspecified":
                try:
                    comm_hour, comm_minute = map(int, commitment.time.split(":"))
                    comm_minutes = comm_hour * 60 + comm_minute
                    if abs(current_minutes - comm_minutes) > 15:
                        continue
                except (ValueError, TypeError):
                    pass

            # Check if other agents in the commitment are at the same location
            other_agents_present = True
            for other_id in commitment.with_agents:
                if other_id == agent_id:
                    continue
                other_location = self.adapter.get_agent_location(other_id)
                if other_location != current_location:
                    other_agents_present = False
                    break

            if other_agents_present:
                # Mark commitment as fulfilled
                commitment.fulfilled = True
                print(f"[COMMITMENT] {agent_id}: Fulfilled '{commitment.activity}' with {commitment.with_agents}")

                # Also mark as fulfilled for the other agents
                for other_id in commitment.with_agents:
                    if other_id == agent_id:
                        continue
                    other_agent = self._agents.get(other_id)
                    if other_agent:
                        other_daily_plan = other_agent.get_daily_plan()
                        if other_daily_plan and hasattr(other_daily_plan, 'social_commitments'):
                            for other_commitment in other_daily_plan.social_commitments:
                                if (not other_commitment.fulfilled and
                                    other_commitment.activity.lower() == commitment.activity.lower() and
                                    agent_id in other_commitment.with_agents):
                                    other_commitment.fulfilled = True
                                    print(f"[COMMITMENT] {other_id}: Fulfilled '{other_commitment.activity}' with {agent_id}")
                                    break

    async def _handle_commitment_outcome(
        self,
        agent_id: str,
        step_decision: "StepDecisions",
        checkpoint_reason: str,
        now: datetime,
    ) -> None:
        """
        Handle relationship consequences for commitment checkpoints (M.2).

        When a commitment checkpoint fires and the agent doesn't execute it
        (defer, skip, or no action taken), this records the impact:
        1. Adds a memory about not meeting the person
        2. Updates relationship sentiment negatively

        Args:
            agent_id: The agent who received the commitment checkpoint
            step_decision: The decision they made
            checkpoint_reason: The checkpoint reason (should start with "commitment:")
            now: Current datetime
        """
        if not checkpoint_reason.startswith("commitment:"):
            return

        agent = self._agents.get(agent_id)
        if not agent:
            return

        activity = checkpoint_reason.split(":", 1)[1] if ":" in checkpoint_reason else "activity"

        # Get commitment details
        daily_plan = agent.get_daily_plan()
        if not daily_plan or not hasattr(daily_plan, 'social_commitments'):
            return

        # Find the matching commitment
        matching_commitment = None
        for commitment in daily_plan.social_commitments:
            if commitment.fulfilled:
                continue
            if activity.lower() in commitment.activity.lower():
                matching_commitment = commitment
                break

        if not matching_commitment:
            return

        partner_agents = matching_commitment.with_agents
        if not partner_agents:
            return

        # Determine what action was taken
        commitment_response = getattr(step_decision, 'commitment_response', None)
        action_taken = "no_action"
        if commitment_response:
            response_action = getattr(commitment_response, 'action', 'no_action')
            if response_action in ("execute", "defer", "skip"):
                action_taken = response_action

        # Check if agent actually went on break/lunch (executed)
        if step_decision.actions:
            for action in step_decision.actions:
                action_type = getattr(action, 'action_type', '')
                if action_type in ("take_break", "go_out_for_break", "go_to_lunch", "go_out_for_lunch"):
                    action_taken = "execute"
                    break

        # If they executed, no negative consequences
        if action_taken == "execute":
            return

        # Get partner name(s) for memory
        partner_names = []
        for pid in partner_agents:
            other_agent = self._agents.get(pid)
            if other_agent:
                partner_names.append(other_agent.first_name)
            else:
                partner_names.append(pid.split("_")[0].capitalize())

        if len(partner_names) == 1:
            partner_str = partner_names[0]
        elif len(partner_names) == 2:
            partner_str = f"{partner_names[0]} and {partner_names[1]}"
        else:
            partner_str = ", ".join(partner_names[:-1]) + f", and {partner_names[-1]}"

        # Determine consequences based on action
        deferred_count = getattr(matching_commitment, 'deferred_count', 0)

        if action_taken == "defer":
            # Deferral - mild negative impact
            sentiment_delta = -0.03 if deferred_count < 2 else -0.05
            memory_desc = (
                f"I had to postpone {activity} with {partner_str} again. "
                f"They were probably expecting me. I should make it up to them."
            )
            importance = 5.0 if deferred_count < 2 else 7.0
            print(f"[COMMITMENT] {agent_id}: Deferred {activity} with {partner_str} (count: {deferred_count + 1})")

            # Update deferral count
            matching_commitment.deferred = True
            matching_commitment.deferred_count = deferred_count + 1

        elif action_taken == "skip":
            # Skip - stronger negative impact
            sentiment_delta = -0.08
            memory_desc = (
                f"I skipped {activity} with {partner_str}. "
                f"They were probably waiting for me. I feel bad about letting them down."
            )
            importance = 8.0
            print(f"[COMMITMENT] {agent_id}: Skipped {activity} with {partner_str}")

            # Mark as fulfilled (broken, but no longer pending)
            matching_commitment.fulfilled = True

        else:
            # No action (no_op) - treated like skip
            sentiment_delta = -0.06
            memory_desc = (
                f"I didn't meet {partner_str} for {activity} as I promised. "
                f"They were probably waiting. I should apologize and reschedule."
            )
            importance = 7.0
            print(f"[COMMITMENT] {agent_id}: Missed {activity} with {partner_str} (no action taken)")

        # Record memory
        agent.memory_stream.add_event(
            description=memory_desc,
            subject="I",
            predicate="missed commitment with",
            obj=partner_str,
            now=now,
            importance=importance,
        )

        # Update relationship sentiment with each partner
        for partner_id in partner_agents:
            agent.update_relationship(partner_id, sentiment_delta=sentiment_delta)
            print(f"[RELATIONSHIP] {agent_id} -> {partner_id}: sentiment {sentiment_delta:+.2f}")

    def _log_action(
        self,
        agent_id: str,
        decision: OccupantStepDecision,
        now: datetime,
    ) -> None:
        """Log action to shared actions.log file and UI demo output."""
        try:
            log_path = self._agent_paths.get("shared", {}).get("action_log")
            if not log_path:
                return

            with open(log_path, "a") as f:
                for action in decision.actions:
                    # Truncate reason to 60 chars
                    reason = decision.brief_rationale or ""
                    if len(reason) > 60:
                        reason = reason[:57] + "..."
                    f.write(f"{now.strftime('%Y-%m-%d %H:%M')} | {agent_id:12} | {action.action_type:20} | {reason}\n")
        except IOError as e:
            print(f"Warning: Failed to log action for {agent_id}: {e}")

        # Write to UI demo output if enabled
        if self._ui_writer:
            agent = self._agents.get(agent_id)
            agent_name = agent.first_name if agent else agent_id.split("_")[0].capitalize()
            for action in decision.actions:
                self._ui_writer.write_agent_action(
                    agent_id=agent_id,
                    action_data={
                        "timestamp": now.isoformat(),
                        "action_type": action.action_type,
                        "details": action.parameters or {},
                        "reason": decision.brief_rationale or "",
                    },
                    agent_name=agent_name,
                )

    def get_window_state(self) -> float:
        """Return current window open fraction (0-1)."""
        return self.adapter.get_window_open_fraction()

    def get_thermostat_offset(self) -> float:
        """Return current thermostat offset from base setpoint (C)."""
        return self.adapter.get_thermostat_offset()

    def get_total_internal_gains_w(self, metabolic_w_per_person: float = 120.0) -> float:
        """
        Get total internal gains from LLM-managed occupants.

        Includes equipment + lighting + metabolic heat.

        Args:
            metabolic_w_per_person: Metabolic heat per person (default 120W sensible)

        Returns:
            Total internal gains in Watts
        """
        return self.adapter.get_total_occupant_gains_w(metabolic_w_per_person)

    def get_equipment_power_w(self) -> float:
        """Get total equipment power consumption."""
        return self.adapter.get_total_equipment_power_w()

    def get_lighting_power_w(self) -> float:
        """Get total lighting power consumption."""
        return self.adapter.get_total_lighting_power_w()

    def get_present_count(self) -> int:
        """Get number of currently present occupants."""
        return len(self.adapter.get_present_occupant_ids())

    def get_present_agents(self) -> List[str]:
        """Get list of currently present agent IDs."""
        return self.adapter.get_present_occupant_ids()

    def had_decisions_this_step(self) -> bool:
        """Check if any agent decisions were made during the current step.

        Used by runner.py to track decision count for progress reporting.
        """
        return self._decisions_made_this_step > 0

    def get_daily_plan(self, agent_id: str) -> Optional[DailyPlan]:
        """Get the daily plan for an agent."""
        return self._daily_plans.get(agent_id)

    def end_of_day_cleanup(self) -> None:
        """Clean up at end of simulated day."""
        self.adapter.end_of_day_cleanup()

    def save_all_agents(self) -> None:
        """Persist all agent state (memory streams, scratch) to files."""
        for agent in self._agents.values():
            agent.save()
        print(f"[LLM] Saved state for {len(self._agents)} agents")

    def get_agent_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all agents (memory counts, etc.)."""
        stats = {}
        for agent_id, agent in self._agents.items():
            stats[agent_id] = agent.get_stats()
        return stats

    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of current LLM occupant status."""
        present = self.adapter.get_present_occupant_ids()
        desk_occupancy = self.desks.get_desk_occupancy()
        meeting_room = self.desks.get_meeting_room_occupants()

        return {
            "total_agents": len(self._agent_ids),
            "present_agents": present,
            "present_count": len(present),
            "desk_occupancy": desk_occupancy,
            "meeting_room_occupants": meeting_room,
            "equipment_power_w": self.get_equipment_power_w(),
            "lighting_power_w": self.get_lighting_power_w(),
            "thermostat_offset_c": self.get_thermostat_offset(),
            "window_open_fraction": self.get_window_state(),
        }

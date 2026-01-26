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

from openai import OpenAI, APIError, APIConnectionError
from agents import Agent, Runner

from bsm.agents.skeleton import (
    CalendarStore,
    ClothingChoice,
    EmbeddingClient,
    OccupantStepDecision,
    DailyPlan,
    SimContext,
    build_step_agent,
    build_day_planner_agent,
    DEFAULT_AGENT_MODEL,
    DEFAULT_EMBED_MODEL,
)
from bsm.agents.generative_agent import GenerativeAgent, initialize_agents_for_simulation
from bsm.agents.cognition.modules import (
    perceive,
    retrieve,
    reflect,
    reflect_on_conversation,
    get_decision_focal_points,
    get_planning_focal_points,
    get_meeting_context,
    format_colleague_context,
    format_step_prompt,
    format_planning_prompt,
    record_decision_to_memory,
    record_plan_to_memory,
)
from bsm.agents.cognition.conversation import (
    consultation_conversation,
    record_agreement_to_memory,
    ConsultationOutcome,
    ConversationResult,
)
from bsm.agents.memory.stream import MemoryStream, MemoryNode
from bsm.agents.equipment_manager import EquipmentManager
from bsm.agents.lighting_manager import LightingManager
from bsm.agents.desk_manager import DeskManager
from bsm.agents.simulation_adapter import ZoneStateProvider, ProductionSimulationAdapter


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
        for agent_id in self._agent_ids:
            log_path = self._agent_paths[agent_id]["decision_log"]
            self._decision_log_paths[agent_id] = log_path
            with open(log_path, 'w') as f:
                f.write(f"# LLM Agent Decision Log - {agent_id}\n")
                f.write(f"# Run: {run_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("# Format: timestamp | action_type | parameters | rationale\n")
                f.write("-" * 80 + "\n")

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
        self.embedder = EmbeddingClient(model=llm_config.get("embed_model", DEFAULT_EMBED_MODEL))

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
            self._step_agents[agent_id] = build_step_agent(agent_id)
            self._planner_agents[agent_id] = build_day_planner_agent(agent_id)

        # Decision interval
        self._decision_interval_minutes = llm_config.get("decision_interval_minutes", 15)
        self._daily_planning_hour = llm_config.get("daily_planning_hour", 0)

        # Track daily plans
        self._daily_plans: Dict[str, DailyPlan] = {}
        self._last_plan_date: Optional[date] = None

        # Track last decision time per agent
        self._last_decision_time: Dict[str, datetime] = {}

        # Note: Core memories are now seeded from markdown files in agent_base_types/
        # via GenerativeAgent._seed_core_memories_from_markdown()

        run_dir = self._agent_paths.get("run_dir", base_dir)
        self._actions_log_path = self._agent_paths.get("shared", {}).get("action_log")
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
        """
        try:
            # Only create if there are invitees or it's a real meeting
            if not meeting.title:
                return

            # Parse meeting times
            start = datetime.fromisoformat(meeting.start_datetime_iso)
            end = datetime.fromisoformat(meeting.end_datetime_iso)

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
        result = await Runner.run(planner_agent, prompt, context=context, max_turns=15)
        plan = result.final_output

        # Record plan to memory
        record_plan_to_memory(agent, plan, now)

        # Store plan
        self._daily_plans[agent_id] = plan
        agent.set_daily_plan(plan.model_dump())

        # Store clothing choice for today (resets daily)
        if plan.clothing:
            agent.set_todays_clothing(plan.clothing.model_dump())
            # Add clothing as a perception for the day
            if agent.memory_stream:
                clothing_memory = f"I'm wearing {plan.clothing.description} today - {plan.clothing.warmth_level} warmth"
                agent.memory_stream.add_event(
                    description=clothing_memory,
                    subject=agent.scratch.get("first_name", agent_id),
                    predicate="is wearing",
                    obj=plan.clothing.description,
                    importance=2.0,  # Low importance - routine
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
        agent.update_daily_plan(updates, reason=event, now=now)
        print(f"[PLAN] {agent_id} plan updated due to: {event}")

    def plan_day(self, day: date, now: datetime) -> Dict[str, DailyPlan]:
        """
        Call daily planning for all agents using cognitive planning.

        Should be called once at the start of each simulated day.

        Args:
            day: The date to plan for
            now: Current simulation datetime

        Returns:
            Dict mapping agent_id to their DailyPlan
        """
        print(f"\n[LLM] Running cognitive daily planning for {len(self._agent_ids)} agents on {day.isoformat()}")

        self._daily_plans.clear()
        self._last_plan_date = day

        # Reset adapter state for new day
        self.adapter.reset_day()
        self.adapter.start_of_day_setup()

        # Run planning for all agents (could be parallelized with asyncio.gather)
        async def plan_all():
            results = []
            for agent_id in self._agent_ids:
                try:
                    plan = await self._plan_agent_day_async(agent_id, day, now)
                    results.append((agent_id, plan, None))
                except Exception as e:
                    results.append((agent_id, None, e))
            return results

        results = asyncio.run(plan_all())

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

    def should_make_decision(self, agent_id: str, now: datetime) -> bool:
        """
        Check if an agent should make a decision at this time.

        Based on decision_interval_minutes from config.
        """
        last_time = self._last_decision_time.get(agent_id)
        if last_time is None:
            return True

        elapsed = (now - last_time).total_seconds() / 60.0
        return elapsed >= self._decision_interval_minutes

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
                # Agent is arriving
                self._handle_arrival(agent_id, now)
            elif not is_working and was_present:
                # Agent is departing
                self._handle_departure(agent_id, now)

            # Make decisions for present agents
            if is_working and (force_all or self.should_make_decision(agent_id, now)):
                self._make_agent_decision(agent_id, now)

        # Resolve any pending votes
        thermostat_offset, window_fraction = self.adapter.resolve_votes()

        # Calculate total power
        equipment_power = self.adapter.get_total_equipment_power_w()
        lighting_power = self.adapter.get_total_lighting_power_w()

        return equipment_power, lighting_power, thermostat_offset, window_fraction

    def _handle_arrival(self, agent_id: str, now: datetime) -> None:
        """Handle an agent arriving at work."""
        print(f"[LLM] {agent_id} arriving at {now.strftime('%H:%M')}")

        # Mark as present
        self.adapter.set_occupant_present(agent_id, True)

        # Assign desk based on agent preferences and variety trait
        # NOTE: Equipment and lights are NOT auto-turned on.
        # The agent decides to turn them on via the cognitive loop.
        agent = self._agents.get(agent_id)
        if agent:
            # Read desk preferences from agent's scratch (loaded from agent_base_types)
            preferred_desk = agent.scratch.get("preferred_desk")
            workspace_variety = agent.scratch.get("workspace_variety", "low")
            likes_variety = agent.scratch.get("likes_desk_variety", False)

            available_desks = self.desks.get_available_desks()
            chosen_desk = None

            if preferred_desk and preferred_desk in available_desks:
                # Preferred desk is available - check variety preference
                if workspace_variety == "high" or likes_variety:
                    # 40% chance of trying different desk for variety-seeking agents
                    if random.random() < 0.4 and len(available_desks) > 1:
                        alternatives = [d for d in available_desks if d != preferred_desk]
                        chosen_desk = random.choice(alternatives)
                        print(f"  {agent_id} trying different desk for variety")
                    else:
                        chosen_desk = preferred_desk
                elif workspace_variety == "medium":
                    # 20% chance for medium variety preference
                    if random.random() < 0.2 and len(available_desks) > 1:
                        alternatives = [d for d in available_desks if d != preferred_desk]
                        chosen_desk = random.choice(alternatives)
                        print(f"  {agent_id} occasionally trying different desk")
                    else:
                        chosen_desk = preferred_desk
                else:
                    # Low variety - always use preferred desk
                    chosen_desk = preferred_desk
            elif available_desks:
                # Preferred desk not available - choose from what's available
                chosen_desk = random.choice(available_desks)
                print(f"  {agent_id}'s preferred desk unavailable, using {chosen_desk}")

            if chosen_desk:
                self.desks.assign_desk(agent_id, chosen_desk)
                print(f"  Assigned to {chosen_desk} (equipment OFF - agent will decide)")

        # Set flag for perceive() to create "equipment off" memory
        if agent:
            agent.set_just_arrived(True)

    def _handle_departure(self, agent_id: str, now: datetime) -> None:
        """Handle an agent departing from work."""
        print(f"[LLM] {agent_id} departing at {now.strftime('%H:%M')}")

        # Handle departure (turns off equipment, releases desk)
        self.adapter.set_occupant_present(agent_id, False)

    async def _make_agent_decision_async(
        self,
        agent_id: str,
        now: datetime,
    ) -> Optional[OccupantStepDecision]:
        """
        Execute cognitive loop for agent decision (async - SDK best practice).

        Perceive -> CheckInteraction -> Retrieve -> Reflect -> Act
        """
        agent = self._agents.get(agent_id)
        if not agent:
            print(f"[LLM] {agent_id}: Agent not found")
            return None

        # Decrement chat buffers at start of each decision cycle
        agent.decrement_all_chat_buffers()

        # 1. Get simulation state
        sim_state = self.adapter.get_state(agent_id, now)

        # 2. PERCEIVE: Convert state to memory events
        perceived_events = perceive(agent, sim_state, now)

        # 3. Accumulate importance for reflection trigger
        for event in perceived_events:
            agent.decrement_importance_trigger(event.importance)

        # 4. Get other occupants for consultation (general conversations disabled)
        other_occupants = sim_state.get("other_occupants_present", [])

        # 5. REFLECT: Generate insights if threshold reached
        await reflect(agent, now)

        # 5.5 Post-conversation reflections (Stanford-style)
        # Process any pending conversation reflections
        pending_reflections = agent.get_pending_conversation_reflections()
        for pending in pending_reflections:
            other_id = pending.get("other_agent_id")
            if other_id:
                await reflect_on_conversation(agent, other_id, now)
        agent.clear_pending_conversation_reflections()

        # 6. Get meeting context
        meeting_context = get_meeting_context(agent_id, self.calendar, now)
        pending_invitations = self.calendar.get_pending_invitations(agent_id)

        # Add focal point for meeting attendance only if meeting is imminent (within 10 min)
        focal_points = get_decision_focal_points(sim_state)
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
        )

        # 10. Call LLM via async Runner.run (SDK best practice)
        # max_turns=5 is sufficient for step decisions (minimal tool use)
        step_agent = self._step_agents[agent_id]
        context = SimContext(
            occupant_id=agent_id,
            now=now,
            simulation=self.adapter,
            calendar=self.calendar,
            configured_agent_ids=list(self._agents.keys()),
        )
        result = await Runner.run(step_agent, prompt, context=context, max_turns=5)
        decision = result.final_output

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

        # 11.1. Handle lunch/break actions (update agent status)
        for action in decision.actions:
            if action.action_type in ("go_to_lunch", "go_out_for_lunch"):
                self.adapter.set_agent_status(agent_id, "at_lunch", True)
                self.adapter.set_agent_status(agent_id, "at_desk", False)
                out_of_building = action.action_type == "go_out_for_lunch"
                self.adapter.set_agent_status(agent_id, "out_of_office", out_of_building)
                location = "outside" if out_of_building else "kitchen/cafeteria"
                print(f"[LLM] {agent_id} going to lunch ({location})")
            elif action.action_type == "return_from_lunch":
                self.adapter.set_agent_status(agent_id, "at_lunch", False)
                self.adapter.set_agent_status(agent_id, "at_desk", True)
                self.adapter.set_agent_status(agent_id, "out_of_office", False)
                print(f"[LLM] {agent_id} returning from lunch")
            elif action.action_type == "take_break":
                self.adapter.set_agent_status(agent_id, "on_break", True)
                self.adapter.set_agent_status(agent_id, "at_desk", False)
                print(f"[LLM] {agent_id} taking a break")
            elif action.action_type == "return_from_break":
                self.adapter.set_agent_status(agent_id, "on_break", False)
                self.adapter.set_agent_status(agent_id, "at_desk", True)
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
        self.adapter.apply_decision(decision)

        # 13. Record decision to memory
        record_decision_to_memory(agent, decision, now)

        # 14. Update tracking
        self._last_decision_time[agent_id] = now

        # 15. Log to file with memory context
        self._log_decision(agent_id, now, decision, retrieved)

        # 16. Log to shared action log
        self._log_action(agent_id, decision, now)

        # 17. Save agent state
        agent.save()

        return decision

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
                proposed_setpoint = action.parameters.get("setpoint_c")
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

        # Build proposed action description
        if shared_space_action.action_type == "thermostat_adjust":
            proposed_action = f"set thermostat to {proposed_setpoint}°C"
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
            record_agreement_to_memory(present_agents, outcome, now)

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

    def _make_agent_decision(self, agent_id: str, now: datetime) -> Optional[OccupantStepDecision]:
        """
        Sync wrapper for cognitive loop (for compatibility with main.py).

        Returns the decision or None if failed.
        """
        try:
            decision = asyncio.run(self._make_agent_decision_async(agent_id, now))

            # Log non-trivial decisions to console
            if decision:
                non_trivial = [a for a in decision.actions if a.action_type != "no_op"]
                if non_trivial:
                    actions_str = ", ".join(a.action_type for a in non_trivial)
                    print(f"[LLM] {agent_id}: {actions_str}")

            return decision

        except Exception as e:
            print(f"[LLM] {agent_id} decision failed: {e}")
            raise RuntimeError(f"Agent decision failed for {agent_id}: {e}")

    def _log_decision(
        self,
        agent_id: str,
        now: datetime,
        decision: OccupantStepDecision,
        retrieved_memories: Optional[Dict[str, List[MemoryNode]]] = None,
    ) -> None:
        """Log a decision to the agent's decision log file with memory context."""
        try:
            log_path = self._decision_log_paths.get(agent_id)
            if not log_path:
                return
            with open(log_path, 'a') as f:
                timestamp = now.strftime('%Y-%m-%d %H:%M')

                # Log retrieved memories (cognitive context)
                if retrieved_memories:
                    f.write(f"\n[{timestamp}] Retrieved memories:\n")
                    for focal_pt, memories in retrieved_memories.items():
                        f.write(f"  '{focal_pt}': {len(memories)} memories\n")
                        # Log first few memory descriptions for context
                        for mem in memories[:3]:
                            desc = mem.description[:80] + "..." if len(mem.description) > 80 else mem.description
                            f.write(f"    - {desc}\n")

                # Log actions
                for action in decision.actions:
                    params_str = str(action.parameters) if action.parameters else "{}"
                    f.write(f"{timestamp} | {action.action_type} | {params_str}\n")
                if decision.brief_rationale:
                    f.write(f"  Rationale: {decision.brief_rationale}\n")
        except IOError as e:
            print(f"Warning: Failed to log decision for {agent_id}: {e}")

    def _log_conversation(
        self,
        conversation: ConversationResult,
        now: datetime,
    ) -> None:
        """Log conversation to shared conversations.log file."""
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

    def _log_action(
        self,
        agent_id: str,
        decision: OccupantStepDecision,
        now: datetime,
    ) -> None:
        """Log action to shared actions.log file."""
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

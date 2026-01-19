"""
cognitive_modules.py

Cognitive modules for generative agents.
Implements the cognitive loop: Perceive -> Retrieve -> Plan -> Reflect -> Act

Based on Stanford Generative Agents architecture, adapted for building simulation.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from openai import OpenAI

from bsm.agents.memory.stream import MemoryNode

if TYPE_CHECKING:
    from bsm.agents.generative_agent import GenerativeAgent
    from bsm.agents.skeleton import CalendarStore

# Module-level OpenAI client for reflections (initialized lazily)
_reflection_client: Optional[OpenAI] = None


def _get_reflection_client() -> OpenAI:
    """Get or create the OpenAI client for reflections."""
    global _reflection_client
    if _reflection_client is None:
        _reflection_client = OpenAI()
    return _reflection_client


def perceive(
    agent: "GenerativeAgent",
    sim_state: Dict[str, Any],
    now: datetime,
) -> List[MemoryNode]:
    """
    Convert simulation state into perceived events.

    This is the first step in the cognitive loop. The agent observes
    the current state and creates memory events for notable observations.

    Args:
        agent: The generative agent
        sim_state: Current simulation state dict
        now: Current simulation datetime

    Returns:
        List of newly created MemoryNode events
    """
    if not agent.memory_stream:
        return []

    perceived_events: List[MemoryNode] = []

    # Perceive thermal comfort
    temp = sim_state.get("indoor_temp_c", 21.0)
    comfort_temp = agent.scratch.get("thermal_comfort_c", 21.0)
    temp_diff = abs(temp - comfort_temp)

    if temp_diff > 2.0:
        # Significant discomfort
        feeling = "too warm" if temp > comfort_temp else "too cold"
        event = agent.memory_stream.add_event(
            description=f"The room temperature is {temp:.1f}C, which feels {feeling} (I prefer {comfort_temp}C)",
            subject="room",
            predicate="feels",
            obj=feeling,
            now=now,
            importance=min(8.0, 5.0 + temp_diff),  # More discomfort = more important
        )
        perceived_events.append(event)
    elif temp_diff > 1.0:
        # Mild discomfort
        feeling = "a bit warm" if temp > comfort_temp else "a bit cool"
        event = agent.memory_stream.add_event(
            description=f"The room temperature is {temp:.1f}C, which is {feeling}",
            subject="room",
            predicate="feels",
            obj=feeling,
            now=now,
            importance=4.0,
        )
        perceived_events.append(event)

    # Perceive other occupants present
    other_occupants = sim_state.get("other_occupants_present", [])
    for other_id in other_occupants:
        if not agent.recently_perceived(other_id, within_minutes=30):
            # Haven't noticed them recently
            event = agent.memory_stream.add_event(
                description=f"{other_id} is in the office",
                subject=other_id,
                predicate="is present in",
                obj="office",
                now=now,
                importance=4.0,
            )
            perceived_events.append(event)

    # Perceive equipment state if just arrived
    if agent.has_just_arrived():
        equipment_status = sim_state.get("equipment_status", {})
        if equipment_status.get("all_off", True):
            event = agent.memory_stream.add_event(
                description="My desk equipment is all off - I need to turn it on to start working",
                subject="I",
                predicate="notice",
                obj="equipment is off",
                now=now,
                importance=6.0,
            )
            perceived_events.append(event)

        # Perceive lighting conditions
        lighting = sim_state.get("lighting_conditions", {})
        natural_light = lighting.get("natural_light_level", "moderate")
        desk_light_on = lighting.get("desk_light_on", False)

        if natural_light in ["dim", "dark"] and not desk_light_on:
            event = agent.memory_stream.add_event(
                description=f"It's {natural_light} in here and my desk light is off",
                subject="workspace",
                predicate="has",
                obj=f"{natural_light} lighting",
                now=now,
                importance=5.0,
            )
            perceived_events.append(event)

        agent.clear_just_arrived()

    # Perceive weather
    weather_desc = sim_state.get("weather_description", "")
    is_sunny = sim_state.get("is_sunny", False)
    outdoor_temp = sim_state.get("outdoor_temp_c", 10.0)

    if weather_desc and not agent.recently_perceived("weather", within_minutes=60):
        event = agent.memory_stream.add_event(
            description=f"The weather outside is {weather_desc}, {outdoor_temp:.1f}C",
            subject="weather",
            predicate="is",
            obj=weather_desc,
            now=now,
            importance=3.0,
        )
        perceived_events.append(event)

    return perceived_events


def retrieve(
    agent: "GenerativeAgent",
    focal_points: List[str],
    now: datetime,
    n_count: int = 20,
    core_memory_count: int = 5,
) -> Dict[str, List[MemoryNode]]:
    """
    Retrieve memories relevant to focal points.

    Focal points are questions or topics the agent is thinking about.
    Returns relevant memories for each focal point.

    This function now retrieves from BOTH:
    - Core memories (permanent, from CoreMemoryStore)
    - Memory stream (decaying events/thoughts/chats)

    Args:
        agent: The generative agent
        focal_points: List of questions/topics to retrieve memories for
        now: Current datetime
        n_count: Number of memories to retrieve per focal point from memory stream
        core_memory_count: Number of core memories to include per focal point

    Returns:
        Dict mapping focal_point -> list of relevant MemoryNodes
        (includes both core and episodic memories)
    """
    if not agent.memory_stream:
        return {}

    recency_w, relevance_w, importance_w = agent.get_retrieval_weights()
    recency_decay = agent.get_recency_decay()

    retrieved: Dict[str, List[MemoryNode]] = {}

    for focal_pt in focal_points:
        nodes = agent.memory_stream.retrieve(
            focal_point=focal_pt,
            now=now,
            n_count=n_count,
            recency_weight=recency_w,
            relevance_weight=relevance_w,
            importance_weight=importance_w,
            recency_decay=recency_decay,
            # Pass the separate core memory store for union
            core_memory_store=agent.core_memory_store,
            core_memory_count=core_memory_count,
        )
        retrieved[focal_pt] = nodes

    return retrieved


def build_decision_context(
    agent: "GenerativeAgent",
    sim_state: Dict[str, Any],
    retrieved_memories: Dict[str, List[MemoryNode]],
    now: datetime,
) -> Dict[str, Any]:
    """
    Build context dict for LLM decision making.

    Combines agent identity, current state, and retrieved memories
    into a format suitable for the decision-making prompt.

    Args:
        agent: The generative agent
        sim_state: Current simulation state
        retrieved_memories: Retrieved memories from retrieve()
        now: Current datetime

    Returns:
        Context dict for prompt construction
    """
    # Format memories into text
    memory_text_parts = []
    for focal_pt, memories in retrieved_memories.items():
        if memories:
            mem_str = agent.memory_stream.format_memories_for_prompt(memories)
            memory_text_parts.append(f"Regarding '{focal_pt}':\n{mem_str}")

    memories_text = "\n\n".join(memory_text_parts) if memory_text_parts else "No specific memories retrieved."

    return {
        "identity": agent.get_identity_stable_set(),
        "schedule": agent.get_schedule_info(),
        "current_state": sim_state,
        "relevant_memories": memories_text,
        "current_plan": agent.get_daily_plan(),
        "datetime": now.isoformat(),
        "day_of_week": now.strftime("%A"),
    }


async def reflect(
    agent: "GenerativeAgent",
    now: datetime,
    use_llm: bool = True,
    reflection_model: str = "gpt-4o-mini",
) -> List[MemoryNode]:
    """
    Generate reflections (thoughts) based on recent experiences.

    Stanford-style LLM-generated reflections that derive high-level insights
    from recent observations. Triggered when accumulated importance of
    perceived events exceeds the reflection threshold.

    Args:
        agent: The generative agent
        now: Current datetime
        use_llm: Whether to use LLM for reflection generation (default True)
        reflection_model: Model to use for reflection generation

    Returns:
        List of newly created thought MemoryNodes
    """
    if not agent.should_reflect():
        return []

    if not agent.memory_stream:
        return []

    # Reset trigger
    agent.reset_importance_trigger()

    # Get recent high-importance events
    recent_events = agent.memory_stream.get_high_importance_events(
        now=now,
        hours=4.0,
        min_importance=5.0,
    )

    if len(recent_events) < 3:
        # Need enough observations to reflect on
        return []

    reflections: List[MemoryNode] = []

    if use_llm:
        # LLM-generated reflection (Stanford-style)
        try:
            insight = await _generate_llm_reflection(
                agent_name=agent.name,
                recent_events=recent_events[:10],  # Limit to 10 most recent
                model=reflection_model,
            )

            if insight:
                thought = agent.memory_stream.add_thought(
                    description=insight,
                    evidence_ids=[e.node_id for e in recent_events[:5]],
                    now=now,
                    importance=7.0,
                )
                reflections.append(thought)
                print(f"[Reflect] {agent.name}: {insight}")

        except Exception as e:
            print(f"[Reflect] LLM reflection failed for {agent.name}: {e}")
            # Fall back to template-based reflection
            reflections.extend(
                _generate_template_reflections(agent, recent_events, now)
            )
    else:
        # Template-based fallback
        reflections.extend(
            _generate_template_reflections(agent, recent_events, now)
        )

    return reflections


async def _generate_llm_reflection(
    agent_name: str,
    recent_events: List[MemoryNode],
    model: str = "gpt-4o-mini",
) -> Optional[str]:
    """
    Generate a reflection using LLM (Stanford-style).

    Args:
        agent_name: Name of the agent
        recent_events: List of recent high-importance events
        model: OpenAI model to use

    Returns:
        Generated insight string, or None if generation failed
    """
    # Format recent observations as numbered statements
    statements = "\n".join([
        f"{i+1}. {event.description}"
        for i, event in enumerate(recent_events[:10])
    ])

    prompt = f"""You are {agent_name}, an office worker.

Recent observations and experiences:
{statements}

Based on these observations, what is ONE high-level insight about your:
- Comfort preferences (temperature, lighting)
- Workspace habits
- Working patterns
- Interactions with colleagues

Write the insight in first person (e.g., "I notice that...", "I prefer...", "I tend to...").
Focus on patterns that would help you make better decisions in the future.
Keep it concise (1-2 sentences).

Output only the insight, nothing else."""

    client = _get_reflection_client()

    # Use synchronous call wrapped in async context
    # (OpenAI client supports async via AsyncOpenAI, but for simplicity we use sync)
    import asyncio
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100,
        )
    )

    insight = response.choices[0].message.content.strip()
    return insight if insight else None


def _generate_template_reflections(
    agent: "GenerativeAgent",
    recent_events: List[MemoryNode],
    now: datetime,
) -> List[MemoryNode]:
    """
    Generate template-based reflections as fallback.

    Used when LLM reflection fails or is disabled.
    """
    reflections: List[MemoryNode] = []

    # Group events by subject/theme
    themes: Dict[str, List[MemoryNode]] = {}
    for event in recent_events:
        theme = event.subject.lower()
        if theme not in themes:
            themes[theme] = []
        themes[theme].append(event)

    # Generate simple reflections for themes with multiple events
    for theme, events in themes.items():
        if len(events) >= 2:
            if theme == "room":
                reflection_text = "I've been noticing the room temperature quite a bit recently."
            elif theme == "i":
                reflection_text = "I've been quite active recently with various tasks."
            else:
                reflection_text = f"I've been thinking about {theme} recently."

            thought = agent.memory_stream.add_thought(
                description=reflection_text,
                evidence_ids=[e.node_id for e in events[:3]],
                now=now,
                importance=7.0,
            )
            reflections.append(thought)

    return reflections


def get_decision_focal_points(
    sim_state: Dict[str, Any],
) -> List[str]:
    """
    Generate focal points for memory retrieval based on current state.

    Args:
        sim_state: Current simulation state

    Returns:
        List of focal point strings
    """
    focal_points = [
        "what should I do right now",
        "my preferences and habits",
    ]

    # Add context-specific focal points
    temp = sim_state.get("indoor_temp_c", 21.0)
    if temp < 18 or temp > 25:
        focal_points.append("my thermal comfort preferences")

    lighting = sim_state.get("lighting_conditions", {})
    if lighting.get("natural_light_level") in ["dim", "dark"]:
        focal_points.append("lighting and workspace setup")

    equipment = sim_state.get("equipment_status", {})
    if equipment.get("all_off", False):
        focal_points.append("my equipment habits when arriving")

    other_occupants = sim_state.get("other_occupants_present", [])
    if other_occupants:
        focal_points.append(f"my relationship with {other_occupants[0]}")
        # Add focal point for agreements when others are present
        focal_points.append("agreements I made with colleagues about temperature or comfort")

    return focal_points


def get_planning_focal_points() -> List[str]:
    """
    Generate focal points for daily planning memory retrieval.

    Returns:
        List of focal point strings for planning context
    """
    return [
        "my typical work schedule and habits",
        "recent events that might affect today",
        "meetings or commitments I have",
        "my preferences for arrival and departure times",
    ]


def get_meeting_context(
    agent_id: str,
    calendar: "CalendarStore",
    now: datetime,
) -> Dict[str, Any]:
    """
    Get meeting context for step decision prompts.

    Queries the calendar to determine:
    - If a meeting is currently happening
    - When the next meeting is
    - All meetings for today
    - Whether the agent should attend a meeting NOW

    Args:
        agent_id: The agent's ID
        calendar: CalendarStore instance
        now: Current simulation datetime

    Returns:
        Dict with meeting context:
        {
            "current_meeting": {...} or None,
            "next_meeting": {...} or None,
            "minutes_to_next_meeting": int or None,
            "meetings_today": [...],
            "should_attend_now": bool,
            "meeting_alert": str or None,
        }
    """
    # Get today's date range
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = now.replace(hour=23, minute=59, second=59, microsecond=999999)

    # Query all meetings for today from shared calendar
    meetings_today = calendar.list_events("shared", day_start, day_end)

    # Filter to meetings the agent is involved in (created by or RSVP'd)
    agent_meetings = []
    for meeting in meetings_today:
        # Include if agent created it
        if meeting.get("created_by") == agent_id:
            agent_meetings.append(meeting)
            continue

        # Check RSVPs for this event
        rsvps = calendar.get_rsvps_for_event(meeting.get("event_id", ""))
        for rsvp in rsvps:
            if rsvp.get("agent_id") == agent_id and rsvp.get("status") == "yes":
                agent_meetings.append(meeting)
                break

    # Determine current meeting and next meeting
    current_meeting = None
    next_meeting = None
    minutes_to_next = None

    for meeting in agent_meetings:
        start_iso = meeting.get("start_datetime_iso", "")
        end_iso = meeting.get("end_datetime_iso", "")

        try:
            start_dt = datetime.fromisoformat(start_iso)
            end_dt = datetime.fromisoformat(end_iso)

            # Ensure timezone aware
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)

            now_utc = now if now.tzinfo else now.replace(tzinfo=timezone.utc)

            # Check if meeting is currently happening
            if start_dt <= now_utc <= end_dt:
                current_meeting = meeting
                current_meeting["_minutes_in"] = int((now_utc - start_dt).total_seconds() / 60)

            # Check if meeting is upcoming
            elif start_dt > now_utc:
                mins_until = int((start_dt - now_utc).total_seconds() / 60)
                if next_meeting is None or mins_until < minutes_to_next:
                    next_meeting = meeting
                    minutes_to_next = mins_until

        except (ValueError, TypeError):
            continue

    # Determine if agent should attend now
    should_attend_now = False
    meeting_alert = None

    if current_meeting:
        should_attend_now = True
        mins_in = current_meeting.get("_minutes_in", 0)
        meeting_alert = (
            f"MEETING IN PROGRESS: '{current_meeting.get('title', 'Meeting')}' "
            f"started {mins_in} minutes ago. You should be attending this meeting!"
        )
    elif next_meeting and minutes_to_next is not None and minutes_to_next <= 5:
        should_attend_now = True
        meeting_alert = (
            f"MEETING STARTING SOON: '{next_meeting.get('title', 'Meeting')}' "
            f"starts in {minutes_to_next} minutes. Head to the meeting room now!"
        )

    return {
        "current_meeting": current_meeting,
        "next_meeting": next_meeting,
        "minutes_to_next_meeting": minutes_to_next,
        "meetings_today": agent_meetings,
        "should_attend_now": should_attend_now,
        "meeting_alert": meeting_alert,
    }


def format_colleague_context(
    agent: "GenerativeAgent",
    present_agent_ids: List[str],
) -> str:
    """
    Format information about colleagues present for the step prompt.

    Uses relationship_models from the agent's scratch to provide
    context about familiarity and sentiment with each colleague.

    Args:
        agent: The generative agent
        present_agent_ids: List of other agent IDs currently present

    Returns:
        Formatted string describing colleagues present
    """
    if not present_agent_ids:
        return "No colleagues currently present."

    lines = []
    for other_id in present_agent_ids:
        rel_info = agent.get_relationship_info(other_id)
        if rel_info:
            familiarity = rel_info.get("familiarity", 0.5)
            sentiment = rel_info.get("sentiment", 0.5)

            # Convert to descriptive terms
            fam_desc = "high" if familiarity > 0.7 else ("moderate" if familiarity > 0.4 else "low")
            sent_desc = "positive" if sentiment > 0.6 else ("neutral" if sentiment > 0.4 else "negative")

            lines.append(f"- {other_id} (familiarity: {fam_desc}, sentiment: {sent_desc})")
        else:
            lines.append(f"- {other_id} (no relationship data)")

    return "\n".join(lines)


def format_meetings_for_prompt(meetings: List[Dict[str, Any]], agent_id: str) -> str:
    """
    Format a list of meetings for inclusion in prompts.

    Args:
        meetings: List of meeting dicts from calendar
        agent_id: The agent's ID (to indicate if they're the organizer)

    Returns:
        Formatted string listing meetings
    """
    if not meetings:
        return "No meetings scheduled today."

    lines = []
    for meeting in meetings:
        title = meeting.get("title", "Untitled meeting")
        start = meeting.get("start_datetime_iso", "")
        end = meeting.get("end_datetime_iso", "")
        location = meeting.get("location", "meeting_room")
        created_by = meeting.get("created_by", "")

        # Extract time portion
        try:
            start_dt = datetime.fromisoformat(start)
            end_dt = datetime.fromisoformat(end)
            time_str = f"{start_dt.strftime('%H:%M')}-{end_dt.strftime('%H:%M')}"
        except (ValueError, TypeError):
            time_str = "time unknown"

        role = "(organizer)" if created_by == agent_id else "(invited)"
        lines.append(f"- {time_str}: {title} at {location} {role}")

    return "\n".join(lines)


def format_pending_invitations_for_prompt(invitations: List[Dict[str, Any]]) -> str:
    """
    Format pending invitations for inclusion in prompts.

    Args:
        invitations: List of pending invitation dicts

    Returns:
        Formatted string listing invitations
    """
    if not invitations:
        return "No pending invitations."

    lines = []
    for inv in invitations:
        title = inv.get("event_title", "Meeting")
        inviter = inv.get("inviter_id", "Unknown")
        start = inv.get("event_start_iso", "")

        try:
            start_dt = datetime.fromisoformat(start)
            time_str = start_dt.strftime("%H:%M")
        except (ValueError, TypeError):
            time_str = "time unknown"

        lines.append(f"- '{title}' from {inviter} at {time_str} - RESPOND (accept/decline)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interaction Triggers (Stanford-style)
# ---------------------------------------------------------------------------

def should_react(
    agent: "GenerativeAgent",
    perceived_agent_ids: List[str],
    sim_state: Dict[str, Any],
    now: datetime,
    agents_dict: Dict[str, "GenerativeAgent"],
) -> Optional[str]:
    """
    Determine if agent should react to perceived agents (Stanford-style).

    This function decides if an agent should initiate a conversation
    with another agent based on various factors.

    Args:
        agent: The agent making the decision
        perceived_agent_ids: List of agent IDs currently perceived (present)
        sim_state: Current simulation state
        now: Current datetime
        agents_dict: Dict mapping agent_id to GenerativeAgent instances

    Returns:
        - "chat with {agent_id}" if should initiate conversation
        - None if no reaction needed
    """
    # Skip if currently in a meeting (don't interrupt with casual chat)
    if sim_state.get("in_meeting_room", False):
        return None

    # Skip if agent is currently chatting with someone
    if agent.get_chatting_with():
        return None

    # Check each perceived agent for potential conversation
    for other_id in perceived_agent_ids:
        other_agent = agents_dict.get(other_id)
        if not other_agent:
            continue

        # Skip if other agent is in a meeting
        # (we'd need their state, so approximate with chatting check)
        if other_agent.get_chatting_with():
            continue

        # Check chat buffer (cooldown period)
        if not agent.can_chat_with(other_id):
            continue

        # Check if we should initiate conversation
        if _should_initiate_chat(agent, other_agent, sim_state, now):
            return f"chat with {other_id}"

    return None


def _should_initiate_chat(
    init_agent: "GenerativeAgent",
    target_agent: "GenerativeAgent",
    sim_state: Dict[str, Any],
    now: datetime,
) -> bool:
    """
    Decide if initiator should start a conversation with target.

    Factors considered:
    - Time of day (not too early or late)
    - Relationship (familiarity and sentiment)
    - Recent interaction history
    - Probability based on relationship strength

    Args:
        init_agent: Agent considering starting conversation
        target_agent: Potential conversation partner
        sim_state: Current simulation state
        now: Current datetime

    Returns:
        True if conversation should be initiated
    """
    import random

    # Don't chat too early (before 8am) or too late (after 6pm)
    hour = now.hour
    if hour < 8 or hour >= 18:
        return False

    # Get relationship info
    rel_info = init_agent.get_relationship_info(target_agent.agent_id)

    # Base probability of chatting
    base_prob = 0.15  # 15% base chance when both present

    if rel_info:
        familiarity = rel_info.get("familiarity", 0.5)
        sentiment = rel_info.get("sentiment", 0.5)

        # Higher familiarity and positive sentiment increase probability
        # familiarity 0.7 + sentiment 0.7 = 0.4 bonus
        relationship_bonus = (familiarity - 0.3) * 0.3 + (sentiment - 0.3) * 0.3
        chat_prob = base_prob + max(0, relationship_bonus)
    else:
        # Unknown relationship - lower chance
        chat_prob = base_prob * 0.5

    # Cap probability
    chat_prob = min(0.5, chat_prob)

    # Random decision
    return random.random() < chat_prob


def generate_relationship_summary(
    agent: "GenerativeAgent",
    other_agent_id: str,
) -> str:
    """
    Generate a text summary of the relationship for conversation context.

    Args:
        agent: The agent whose perspective we're summarizing from
        other_agent_id: The other agent in the relationship

    Returns:
        Text summary of the relationship
    """
    rel_info = agent.get_relationship_info(other_agent_id)

    if not rel_info:
        return f"{agent.first_name} doesn't know {other_agent_id} very well yet."

    familiarity = rel_info.get("familiarity", 0.5)
    sentiment = rel_info.get("sentiment", 0.5)

    # Describe familiarity
    if familiarity > 0.7:
        fam_desc = f"{agent.first_name} knows {other_agent_id} quite well"
    elif familiarity > 0.4:
        fam_desc = f"{agent.first_name} is somewhat familiar with {other_agent_id}"
    else:
        fam_desc = f"{agent.first_name} doesn't know {other_agent_id} very well"

    # Describe sentiment
    if sentiment > 0.7:
        sent_desc = "and has positive feelings toward them"
    elif sentiment > 0.5:
        sent_desc = "and generally gets along with them"
    elif sentiment > 0.3:
        sent_desc = "and has neutral feelings toward them"
    else:
        sent_desc = "and has some reservations about them"

    return f"{fam_desc} {sent_desc}."


def format_step_prompt(
    agent: "GenerativeAgent",
    sim_state: Dict[str, Any],
    retrieved_memories: Dict[str, List[MemoryNode]],
    now: datetime,
    meeting_context: Optional[Dict[str, Any]] = None,
    pending_invitations: Optional[List[Dict[str, Any]]] = None,
    colleague_context: Optional[str] = None,
) -> str:
    """
    Format the prompt for step decision making.

    Args:
        agent: The generative agent
        sim_state: Current simulation state
        retrieved_memories: Retrieved memories
        now: Current datetime
        meeting_context: Meeting context from get_meeting_context()
        pending_invitations: Pending meeting invitations
        colleague_context: Formatted colleague context string

    Returns:
        Formatted prompt string
    """
    context = build_decision_context(agent, sim_state, retrieved_memories, now)

    # Build meeting status section
    meeting_status_section = ""
    if meeting_context:
        meetings_today = meeting_context.get("meetings_today", [])
        meeting_alert = meeting_context.get("meeting_alert")
        next_meeting = meeting_context.get("next_meeting")
        minutes_to_next = meeting_context.get("minutes_to_next_meeting")

        meeting_lines = []

        # Today's meetings
        if meetings_today:
            meeting_lines.append("Your meetings today:")
            meeting_lines.append(format_meetings_for_prompt(meetings_today, agent.agent_id))
        else:
            meeting_lines.append("No meetings scheduled today.")

        # Alert for current or imminent meeting
        if meeting_alert:
            meeting_lines.append("")
            meeting_lines.append(f"*** {meeting_alert} ***")

        # Next meeting info
        elif next_meeting and minutes_to_next is not None:
            meeting_lines.append("")
            meeting_lines.append(
                f"Next meeting: '{next_meeting.get('title', 'Meeting')}' "
                f"in {minutes_to_next} minutes"
            )

        meeting_status_section = "\n".join(meeting_lines)
    else:
        meeting_status_section = "No meeting information available."

    # Build pending invitations section
    invitations_section = ""
    if pending_invitations:
        invitations_section = format_pending_invitations_for_prompt(pending_invitations)
    else:
        invitations_section = "No pending invitations."

    # Build colleague section
    if colleague_context is None:
        other_occupants = sim_state.get("other_occupants_present", [])
        colleague_context = format_colleague_context(agent, other_occupants)

    # Extract recent agreements from retrieved memories
    agreements_section = _extract_agreements_from_memories(retrieved_memories, now)

    prompt = f"""
=== WHO YOU ARE ===
{context['identity']}

=== YOUR SCHEDULE ===
{context['schedule']}

=== MEETING STATUS ===
{meeting_status_section}

=== PENDING INVITATIONS ===
{invitations_section}

=== COLLEAGUES PRESENT ===
{colleague_context}

=== RECENT AGREEMENTS ===
{agreements_section}

=== CURRENT STATE ===
DateTime: {context['datetime']} ({context['day_of_week']})
Indoor temperature: {sim_state.get('indoor_temp_c', 'N/A')}C
Outdoor temperature: {sim_state.get('outdoor_temp_c', 'N/A')}C
Weather: {sim_state.get('weather_description', 'N/A')}
Your desk: {sim_state.get('current_desk', 'N/A')}
Lighting: {sim_state.get('lighting_conditions', {})}
Equipment status: {sim_state.get('equipment_status', {})}
Window state: {sim_state.get('window_open_fraction', 0)}
Thermostat: {sim_state.get('thermostat_setpoint_c', 'N/A')}C

=== RELEVANT MEMORIES ===
{context['relevant_memories']}

Based on who you are, your memories, the meeting status, and the current state, decide what actions to take.
IMPORTANT: If a meeting is in progress or starting soon, use 'attend_meeting' action to join it.
If you have pending invitations, use 'respond_to_invitation' to accept or decline.
IMPORTANT: Respect prior agreements with colleagues. Do not take actions that contradict recent agreements unless circumstances have changed significantly.
""".strip()

    return prompt


def _extract_agreements_from_memories(
    retrieved_memories: Dict[str, List[MemoryNode]],
    now: datetime,
) -> str:
    """
    Extract recent agreements from retrieved memories.

    Args:
        retrieved_memories: Retrieved memories by focal point
        now: Current datetime

    Returns:
        Formatted string of recent agreements
    """
    agreements = []

    # Look for agreement-related memories
    for focal_pt, memories in retrieved_memories.items():
        if "agreement" in focal_pt.lower():
            for mem in memories:
                # Check if this is an agreement (contains "Agreed" or "agreed")
                if "agreed" in mem.description.lower():
                    # Check if it's recent (within last 2 hours)
                    if hasattr(mem, 'created') and mem.created is not None:
                        try:
                            # mem.created may be a float timestamp or datetime
                            if isinstance(mem.created, (int, float)):
                                # Convert timestamp to datetime
                                mem_time = datetime.fromtimestamp(mem.created, tz=timezone.utc)
                                # Make now timezone-aware if it isn't
                                now_aware = now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now
                                time_diff = (now_aware - mem_time).total_seconds() / 3600
                            else:
                                # Assume it's already a datetime
                                time_diff = (now - mem.created).total_seconds() / 3600

                            if time_diff <= 2:
                                agreements.append(f"- {mem.description}")
                        except (TypeError, ValueError):
                            # If any error, include it anyway
                            agreements.append(f"- {mem.description}")
                    else:
                        # If no timestamp, include it anyway as it was retrieved as relevant
                        agreements.append(f"- {mem.description}")

    if not agreements:
        return "No recent agreements with colleagues."

    return "\n".join(agreements)


def format_planning_prompt(
    agent: "GenerativeAgent",
    day: str,
    day_of_week: str,
    retrieved_memories: Dict[str, List[MemoryNode]],
    calendar_events: List[Dict[str, Any]],
    pending_invitations: List[Dict[str, Any]],
    my_meetings: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Format the prompt for daily planning.

    Args:
        agent: The generative agent
        day: ISO date string
        day_of_week: Day name (Monday, Tuesday, etc.)
        retrieved_memories: Retrieved memories
        calendar_events: Existing calendar events for today
        pending_invitations: Pending meeting invitations
        my_meetings: Meetings agent has already created or accepted

    Returns:
        Formatted prompt string
    """
    # Format memories
    memory_text_parts = []
    for focal_pt, memories in retrieved_memories.items():
        if memories and agent.memory_stream:
            mem_str = agent.memory_stream.format_memories_for_prompt(memories)
            memory_text_parts.append(f"Regarding '{focal_pt}':\n{mem_str}")

    memories_text = "\n\n".join(memory_text_parts) if memory_text_parts else "No specific memories."

    # Format my confirmed meetings (created or accepted)
    if my_meetings:
        my_meetings_text = "\n".join([
            f"- {m.get('title', 'Meeting')} ({m.get('start_datetime_iso', 'N/A')}) - role: {m.get('role', 'attendee')}"
            for m in my_meetings
        ])
    else:
        my_meetings_text = "No confirmed meetings yet."

    # Format all calendar events (for visibility)
    if calendar_events:
        calendar_text = "\n".join([
            f"- {e.get('title', 'Event')} ({e.get('start_datetime_iso', 'N/A')} - {e.get('end_datetime_iso', 'N/A')})"
            for e in calendar_events
        ])
    else:
        calendar_text = "No scheduled events."

    # Format invitations
    if pending_invitations:
        invitations_text = "\n".join([
            f"- {inv.get('event_title', 'Meeting')} from {inv.get('inviter_id', 'Unknown')} "
            f"({inv.get('event_start_iso', 'N/A')})"
            for inv in pending_invitations
        ])
    else:
        invitations_text = "No pending invitations."

    prompt = f"""
=== WHO YOU ARE ===
{agent.get_identity_stable_set()}

=== YOUR TYPICAL SCHEDULE ===
{agent.get_schedule_info()}

=== TODAY ===
Date: {day} ({day_of_week})

=== RELEVANT MEMORIES ===
{memories_text}

=== YOUR CONFIRMED MEETINGS ===
{my_meetings_text}
NOTE: Do NOT respond to invitations for meetings already listed above.
Do NOT create new meetings that duplicate those listed above.

=== ALL CALENDAR EVENTS (shared) ===
{calendar_text}

=== PENDING INVITATIONS (need response) ===
{invitations_text}

Based on who you are and your memories, plan your day.
Consider your typical schedule as a guideline, but you may adjust based on circumstances.
""".strip()

    return prompt


def record_decision_to_memory(
    agent: "GenerativeAgent",
    decision: Any,  # OccupantStepDecision
    now: datetime,
) -> None:
    """
    Record a decision to the agent's memory stream.

    Args:
        agent: The generative agent
        decision: The decision that was made
        now: Current datetime
    """
    if not agent.memory_stream:
        return

    # Extract action descriptions
    actions = getattr(decision, 'actions', [])
    non_trivial = [a for a in actions if getattr(a, 'action_type', '') != 'no_op']

    if not non_trivial:
        return

    action_desc = ", ".join([getattr(a, 'action_type', 'unknown') for a in non_trivial])
    rationale = getattr(decision, 'brief_rationale', '')

    description = f"I decided to: {action_desc}"
    if rationale:
        description += f". Reason: {rationale}"

    agent.memory_stream.add_event(
        description=description,
        subject="I",
        predicate="decided",
        obj=action_desc,
        now=now,
        importance=5.0,
    )


def record_plan_to_memory(
    agent: "GenerativeAgent",
    plan: Any,  # DailyPlan
    now: datetime,
) -> None:
    """
    Record a daily plan to the agent's memory stream.

    Args:
        agent: The generative agent
        plan: The daily plan that was created
        now: Current datetime
    """
    if not agent.memory_stream:
        return

    arrival = getattr(plan, 'actual_arrival_time', 'N/A')
    departure = getattr(plan, 'actual_departure_time', 'N/A')
    meetings = getattr(plan, 'meetings', [])

    description = f"I planned my day: arrive at {arrival}, depart at {departure}"
    if meetings:
        description += f", with {len(meetings)} meeting(s)"

    agent.memory_stream.add_event(
        description=description,
        subject="I",
        predicate="planned",
        obj="my day",
        now=now,
        importance=4.0,
    )

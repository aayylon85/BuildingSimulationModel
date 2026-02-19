"""
Prompt formatting module for generative agents.

Implements prompt construction for:
- Step decision prompts (the main decision-making prompt)
- Daily planning prompts
- Meeting context formatting
- Various section formatters
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from bsm.agents.cognition.retrieval import retrieve, deduplicate_retrieved_memories
from bsm.agents.cognition.social import format_colleague_context
from bsm.agents.memory.stream import MemoryNode
from bsm.agents.skeleton import LOCATION_TRANSITIONS

if TYPE_CHECKING:
    from bsm.agents.generative_agent import GenerativeAgent
    from bsm.agents.skeleton import CalendarStore
    from bsm.agents.cognition.checkpoint_state import CheckpointState


# ---------------------------------------------------------------------------
# Focal Point Generation
# ---------------------------------------------------------------------------

def get_decision_focal_points(
    sim_state: Dict[str, Any],
    checkpoint_reason: Optional[str] = None,
) -> List[str]:
    """
    Generate focal points for memory retrieval based on current state and checkpoint reason.

    Args:
        sim_state: Current simulation state
        checkpoint_reason: Why this decision checkpoint was triggered (e.g., 'hourly', 'meeting_start')

    Returns:
        List of focal point strings
    """
    focal_points = [
        "what should I do right now",
        "my preferences and habits",
    ]

    # Add checkpoint-specific focal points
    if checkpoint_reason:
        checkpoint_base = checkpoint_reason.split(":")[0]  # Handle "meeting_start:Meeting Name" format
        checkpoint_focal_points = {
            "hourly": ["my typical hourly routine"],
            "first_decision": ["what I usually do when I first arrive at work", "my morning startup routine"],
            "meeting_prep": ["preparing for meetings", "getting ready for meetings", "what I need for the next meeting"],
            "meeting_start": ["how I prepare for meetings", "my meeting habits"],
            "meeting_end": ["what I do after meetings end", "refocusing after meetings"],
            "lunch_time": ["my lunch preferences and habits", "where I like to eat lunch"],
            "return_from_lunch": ["when I return from lunch", "my post-lunch routine"],
            "take_break": ["my break habits", "when I take coffee or tea breaks", "staying energized"],
            "return_from_break": ["returning to work after breaks"],
            # M.1: Enhanced commitment focal points for relationship-aware memory retrieval
            "commitment": [
                "promises I made to colleagues",
                "someone counting on me",
                "times I kept my word",
                "times I let someone down",
                "how it feels when someone doesn't show up",
            ],
            # M.3: Focal points for when waiting for a colleague
            "commitment_waiting": [
                "when someone didn't show up to meet me",
                "waiting for colleagues",
                "being stood up",
                "understanding when people are late",
            ],
            # F.2: Focal points for upcoming commitment
            "commitment_prep": [
                "upcoming plans with colleagues",
                "getting ready for social activities",
                "wrapping up work before breaks",
            ],
            "departure_prep": ["wrapping up for the day", "end of day routine", "turning off equipment before leaving"],
            "forced": ["urgent matters that need attention"],
        }
        focal_points.extend(checkpoint_focal_points.get(checkpoint_base, []))

    # Add context-specific focal points
    temp = sim_state.get("indoor_temp_c", 21.0)
    if temp < 18 or temp > 25:
        focal_points.append("my thermal comfort preferences")

    lighting = sim_state.get("lighting_conditions", {})
    if lighting.get("natural_light_level") in ["dim", "dark"]:
        focal_points.append("lighting and workspace setup")

    equipment = sim_state.get("equipment_status", {})
    equipment_items = equipment.get("items", {})
    if equipment_items and not any(equipment_items.values()):
        focal_points.append("my equipment habits when arriving")

    other_occupants = sim_state.get("other_occupants_present", [])
    if other_occupants:
        focal_points.append(f"my relationship with {other_occupants[0]}")
        focal_points.append("agreements I made with colleagues about temperature or comfort")
        focal_points.append("conversations I had today")
        focal_points.append("topics I've already discussed with colleagues")

    # Time-aware focal points for lunch and breaks
    datetime_str = sim_state.get("datetime", "")
    if datetime_str:
        try:
            from datetime import datetime as dt
            now = dt.fromisoformat(datetime_str.replace("Z", "+00:00"))
            hour = now.hour

            if 11 <= hour <= 14:
                focal_points.append("my lunch habits and food preferences")
                others_at_lunch = sim_state.get("other_occupants_at_lunch", [])
                if others_at_lunch:
                    focal_points.append(f"going to lunch with colleagues")

            if 14 <= hour <= 16:
                focal_points.append("taking breaks and energy levels")

            if 16 <= hour <= 19:
                focal_points.append("my end of day routine and departure habits")
                office_occupancy = sim_state.get("office_occupancy", {})
                if office_occupancy.get("you_would_be_last_to_leave", False):
                    focal_points.append("responsibilities when leaving the office last")

            if 8 <= hour <= 10:
                focal_points.append("my morning routine at work")
        except (ValueError, TypeError):
            pass

    # If currently at lunch or on break, add relevant focal points
    agent_status = sim_state.get("agent_status", {})
    if agent_status.get("at_lunch"):
        focal_points.append("when I usually return from lunch")
    if agent_status.get("on_break"):
        focal_points.append("how long I usually take breaks")

    # Limit focal points to reduce memory retrieval overhead
    MAX_FOCAL_POINTS = 6
    if len(focal_points) > MAX_FOCAL_POINTS:
        focal_points = focal_points[:MAX_FOCAL_POINTS]

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
        "my clothing style and what I typically wear",
        "my lunch and break habits",
    ]


# ---------------------------------------------------------------------------
# Meeting Context
# ---------------------------------------------------------------------------

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
        Dict with meeting context
    """
    # Get today's date range
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = now.replace(hour=23, minute=59, second=59, microsecond=999999)

    # Query all meetings for today from shared calendar
    meetings_today = calendar.list_events("shared", day_start, day_end)

    # Filter to meetings the agent is involved in
    agent_meetings = []
    for meeting in meetings_today:
        if meeting.get("created_by") == agent_id:
            agent_meetings.append(meeting)
            continue

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

            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)

            now_utc = now if now.tzinfo else now.replace(tzinfo=timezone.utc)

            if start_dt <= now_utc <= end_dt:
                current_meeting = meeting
                current_meeting["_minutes_in"] = int((now_utc - start_dt).total_seconds() / 60)

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
            f"Meeting '{current_meeting.get('title', 'Meeting')}' is in progress "
            f"(started {mins_in} min ago)"
        )
    elif next_meeting and minutes_to_next is not None and minutes_to_next <= 5:
        should_attend_now = True
        meeting_alert = (
            f"Meeting '{next_meeting.get('title', 'Meeting')}' starts in {minutes_to_next} minutes"
        )

    return {
        "current_meeting": current_meeting,
        "next_meeting": next_meeting,
        "minutes_to_next_meeting": minutes_to_next,
        "meetings_today": agent_meetings,
        "should_attend_now": should_attend_now,
        "meeting_alert": meeting_alert,
    }


def get_meeting_host_equipment_context(
    agent_id: str,
    meeting_context: Dict[str, Any],
    checkpoint_reason: str,
) -> Optional[str]:
    """
    Get equipment context for meeting host at meeting start/end.

    Args:
        agent_id: The agent's ID
        meeting_context: Meeting context from get_meeting_context()
        checkpoint_reason: The checkpoint reason (e.g., "meeting_start:Team Standup")

    Returns:
        Formatted prompt section for meeting equipment, or None if not applicable
    """
    if not checkpoint_reason.startswith(("meeting_start:", "meeting_end:")):
        return None

    parts = checkpoint_reason.split(":", 1)
    if len(parts) < 2:
        return None

    checkpoint_type = parts[0]
    meeting_name = parts[1]

    meetings_today = meeting_context.get("meetings_today", [])

    target_meeting = None
    for meeting in meetings_today:
        if meeting.get("title") == meeting_name:
            target_meeting = meeting
            break

    if not target_meeting:
        return None

    is_host = target_meeting.get("created_by") == agent_id

    if not is_host:
        return None

    if checkpoint_type == "meeting_start":
        return f"""
=== MEETING HOST EQUIPMENT SETUP ===
You are the HOST of meeting "{meeting_name}" starting now.
Location: {target_meeting.get('location', 'meeting_room')}

As the meeting host, consider equipment setup:
- Projector: Turn on if you have a presentation
- Lighting: Adjust for visibility (dim for projection, bright for discussion)
- Conference phone: Turn on if remote participants expected
- Thermostat: Adjust if needed for comfort during meeting

Include your equipment requests in your MeetingEquipmentDecision.
"""
    else:  # meeting_end
        return f"""
=== MEETING HOST EQUIPMENT TEARDOWN ===
Meeting "{meeting_name}" is ending.

As the meeting host, please ensure:
- Turn off projector if you turned it on
- Return lighting to normal levels
- Turn off conference phone if not needed
- Leave the meeting room ready for the next user

Include your equipment changes in your MeetingEquipmentDecision.
"""


# ---------------------------------------------------------------------------
# Section Formatters (Decomposed from format_step_prompt)
# ---------------------------------------------------------------------------

def _format_equipment_section(
    sim_state: Dict[str, Any],
) -> tuple[str, str]:
    """
    Format equipment status section.

    Returns:
        Tuple of (equipment_section, equipment_note)
    """
    equipment_status = sim_state.get("equipment_status", {})
    equipment_items = equipment_status.get("items", {})
    current_desk = sim_state.get("current_desk", "")

    desk_equipment_lines = []
    shared_equipment_lines = []

    for equipment_name, is_on in equipment_items.items():
        state_str = "ON" if is_on else "OFF"
        is_desk_equipment = any(equipment_name.endswith(f"_{suffix}") for suffix in ['A', 'B', 'C', 'D', 'E'])
        if is_desk_equipment:
            desk_equipment_lines.append(f"    - {equipment_name}: {state_str}")
        else:
            shared_equipment_lines.append(f"    - {equipment_name}: {state_str}")

    equipment_section_parts = []
    if current_desk:
        equipment_section_parts.append(f"  Your desk ({current_desk}):")
        if desk_equipment_lines:
            equipment_section_parts.extend(desk_equipment_lines)
        else:
            equipment_section_parts.append("    (no equipment assigned)")
    if shared_equipment_lines:
        equipment_section_parts.append("  Shared equipment:")
        equipment_section_parts.extend(shared_equipment_lines)

    equipment_section = "\n".join(equipment_section_parts) if equipment_section_parts else "  No equipment tracked"

    # Add warning if work equipment is off
    laptop_on = any(equipment_items.get(f"laptop_{s}", False) for s in ['A', 'B', 'C', 'D', 'E'])
    monitor_on = any(equipment_items.get(f"monitor_{s}", False) for s in ['A', 'B', 'C', 'D', 'E'])
    if current_desk and (not laptop_on or not monitor_on):
        equipment_section += "\n  >>> NOTE: Your laptop/monitor may be OFF. Turn them ON to work."

    # Equipment note if all off
    equipment_note = ""
    if equipment_items and not any(equipment_items.values()):
        equipment_note = "\n>>> Note: Your equipment is ALL OFF. You need laptop and monitor ON to work."

    return equipment_section, equipment_note


def _format_location_section(
    agent: "GenerativeAgent",
    sim_state: Dict[str, Any],
) -> str:
    """Format location section with available locations and who's where."""
    current_location = sim_state.get("current_location", "desk_area")
    location_info = sim_state.get("location_info", {})
    available_locations = sim_state.get("available_locations", [])
    agents_by_location = sim_state.get("agents_by_location", {})
    location_equipment = sim_state.get("location_equipment", [])

    location_lines = [f"You are at: {current_location}"]
    current_loc_info = location_info.get(current_location, {})
    if current_loc_info.get("description"):
        location_lines.append(f"  ({current_loc_info['description']})")

    # Show who else is at current location
    agents_at_current = agents_by_location.get(current_location, [])
    others_here = [a for a in agents_at_current if a != agent.agent_id]
    if others_here:
        location_lines.append(f"\n  >>> COLLEAGUES HERE WITH YOU: {', '.join(others_here)}")
    else:
        location_lines.append(f"\n  >>> You are alone at this location.")

    # Show equipment available at current location
    if location_equipment:
        location_lines.append("\nEquipment available here:")
        for eq in location_equipment:
            state_str = "ON" if eq.get("is_on") else "OFF"
            in_use = eq.get("in_use_by")
            use_str = f" (in use by {in_use})" if in_use else ""
            location_lines.append(f"  - {eq['name']}: {state_str}{use_str}")

    # Show other locations
    location_lines.append("\nOther locations you can go to:")
    for loc_name in available_locations:
        if loc_name == current_location:
            continue
        loc_info = location_info.get(loc_name, {})
        loc_desc = loc_info.get("description", "")
        agents_there = agents_by_location.get(loc_name, [])
        agents_str = f" - {', '.join(agents_there)} there" if agents_there else ""
        appliances = loc_info.get("appliances", [])
        appliances_str = f" (has: {', '.join(appliances)})" if appliances else ""
        location_lines.append(f"  - {loc_name}: {loc_desc}{appliances_str}{agents_str}")

    return "\n".join(location_lines)


def _format_meetings_section(
    meeting_context: Optional[Dict[str, Any]],
    agent_id: str,
) -> str:
    """Format meeting status section."""
    if not meeting_context:
        return "No meeting information available."

    meetings_today = meeting_context.get("meetings_today", [])
    meeting_alert = meeting_context.get("meeting_alert")
    next_meeting = meeting_context.get("next_meeting")
    minutes_to_next = meeting_context.get("minutes_to_next_meeting")

    meeting_lines = []

    if meetings_today:
        meeting_lines.append("Your meetings today:")
        meeting_lines.append(format_meetings_for_prompt(meetings_today, agent_id))
    else:
        meeting_lines.append("No meetings scheduled today.")

    if meeting_alert:
        meeting_lines.append("")
        meeting_lines.append(f">>> IMPORTANT: {meeting_alert}")
        current_meeting = meeting_context.get("current_meeting")
        if current_meeting:
            meeting_lines.append("")
            meeting_lines.append(">>> ACTION REQUIRED: Use 'attend_meeting' NOW to physically go to the meeting room!")
            meeting_lines.append("    (Remember: accepting an invitation doesn't move you there - you must use attend_meeting)")
        elif meeting_context.get("should_attend_now"):
            meeting_lines.append("")
            meeting_lines.append(">>> ACTION REQUIRED: Use 'attend_meeting' to go to the meeting room!")

    elif next_meeting and minutes_to_next is not None:
        meeting_lines.append("")
        meeting_lines.append(
            f"Next meeting: '{next_meeting.get('title', 'Meeting')}' "
            f"in {minutes_to_next} minutes"
        )

    return "\n".join(meeting_lines)


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


def format_device_state(
    sim_state: Dict[str, Any],
    current_location: Optional[str] = None,
) -> str:
    """
    Format device availability and status at agent's current location.

    Args:
        sim_state: Current simulation state dict
        current_location: Current location (if None, uses sim_state)

    Returns:
        Formatted string describing devices at location
    """
    if current_location is None:
        current_location = sim_state.get("current_location", "unknown")

    location_equipment = sim_state.get("location_equipment", [])

    if not location_equipment:
        return f"No controllable devices at {current_location}."

    lines = [f"Devices at {current_location}:"]
    for device in location_equipment:
        name = device.get("name", "unknown")
        status = "ON" if device.get("is_on") else "OFF"
        device_type = device.get("type", "")
        in_use_by = device.get("in_use_by")

        line = f"- {name}: {status}"
        if device_type and device_type != name:
            line += f" (type: {device_type})"
        if in_use_by:
            line += f" (in use by {in_use_by})"

        lines.append(line)

    return "\n".join(lines)


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

    for focal_pt, memories in retrieved_memories.items():
        if "agreement" in focal_pt.lower():
            for mem in memories:
                if "agreed" in mem.description.lower():
                    if hasattr(mem, 'created') and mem.created is not None:
                        try:
                            if isinstance(mem.created, (int, float)):
                                mem_time = datetime.fromtimestamp(mem.created, tz=timezone.utc)
                                now_aware = now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now
                                time_diff = (now_aware - mem_time).total_seconds() / 3600
                            else:
                                time_diff = (now - mem.created).total_seconds() / 3600

                            if time_diff <= 2:
                                agreements.append(f"- {mem.description}")
                        except (TypeError, ValueError):
                            agreements.append(f"- {mem.description}")
                    else:
                        agreements.append(f"- {mem.description}")

    if not agreements:
        return "No recent agreements with colleagues."

    return "\n".join(agreements)


def _format_comfort_preferences_section(agent: "GenerativeAgent") -> str:
    """
    Extract comfort preferences from the agent's daily plan.

    The comfort_preferences field captures thermal/lighting preferences set during
    daily planning and should inform thermostat and lighting decisions.

    Args:
        agent: The generative agent

    Returns:
        Formatted string with comfort preferences, or empty string if none
    """
    daily_plan = agent.get_daily_plan() if hasattr(agent, 'get_daily_plan') else None

    if not daily_plan:
        return ""

    # Handle both dict and Pydantic model
    if isinstance(daily_plan, dict):
        comfort_prefs = daily_plan.get("comfort_preferences", "")
    elif hasattr(daily_plan, 'comfort_preferences'):
        comfort_prefs = daily_plan.comfort_preferences
    else:
        comfort_prefs = ""

    if not comfort_prefs or comfort_prefs.strip() == "":
        return ""

    return f"\n<your_comfort_preferences>{comfort_prefs}</your_comfort_preferences>"


# ---------------------------------------------------------------------------
# 6-Step Checkpoint Prompt Builders
# ---------------------------------------------------------------------------

def format_step1_prompt(state: "CheckpointState") -> str:
    """
    Step 1: Plan Review - understand priorities and commitment status.

    This is the first step of the 6-step checkpoint flow. It focuses on:
    - Reviewing the daily plan
    - Identifying current priorities
    - Checking commitment status (approaching? active? waiting?)

    Args:
        state: CheckpointState containing agent and simulation context

    Returns:
        Formatted prompt string for Step 1
    """
    agent = state.agent

    # Get commitment info from sync manager if applicable
    commitment_info = ""
    if state.daily_plan:
        # Handle both dict and object access patterns
        social_commitments = (
            state.daily_plan.get("social_commitments", [])
            if isinstance(state.daily_plan, dict)
            else getattr(state.daily_plan, 'social_commitments', [])
        )

        for c in social_commitments:
            # Handle both dict and object access
            if isinstance(c, dict):
                fulfilled = c.get("fulfilled", False)
                activity = c.get("activity", "activity")
                time_str = c.get("time", "unspecified")
                with_agents = c.get("with_agents", [])
                location = c.get("location", "break_area")
            else:
                fulfilled = getattr(c, 'fulfilled', False)
                activity = getattr(c, 'activity', "activity")
                time_str = getattr(c, 'time', "unspecified")
                with_agents = getattr(c, 'with_agents', [])
                location = getattr(c, 'location', "break_area")

            if not fulfilled:
                partner_names = ", ".join(
                    a.split("_")[0].capitalize() for a in with_agents
                )
                commitment_info += f"\n- {activity} with {partner_names} at {time_str}"
                commitment_info += f" (location: {location})"

                # Get partner status from sync manager
                commitment_id = f"{activity}_{time_str}"
                partner_status = state.sync_manager.get_partner_status(
                    state.agent_id, commitment_id
                )
                if partner_status:
                    commitment_info += f"\n  Partner status: {partner_status}"

    # Core memory retrieval for reliability traits
    core_traits = ""
    if agent.core_memory_store:
        traits = agent.core_memory_store.retrieve_relevant(
            "reliability punctuality keeping promises social commitments", n_count=2
        )
        if traits:
            core_traits = "\n".join(f"- {t['description']}" for _, t in traits)

    # Retrieve step-specific memories
    step_memories = state.retrieve_for_step("plan_review")
    memories_text = _format_memories_for_step(step_memories, limit=5)

    # Format filtered daily plan (current and future only)
    filtered_plan = _filter_plan_to_current_and_future(state.daily_plan, state.now)
    plan_text = _format_filtered_plan(filtered_plan, state.daily_plan)

    # Get physical environment context
    indoor_temp = state.sim_state.get("indoor_temp_c", 22)
    thermostat_setpoint = state.sim_state.get("thermostat_setpoint_c", 22)
    preferred_temp = state.sim_state.get("preferred_temp_c", 22)
    lighting_conditions = state.sim_state.get("lighting_conditions", {})
    natural_light = lighting_conditions.get("natural_light_level", "moderate")
    current_location = state.current_location

    # Get equipment at current location
    equipment_list = state.get_equipment_at_location(current_location)
    equipment_summary = ", ".join(
        f"{eq.get('name')} ({'ON' if eq.get('is_on') else 'OFF'})"
        for eq in equipment_list
    ) if equipment_list else "none nearby"

    # Get colleagues at current location
    colleagues_here = state.get_colleagues_at_current_location()
    colleagues_text = ", ".join(colleagues_here) if colleagues_here else "none"

    return f"""
<checkpoint reason="{state.checkpoint_reason}" time="{state.now.strftime('%H:%M')}" />

<current_situation>
<location>{current_location}</location>
<indoor_temp>{indoor_temp:.1f}C</indoor_temp>
<thermostat_setpoint>{thermostat_setpoint:.1f}C</thermostat_setpoint>
<your_preferred_temp>{preferred_temp:.1f}C</your_preferred_temp>
<natural_light>{natural_light}</natural_light>
<equipment_here>{equipment_summary}</equipment_here>
<colleagues_here>{colleagues_text}</colleagues_here>
</current_situation>

<your_plan>
{plan_text}
</your_plan>

<commitments>
{commitment_info.strip() if commitment_info else "No active commitments."}
</commitments>

<your_traits>
{core_traits if core_traits else "No specific traits retrieved."}
</your_traits>

<recent_memories>
{memories_text}
</recent_memories>

<instructions>
Review your plan and current physical situation.

Your priorities should focus on ACTIONABLE items in this building simulation:
- Physical comfort: Is temperature OK? Need thermostat adjustment?
- Equipment: Need to turn on laptop/monitor? Turn off unused equipment?
- Lighting: Is natural light sufficient or do you need desk lamp?
- Location: Should you stay or move somewhere else?
- Commitments: Any meetings or social plans approaching?
- Colleagues: Anyone here you should interact with?

DO NOT prioritize imaginary work tasks like "check email" or "review documents".
Focus on physical environment, equipment, and schedule.

Output a PlanReviewDecision with:
- checkpoint_summary: Brief description of what's happening now
- plan_alignment: Are you on_track, need_adjustment, or off_track?
- priorities: Your top 2-3 ACTIONABLE priorities (comfort, equipment, location, social)
- active_commitment: If you have an upcoming commitment, describe it
- commitment_status: not_yet, approaching, now, waiting, or overdue
- reasoning: Your thought process
</instructions>
""".strip()


def format_step2_prompt(state: "CheckpointState") -> str:
    """
    Step 2: Current Location Decisions - equipment, comfort, breaks.

    Handles decisions at the agent's CURRENT location:
    - Thermostat adjustments
    - Lighting adjustments
    - Equipment ON/OFF decisions
    - Break actions (if taking break HERE)
    - Commitment responses

    Args:
        state: CheckpointState with step1 already completed

    Returns:
        Formatted prompt string for Step 2
    """
    location = state.current_location
    equipment = state.get_equipment_at_location(location)

    # Format equipment list
    equipment_lines = []
    for eq in equipment:
        name = eq.get("name", "unknown")
        is_on = eq.get("is_on", False)
        state_str = "ON" if is_on else "OFF"
        in_use = eq.get("in_use_by")
        use_str = f" (in use by {in_use})" if in_use else ""
        equipment_lines.append(f"- {name}: {state_str}{use_str}")
    equipment_text = "\n".join(equipment_lines) if equipment_lines else "No equipment at this location."

    # Format lighting state
    lighting_lines = []
    lighting_conditions = state.sim_state.get("lighting_conditions", {})
    natural_light = lighting_conditions.get("natural_light_level", "unknown")

    # Desk light
    desk_light = state.sim_state.get("desk_light")
    if desk_light:
        desk_light_on = desk_light.get("is_on", False)
        lighting_lines.append(f"- desk_light: {'ON' if desk_light_on else 'OFF'}")

    # Zone lights
    zone_lights = state.sim_state.get("zone_lights", {})
    for light_name, light_info in zone_lights.items():
        if isinstance(light_info, dict):
            is_on = light_info.get("is_on", False)
            lighting_lines.append(f"- {light_name}: {'ON' if is_on else 'OFF'}")

    lighting_text = "\n".join(lighting_lines) if lighting_lines else "No controllable lights at this location."

    # Get comfort info
    indoor_temp = state.sim_state.get("indoor_temp_c", 22)
    thermostat_setpoint = state.sim_state.get("thermostat_setpoint_c", 22)
    preferred_temp = state.sim_state.get("preferred_temp_c", 22)

    # Format priorities from step 1
    priorities = state.get_priorities()
    priorities_text = "\n".join(f"- {p}" for p in priorities) if priorities else "- Continue with current activities"

    # Commitment status from step 1
    commitment_status = state.get_commitment_status() or "none"

    # Retrieve step-specific memories
    step_memories = state.retrieve_for_step("current_location_actions")
    memories_text = _format_memories_for_step(step_memories, limit=4)

    return f"""
<location>{location}</location>

<relevant_memories>
{memories_text}
</relevant_memories>

<priorities>
{priorities_text}
</priorities>

<comfort>
<indoor_temp>{indoor_temp:.1f}C</indoor_temp>
<thermostat_setpoint>{thermostat_setpoint:.1f}C</thermostat_setpoint>
<your_preferred_temp>{preferred_temp:.1f}C</your_preferred_temp>
</comfort>

<lighting_here>
<natural_light>{natural_light}</natural_light>
{lighting_text}
<note>zone_main is the main overhead light for the ENTIRE office - turning it on/off affects everyone. Desk lights (desk_light_A/B/C) are personal.</note>
</lighting_here>

<equipment_here>
{equipment_text}
</equipment_here>

<commitment_status>{commitment_status}</commitment_status>

<instructions>
What do you want to do HERE at {location}?

IMPORTANT - Lighting decisions:
- Check the current state of each light listed above (ON or OFF)
- If a light is ALREADY ON and you want it on, use "keep_current" - do NOT turn_on again
- If a light is ALREADY OFF and you want it off, use "keep_current" - do NOT turn_off again
- Only use "turn_on" or "turn_off" when you want to CHANGE the lighting state
- Natural light level: {natural_light} (bright/moderate = lights probably not needed; dim/dark = lights may help)

IMPORTANT - Equipment decisions:
- Check the current state of each device listed above (ON or OFF)
- If a device is ALREADY ON and you want it on, use "keep_current" - do NOT turn_on again
- If a device is ALREADY OFF and you want it off, use "keep_current" - do NOT turn_off again
- Only use "turn_on" or "turn_off" when you want to CHANGE the current state
- You can only control equipment at YOUR current location ({location})

Consider:
- Adjust thermostat if uncomfortable (direction: warmer/cooler, amount: small/medium/large)
- Turn lights/equipment ON or OFF ONLY if you need to CHANGE the current state
- Take a break here (if not going elsewhere)
- Respond to commitment (if one is active)
- Update plan (if needed)

If at break_area:
- activity="tea" → kettle is used automatically (auto-off after 2 min)
- activity="coffee" → coffee_machine is used automatically (auto-off after 10 min)
- You do NOT need to manually turn off kitchen appliances - they have automatic timers

Do NOT decide about moving yet - that comes in Step 4.

Output a CurrentLocationDecision with your choices.
</instructions>
""".strip()


def format_step3_prompt(state: "CheckpointState") -> Optional[str]:
    """
    Step 3: Current Location Conversations.

    Decides whether to initiate a conversation with colleagues
    at the current location before potentially moving.

    Args:
        state: CheckpointState with steps 1-2 completed

    Returns:
        Formatted prompt string, or None if no colleagues present (skip step)
    """
    colleagues = state.get_colleagues_at_current_location()

    if not colleagues:
        return None  # Skip this step - no one to talk to

    # Format priorities
    priorities = state.get_priorities()
    priorities_text = "\n".join(f"- {p}" for p in priorities) if priorities else "- Continue with current activities"

    # Format colleagues list
    colleagues_text = "\n".join(f"- {c}" for c in colleagues)

    # Retrieve step-specific memories
    step_memories = state.retrieve_for_step("current_location_conversation")
    memories_text = _format_memories_for_step(step_memories, limit=4)

    return f"""
<location>{state.current_location}</location>

<colleagues_here>
{colleagues_text}
</colleagues_here>

<relevant_memories>
{memories_text}
</relevant_memories>

<priorities>
{priorities_text}
</priorities>

<instructions>
Do you want to talk to someone HERE before potentially moving?

Only initiate if there's a good reason:
- Commitment to fulfill with them
- Important topic to discuss
- Haven't chatted recently and want to connect

Don't initiate conversations just to "check in" about existing plans.

Output a StepConversationDecision:
- action: "initiate" or "none"
- target_agent: Who to talk to (if initiating)
- topic: What to discuss (if initiating)
- reasoning: Your thought process
</instructions>
""".strip()


def format_step4_prompt(state: "CheckpointState") -> str:
    """
    Step 4: Move Decision.

    Decides whether to move to a different location and where.
    Uses LOCATION_TRANSITIONS to enforce valid moves.

    Args:
        state: CheckpointState with steps 1-3 completed

    Returns:
        Formatted prompt string for Step 4
    """
    current = state.current_location
    valid_destinations = state.get_valid_destinations()

    # Format destinations
    destinations_text = "\n".join(f"- {loc}" for loc in valid_destinations)

    # Format priorities
    priorities = state.get_priorities()
    priorities_text = "\n".join(f"- {p}" for p in priorities) if priorities else "- Continue with current activities"

    # Format upcoming events from daily plan
    upcoming_text = _format_upcoming_events(state.daily_plan, state.now)

    # Get commitment status and partner info
    commitment_status = state.get_commitment_status() or "none"
    partner_status = ""
    if state.step1 and state.step1.active_commitment:
        # Try to get partner status for the active commitment
        if state.daily_plan:
            social_commitments = (
                state.daily_plan.get("social_commitments", [])
                if isinstance(state.daily_plan, dict)
                else getattr(state.daily_plan, 'social_commitments', [])
            )
            for c in social_commitments:
                if isinstance(c, dict):
                    activity = c.get("activity", "")
                    time_str = c.get("time", "")
                else:
                    activity = getattr(c, 'activity', "")
                    time_str = getattr(c, 'time', "")

                commitment_id = f"{activity}_{time_str}"
                ps = state.sync_manager.get_partner_status(state.agent_id, commitment_id)
                if ps:
                    partner_status = f"Partner: {ps}"
                    break

    # Retrieve step-specific memories
    step_memories = state.retrieve_for_step("move_decision")
    memories_text = _format_memories_for_step(step_memories, limit=4)

    checkpoint_reason = state.checkpoint_reason

    return f"""
<checkpoint_reason>{checkpoint_reason}</checkpoint_reason>

<current_location>{current}</current_location>

<relevant_memories>
{memories_text}
</relevant_memories>

<can_move_to>
{destinations_text}
</can_move_to>

<priorities>
{priorities_text}
</priorities>

<upcoming>
{upcoming_text}
</upcoming>

<commitment_status>
Status: {commitment_status}
{partner_status}
</commitment_status>

<instructions>
You are at {current}.

**Staying is the default.** You do NOT need to move unless you have a specific reason.

Reasons to move:
- A meeting starting soon (checkpoint indicates "meeting_start:" or "meeting_prep:")
- A commitment checkpoint (checkpoint indicates "commitment_prep:" or "commitment_start:")
- Equipment you need is only available elsewhere

**IMPORTANT - Timing for Social Commitments:**
- Do NOT move to break_area or other locations just because you have a future commitment there
- If a commitment (coffee run, break with colleague) is more than 15 minutes away, STAY at your current location
- The system will trigger a "commitment_prep:" checkpoint ~5 minutes before the commitment time
- Wait for that checkpoint before moving to the commitment location
- Only move early if the checkpoint reason specifically indicates it's time

**IMPORTANT - Meeting Attendance:**
If your checkpoint reason indicates a meeting is starting (e.g., "meeting_start:", "meeting_prep:"),
you MUST move to meeting_room unless you are already there. Meetings happen in meeting_room.

**IMPORTANT - Commitment Fulfillment:**
If your checkpoint reason indicates a commitment (e.g., "commitment_prep:", "commitment_start:"),
check the commitment_status and move to the appropriate location to fulfill it.

If staying: Simply output action="stay" - no justification needed.
If moving: Specify destination and purpose.

Note: To go outside, you must be at entrance first, then move to outside.
To return inside: outside → entrance → desk_area (or other locations).

Output a MoveDecision:
- action: "move" or "stay"
- destination: Where to go (if moving, must be in can_move_to list)
- purpose: Why you're moving (if moving)
- reasoning: Your thought process
</instructions>
""".strip()


def format_step5_prompt(state: "CheckpointState") -> str:
    """
    Step 5: New Location Decisions.

    Only called if agent decided to move in Step 4.
    Handles decisions at the DESTINATION location:
    - Equipment ON/OFF decisions
    - Meeting equipment (if meeting_room)
    - Break actions at new location

    Args:
        state: CheckpointState with step4.action == "move"

    Returns:
        Formatted prompt string for Step 5
    """
    location = state.step4.destination
    purpose = state.step4.purpose or "general activity"
    equipment = state.get_equipment_at_location(location)

    # Format equipment list
    equipment_lines = []
    for eq in equipment:
        name = eq.get("name", "unknown")
        is_on = eq.get("is_on", False)
        state_str = "ON" if is_on else "OFF"
        in_use = eq.get("in_use_by")
        use_str = f" (in use by {in_use})" if in_use else ""
        equipment_lines.append(f"- {name}: {state_str}{use_str}")
    equipment_text = "\n".join(equipment_lines) if equipment_lines else "No equipment at this location."

    # Format lighting state at new location
    lighting_lines = []
    lighting_conditions = state.sim_state.get("lighting_conditions", {})
    natural_light = lighting_conditions.get("natural_light_level", "unknown")

    # Get zone lights from sim_state (includes lights at all locations)
    zone_lights = state.sim_state.get("zone_lights", {})

    # Show lights relevant to the destination location
    if location == "meeting_room":
        meeting_light = zone_lights.get("meeting_room", {})
        if isinstance(meeting_light, dict):
            is_on = meeting_light.get("is_on", False)
            lighting_lines.append(f"- meeting_room: {'ON' if is_on else 'OFF'}")
    elif location == "desk_area":
        # Show desk lights
        desk_light = state.sim_state.get("desk_light")
        if desk_light:
            is_on = desk_light.get("is_on", False)
            lighting_lines.append(f"- desk_light: {'ON' if is_on else 'OFF'}")
        zone_main = zone_lights.get("zone_main", {})
        if isinstance(zone_main, dict):
            is_on = zone_main.get("is_on", False)
            lighting_lines.append(f"- zone_main: {'ON' if is_on else 'OFF'}")

    lighting_text = "\n".join(lighting_lines) if lighting_lines else "No controllable lights at this location."

    # Add meeting-specific context
    meeting_context = ""
    if location == "meeting_room":
        meeting_context = """
<meeting_equipment>
Available for meetings: projector, conference_phone
Turn on what you need for your meeting.
</meeting_equipment>
"""

    # Retrieve step-specific memories
    step_memories = state.retrieve_for_step("new_location_actions")
    memories_text = _format_memories_for_step(step_memories, limit=4)

    return f"""
<arrived_at>{location}</arrived_at>
<purpose>{purpose}</purpose>

<relevant_memories>
{memories_text}
</relevant_memories>

<lighting_here>
<natural_light>{natural_light}</natural_light>
{lighting_text}
<note>zone_main is the main overhead light for the ENTIRE office - turning it on/off affects everyone. Desk lights (desk_light_A/B/C) are personal.</note>
</lighting_here>

<equipment_here>
{equipment_text}
</equipment_here>
{meeting_context}
<instructions>
You arrived at {location} for: {purpose}

IMPORTANT - Lighting decisions:
- Check the current state of each light listed above (ON or OFF)
- If a light is ALREADY ON and you want it on, use "keep_current" - do NOT turn_on again
- If a light is ALREADY OFF and you want it off, use "keep_current" - do NOT turn_off again
- Only use "turn_on" or "turn_off" when you want to CHANGE the lighting state

IMPORTANT - Equipment decisions:
- Check the current state of each device listed above (ON or OFF)
- If a device is ALREADY ON and you want it on, use "keep_current" - do NOT turn_on again
- If a device is ALREADY OFF and you want it off, use "keep_current" - do NOT turn_off again
- Only use "turn_on" or "turn_off" when you want to CHANGE the current state
- You can only control equipment at THIS location ({location})

What do you need to do here?
- Turn lights/equipment ON or OFF ONLY if you need to CHANGE the current state
- Set up meeting equipment (if in meeting_room)
- Take a break action (if here for a break)

If at break_area:
- activity="tea" → kettle is used automatically (auto-off after 2 min)
- activity="coffee" → coffee_machine is used automatically (auto-off after 10 min)
- You do NOT need to manually turn off kitchen appliances - they have automatic timers

Output a NewLocationDecision:
- equipment_decisions: List of equipment to turn ON or OFF (or keep_current if no change needed)
- meeting_equipment: Meeting setup (if applicable)
- break_action: Break to take (if applicable)
- reasoning: Your thought process
</instructions>
""".strip()


def format_step6_prompt(state: "CheckpointState") -> Optional[str]:
    """
    Step 6: New Location Conversations.

    Only called if agent moved (Step 4) and there are colleagues
    at the new location. Decides whether to initiate conversations.

    Args:
        state: CheckpointState with step4.action == "move"

    Returns:
        Formatted prompt string, or None if no colleagues present (skip step)
    """
    location = state.step4.destination
    purpose = state.step4.purpose or "general activity"

    # Get colleagues at the new location
    colleagues_by_location = state.sim_state.get("colleagues_by_location", {})
    colleagues = colleagues_by_location.get(location, [])

    if not colleagues:
        return None  # Skip - no one to talk to

    # Format colleagues list
    colleagues_text = "\n".join(f"- {c}" for c in colleagues)

    # Retrieve step-specific memories
    step_memories = state.retrieve_for_step("new_location_conversation")
    memories_text = _format_memories_for_step(step_memories, limit=4)

    return f"""
<location>{location}</location>
<purpose>{purpose}</purpose>

<colleagues_here>
{colleagues_text}
</colleagues_here>

<relevant_memories>
{memories_text}
</relevant_memories>

<instructions>
You're at {location} for: {purpose}

Do you want to talk to anyone here?
This is especially relevant if:
- You came to meet someone for a commitment
- You have something to discuss with them
- You haven't talked to them today

Output a StepConversationDecision:
- action: "initiate" or "none"
- target_agent: Who to talk to (if initiating)
- topic: What to discuss (if initiating)
- reasoning: Your thought process
</instructions>
""".strip()


# ---------------------------------------------------------------------------
# Step Prompt Helper Functions
# ---------------------------------------------------------------------------

def _format_memories_for_step(
    memories: Dict[str, List[MemoryNode]],
    limit: int = 5,
) -> str:
    """Format memories for inclusion in step prompts."""
    all_memories = []
    for focal_pt, mem_list in memories.items():
        for mem in mem_list[:limit]:
            all_memories.append(mem)

    if not all_memories:
        return "No specific memories retrieved."

    # Deduplicate and limit
    seen = set()
    unique = []
    for mem in all_memories:
        if mem.description not in seen:
            seen.add(mem.description)
            unique.append(mem)

    # Format as text
    lines = []
    for mem in unique[:limit]:
        lines.append(f"- {mem.description}")

    return "\n".join(lines)


def _format_daily_plan_for_step(daily_plan: Optional[Any]) -> str:
    """Format daily plan for inclusion in step prompts."""
    if not daily_plan:
        return "No daily plan set."

    # Handle both dict and object access
    if isinstance(daily_plan, dict):
        arrival = daily_plan.get("arrival_time", "not set")
        departure = daily_plan.get("departure_time", "not set")
        morning_break = daily_plan.get("morning_break", "not set")
        afternoon_break = daily_plan.get("afternoon_break", "not set")
        lunch_plan = daily_plan.get("lunch_plan", {})
        lunch_time = lunch_plan.get("time", "not set") if isinstance(lunch_plan, dict) else "not set"
    else:
        arrival = getattr(daily_plan, 'arrival_time', "not set")
        departure = getattr(daily_plan, 'departure_time', "not set")
        morning_break = getattr(daily_plan, 'morning_break', "not set")
        afternoon_break = getattr(daily_plan, 'afternoon_break', "not set")
        lunch_plan = getattr(daily_plan, 'lunch_plan', None)
        lunch_time = lunch_plan.time if lunch_plan and hasattr(lunch_plan, 'time') else "not set"

    return f"""Arrival: {arrival}
Morning break: {morning_break}
Lunch: {lunch_time}
Afternoon break: {afternoon_break}
Departure: {departure}"""


def _format_upcoming_events(daily_plan: Optional[Any], now: datetime) -> str:
    """Format upcoming events from daily plan for move decision context."""
    if not daily_plan:
        return "No upcoming events."

    events = []
    current_minutes = now.hour * 60 + now.minute

    # Check social commitments
    social_commitments = (
        daily_plan.get("social_commitments", [])
        if isinstance(daily_plan, dict)
        else getattr(daily_plan, 'social_commitments', [])
    )

    for c in social_commitments:
        if isinstance(c, dict):
            fulfilled = c.get("fulfilled", False)
            time_str = c.get("time", "")
            activity = c.get("activity", "")
            with_agents = c.get("with_agents", [])
        else:
            fulfilled = getattr(c, 'fulfilled', False)
            time_str = getattr(c, 'time', "")
            activity = getattr(c, 'activity', "")
            with_agents = getattr(c, 'with_agents', [])

        if fulfilled or not time_str or time_str == "needs_confirmation":
            continue

        try:
            hour, minute = map(int, time_str.split(":"))
            event_minutes = hour * 60 + minute
            minutes_until = event_minutes - current_minutes

            if 0 <= minutes_until <= 60:
                partner_names = ", ".join(a.split("_")[0].capitalize() for a in with_agents)
                events.append(f"- In {minutes_until} min: {activity} with {partner_names}")
        except (ValueError, TypeError):
            continue

    # Check for meetings (if available in daily plan)
    # This would need meeting_context passed in for full implementation

    if not events:
        return "No events in the next hour."

    return "\n".join(events)


def _filter_plan_to_current_and_future(
    daily_plan: Optional[Any],
    now: datetime,
) -> Dict[str, Any]:
    """
    Filter plan to show only relevant items (not past).

    Agents should only see and update parts of their plan that are current
    or in the future. This prevents cluttering prompts with completed activities.

    Args:
        daily_plan: The agent's daily plan
        now: Current simulation datetime

    Returns:
        Dict with filtered plan elements:
        - upcoming_meetings: Meetings not yet ended
        - pending_commitments: Unfulfilled social commitments
        - remaining_breaks: Future break times
        - lunch: Lunch plan if not past
    """
    if not daily_plan:
        return {
            "upcoming_meetings": [],
            "pending_commitments": [],
            "remaining_breaks": [],
            "lunch": None,
        }

    current_minutes = now.hour * 60 + now.minute

    def _time_str_to_minutes(time_str: str) -> int:
        """Convert HH:MM to minutes since midnight."""
        try:
            hour, minute = map(int, time_str.split(":"))
            return hour * 60 + minute
        except (ValueError, TypeError):
            return 0

    def _is_meeting_past(meeting) -> bool:
        """Check if meeting has ended."""
        try:
            end_dt = datetime.fromisoformat(meeting.end_datetime_iso)
            end_minutes = end_dt.hour * 60 + end_dt.minute
            return current_minutes > end_minutes
        except (ValueError, TypeError, AttributeError):
            return False

    # Filter meetings
    meetings = getattr(daily_plan, 'meetings', []) if not isinstance(daily_plan, dict) else daily_plan.get('meetings', [])
    upcoming_meetings = [m for m in meetings if not _is_meeting_past(m)]

    # Filter social commitments
    social_commitments = (
        daily_plan.get("social_commitments", [])
        if isinstance(daily_plan, dict)
        else getattr(daily_plan, 'social_commitments', [])
    )
    pending_commitments = []
    for c in social_commitments:
        if isinstance(c, dict):
            fulfilled = c.get("fulfilled", False)
        else:
            fulfilled = getattr(c, 'fulfilled', False)
        if not fulfilled:
            pending_commitments.append(c)

    # Filter breaks
    remaining_breaks = []
    if isinstance(daily_plan, dict):
        morning_break = daily_plan.get("morning_break")
        afternoon_break = daily_plan.get("afternoon_break")
    else:
        morning_break = getattr(daily_plan, 'morning_break', None)
        afternoon_break = getattr(daily_plan, 'afternoon_break', None)

    if morning_break:
        break_time = morning_break.get("preferred_time") if isinstance(morning_break, dict) else getattr(morning_break, 'preferred_time', None)
        if break_time and _time_str_to_minutes(break_time) > current_minutes:
            remaining_breaks.append(("morning", morning_break))

    if afternoon_break:
        break_time = afternoon_break.get("preferred_time") if isinstance(afternoon_break, dict) else getattr(afternoon_break, 'preferred_time', None)
        if break_time and _time_str_to_minutes(break_time) > current_minutes:
            remaining_breaks.append(("afternoon", afternoon_break))

    # Filter lunch
    lunch = None
    if isinstance(daily_plan, dict):
        lunch_plan = daily_plan.get("lunch_plan")
    else:
        lunch_plan = getattr(daily_plan, 'lunch_plan', None)

    if lunch_plan:
        lunch_time = lunch_plan.get("time") if isinstance(lunch_plan, dict) else getattr(lunch_plan, 'time', None)
        if lunch_time and _time_str_to_minutes(lunch_time) > current_minutes - 30:  # Include if within 30 min
            lunch = lunch_plan

    return {
        "upcoming_meetings": upcoming_meetings,
        "pending_commitments": pending_commitments,
        "remaining_breaks": remaining_breaks,
        "lunch": lunch,
    }


def _format_filtered_plan(filtered: Dict[str, Any], daily_plan: Optional[Any]) -> str:
    """
    Format the filtered plan for display in prompts.

    Args:
        filtered: Output from _filter_plan_to_current_and_future
        daily_plan: Original daily plan for arrival/departure times

    Returns:
        Formatted string showing relevant plan items
    """
    lines = []

    # Get arrival/departure from original plan
    if daily_plan:
        if isinstance(daily_plan, dict):
            arrival = daily_plan.get("arrival_time", "not set")
            departure = daily_plan.get("actual_departure_time") or daily_plan.get("departure_time", "not set")
        else:
            arrival = getattr(daily_plan, 'arrival_time', "not set")
            departure = getattr(daily_plan, 'actual_departure_time', None) or getattr(daily_plan, 'departure_time', "not set")
        lines.append(f"Arrival: {arrival}, Departure: {departure}")

    # Upcoming meetings
    if filtered["upcoming_meetings"]:
        lines.append("Upcoming meetings:")
        for m in filtered["upcoming_meetings"]:
            title = getattr(m, 'title', 'Meeting') if not isinstance(m, dict) else m.get('title', 'Meeting')
            start = getattr(m, 'start_datetime_iso', '') if not isinstance(m, dict) else m.get('start_datetime_iso', '')
            try:
                start_dt = datetime.fromisoformat(start)
                time_str = start_dt.strftime("%H:%M")
            except:
                time_str = "unknown"
            lines.append(f"  - {title} at {time_str}")

    # Pending commitments
    if filtered["pending_commitments"]:
        lines.append("Pending commitments:")
        for c in filtered["pending_commitments"]:
            if isinstance(c, dict):
                activity = c.get("activity", "activity")
                time_str = c.get("time", "unspecified")
                with_agents = c.get("with_agents", [])
            else:
                activity = getattr(c, 'activity', "activity")
                time_str = getattr(c, 'time', "unspecified")
                with_agents = getattr(c, 'with_agents', [])
            partners = ", ".join(a.split("_")[0].capitalize() for a in with_agents)
            lines.append(f"  - {activity} with {partners} at {time_str}")

    # Lunch
    if filtered["lunch"]:
        lunch = filtered["lunch"]
        lunch_time = lunch.get("time") if isinstance(lunch, dict) else getattr(lunch, 'time', "not set")
        lines.append(f"Lunch: {lunch_time}")

    # Remaining breaks
    if filtered["remaining_breaks"]:
        for break_type, brk in filtered["remaining_breaks"]:
            break_time = brk.get("preferred_time") if isinstance(brk, dict) else getattr(brk, 'preferred_time', "not set")
            lines.append(f"{break_type.capitalize()} break: {break_time}")

    if not lines:
        return "No plan set for today."

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Decision Context Builder
# ---------------------------------------------------------------------------

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
    # Deduplicate memories across focal points to reduce prompt size
    retrieved_memories = deduplicate_retrieved_memories(retrieved_memories)

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


# ---------------------------------------------------------------------------
# Main Prompt Formatters
# ---------------------------------------------------------------------------

def format_step_prompt(
    agent: "GenerativeAgent",
    sim_state: Dict[str, Any],
    retrieved_memories: Dict[str, List[MemoryNode]],
    now: datetime,
    meeting_context: Optional[Dict[str, Any]] = None,
    pending_invitations: Optional[List[Dict[str, Any]]] = None,
    colleague_context: Optional[str] = None,
    checkpoint_reason: str = "interval",
) -> str:
    """
    Format the prompt for step decision making.

    This is the main prompt that agents use to decide their next actions.

    Args:
        agent: The generative agent
        sim_state: Current simulation state
        retrieved_memories: Retrieved memories
        now: Current datetime
        meeting_context: Meeting context from get_meeting_context()
        pending_invitations: Pending meeting invitations
        colleague_context: Formatted colleague context string
        checkpoint_reason: Why decision is being made (hourly, meeting_start, lunch_time, etc.)

    Returns:
        Formatted prompt string
    """
    context = build_decision_context(agent, sim_state, retrieved_memories, now)

    # Build sections using helper functions
    equipment_section, equipment_note = _format_equipment_section(sim_state)
    location_section = _format_location_section(agent, sim_state)
    meeting_status_section = _format_meetings_section(meeting_context, agent.agent_id)

    # Build pending invitations section
    invitations_section = format_pending_invitations_for_prompt(pending_invitations) if pending_invitations else "No pending invitations."

    # Build colleague section
    if colleague_context is None:
        other_occupants = sim_state.get("other_occupants_present", [])
        agents_by_loc = sim_state.get("agents_by_location", {})
        colleague_context = format_colleague_context(agent, other_occupants, agents_by_loc)

    # Extract recent agreements
    agreements_section = _extract_agreements_from_memories(retrieved_memories, now)

    # Build social commitments section
    social_commitments_section = _format_social_commitments_section(agent)

    # Build upcoming commitments section (M.4 - anticipatory awareness)
    upcoming_commitments_section = _format_upcoming_commitments_section(agent, now, checkpoint_reason)

    # Format lighting
    lighting = sim_state.get("lighting_conditions", {})
    natural_light = lighting.get("natural_light_level", "moderate")
    desk_light_on = lighting.get("desk_light_on", False)
    lighting_section = f"  Natural light: {natural_light}, Desk light: {'ON' if desk_light_on else 'OFF'}"

    # Get work preferences from core memories
    work_preferences_section = _format_work_preferences_section(agent)

    # Get clothing section
    clothing_section = _format_clothing_section(agent)

    # Get agent status
    status_str = _format_agent_status(sim_state)

    # Get colleague status
    colleague_status_section = _format_colleague_status(sim_state)

    # Format checkpoint reason
    checkpoint_display = _format_checkpoint_reason(checkpoint_reason)

    # Get thermostat info
    thermostat_info = sim_state.get('thermostat', {})
    heating_setpoint = thermostat_info.get('heating_setpoint_c', 21)
    cooling_setpoint = thermostat_info.get('cooling_setpoint_c', 24)

    # Get comfort preferences from daily plan (for thermal decisions)
    comfort_preferences_section = _format_comfort_preferences_section(agent)

    # Build the main prompt
    prompt = f"""
<current_time>
{now.strftime('%H:%M')} on {context['day_of_week']}, {now.strftime('%Y-%m-%d')}
Checkpoint: {checkpoint_display}
</current_time>

<identity>
{context['identity']}
</identity>

<current_state>
<status>{status_str}</status>
<temperature indoor="{sim_state.get('indoor_temp_c', 'N/A')}C" outdoor="{sim_state.get('outdoor_temp_c', 'N/A')}C" />
<weather>{sim_state.get('weather_description', 'N/A')}</weather>{clothing_section}
<desk>{sim_state.get('current_desk', 'N/A')}</desk>
<thermostat heating="{heating_setpoint}C" cooling="{cooling_setpoint}C" />{comfort_preferences_section}
</current_state>

<location>
{location_section}
</location>

<your_equipment>
{equipment_section}{equipment_note}
</your_equipment>

<lighting>
{lighting_section}
</lighting>

<colleagues>
{colleague_context}
{colleague_status_section}
</colleagues>

<schedule>
{context['schedule']}
</schedule>

<meetings>
{meeting_status_section}
</meetings>

<pending_invitations>
{invitations_section}
</pending_invitations>

<recent_agreements>
{agreements_section}
</recent_agreements>

<social_commitments>
{social_commitments_section}
</social_commitments>
{upcoming_commitments_section}
<relevant_memories purpose="context for your decisions">
{context['relevant_memories']}{work_preferences_section}
</relevant_memories>

<direct_controls>
You have DIRECT CONTROL over these systems - your changes take effect immediately:

THERMOSTAT - Trust your comfort perception:
- Set thermostat.action="adjust" with adjustment_direction and adjustment_amount
- direction: "warmer" or "cooler"
- amount: "small" (0.5C), "medium" (1.0C), or "large" (1.5C)
- Your adjustment changes the setpoint instantly. You are not requesting - you are controlling.

Consider these factors when deciding if you're comfortable:
1. OUTSIDE WEATHER: If it's hot outside, cooler indoor temps (even 16-17C) may feel refreshing.
   If it's cold outside, warmer indoor temps (up to 23-24C) may feel cozy.
2. SEASON: In summer you may prefer cooler indoors; in winter, warmer.
3. YOUR PREFERENCE: Trust how YOU feel based on your thermal memories.
4. CURRENT CONDITIONS: Consider both indoor AND outdoor temperature.

Key principle: Don't tolerate discomfort! If you feel too hot or too cold, adjust the thermostat.
Let YOUR perception guide you, considering the weather context.

EQUIPMENT:
- You can turn ON/OFF any equipment at your current location or your assigned desk
- Use equipment_decisions list with equipment_name and action (turn_on/turn_off/keep_current)
- Changes happen immediately - you control these devices directly.

LIGHTING:
- Set lighting.action to turn_on, turn_off, or keep_current
- Lights are ON/OFF only (no dimming). Desk lights and zone_main respond immediately to your control.
</direct_controls>

<constraint_spec>
FORBIDDEN:
- Controlling equipment that is not at your location or desk
- Making more than one thermostat adjustment per hour without significant temperature change
- Starting conversations during active meetings
</constraint_spec>

<action_guidance>
Based on your identity, memories, and current state, decide what to do:

MEETINGS:
- Meeting starting/in progress: Use 'attend_meeting' to physically go there
- Pending invitations: Use 'respond_to_invitation' to accept or decline

BREAKS:
Use 'take_break' with location="break_area" and specify your activity:
- activity="tea" → kettle will be turned on automatically for you
- activity="coffee" → coffee_machine will be turned on automatically for you
- activity="snack" or other → equipment as needed

The break room appliances have automatic timers:
- Kettle: auto-off after 2 minutes
- Coffee machine: auto-off after 10 minutes
- Microwave: auto-off after 5 minutes
You do NOT need to turn them off manually.

When done, use 'return_from_break' to go back to your desk.

BREAK AWARENESS:
- Consider your recent break history before taking another break
- Typically people take a morning break and an afternoon break
- Check your memories: "When did I last take a break?"
- Balance your need for breaks with your work responsibilities

LUNCH:
- Use 'go_to_lunch' (stay at break_area) or 'go_out_for_lunch' (leave building)
- If food needs heating, use the microwave in break_area
- When done, use 'return_from_lunch' to return to your desk

WORK ACTIVITIES (photocopying, filing, etc. in shared_area):
1. Use 'move_to' with destination="shared_area"
2. Use the equipment (photocopier, etc.)
3. When done, use 'move_to' with destination="desk_area" to RETURN to work

After completing any task or break away from your desk, return to desk_area.
</action_guidance>

<plan_update_guidance>
UPDATE your plan (plan_update.action="update") if ANY of these apply:
- Weather changed (now raining, you planned outdoor lunch)
- Colleague invited you to lunch and you want to join
- Meeting was cancelled or rescheduled
- Feeling tired and want to adjust break timing
- 2+ hours since making plan and circumstances changed

Keep current ONLY if nothing significant has changed.
If in doubt and something has changed, UPDATE your plan!
</plan_update_guidance>

<conversations>
You can chat with colleagues naturally - topics might include:
- Work projects, deadlines, upcoming meetings
- Lunch or coffee plans
- Weekend plans or hobbies
- General office observations
Use conversation.action="initiate" with a topic to start talking.
Note: Minor thermostat adjustments don't need discussion - just make the change directly.

BEFORE INITIATING A CONVERSATION, CHECK:

1. **Already have a commitment?** If you already have a confirmed social commitment
   with this person today (check <social_commitments> above), DO NOT initiate another
   conversation to "confirm" or "check in" about it. Just execute the plan when the
   time comes.

2. **Recently talked?** If you talked to this person within the last hour
   (check relevant_memories), only initiate if you have NEW information to share.
   Don't repeat the same conversation.

3. **Good reason?** Only initiate conversations for:
   - New work coordination needs (scheduling, questions, updates)
   - Questions that genuinely need answers
   - Social connection (if you haven't talked today)
   - Spontaneous observations worth sharing

   DO NOT initiate for:
   - Re-confirming existing plans
   - Checking if they're "ready" for something already planned
   - Reminding them about commitments they already know about
</conversations>
"""

    # Add checkpoint-specific guidance
    prompt += _format_checkpoint_specific_guidance(agent, sim_state, checkpoint_reason, meeting_context)

    return prompt.strip()


def _format_social_commitments_section(agent: "GenerativeAgent") -> str:
    """Format social commitments section.

    Note: daily_plan is stored as a dict (from model_dump()), not a DailyPlan object.
    """
    daily_plan = agent.get_daily_plan() if hasattr(agent, 'get_daily_plan') else None
    if not daily_plan:
        return "No pending social commitments."

    # Handle both dict and object access patterns
    social_commitments = (
        daily_plan.get("social_commitments", []) if isinstance(daily_plan, dict)
        else getattr(daily_plan, 'social_commitments', [])
    )

    if social_commitments:
        # Filter to unfulfilled commitments
        unfulfilled = []
        for c in social_commitments:
            # Handle both dict and object access
            if isinstance(c, dict):
                if not c.get("fulfilled", False):
                    unfulfilled.append(c)
            else:
                if not getattr(c, 'fulfilled', False):
                    unfulfilled.append(c)

        if unfulfilled:
            # Separate confirmed and unconfirmed commitments
            confirmed = []
            needs_time = []
            for c in unfulfilled:
                # Handle both dict and object access
                if isinstance(c, dict):
                    time = c.get("time", "unspecified")
                else:
                    time = getattr(c, 'time', "unspecified")
                if time == "needs_confirmation":
                    needs_time.append(c)
                else:
                    confirmed.append(c)

            commitment_lines = []

            # Show confirmed commitments (have specific times)
            if confirmed:
                commitment_lines.append("=== CONFIRMED COMMITMENTS (specific times) ===")
                for c in confirmed:
                    if isinstance(c, dict):
                        with_agents = c.get("with_agents", [])
                        time = c.get("time", "unspecified")
                        activity = c.get("activity", "activity")
                        location = c.get("location", "unspecified")
                        has_conflict = c.get("has_conflict", False)
                    else:
                        with_agents = getattr(c, 'with_agents', [])
                        time = getattr(c, 'time', "unspecified")
                        activity = getattr(c, 'activity', "activity")
                        location = getattr(c, 'location', "unspecified")
                        has_conflict = getattr(c, 'has_conflict', False)

                    with_names = ", ".join(a.split("_")[0].capitalize() for a in with_agents)
                    line = f"  - {time}: {activity} with {with_names}"
                    if location != "unspecified":
                        line += f" at {location}"
                    if has_conflict:
                        line += " (CONFLICT - may overlap with other plans)"
                    commitment_lines.append(line)
                commitment_lines.append(">>> At the scheduled time, use take_break or go_out_for_break to fulfill these!")

            # Show unconfirmed commitments (need time confirmation)
            if needs_time:
                commitment_lines.append("")
                commitment_lines.append("=== UNCONFIRMED COMMITMENTS (need time) ===")
                for c in needs_time:
                    if isinstance(c, dict):
                        with_agents = c.get("with_agents", [])
                        activity = c.get("activity", "activity")
                        location = c.get("location", "unspecified")
                    else:
                        with_agents = getattr(c, 'with_agents', [])
                        activity = getattr(c, 'activity', "activity")
                        location = getattr(c, 'location', "unspecified")

                    with_names = ", ".join(a.split("_")[0].capitalize() for a in with_agents)
                    line = f"  - {activity} with {with_names}"
                    if location != "unspecified":
                        line += f" (likely at {location})"
                    commitment_lines.append(line)
                commitment_lines.append(">>> YOU AGREED but no time was set! Suggest a specific time (e.g., '10:30') and invite them!")
                commitment_lines.append(">>> Use conversation.action='initiate' to coordinate the time with your colleague.")

            return "\n".join(commitment_lines)

    return "No pending social commitments."


def _format_upcoming_commitments_section(
    agent: "GenerativeAgent",
    now: datetime,
    checkpoint_reason: str
) -> str:
    """Format upcoming commitments section (M.4).

    Shows commitments coming up in 10-60 minutes to enable anticipatory planning.
    Skips if we're already at a commitment checkpoint (detailed guidance shown there).

    Args:
        agent: The generative agent
        now: Current datetime
        checkpoint_reason: Current checkpoint reason

    Returns:
        Formatted upcoming commitments section or empty string
    """
    # Don't show if we're already at a commitment checkpoint (redundant)
    if checkpoint_reason.startswith("commitment:"):
        return ""

    daily_plan = agent.get_daily_plan() if hasattr(agent, 'get_daily_plan') else None
    if not daily_plan:
        return ""

    # Handle both dict and object access patterns
    social_commitments = (
        daily_plan.get("social_commitments", []) if isinstance(daily_plan, dict)
        else getattr(daily_plan, 'social_commitments', [])
    )

    if not social_commitments:
        return ""

    current_minutes = now.hour * 60 + now.minute
    upcoming = []

    for c in social_commitments:
        # Handle both dict and object access
        if isinstance(c, dict):
            fulfilled = c.get("fulfilled", False)
            time_str = c.get("time", "")
            activity = c.get("activity", "activity")
            with_agents = c.get("with_agents", [])
        else:
            fulfilled = getattr(c, 'fulfilled', False)
            time_str = getattr(c, 'time', "")
            activity = getattr(c, 'activity', "activity")
            with_agents = getattr(c, 'with_agents', [])

        if fulfilled:
            continue

        if not time_str or time_str == "needs_confirmation":
            continue

        # Parse commitment time
        try:
            comm_hour, comm_minute = map(int, time_str.split(":"))
            comm_minutes = comm_hour * 60 + comm_minute
            minutes_until = comm_minutes - current_minutes

            # Show commitments 10-60 minutes away
            if 10 <= minutes_until <= 60:
                # Format partner names naturally
                if len(with_agents) == 1:
                    partner_str = with_agents[0].split("_")[0].capitalize()
                elif len(with_agents) == 2:
                    partner_str = f"{with_agents[0].split('_')[0].capitalize()} and {with_agents[1].split('_')[0].capitalize()}"
                elif with_agents:
                    names = [a.split("_")[0].capitalize() for a in with_agents]
                    partner_str = ", ".join(names[:-1]) + f", and {names[-1]}"
                else:
                    partner_str = "your colleague"

                upcoming.append({
                    "minutes": minutes_until,
                    "activity": activity,
                    "partner": partner_str,
                    "time": time_str
                })
        except (ValueError, TypeError):
            continue

    if not upcoming:
        return ""

    # Sort by how soon they are
    upcoming.sort(key=lambda x: x["minutes"])

    lines = ["\n<upcoming_commitments>"]
    lines.append("⏰ COMING UP SOON:")

    for item in upcoming:
        lines.append(
            f"  In {item['minutes']} min ({item['time']}): {item['activity']} with {item['partner']}"
        )

    lines.append("")
    lines.append(f"💭 {upcoming[0]['partner']} is counting on you. Start wrapping up your current task")
    lines.append("   so you're ready when it's time to meet.")
    lines.append("</upcoming_commitments>")

    return "\n".join(lines)


def _format_work_preferences_section(agent: "GenerativeAgent") -> str:
    """Format work preferences from core memories."""
    if agent.core_memory_store:
        work_memories = agent.core_memory_store.retrieve_relevant(
            focal_point="work style equipment preferences routine",
            n_count=2
        )
        if work_memories:
            prefs = [f"  - {mem['description']}" for _, mem in work_memories]
            return f"\n\n=== YOUR WORK PREFERENCES ===\n" + "\n".join(prefs)
    return ""


def _format_clothing_section(agent: "GenerativeAgent") -> str:
    """Format today's clothing section."""
    todays_clothing = agent.get_todays_clothing() if hasattr(agent, 'get_todays_clothing') else None
    if todays_clothing:
        clothing_desc = todays_clothing.get("description", "Not specified")
        warmth = todays_clothing.get("warmth_level", "medium")
        layers_removable = "yes" if todays_clothing.get("layers_removable", True) else "no"
        return f"\nYour clothing today: {clothing_desc} (warmth: {warmth}, can remove layers: {layers_removable})"
    return ""


def _format_agent_status(sim_state: Dict[str, Any]) -> str:
    """Format agent status string."""
    agent_status = sim_state.get("agent_status", {})
    status_parts = []
    if agent_status.get("at_lunch"):
        status_parts.append("at lunch")
    elif agent_status.get("on_break"):
        status_parts.append("on break")
    elif agent_status.get("at_desk", True):
        status_parts.append("at your desk")
    if agent_status.get("out_of_office"):
        status_parts.append("outside the building")
    return ", ".join(status_parts) if status_parts else "at your desk"


def _format_colleague_status(sim_state: Dict[str, Any]) -> str:
    """Format colleague status section."""
    others_at_lunch = sim_state.get("other_occupants_at_lunch", [])
    others_on_break = sim_state.get("other_occupants_on_break", [])
    colleague_status_lines = []
    if others_at_lunch:
        colleague_status_lines.append(f"  At lunch: {', '.join(others_at_lunch)}")
    if others_on_break:
        colleague_status_lines.append(f"  On break: {', '.join(others_on_break)}")
    return "\n".join(colleague_status_lines) if colleague_status_lines else ""


def _format_checkpoint_reason(checkpoint_reason: str) -> str:
    """Format checkpoint reason for display."""
    checkpoint_display = {
        "hourly": "Regular hourly check-in",
        "meeting_start": "A meeting is starting",
        "meeting_end": "A meeting is ending",
        "lunch_time": "It's time for your planned lunch",
        "return_from_lunch": "Time to return from lunch",
        "take_break": "It's time for a break",
        "return_from_break": "Time to return from break",
        "interval": "Regular decision interval",
        "first_decision": "First decision of the day",
        "forced": "Decision requested",
    }.get(checkpoint_reason.split(":")[0], checkpoint_reason)

    if ":" in checkpoint_reason:
        meeting_name = checkpoint_reason.split(":", 1)[1]
        checkpoint_display += f" - '{meeting_name}'"

    return checkpoint_display


def _format_checkpoint_specific_guidance(
    agent: "GenerativeAgent",
    sim_state: Dict[str, Any],
    checkpoint_reason: str,
    meeting_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Format checkpoint-specific guidance sections."""
    guidance = ""

    # Lunch time guidance
    if checkpoint_reason.startswith("lunch_time"):
        daily_plan = agent.get_daily_plan() or {}
        lunch_plan = daily_plan.get('lunch_plan', {}) if hasattr(daily_plan, 'get') else {}
        if lunch_plan:
            lunch_location = lunch_plan.get('location', 'break_area')
            lunch_food = lunch_plan.get('food_description', '')
            guidance += f"\n=== LUNCH TIME ===\nYour planned lunch location: {lunch_location}"
            if lunch_food:
                guidance += f"\nFood you brought: {lunch_food}"
                guidance += "\nIf your food needs heating (leftovers, soup, etc.), use the microwave in break_area."

    # Return from lunch guidance
    if checkpoint_reason == "return_from_lunch":
        current_desk = sim_state.get('current_desk', 'your assigned desk')
        guidance += f"""

<lunch_return>
Your lunch break is over. Return to YOUR assigned desk: {current_desk}

ACTION REQUIRED: Set break_decision.action to "return_from_lunch"

IMPORTANT:
- You already have a desk assigned ({current_desk}). Do NOT try to select a new desk.
- Your desk equipment is still there waiting for you.
- Turn your equipment back on (laptop, monitor) when you return.

This action will move you back to desk_area and set your status to at_desk.
</lunch_return>
"""

    # Return from break guidance
    if checkpoint_reason == "return_from_break":
        current_desk = sim_state.get('current_desk', 'your assigned desk')
        guidance += f"""

<break_return>
Your break is over. Return to YOUR assigned desk: {current_desk}

ACTION REQUIRED: Set break_decision.action to "return_from_break"

IMPORTANT:
- You already have a desk assigned ({current_desk}). Do NOT try to select a new desk.
- Your desk equipment is still there waiting for you.

This action will move you back to desk_area and set your status to at_desk.
</break_return>
"""

    # F.2: Commitment prep guidance (10-15 min before commitment)
    if checkpoint_reason.startswith("commitment_prep:"):
        activity = checkpoint_reason.split(":", 1)[1] if ":" in checkpoint_reason else "activity"

        # Get partner names and time from the commitment
        partner_names = []
        commitment_time = ""
        daily_plan = agent.get_daily_plan()
        if daily_plan:
            social_commitments = (
                daily_plan.get("social_commitments", []) if isinstance(daily_plan, dict)
                else getattr(daily_plan, 'social_commitments', [])
            )
            for commitment in social_commitments:
                comm_activity = commitment.get("activity", "") if isinstance(commitment, dict) else getattr(commitment, "activity", "")
                if comm_activity and activity.lower() in comm_activity.lower():
                    if isinstance(commitment, dict):
                        partner_names = commitment.get("with_agents", [])
                        commitment_time = commitment.get("time", "")
                    else:
                        partner_names = getattr(commitment, "with_agents", [])
                        commitment_time = getattr(commitment, "time", "")
                    break

        # Format partner names naturally
        if len(partner_names) == 1:
            partner_str = partner_names[0].split("_")[0].capitalize()
        elif len(partner_names) == 2:
            partner_str = f"{partner_names[0].split('_')[0].capitalize()} and {partner_names[1].split('_')[0].capitalize()}"
        elif partner_names:
            names = [n.split("_")[0].capitalize() for n in partner_names]
            partner_str = ", ".join(names[:-1]) + f", and {names[-1]}"
        else:
            partner_str = "your colleague"

        guidance += f"""

=== UPCOMING: {activity.upper()} WITH {partner_str.upper()} ===

⏰ In about 10-15 minutes, you have {activity} with {partner_str}{f" at {commitment_time}" if commitment_time else ""}.

START WRAPPING UP:
- Finish your current task or find a good stopping point
- Save your work
- {partner_str} is counting on you to be there

This is a heads-up so you're ready when it's time to go.
Don't start any new tasks that will take a long time.
"""

    # Commitment guidance - relationship-focused framing (M.5)
    if checkpoint_reason.startswith("commitment:"):
        activity = checkpoint_reason.split(":", 1)[1] if ":" in checkpoint_reason else "activity"
        current_location = sim_state.get("current_location", "desk_area")
        in_meeting_room = current_location == "meeting_room"

        # Get partner name and deferral count from the commitment
        partner_names = []
        deferred_count = 0
        commitment_time = ""
        daily_plan = agent.get_daily_plan()
        if daily_plan and hasattr(daily_plan, 'social_commitments'):
            for commitment in daily_plan.social_commitments:
                # Match by activity (normalized)
                comm_activity = commitment.get("activity", "") if isinstance(commitment, dict) else getattr(commitment, "activity", "")
                if comm_activity and activity.lower() in comm_activity.lower():
                    if isinstance(commitment, dict):
                        partner_names = commitment.get("with_agents", [])
                        deferred_count = commitment.get("deferred_count", 0)
                        commitment_time = commitment.get("time", "")
                    else:
                        partner_names = getattr(commitment, "with_agents", [])
                        deferred_count = getattr(commitment, "deferred_count", 0)
                        commitment_time = getattr(commitment, "time", "")
                    break

        # Format partner names naturally
        if len(partner_names) == 1:
            partner_str = partner_names[0]
        elif len(partner_names) == 2:
            partner_str = f"{partner_names[0]} and {partner_names[1]}"
        elif partner_names:
            partner_str = ", ".join(partner_names[:-1]) + f", and {partner_names[-1]}"
        else:
            partner_str = "your colleague"

        guidance += f"""

=== TIME FOR {activity.upper()} WITH {partner_str.upper()} ===

You promised {partner_str} you'd have {activity} together.
{f"The time you agreed on was {commitment_time}." if commitment_time else ""}
Right now, {partner_str} is probably heading to the break area, expecting you.

Think about this:
- You made a promise to {partner_str}. They're counting on you.
- How would you feel if you went to meet someone and they never showed up?
- This is a chance to strengthen your relationship with {partner_str}.
- Keeping your word matters - it's how trust is built.
"""

        # F.5: Limit rescheduling - after 3+ deferrals, force a real decision
        if deferred_count >= 3:
            guidance += f"""
🚨 FINAL NOTICE - NO MORE RESCHEDULING 🚨

You've rescheduled this commitment {deferred_count} times already.
{partner_str} has been waiting for you to follow through, and each
time you've pushed it back. This pattern is damaging your relationship.

This is your LAST chance. You MUST choose one of these options:

1. **EXECUTE NOW**: Go fulfill this commitment RIGHT NOW.
   - Drop what you're doing (unless it's truly an emergency)
   - Go to break_area and have {activity} with {partner_str}
   - Show {partner_str} that you value your word and their time

2. **CANCEL HONESTLY**: If you truly cannot do this today, be honest.
   - Set commitment_response.action = "skip"
   - In your reasoning, acknowledge you couldn't follow through
   - Consider reaching out to {partner_str} to apologize

NO MORE DEFERRALS. Rescheduling again is NOT an option.
Ask yourself: What kind of colleague do you want to be?
"""
        elif deferred_count > 0:
            guidance += f"""
⚠️ You've already rescheduled this {deferred_count} time(s).
{partner_str} has been waiting for you to follow through.
Rescheduling again will likely disappoint them and hurt your relationship.
"""

        if in_meeting_room:
            guidance += f"""
You're currently in the meeting room. If you're in an active meeting, that's
a legitimate reason to delay - but be honest with yourself: Is this meeting
truly more important than keeping your promise to {partner_str}?

If you must defer, let {partner_str} know as soon as the meeting ends.
"""
        else:
            guidance += f"""
You're at {current_location}. Unless you're in the middle of something truly
critical, now is the time to go. {partner_str} is waiting.
"""

        guidance += f"""
TO FULFILL THIS COMMITMENT:
- Go to break_area using move_to or take_break action
- Use take_break with activity="{activity}"
- For outside activities: use go_out_for_break

If you absolutely cannot go (real emergency, critical deadline):
- Set commitment_response.action = "defer" with a specific defer_until time
- Ask yourself honestly: Is this worth disappointing {partner_str}?
- Think about how you'd feel if the situation were reversed
"""

    # M.3: Commitment waiting guidance - when agent is waiting for colleague
    if checkpoint_reason.startswith("commitment_waiting:"):
        parts = checkpoint_reason.split(":")
        activity = parts[1] if len(parts) > 1 else "activity"
        missing_partner_id = parts[2] if len(parts) > 2 else None

        # Try to get partner name
        if missing_partner_id:
            partner_name = missing_partner_id.split("_")[0].capitalize()
        else:
            partner_name = "your colleague"

        guidance += f"""

=== WAITING FOR {partner_name.upper()} ===

You're at the break area for {activity} as planned, but {partner_name} hasn't arrived yet.
It's been 5-10 minutes past the agreed time.

Think about this:
- {partner_name} might just be running late
- They could be stuck in something important
- Or they might have forgotten

YOUR OPTIONS:

1. **Wait a bit longer**: Give them a few more minutes. Sometimes people get delayed.
   - Stay where you are
   - Maybe enjoy your {activity.split()[0]} while waiting

2. **Check in with them**: Send a quick message or go find them.
   - Use initiate_conversation to ask: "Hey, are we still on for {activity}?"
   - This is natural - you're not being pushy, just checking in

3. **Head back to work**: If you've been waiting a while and have things to do.
   - Use return_from_break to go back to your desk
   - You can try to catch up with {partner_name} later

It's okay to feel a bit disappointed if someone doesn't show up as planned.
That's a natural reaction when you were looking forward to spending time together.
"""

    # Meeting prep guidance
    if checkpoint_reason.startswith("meeting_prep:"):
        meeting_title = checkpoint_reason.split(":", 1)[1] if ":" in checkpoint_reason else "meeting"
        current_location = sim_state.get("current_location", "desk_area")
        guidance += f"""

=== MEETING PREPARATION ===
Your meeting "{meeting_title}" starts in about 10 minutes.

PREPARE NOW:
1. Wrap up your current task
2. Gather any materials you need
"""
        if current_location == "outside":
            guidance += """3. URGENT: You are outside the building! Start heading back NOW to avoid being late.
"""
        elif current_location != "meeting_room":
            guidance += """3. Consider moving towards the meeting room soon
"""
        guidance += """
Use this time to prepare so you're not rushing when the meeting starts.
"""

    # Departure prep guidance
    if checkpoint_reason == "departure_prep":
        guidance += """

=== END OF DAY PREPARATION ===
Your departure time is approaching (10-20 minutes away).

Before you leave, please:
1. Save your work and wrap up current tasks
2. Turn OFF your laptop and monitor using equipment_decisions
3. Turn OFF your desk light if it's on
4. Return any shared equipment you borrowed

Use equipment_decisions with action="turn_off" for each piece of equipment at your desk.
Example: equipment_name="laptop_A", action="turn_off"

Being a good colleague means leaving your workspace ready for tomorrow!
"""

    # Meeting host equipment context
    if meeting_context:
        host_equipment_context = get_meeting_host_equipment_context(
            agent_id=agent.agent_id,
            meeting_context=meeting_context,
            checkpoint_reason=checkpoint_reason,
        )
        if host_equipment_context:
            guidance += f"\n{host_equipment_context}"

    return guidance


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

    # Format my confirmed meetings
    if my_meetings:
        my_meetings_text = "\n".join([
            f"- {m.get('title', 'Meeting')} ({m.get('start_datetime_iso', 'N/A')}) - role: {m.get('role', 'attendee')}"
            for m in my_meetings
        ])
    else:
        my_meetings_text = "No confirmed meetings yet."

    # Format all calendar events
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
<identity>
{agent.get_identity_stable_set()}
</identity>

<typical_schedule>
{agent.get_schedule_info()}
</typical_schedule>

<today>
Date: {day} ({day_of_week})
</today>

<relevant_memories>
{memories_text}
</relevant_memories>

<confirmed_meetings>
{my_meetings_text}
NOTE: Do NOT respond to invitations for meetings already listed above.
Do NOT create new meetings that duplicate those listed above.
</confirmed_meetings>

<calendar_events>
{calendar_text}
</calendar_events>

<pending_invitations>
{invitations_text}
</pending_invitations>

<planning_guidance>
Based on who you are and your memories, plan your day.
Consider your typical schedule as a guideline, but you may adjust based on circumstances.

BREAKS AND MEETINGS:
- Do NOT schedule breaks (morning_break or afternoon_break) that overlap with your meetings
- Check your confirmed meetings above BEFORE setting break times
- If you have a meeting at 10:00, do NOT schedule morning_break at 10:00
- If you have a meeting at 15:00, do NOT schedule afternoon_break at 15:00
- Choose break times that are at least 30 minutes away from any meeting

LUNCH PLANNING:
- Similarly, avoid scheduling lunch during meetings
- Check your meetings when choosing your lunch time

CLOTHING:
Decide what you'll wear today. Consider:
- The weather forecast (outdoor temperature)
- Any meetings you have (formality)
- Your personal style and comfort preferences
Provide a description of your outfit and its warmth level (very_light, light, medium, warm, very_warm).

WORK ACTIVITIES:
Plan any activities that require you to use specific equipment or move to a different location:
- Photocopying documents (requires photocopier in shared_area)
- Filing paperwork, picking up prints
- Any other tasks that take you away from your desk

For each activity, specify what, when (HH:MM), where, and how long.
Example: "10:30 - Photocopy meeting handouts (photocopier in shared_area, 10 min)"

RETURN TO DESK:
After any break, lunch, or work activity away from your desk, you should return.
</planning_guidance>
""".strip()

    return prompt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Focal point generation
    "get_decision_focal_points",
    "get_planning_focal_points",
    # Meeting context
    "get_meeting_context",
    "get_meeting_host_equipment_context",
    # Formatting helpers
    "format_colleague_context",
    "format_device_state",
    "format_meetings_for_prompt",
    "format_pending_invitations_for_prompt",
    # Legacy monolithic prompt (to be deprecated)
    "format_step_prompt",
    "format_planning_prompt",
    "build_decision_context",
    # 6-step checkpoint prompts (new)
    "format_step1_prompt",
    "format_step2_prompt",
    "format_step3_prompt",
    "format_step4_prompt",
    "format_step5_prompt",
    "format_step6_prompt",
]

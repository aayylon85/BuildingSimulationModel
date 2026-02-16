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

if TYPE_CHECKING:
    from bsm.agents.generative_agent import GenerativeAgent
    from bsm.agents.skeleton import CalendarStore


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
            "commitment": ["plans I made with colleagues", "social commitments", "what I promised to do"],
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
<thermostat heating="{heating_setpoint}C" cooling="{cooling_setpoint}C" />
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
- Set lighting.action to turn_on, turn_off, adjust_brightness, or keep_current
- Desk lights respond immediately to your control.
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
            commitment_lines = ["You have committed to these activities with colleagues:"]
            for c in unfulfilled:
                # Handle both dict and object access
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

            commitment_lines.append(">>> HONOR YOUR COMMITMENTS - use take_break or go_out_for_break to fulfill these!")
            return "\n".join(commitment_lines)

    return "No pending social commitments."


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

    # Commitment guidance
    if checkpoint_reason.startswith("commitment:"):
        activity = checkpoint_reason.split(":", 1)[1] if ":" in checkpoint_reason else "activity"
        guidance += f"""

=== SOCIAL COMMITMENT ===
You have a commitment for: {activity}

ACTION REQUIRED: Honor your commitment!
- If it's a coffee/tea break: Use take_break with location="break_area" and the appropriate activity
- If it's lunch: Use go_to_lunch or go_out_for_lunch
- If it's with a colleague: Make sure to go to the agreed location

Check your social_commitments above for details on who you're meeting and where.
Your colleagues are counting on you - don't let them down!
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
    "get_decision_focal_points",
    "get_planning_focal_points",
    "get_meeting_context",
    "get_meeting_host_equipment_context",
    "format_colleague_context",
    "format_device_state",
    "format_meetings_for_prompt",
    "format_pending_invitations_for_prompt",
    "format_step_prompt",
    "format_planning_prompt",
    "build_decision_context",
]

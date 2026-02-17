"""
Memory operations module for generative agents.

Implements functions for recording various events to agent memory:
- record_decision_to_memory: Record step decisions
- record_plan_to_memory: Record daily plans
- record_conversation_to_memory: Record conversations
- record_agreement_to_memory: Record consultation agreements
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from bsm.agents.cognition.perception import get_importance
from bsm.agents.cognition.schemas import ConversationResult, ConsultationOutcome, ConversationCommitmentOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Commitment Deduplication Helpers
# ---------------------------------------------------------------------------

# Activity keywords that map to normalized activity types
ACTIVITY_KEYWORDS = {
    "coffee": ["coffee", "coffee break", "coffee run", "grab coffee", "get coffee", "coffee with"],
    "lunch": ["lunch", "eat lunch", "lunch break", "grab lunch", "lunch with", "lunch and"],
    "break": ["break", "quick break", "take a break", "short break", "rest"],
    "walk": ["walk", "walk outside", "go for a walk", "take a walk", "stroll",
             "outdoor walk", "lap outside", "stretch legs", "stretch our legs"],
    "chat": ["chat", "talk", "catch up", "quick chat"],
    "meeting": ["meeting", "meet", "meet up", "meetup"],
}

# Compound activity patterns - normalize to primary activity
# These patterns match activities like "lunch with outdoor walk" and normalize to "lunch"
COMPOUND_PATTERNS = [
    # lunch + walk combinations → lunch (primary activity)
    (re.compile(r'lunch.*walk|walk.*lunch', re.IGNORECASE), "lunch"),
    # coffee + walk combinations → coffee
    (re.compile(r'coffee.*walk|walk.*coffee', re.IGNORECASE), "coffee"),
    # break + walk combinations → break
    (re.compile(r'break.*walk|walk.*break', re.IGNORECASE), "break"),
    # lunch + chat combinations → lunch
    (re.compile(r'lunch.*chat|chat.*lunch', re.IGNORECASE), "lunch"),
    # coffee + chat combinations → coffee
    (re.compile(r'coffee.*chat|chat.*coffee', re.IGNORECASE), "coffee"),
]


def _normalize_activity(activity: str) -> str:
    """
    Normalize activity string to a canonical form for comparison.
    Handles compound activities by normalizing to the primary activity.

    Examples:
        "grab coffee" -> "coffee"
        "lunch with outdoor walk" -> "lunch"
        "walk outside and chat" -> "walk"

    Args:
        activity: Raw activity string (e.g., "grab coffee", "lunch with outdoor walk")

    Returns:
        Normalized activity type (e.g., "coffee", "lunch")
    """
    activity_lower = activity.lower().strip()

    # First check compound patterns - these take priority
    for pattern, canonical in COMPOUND_PATTERNS:
        if pattern.search(activity_lower):
            return canonical

    # Then check single-activity keywords
    for canonical, keywords in ACTIVITY_KEYWORDS.items():
        for kw in keywords:
            if kw in activity_lower or activity_lower in kw:
                return canonical

    # If no match, return the original lowercased
    return activity_lower


def _parse_commitment_time(time_str: str) -> Optional[Tuple[int, int]]:
    """
    Parse a commitment time string into (hour, minute) tuple.

    Args:
        time_str: Time string in HH:MM format or "needs_confirmation"

    Returns:
        Tuple of (hour, minute) or None if time is unspecified
    """
    if not time_str or time_str in ("needs_confirmation", "unspecified", ""):
        return None

    # Try to parse HH:MM format
    match = re.match(r'^(\d{1,2}):(\d{2})$', time_str.strip())
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return (hour, minute)

    return None


def _times_within_window(
    time1: Optional[Tuple[int, int]],
    time2: Optional[Tuple[int, int]],
    window_minutes: int = 15
) -> bool:
    """
    Check if two times are within a given window of each other.

    Args:
        time1: First time as (hour, minute) tuple or None
        time2: Second time as (hour, minute) tuple or None
        window_minutes: Maximum difference in minutes to consider as "same time"

    Returns:
        True if times are within window (or if either is None/unspecified)
    """
    # If either time is unspecified, consider them potentially matching
    if time1 is None or time2 is None:
        return True

    # Convert to minutes since midnight for easy comparison
    mins1 = time1[0] * 60 + time1[1]
    mins2 = time2[0] * 60 + time2[1]

    return abs(mins1 - mins2) <= window_minutes


def _find_similar_commitment(
    existing_commitments: List[Dict[str, Any]],
    new_activity: str,
    new_time: str,
    new_with_agents: List[str],
    window_minutes: int = 15
) -> Optional[int]:
    """
    Find an existing commitment similar to a new one being added.

    A commitment is considered similar if:
    - Activity type matches (after normalization)
    - Time is within the window (or either is unspecified)
    - At least one agent in common

    Args:
        existing_commitments: List of existing commitment dicts
        new_activity: Activity for the new commitment
        new_time: Time for the new commitment
        new_with_agents: Agent IDs for the new commitment
        window_minutes: Time window for considering times as "same"

    Returns:
        Index of similar commitment if found, None otherwise
    """
    new_activity_norm = _normalize_activity(new_activity)
    new_time_parsed = _parse_commitment_time(new_time)
    new_agents_set = set(new_with_agents)

    for idx, existing in enumerate(existing_commitments):
        # Skip already fulfilled commitments
        if existing.get("fulfilled", False):
            continue

        # Check activity match
        existing_activity_norm = _normalize_activity(existing.get("activity", ""))
        if existing_activity_norm != new_activity_norm:
            continue

        # Check time proximity
        existing_time_parsed = _parse_commitment_time(existing.get("time", ""))
        if not _times_within_window(existing_time_parsed, new_time_parsed, window_minutes):
            continue

        # Check agent overlap
        existing_agents = set(existing.get("with_agents", []))
        if not (existing_agents & new_agents_set):
            continue

        # Found a similar commitment
        return idx

    return None


def _merge_commitment(
    existing: Dict[str, Any],
    new_activity: str,
    new_time: str,
    new_with_agents: List[str],
    new_location: str,
) -> Dict[str, Any]:
    """
    Merge a new commitment into an existing one.

    Prefers more specific information (e.g., actual time over "needs_confirmation").

    Args:
        existing: Existing commitment dict
        new_activity: Activity from new commitment
        new_time: Time from new commitment
        new_with_agents: Agents from new commitment
        new_location: Location from new commitment

    Returns:
        Updated commitment dict
    """
    # Prefer more specific time
    existing_time = existing.get("time", "")
    if _parse_commitment_time(new_time) is not None:
        # New time is specific, use it
        if _parse_commitment_time(existing_time) is None:
            existing["time"] = new_time
        # If both are specific, prefer the new one (more recent agreement)
        else:
            existing["time"] = new_time

    # Merge agents (union)
    existing_agents = set(existing.get("with_agents", []))
    existing_agents.update(new_with_agents)
    existing["with_agents"] = list(existing_agents)

    # Update location if new one is more specific
    if new_location and new_location != existing.get("location"):
        existing["location"] = new_location

    # Track that this was updated
    existing["last_updated"] = datetime.now().isoformat()

    return existing

if TYPE_CHECKING:
    from bsm.agents.generative_agent import GenerativeAgent


# ---------------------------------------------------------------------------
# Decision and Plan Recording
# ---------------------------------------------------------------------------

async def record_decision_to_memory(
    agent: "GenerativeAgent",
    decision: Any,  # StepDecisions or OccupantStepDecision
    now: datetime,
) -> None:
    """
    Record a decision to the agent's memory stream.

    Supports both StepDecisions (structured category decisions) and
    OccupantStepDecision (legacy action list) for backward compatibility.

    NOTE: This function does NOT call agent.save(). The caller is responsible
    for calling agent.save() to persist the memory to disk. This is intentional
    to allow batching multiple memory operations before a single save.

    Args:
        agent: The generative agent
        decision: The decision that was made (StepDecisions or OccupantStepDecision)
        now: Current datetime
    """
    if not agent.memory_stream:
        return

    # Check if this is a StepDecisions (has category decisions) or OccupantStepDecision (has actions list)
    if hasattr(decision, 'thermostat') and hasattr(decision, 'lighting'):
        # StepDecisions - record each non-trivial decision with its reasoning
        decisions_made = []

        if decision.thermostat.action != "maintain_current":
            desc = f"Thermostat: {decision.thermostat.action} - {decision.thermostat.reasoning}"
            decisions_made.append(desc)

        if decision.lighting.action != "keep_current":
            desc = f"Lighting: {decision.lighting.action} - {decision.lighting.reasoning}"
            decisions_made.append(desc)

        # Process equipment_decisions list
        for eq_dec in decision.equipment_decisions:
            if eq_dec.action != "keep_current":
                desc = f"Equipment ({eq_dec.equipment_name}): {eq_dec.action} - {eq_dec.reasoning}"
                decisions_made.append(desc)

        if decision.location.action == "move":
            desc = f"Moved to {decision.location.destination}: {decision.location.reasoning}"
            decisions_made.append(desc)

        if decision.conversation.action == "initiate":
            desc = f"Started conversation: {decision.conversation.reasoning}"
            decisions_made.append(desc)

        if decision.plan_update.action == "update":
            desc = f"Updated plan: {decision.plan_update.reasoning}"
            decisions_made.append(desc)

        if not decisions_made:
            return  # No non-trivial decisions

        # Create combined memory with all decisions
        full_desc = f"At {now.strftime('%H:%M')}: " + "; ".join(decisions_made)
        importance = await get_importance(agent, full_desc, 5.0, use_llm=True)

        # Extract action types for predicate/object
        action_types = []
        if decision.thermostat.action != "maintain_current":
            action_types.append("thermostat")
        # Check equipment_decisions list for any non-trivial actions
        for eq_dec in decision.equipment_decisions:
            if eq_dec.action != "keep_current":
                action_types.append(f"equipment:{eq_dec.equipment_name}")
                break  # Only add one "equipment" to avoid duplication
        if decision.lighting.action != "keep_current":
            action_types.append("lighting")
        if decision.location.action == "move":
            action_types.append("move")

        agent.memory_stream.add_event(
            description=full_desc,
            subject="I",
            predicate="decided",
            obj=", ".join(action_types) if action_types else "no change",
            now=now,
            importance=importance,
        )
    else:
        # Legacy OccupantStepDecision with actions list
        actions = getattr(decision, 'actions', [])
        non_trivial = [a for a in actions if getattr(a, 'action_type', '') != 'no_op']

        if not non_trivial:
            return

        action_desc = ", ".join([getattr(a, 'action_type', 'unknown') for a in non_trivial])
        rationale = getattr(decision, 'brief_rationale', '')

        description = f"I decided to: {action_desc}"
        if rationale:
            description += f". Reason: {rationale}"

        # Use LLM to assess importance of this decision
        importance = await get_importance(agent, description, 5.0, use_llm=True)
        agent.memory_stream.add_event(
            description=description,
            subject="I",
            predicate="decided",
            obj=action_desc,
            now=now,
            importance=importance,
        )


def _get_plan_attr(obj: Any, key: str, default: Any = None) -> Any:
    """Get attribute from object or dict, with default fallback."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _format_plan_narrative(plan: Any, date: datetime) -> str:
    """
    Convert DailyPlan (object or dict) to natural language narrative for memory storage.

    Args:
        plan: The DailyPlan object or dict
        date: The date of the plan

    Returns:
        Natural language description of the plan
    """
    parts = [f"My plan for {date.strftime('%A, %B %d')}:"]

    # Arrival time
    arrival = _get_plan_attr(plan, 'intended_arrival_time') or _get_plan_attr(plan, 'actual_arrival_time')
    if arrival:
        parts.append(f"Arrive around {arrival}.")

    # Meetings
    meetings = _get_plan_attr(plan, 'meetings', [])
    if meetings:
        meeting_strs = []
        for m in meetings:
            title = _get_plan_attr(m, 'title', 'meeting')
            start = _get_plan_attr(m, 'start_time')
            if start:
                # Handle both ISO datetime and HH:MM format
                if hasattr(start, 'strftime'):
                    start = start.strftime('%H:%M')
                elif 'T' in str(start):
                    start = str(start).split('T')[1][:5]
                meeting_strs.append(f"{title} at {start}")
            else:
                meeting_strs.append(title)
        parts.append(f"Meetings: {', '.join(meeting_strs)}.")

    # Morning break
    morning_break = _get_plan_attr(plan, 'morning_break')
    if morning_break:
        activity = _get_plan_attr(morning_break, 'activity', 'break')
        pref_time = _get_plan_attr(morning_break, 'preferred_time')
        if pref_time:
            parts.append(f"Morning {activity} around {pref_time}.")

    # Lunch
    lunch = _get_plan_attr(plan, 'lunch_plan')
    if lunch:
        location = _get_plan_attr(lunch, 'location', 'desk')
        time = _get_plan_attr(lunch, 'time')
        if time:
            parts.append(f"Lunch at {location} around {time}.")

    # Afternoon break
    afternoon_break = _get_plan_attr(plan, 'afternoon_break')
    if afternoon_break:
        activity = _get_plan_attr(afternoon_break, 'activity', 'break')
        pref_time = _get_plan_attr(afternoon_break, 'preferred_time')
        if pref_time:
            parts.append(f"Afternoon {activity} around {pref_time}.")

    # Social commitments (unfulfilled only)
    commitments = _get_plan_attr(plan, 'social_commitments', [])
    for c in commitments:
        if not _get_plan_attr(c, 'fulfilled', False):
            activity = _get_plan_attr(c, 'activity', 'activity')
            with_agents = _get_plan_attr(c, 'with_agents', [])
            time = _get_plan_attr(c, 'time')
            with_str = ', '.join(with_agents) if with_agents else "colleagues"
            if time and time not in ('needs_confirmation', 'unspecified'):
                parts.append(f"{activity.capitalize()} with {with_str} at {time}.")
            elif time == 'needs_confirmation':
                parts.append(f"{activity.capitalize()} with {with_str} (time to be confirmed).")

    # Departure time
    departure = _get_plan_attr(plan, 'intended_departure_time') or _get_plan_attr(plan, 'actual_departure_time')
    if departure:
        parts.append(f"Leave around {departure}.")

    return " ".join(parts)


async def record_plan_to_memory(
    agent: "GenerativeAgent",
    plan: Any,  # DailyPlan
    now: datetime,
) -> None:
    """
    Record a daily plan as a comprehensive thought in memory.

    The plan is stored as a single thought that can be semantically retrieved
    and updated when commitments change. If a plan thought already exists for
    today, it will be updated rather than creating a new one.

    NOTE: This function does NOT call agent.save(). The caller is responsible
    for calling agent.save() to persist the memory to disk.

    Args:
        agent: The generative agent
        plan: The daily plan that was created
        now: Current datetime
    """
    if not agent.memory_stream:
        return

    narrative = _format_plan_narrative(plan, now)
    date_str = now.strftime("%Y-%m-%d")
    keywords = {"daily_plan", "plan", date_str}

    # Check if plan thought already exists for today
    existing = agent.memory_stream.find_plan_thought(now)

    if existing:
        # Update existing thought
        agent.memory_stream.update_node(
            node_id=existing.node_id,
            description=narrative,
            keywords=keywords,
            now=now,
        )
        logger.debug(f"{agent.agent_id}: Updated plan thought for {date_str}")
    else:
        # Create new thought
        node = agent.memory_stream.add_thought(
            description=narrative,
            evidence_ids=[],
            now=now,
            importance=5.0,  # Plans are important for guiding behavior
        )
        # Add keywords to the new node
        node.keywords = keywords
        agent.memory_stream._update_keyword_index(node)
        logger.debug(f"{agent.agent_id}: Created plan thought for {date_str}")


async def update_plan_in_memory(
    agent: "GenerativeAgent",
    now: datetime,
) -> None:
    """
    Update the plan thought in memory to reflect current plan state.

    Called after commitments are added to the agent's daily plan. This
    re-generates the plan narrative and updates the existing thought node.

    Args:
        agent: The generative agent
        now: Current datetime
    """
    if not agent.memory_stream:
        return

    # Get current plan from agent's scratch (as dict)
    plan_dict = agent.scratch.get("daily_plan")
    if not plan_dict:
        logger.debug(f"{agent.agent_id}: No daily_plan in scratch, skipping update")
        return

    # Pass the dict directly - _format_plan_narrative handles both dicts and objects
    try:
        await record_plan_to_memory(agent, plan_dict, now)
    except Exception as e:
        logger.warning(f"{agent.agent_id}: Failed to update plan in memory: {e}")


# ---------------------------------------------------------------------------
# Conversation and Agreement Recording
# ---------------------------------------------------------------------------

async def record_conversation_to_memory(
    agent: "GenerativeAgent",
    conversation: ConversationResult,
    other_agent_id: str,
    now: datetime,
) -> None:
    """
    Record a conversation to an agent's memory stream.

    Creates a 'chat' type memory node with conversation summary.
    Also marks the conversation end for post-conversation reflection (Stanford-style).

    Args:
        agent: Agent whose memory to update
        conversation: The conversation result
        other_agent_id: ID of the other participant
        now: Current datetime
    """
    # Import here to avoid circular imports
    from bsm.agents.skeleton import SocialCommitment
    from bsm.agents.cognition.conversation import (
        extract_conversation_commitments,
        check_commitment_conflicts,
    )
    from bsm.agents.cognition.reflection import reflect_on_conversation

    if not agent.memory_stream:
        return

    other_first_name = other_agent_id.split("_")[0].capitalize()

    # Extract commitments FIRST so we can include them in the summary
    participant_ids = [agent.agent_id, other_agent_id]
    logger.debug(f"{agent.agent_id}: Starting commitment extraction...")
    commitments = await extract_conversation_commitments(
        conversation=conversation,
        participant_ids=participant_ids,
    )
    logger.debug(f"{agent.agent_id}: Received {len(commitments)} commitments from extraction")

    # Build enhanced description that includes commitment details
    topics_str = ", ".join(conversation.topics_discussed) if conversation.topics_discussed else "general topics"

    if commitments:
        # Build a summary that includes what was agreed
        commitment_parts = []
        for c in commitments:
            if c.time and c.time not in ("needs_confirmation", "unspecified", ""):
                commitment_parts.append(f"{c.activity} at {c.time}")
            else:
                commitment_parts.append(f"{c.activity} (time to be confirmed)")

        if commitment_parts:
            commitment_summary = f"Agreed to: {', '.join(commitment_parts)}."
        else:
            commitment_summary = ""

        description = (
            f"Had a conversation with {other_first_name} about {topics_str}. "
            f"{commitment_summary}"
        )
    else:
        # No commitments - use topics-based summary
        description = (
            f"Had a conversation with {other_first_name} about {topics_str}. "
            f"{conversation.summary}"
        )

    # Use LLM to assess importance of this conversation
    importance = await get_importance(agent, description, 5.0, use_llm=True)

    # Add as chat memory
    agent.memory_stream.add_chat(
        description=description,
        other_agent=other_agent_id,
        now=now,
        importance=importance,
    )

    # Track conversation end for compatibility, but we'll process reflection synchronously
    # (The pending_conversation_reflections queue is no longer needed since we process immediately)

    # Update relationship: familiarity increases with each conversation
    agent.update_relationship(
        other_agent_id=other_agent_id,
        familiarity_delta=0.05,  # Small increase per conversation
        sentiment_delta=0.0,     # Sentiment neutral for general conversations
    )

    # Add commitments to the agent's daily plan (with conflict checking)
    # Note: daily_plan may be a dict (from model_dump()) or a Pydantic model
    daily_plan = agent.get_daily_plan() if hasattr(agent, 'get_daily_plan') else None
    logger.debug(f"{agent.agent_id}: daily_plan exists: {daily_plan is not None}")

    # Convert Pydantic model to dict if needed
    if daily_plan and not isinstance(daily_plan, dict):
        if hasattr(daily_plan, 'model_dump'):
            daily_plan = daily_plan.model_dump()
        elif hasattr(daily_plan, '__dict__'):
            daily_plan = vars(daily_plan)
        else:
            logger.warning(f"{agent.agent_id}: daily_plan is type {type(daily_plan).__name__}, cannot convert to dict")
            daily_plan = None

    if daily_plan:
        logger.debug(f"{agent.agent_id}: daily_plan keys: {list(daily_plan.keys())}")
        logger.debug(f"{agent.agent_id}: existing social_commitments: {daily_plan.get('social_commitments', [])}")

    if commitments and daily_plan:
        # Ensure social_commitments list exists in the dict
        if "social_commitments" not in daily_plan:
            daily_plan["social_commitments"] = []

        for commitment in commitments:
            # Check for similar existing commitment (deduplication)
            existing_commitments = daily_plan["social_commitments"]
            similar_idx = _find_similar_commitment(
                existing_commitments=existing_commitments,
                new_activity=commitment.activity,
                new_time=commitment.time,
                new_with_agents=[other_agent_id],
                window_minutes=15,  # Commitments within 15 min are considered duplicates
            )

            if similar_idx is not None:
                # Update existing commitment instead of creating duplicate
                existing = existing_commitments[similar_idx]
                logger.info(
                    f"{agent.agent_id}: DEDUP - Found similar commitment '{existing.get('activity')}' "
                    f"at {existing.get('time')}, merging with '{commitment.activity}' at {commitment.time}"
                )
                _merge_commitment(
                    existing=existing,
                    new_activity=commitment.activity,
                    new_time=commitment.time,
                    new_with_agents=[other_agent_id],
                    new_location=commitment.location,
                )
                continue  # Skip adding new entry

            # No similar commitment found - create new entry
            social_commitment_dict = {
                "activity": commitment.activity,
                "time": commitment.time,
                "with_agents": [other_agent_id],
                "location": commitment.location,
                "source": "conversation",
                "fulfilled": False,
            }

            # Check for conflicts before adding
            social_commitment = SocialCommitment(**social_commitment_dict)
            conflicts = check_commitment_conflicts(agent, social_commitment, now)

            
            has_meeting_conflict = any("Conflicts with meeting" in c for c in conflicts)

            if has_meeting_conflict:
                # Flag the commitment so the agent can see and address the conflict
                meeting_conflicts = [c for c in conflicts if "Conflicts with meeting" in c]
                logger.warning(
                    f"{agent.agent_id}: Commitment '{commitment.activity}' at {commitment.time} "
                    f"conflicts with meeting - flagging: {meeting_conflicts[0]}"
                )
                print(
                    f"[COMMITMENT] {agent.agent_id}: {commitment.activity} at {commitment.time} conflicts with meeting - "
                    f"agent should address this"
                )
                social_commitment_dict["has_meeting_conflict"] = True
                social_commitment_dict["meeting_conflict_details"] = meeting_conflicts[0]

            if conflicts:
                # Non-meeting conflicts (e.g., other commitments) - still add but mark
                logger.info(f"{agent.agent_id}: CONFLICT detected for '{commitment.activity}' at {commitment.time}:")
                for conflict in conflicts:
                    logger.info(f"        - {conflict}")
                social_commitment_dict["has_conflict"] = True

            # Add to the daily_plan dict (it's stored as a dict, not a Pydantic model)
            daily_plan["social_commitments"].append(social_commitment_dict)
            logger.info(f"{agent.agent_id}: Added NEW commitment - {commitment.activity} at {commitment.time} with {other_agent_id}")

        logger.debug(f"{agent.agent_id}: After adding, social_commitments = {daily_plan.get('social_commitments', [])}")
    elif commitments and not daily_plan:
        logger.warning(f"{agent.agent_id}: {len(commitments)} commitments extracted but daily_plan is None!")
    elif not commitments:
        logger.debug(f"{agent.agent_id}: No commitments extracted from conversation")

    # Update the plan thought in memory if commitments were added
    # This keeps the plan thought synchronized with the agent's current commitments
    if commitments and daily_plan:
        await update_plan_in_memory(agent, now)

    # Process reflection SYNCHRONOUSLY after conversation ends
    # This generates planning thoughts and memos about the other person
    # Pass commitments so reflection can reference specific agreements made
    try:
        reflections = await reflect_on_conversation(
            agent=agent,
            other_agent_id=other_agent_id,
            now=now,
            commitments=commitments,  # Include extracted commitments for context
        )
        if reflections:
            logger.info(f"{agent.agent_id}: Generated {len(reflections)} reflection(s) from conversation with {other_agent_id}")
    except Exception as e:
        logger.warning(f"{agent.agent_id}: Reflection after conversation failed: {e}")


async def record_agreement_to_memory(
    agents: List["GenerativeAgent"],
    outcome: ConsultationOutcome,
    now: datetime,
) -> None:
    """
    Record a consultation agreement to ALL participating agents' memories.

    This ensures all agents remember the agreement and can respect it
    in future decisions.

    Args:
        agents: List of all participating agents
        outcome: The consultation outcome
        now: Current datetime
    """
    if not outcome.consensus_reached or not outcome.agreed_action:
        # Only record actual agreements
        return

    for agent in agents:
        if not agent.memory_stream:
            continue

        # Build description from agent's perspective
        other_names = [
            a.first_name for a in agents
            if a.agent_id != agent.agent_id
        ]
        others_str = " and ".join(other_names) if other_names else "colleagues"

        description = f"Agreed with {others_str}: {outcome.summary}"

        # Use LLM to assess importance of this agreement
        importance = await get_importance(agent, description, 7.0, use_llm=True)

        # Record as high-importance event (should influence future decisions)
        agent.memory_stream.add_event(
            description=description,
            subject="we",
            predicate="agreed",
            obj=outcome.agreed_action,
            now=now,
            importance=importance,
        )

        # Update relationships: agreement improves familiarity and sentiment
        for other_agent in agents:
            if other_agent.agent_id != agent.agent_id:
                agent.update_relationship(
                    other_agent_id=other_agent.agent_id,
                    familiarity_delta=0.03,  # Small familiarity boost
                    sentiment_delta=0.05,    # Positive sentiment from agreement
                )

        print(f"  [MEMORY] Recorded agreement for {agent.first_name}")


# ---------------------------------------------------------------------------
# Commitment Fulfillment Tracking
# ---------------------------------------------------------------------------

def mark_commitment_fulfilled(
    agent: "GenerativeAgent",
    activity_type: str,
    with_agent_id: Optional[str],
    at_time: datetime,
    window_minutes: int = 30,
) -> bool:
    """
    Mark a matching commitment as fulfilled.

    Searches for an unfulfilled commitment that matches the activity type,
    involves the specified agent (if provided), and is within the time window.

    Args:
        agent: The agent whose commitments to search
        activity_type: Type of activity (e.g., "coffee", "break", "lunch")
        with_agent_id: ID of the other agent involved (None if solo activity)
        at_time: When the activity was executed
        window_minutes: Time window for matching (default 30 min)

    Returns:
        True if a commitment was found and marked fulfilled, False otherwise
    """
    daily_plan = agent.get_daily_plan() if hasattr(agent, 'get_daily_plan') else None

    # Convert Pydantic model to dict if needed
    if daily_plan and not isinstance(daily_plan, dict):
        if hasattr(daily_plan, 'model_dump'):
            daily_plan = daily_plan.model_dump()
        elif hasattr(daily_plan, '__dict__'):
            daily_plan = vars(daily_plan)
        else:
            return False

    if not daily_plan:
        return False

    commitments = daily_plan.get("social_commitments", [])
    if not commitments:
        return False

    activity_norm = _normalize_activity(activity_type)
    execution_time = (at_time.hour, at_time.minute)

    for commitment in commitments:
        # Skip already fulfilled
        if commitment.get("fulfilled", False):
            continue

        # Check activity match
        if _normalize_activity(commitment.get("activity", "")) != activity_norm:
            continue

        # Check agent match (if specified)
        if with_agent_id:
            commitment_agents = commitment.get("with_agents", [])
            if with_agent_id not in commitment_agents:
                continue

        # Check time proximity
        commitment_time = _parse_commitment_time(commitment.get("time", ""))
        if not _times_within_window(commitment_time, execution_time, window_minutes):
            continue

        # Found matching commitment - mark as fulfilled
        commitment["fulfilled"] = True
        commitment["fulfilled_at"] = at_time.isoformat()
        logger.info(
            f"{agent.agent_id}: Marked commitment fulfilled - {commitment.get('activity')} "
            f"at {commitment.get('time')} with {commitment.get('with_agents')}"
        )
        return True

    return False


def get_unfulfilled_commitments(
    agent: "GenerativeAgent",
    activity_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Get all unfulfilled commitments for an agent.

    Args:
        agent: The agent to query
        activity_filter: Optional activity type to filter by (e.g., "coffee")

    Returns:
        List of unfulfilled commitment dicts
    """
    daily_plan = agent.get_daily_plan() if hasattr(agent, 'get_daily_plan') else None

    # Convert Pydantic model to dict if needed
    if daily_plan and not isinstance(daily_plan, dict):
        if hasattr(daily_plan, 'model_dump'):
            daily_plan = daily_plan.model_dump()
        elif hasattr(daily_plan, '__dict__'):
            daily_plan = vars(daily_plan)
        else:
            return []

    if not daily_plan:
        return []

    commitments = daily_plan.get("social_commitments", [])
    unfulfilled = [c for c in commitments if not c.get("fulfilled", False)]

    if activity_filter:
        filter_norm = _normalize_activity(activity_filter)
        unfulfilled = [
            c for c in unfulfilled
            if _normalize_activity(c.get("activity", "")) == filter_norm
        ]

    return unfulfilled


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "record_decision_to_memory",
    "record_plan_to_memory",
    "record_conversation_to_memory",
    "record_agreement_to_memory",
    # Commitment helpers
    "mark_commitment_fulfilled",
    "get_unfulfilled_commitments",
]

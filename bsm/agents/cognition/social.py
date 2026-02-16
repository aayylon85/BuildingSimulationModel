"""
Social interaction module for generative agents.

Implements social awareness and interaction triggers:
- Determine when agents should initiate conversations
- Generate relationship summaries for context
- Format colleague information for prompts
"""

from __future__ import annotations

import random
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from bsm.agents.generative_agent import GenerativeAgent


# ---------------------------------------------------------------------------
# Relationship Description Helpers (DRY up duplicate logic)
# ---------------------------------------------------------------------------

def _describe_familiarity(level: float) -> str:
    """
    Convert familiarity score to descriptive text.

    Args:
        level: Familiarity level (0.0 to 1.0)

    Returns:
        Description string
    """
    if level > 0.7:
        return "well"
    elif level > 0.4:
        return "somewhat"
    else:
        return "not well"


def _describe_sentiment(level: float) -> str:
    """
    Convert sentiment score to descriptive text.

    Args:
        level: Sentiment level (0.0 to 1.0)

    Returns:
        Description string
    """
    if level > 0.7:
        return "positive feelings"
    elif level > 0.5:
        return "generally gets along"
    elif level > 0.3:
        return "neutral feelings"
    else:
        return "some reservations"


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


# ---------------------------------------------------------------------------
# Relationship Summaries
# ---------------------------------------------------------------------------

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

    # Use shared helpers for description
    fam_desc = _describe_familiarity(familiarity)

    # Describe sentiment
    if sentiment > 0.7:
        sent_desc = "and has positive feelings toward them"
    elif sentiment > 0.5:
        sent_desc = "and generally gets along with them"
    elif sentiment > 0.3:
        sent_desc = "and has neutral feelings toward them"
    else:
        sent_desc = "and has some reservations about them"

    # Build full sentence
    if fam_desc == "well":
        fam_text = f"{agent.first_name} knows {other_agent_id} quite well"
    elif fam_desc == "somewhat":
        fam_text = f"{agent.first_name} is somewhat familiar with {other_agent_id}"
    else:
        fam_text = f"{agent.first_name} doesn't know {other_agent_id} very well"

    return f"{fam_text} {sent_desc}."


def format_colleague_context(
    agent: "GenerativeAgent",
    present_agent_ids: List[str],
    agents_by_location: Optional[Dict[str, List[str]]] = None,
) -> str:
    """
    Format information about colleagues present for the step prompt.

    Uses relationship_models from the agent's scratch to provide
    context about familiarity and sentiment with each colleague.

    Args:
        agent: The generative agent
        present_agent_ids: List of other agent IDs currently present
        agents_by_location: Optional dict mapping location to list of agent IDs there

    Returns:
        Formatted string describing colleagues present
    """
    if not present_agent_ids:
        return """COLLEAGUES: None currently in the building.
You are ALONE in the office right now. Do NOT mention, reference, or interact with anyone."""

    lines = ["=== COLLEAGUES IN BUILDING ==="]
    lines.append("The following people are the ONLY colleagues present:")
    for other_id in present_agent_ids:
        rel_info = agent.get_relationship_info(other_id)
        if rel_info:
            familiarity = rel_info.get("familiarity", 0.5)
            sentiment = rel_info.get("sentiment", 0.5)

            # Convert to descriptive terms with more detail
            if familiarity > 0.7:
                fam_desc = "you know them well"
            elif familiarity > 0.4:
                fam_desc = "you've worked together"
            else:
                fam_desc = "new colleague"

            if sentiment > 0.6:
                sent_desc = "good relationship"
            elif sentiment > 0.4:
                sent_desc = "neutral"
            else:
                sent_desc = "some tension"

            lines.append(f"  - {other_id} ({fam_desc}, {sent_desc})")
        else:
            lines.append(f"  - {other_id} (haven't met yet)")

    # Add location breakdown if provided
    if agents_by_location:
        lines.append("\nBy location:")
        for loc_name, agents in agents_by_location.items():
            others = [a for a in agents if a != agent.agent_id]
            if others:
                lines.append(f"  - {loc_name}: {', '.join(others)}")

    # Add explicit constraint to prevent hallucination
    lines.append("")
    lines.append("These are the ONLY people you can interact with today.")
    lines.append("Do NOT reference or mention any other names or colleagues.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "should_react",
    "generate_relationship_summary",
    "format_colleague_context",
    "_describe_familiarity",
    "_describe_sentiment",
]

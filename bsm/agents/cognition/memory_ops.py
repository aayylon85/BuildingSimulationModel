"""
Memory operations module for generative agents.

Implements functions for recording various events to agent memory:
- record_decision_to_memory: Record step decisions
- record_plan_to_memory: Record daily plans
- record_conversation_to_memory: Record conversations
- record_agreement_to_memory: Record consultation agreements
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, TYPE_CHECKING

from bsm.agents.cognition.perception import get_importance
from bsm.agents.cognition.schemas import ConversationResult, ConsultationOutcome

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


async def record_plan_to_memory(
    agent: "GenerativeAgent",
    plan: Any,  # DailyPlan
    now: datetime,
) -> None:
    """
    Record a daily plan to the agent's memory stream.

    NOTE: This function does NOT call agent.save(). The caller is responsible
    for calling agent.save() to persist the memory to disk. This is intentional
    to allow batching multiple memory operations before a single save.

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

    # Use LLM to assess importance of this plan
    importance = await get_importance(agent, description, 4.0, use_llm=True)
    agent.memory_stream.add_event(
        description=description,
        subject="I",
        predicate="planned",
        obj="my day",
        now=now,
        importance=importance,
    )


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

    if not agent.memory_stream:
        return

    # Build description
    other_first_name = other_agent_id.split("_")[0].capitalize()
    topics_str = ", ".join(conversation.topics_discussed) if conversation.topics_discussed else "general topics"

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

    # Mark for post-conversation reflection (Stanford-style)
    agent.mark_conversation_end(other_agent_id, now)

    # Update relationship: familiarity increases with each conversation
    agent.update_relationship(
        other_agent_id=other_agent_id,
        familiarity_delta=0.05,  # Small increase per conversation
        sentiment_delta=0.0,     # Sentiment neutral for general conversations
    )

    # Extract any commitments made during the conversation
    participant_ids = [agent.agent_id, other_agent_id]
    commitments = await extract_conversation_commitments(
        conversation=conversation,
        participant_ids=participant_ids,
    )

    # Add commitments to the agent's daily plan (with conflict checking)
    # Note: daily_plan is stored as a dict (from model_dump()), not a DailyPlan object
    daily_plan = agent.get_daily_plan() if hasattr(agent, 'get_daily_plan') else None
    if commitments and daily_plan:
        # Ensure social_commitments list exists in the dict
        if "social_commitments" not in daily_plan:
            daily_plan["social_commitments"] = []

        for commitment in commitments:
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
            if conflicts:
                print(f"[CONV] {agent.agent_id}: CONFLICT detected for '{commitment.activity}' at {commitment.time}:")
                for conflict in conflicts:
                    print(f"        - {conflict}")
                # Still add but mark it so the agent is aware of the conflict
                social_commitment_dict["has_conflict"] = True

            # Add to the daily_plan dict (it's stored as a dict, not a Pydantic model)
            daily_plan["social_commitments"].append(social_commitment_dict)
            print(f"[CONV] {agent.agent_id}: Added commitment - {commitment.activity} with {other_agent_id}")

    # Generate post-conversation reflections with commitments for action-oriented thoughts
    from bsm.agents.cognition.reflection import reflect_on_conversation
    await reflect_on_conversation(
        agent=agent,
        other_agent_id=other_agent_id,
        now=now,
        commitments=commitments if commitments else None,
    )


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
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "record_decision_to_memory",
    "record_plan_to_memory",
    "record_conversation_to_memory",
    "record_agreement_to_memory",
]

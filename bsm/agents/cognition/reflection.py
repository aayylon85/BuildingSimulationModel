"""
Reflection module for generative agents.

Implements the reflection stage of the cognitive loop:
- Generate reflections (thoughts) based on recent experiences
- Post-conversation reflection (Stanford-style planning and memo thoughts)
- Memory consolidation assessment
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

from agents import Agent, Runner, ModelSettings
from agents.agent_output import AgentOutputSchema

logger = logging.getLogger(__name__)

from bsm.agents.cognition.schemas import (
    ReflectionOutput,
    PlanningThoughtOutput,
    MemoThoughtOutput,
    ConsolidationBatch,
)
from bsm.agents.cognition.perception import get_importance, _build_simple_agent
from bsm.agents.cognition.retrieval import retrieve
from bsm.agents.memory.stream import MemoryNode
from bsm.agents.skeleton import DEFAULT_AGENT_MODEL

if TYPE_CHECKING:
    from bsm.agents.generative_agent import GenerativeAgent


# ---------------------------------------------------------------------------
# Memory Consolidation (LLM-Based)
# ---------------------------------------------------------------------------

def _build_consolidation_assessor_agent() -> Agent:
    """Build an agent for assessing which memories should become core identity."""
    instructions = """You are analyzing a person's memories from today to determine which
should become part of their permanent core identity/personality.

ONLY consolidate memories that would genuinely change WHO THE PERSON IS:
- Major realizations about their preferences (e.g., "I realized I work best in quiet")
- Significant relationship changes (e.g., "Bob and I had a serious disagreement")
- Important decisions about their work style (e.g., "I've decided to start taking breaks")
- Memorable experiences that shape identity (e.g., "I successfully led my first meeting")

DO NOT consolidate:
- Routine events (meetings attended, thermostat adjustments)
- Temporary states (felt cold, was busy)
- Minor observations (colleague arrived, equipment turned on)

Be very selective - most days should consolidate 0-2 memories.
Return the indices of memories that should be consolidated."""

    return _build_simple_agent(
        name="memory_consolidator",
        instructions=instructions,
        output_type=ConsolidationBatch,
        reasoning_effort="medium",
    )


async def assess_memories_for_consolidation(
    agent_name: str,
    identity: str,
    memories: List[tuple[int, str, float]],  # (index, description, importance)
) -> ConsolidationBatch:
    """
    Use LLM to assess which memories should be consolidated into core identity.

    Args:
        agent_name: Name of the agent
        identity: Current identity description
        memories: List of (index, description, importance) tuples

    Returns:
        ConsolidationBatch with indices to consolidate
    """
    if not memories:
        return ConsolidationBatch(memories_to_consolidate=[], reasoning="No memories to assess")

    memory_list = "\n".join([
        f"[{idx}] (importance: {imp:.1f}) {desc}"
        for idx, desc, imp in memories
    ])

    prompt = f"""Agent: {agent_name}
Current Identity: {identity}

Today's Memories:
{memory_list}

Which memories (if any) should become part of {agent_name}'s permanent core identity?
Return the indices of memories to consolidate, or empty list if none qualify."""

    try:
        assessor = _build_consolidation_assessor_agent()
        result = await Runner.run(assessor, prompt)
        return result.final_output
    except Exception as e:
        logger.error(f"LLM consolidation assessment failed: {e}")
        return ConsolidationBatch(memories_to_consolidate=[], reasoning=f"Assessment failed: {e}")


# ---------------------------------------------------------------------------
# Reflection Generation
# ---------------------------------------------------------------------------

def _build_reflection_agent() -> Agent:
    """Build an agent for generating reflections."""
    instructions = """You are an office worker reflecting on your experiences.

Generate ONE high-level insight about:
- Comfort preferences (temperature, lighting)
- Workspace habits
- Working patterns
- Interactions with colleagues

Write in first person (e.g., "I notice that...", "I prefer...", "I tend to...").
Focus on patterns that help make better decisions in the future.
Keep it concise (1-2 sentences)."""

    return _build_simple_agent(
        name="reflection_agent",
        instructions=instructions,
        output_type=ReflectionOutput,
        reasoning_effort="medium",
    )


async def _generate_llm_reflection(
    agent_name: str,
    recent_events: List[MemoryNode],
    focal_points: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Generate a reflection using LLM (Stanford-style).

    Args:
        agent_name: Name of the agent
        recent_events: List of recent high-importance events
        focal_points: Optional list of keywords/topics to focus reflection on

    Returns:
        Generated insight string, or None if generation failed
    """
    # Format recent observations as numbered statements
    statements = "\n".join([
        f"{i+1}. {event.description}"
        for i, event in enumerate(recent_events[:10])
    ])

    # Include focal points if provided (Stanford-style keyword-based reflection)
    focal_context = ""
    if focal_points:
        focal_str = ", ".join(focal_points)
        focal_context = f"\nKey themes to consider: {focal_str}\n"

    prompt = f"""You are {agent_name}, an office worker.

Recent observations and experiences:
{statements}
{focal_context}
Based on these observations, generate an insight."""

    try:
        agent = _build_reflection_agent()
        result = await Runner.run(agent, prompt)
        output: ReflectionOutput = result.final_output
        return output.insight if output.insight else None
    except Exception as e:
        logger.error(f"LLM reflection failed: {e}")
        return None


async def _generate_template_reflections(
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

            # Use LLM to assess importance (even for template reflections)
            importance = await get_importance(agent, reflection_text, 7.0, use_llm=True)
            thought = agent.memory_stream.add_thought(
                description=reflection_text,
                evidence_ids=[e.node_id for e in events[:3]],
                now=now,
                importance=importance,
            )
            reflections.append(thought)

    return reflections


async def reflect(
    agent: "GenerativeAgent",
    now: datetime,
    use_llm: bool = True,
) -> List[MemoryNode]:
    """
    Generate reflections (thoughts) based on recent experiences.

    Stanford-style LLM-generated reflections that derive high-level insights
    from recent observations. Triggered when accumulated importance of
    perceived events exceeds the reflection threshold.

    Uses keyword strength tracking (Stanford-style) to generate focal points
    for memory retrieval, ensuring reflections are grounded in the most
    salient concepts the agent has been encountering.

    Args:
        agent: The generative agent
        now: Current datetime
        use_llm: Whether to use LLM for reflection generation (default True)

    Returns:
        List of newly created thought MemoryNodes
    """
    if not agent.should_reflect():
        return []

    if not agent.memory_stream:
        return []

    # Reset trigger
    agent.reset_importance_trigger()

    # Get focal points from keyword strength tracking (Stanford-style)
    focal_points = agent.memory_stream.get_reflection_focal_points(top_n=3)

    if not focal_points:
        # Fallback to generic focal points if no keyword strengths
        focal_points = ["recent observations", "patterns in my behavior"]

    # Retrieve memories related to focal points
    retrieved = retrieve(agent, focal_points, now, n_count=10)

    # Get recent high-importance events (for evidence linking)
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
                focal_points=focal_points,
            )

            if insight:
                # Use LLM to assess importance of this reflection
                importance = await get_importance(agent, insight, 7.0, use_llm=True)
                thought = agent.memory_stream.add_thought(
                    description=insight,
                    evidence_ids=[e.node_id for e in recent_events[:5]],
                    now=now,
                    importance=importance,
                )
                reflections.append(thought)
                logger.info(f"{agent.name}: {insight}")

        except Exception as e:
            logger.error(f"LLM reflection failed for {agent.name}: {e}")
            # Fall back to template-based reflection
            reflections.extend(
                await _generate_template_reflections(agent, recent_events, now)
            )
    else:
        # Template-based fallback
        reflections.extend(
            await _generate_template_reflections(agent, recent_events, now)
        )

    # Persist keyword strengths after reflection (ensures reflection focal points
    # are stable across sessions)
    if reflections and agent.memory_stream:
        agent.memory_stream.save()

    return reflections


# ---------------------------------------------------------------------------
# Post-Conversation Reflection (Stanford-style)
# ---------------------------------------------------------------------------

def _build_planning_thought_agent() -> Agent:
    """Build an agent for generating planning thoughts from conversations."""
    instructions = """Generate ONE SPECIFIC, ACTIONABLE planning thought based on the conversation.

Your thought MUST include:
- What ACTION to take
- WHEN to do it (specific time if available, or relative timing like "after lunch")
- WHERE if relevant

GOOD examples:
- "I should head to the break room at 2:30 for coffee with Bob."
- "I need to prepare my notes before the 3pm project discussion with Alice."
- "Bob suggested lunch together - I'll meet him at the break area around 12:15."

BAD examples (too vague - DO NOT generate these):
- "I should keep in mind what we discussed."
- "The conversation was productive."
- "I'll follow up on our discussion."

Write in first person. Be specific about times, locations, and actions."""

    return _build_simple_agent(
        name="planning_thought_agent",
        instructions=instructions,
        output_type=PlanningThoughtOutput,
        reasoning_effort="medium",
    )


async def _generate_conversation_planning_thought(
    agent_name: str,
    other_name: str,
    conversation_memories: List[MemoryNode],
    commitments: Optional[List[Any]] = None,
) -> Optional[str]:
    """Generate a planning thought based on conversation.

    Args:
        agent_name: Name of the agent reflecting
        other_name: Name of the conversation partner
        conversation_memories: Recent memories about the conversation
        commitments: Optional list of ConversationCommitmentOutput from the conversation
    """
    statements = "\n".join([
        f"- {mem.description}"
        for mem in conversation_memories[:5]
    ])

    # Format commitments if provided
    commitments_section = ""
    if commitments:
        commitment_lines = []
        for c in commitments:
            line = f"- {c.activity}"
            if hasattr(c, 'time') and c.time != "unspecified":
                line += f" at {c.time}"
            if hasattr(c, 'location') and c.location != "unspecified":
                line += f" in {c.location}"
            commitment_lines.append(line)
        if commitment_lines:
            commitments_section = f"""

<commitments_made>
These are the specific agreements/plans made during the conversation:
{chr(10).join(commitment_lines)}
</commitments_made>"""

    prompt = f"""You are {agent_name}, an office worker who just finished talking with {other_name}.

<conversation_summary>
{statements}
</conversation_summary>{commitments_section}

Generate a SPECIFIC planning thought about what you should do and WHEN."""

    try:
        agent = _build_planning_thought_agent()
        result = await Runner.run(agent, prompt)
        output: PlanningThoughtOutput = result.final_output
        return output.thought if output.thought else None
    except Exception as e:
        logger.error(f"Planning thought generation failed: {e}")
        return None


def _build_memo_thought_agent() -> Agent:
    """Build an agent for generating memo thoughts about other people."""
    instructions = """Generate ONE ACTIONABLE observation about the other person that will be useful for future interactions.

Focus on information that helps you work better with them:
- Their preferences (temperature, scheduling, work style)
- Current projects or workload they mentioned
- Topics they're interested in or concerned about
- Their schedule patterns (when they take breaks, prefer meetings)

GOOD examples:
- "Bob prefers morning meetings and is currently stressed about the quarterly report deadline."
- "Alice mentioned she likes her tea at 10am - good to know for future break invitations."
- "Charlie is working from home on Fridays and prefers async communication."

BAD examples (too generic - DO NOT generate these):
- "Bob seems like a nice person."
- "Alice was friendly during our chat."
- "We had a good conversation."

Write in first person. Be specific about what you learned."""

    return _build_simple_agent(
        name="memo_thought_agent",
        instructions=instructions,
        output_type=MemoThoughtOutput,
        reasoning_effort="medium",
    )


async def _generate_conversation_memo_thought(
    agent_name: str,
    other_name: str,
    conversation_memories: List[MemoryNode],
) -> Optional[str]:
    """Generate a memo thought about the other person."""
    statements = "\n".join([
        f"- {mem.description}"
        for mem in conversation_memories[:5]
    ])

    prompt = f"""You are {agent_name}, an office worker who just finished talking with {other_name}.

What you remember from the conversation:
{statements}

Generate an observation about {other_name}."""

    try:
        agent = _build_memo_thought_agent()
        result = await Runner.run(agent, prompt)
        output: MemoThoughtOutput = result.final_output
        return output.observation if output.observation else None
    except Exception as e:
        logger.error(f"Memo thought generation failed: {e}")
        return None


def _to_timestamp(created_value: Union[int, float, datetime, str]) -> float:
    """Safely convert a created value to a Unix timestamp.

    Handles multiple possible types that memory nodes may store:
    - float/int: Already a timestamp, return as-is
    - datetime: Convert to timestamp
    - str: Try to parse as ISO format datetime

    Returns:
        float timestamp, or 0.0 if conversion fails
    """
    if isinstance(created_value, (int, float)):
        return float(created_value)
    elif isinstance(created_value, datetime):
        return created_value.timestamp()
    elif isinstance(created_value, str):
        try:
            return datetime.fromisoformat(created_value).timestamp()
        except (ValueError, TypeError):
            return 0.0
    return 0.0


async def reflect_on_conversation(
    agent: "GenerativeAgent",
    other_agent_id: str,
    now: datetime,
    commitments: Optional[List[Any]] = None,
) -> List[MemoryNode]:
    """
    Generate reflections after a conversation (Stanford-style).

    Creates two types of thoughts:
    1. Planning thoughts: What should I do based on this conversation?
    2. Memo thoughts: What did I learn about the other person?

    Args:
        agent: The generative agent
        other_agent_id: ID of the agent the conversation was with
        now: Current simulation datetime
        commitments: Optional list of ConversationCommitmentOutput extracted from conversation

    Returns:
        List of newly created thought MemoryNodes
    """
    if not agent.memory_stream:
        return []

    # Deduplication: Track conversations we've already reflected on this hour
    # Store in agent.scratch so it persists across reloads
    conv_key = f"conv_{other_agent_id}_{now.strftime('%Y%m%d_%H')}"
    reflected_set = set(agent.scratch.get('_reflected_conversations', []))

    if conv_key in reflected_set:
        logger.debug(f"{agent.name}: Skipping duplicate reflection for {other_agent_id} (same hour)")
        return []

    # Check for similar existing planning thoughts to avoid memory clutter
    if agent.memory_stream:
        all_nodes = agent.memory_stream.nodes
        now_ts = now.timestamp()
        recent_thoughts = [
            n for n in all_nodes
            if n.node_type == "thought"
            and (now_ts - _to_timestamp(n.created)) < 7200  # Last 2 hours
        ]
        # Count planning thoughts about this person
        other_name_lower = other_agent_id.lower()
        similar_count = sum(
            1 for t in recent_thoughts
            if other_name_lower in t.description.lower()
            and any(w in t.description.lower() for w in ['should', 'will meet', 'at ', 'around '])
        )
        if similar_count >= 2:
            logger.debug(f"{agent.name}: Skipping reflection - already have {similar_count} planning thoughts about {other_agent_id}")
            return []

    # Mark this conversation as reflected (persisted in scratch)
    reflected_set.add(conv_key)
    agent.scratch['_reflected_conversations'] = list(reflected_set)

    reflections: List[MemoryNode] = []

    # Retrieve recent conversation memories with this person
    focal_points = [
        f"my conversation with {other_agent_id}",
        f"what {other_agent_id} mentioned",
        f"agreements with {other_agent_id}",
    ]
    retrieved = retrieve(agent, focal_points, now, n_count=10)

    # Flatten retrieved memories
    conv_memories: List[MemoryNode] = []
    for fp, nodes in retrieved.items():
        conv_memories.extend(nodes)

    if not conv_memories:
        return []

    # Generate planning thought (include commitments for specific action planning)
    try:
        planning_insight = await _generate_conversation_planning_thought(
            agent_name=agent.name,
            other_name=other_agent_id,
            conversation_memories=conv_memories[:5],
            commitments=commitments,
        )

        if planning_insight:
            # Use LLM to assess importance of this planning thought
            importance = await get_importance(agent, planning_insight, 6.0, use_llm=True)
            thought = agent.memory_stream.add_thought(
                description=planning_insight,
                evidence_ids=[m.node_id for m in conv_memories[:3] if m.node_id >= 0],
                now=now,
                importance=importance,
            )
            reflections.append(thought)
            logger.info(f"{agent.name} planning: {planning_insight}")

    except Exception as e:
        logger.error(f"Planning thought failed for {agent.name}: {e}")

    # Generate memo thought about the other person
    try:
        memo_insight = await _generate_conversation_memo_thought(
            agent_name=agent.name,
            other_name=other_agent_id,
            conversation_memories=conv_memories[:5],
        )

        if memo_insight:
            # Use LLM to assess importance of this memo thought
            importance = await get_importance(agent, memo_insight, 5.0, use_llm=True)
            thought = agent.memory_stream.add_thought(
                description=memo_insight,
                evidence_ids=[m.node_id for m in conv_memories[:3] if m.node_id >= 0],
                now=now,
                importance=importance,
            )
            reflections.append(thought)
            logger.info(f"{agent.name} memo: {memo_insight}")

    except Exception as e:
        logger.error(f"Memo thought failed for {agent.name}: {e}")

    return reflections


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "reflect",
    "reflect_on_conversation",
    "assess_memories_for_consolidation",
]

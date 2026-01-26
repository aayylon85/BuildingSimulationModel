"""
cognitive_modules.py

Cognitive modules for generative agents.
Implements the cognitive loop: Perceive -> Retrieve -> Plan -> Reflect -> Act

Based on Stanford Generative Agents architecture, adapted for building simulation.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

from agents import Agent, Runner, ModelSettings
from agents.agent_output import AgentOutputSchema

from bsm.agents.memory.stream import MemoryNode
from bsm.agents.skeleton import DEFAULT_AGENT_MODEL

if TYPE_CHECKING:
    from bsm.agents.generative_agent import GenerativeAgent
    from bsm.agents.skeleton import CalendarStore


# ---------------------------------------------------------------------------
# Pydantic Output Schemas for Cognitive Module Agents
# ---------------------------------------------------------------------------

class ReflectionOutput(BaseModel):
    """Output schema for reflection generation."""
    insight: str = Field(description="The reflection insight in first person")


class PlanningThoughtOutput(BaseModel):
    """Output schema for conversation planning thought."""
    thought: str = Field(description="What to do or keep in mind from the conversation")


class MemoThoughtOutput(BaseModel):
    """Output schema for memo thought about another person."""
    observation: str = Field(description="Observation about the other person")


class ImportanceAssessment(BaseModel):
    """Output schema for LLM-based importance scoring."""
    importance: int = Field(ge=1, le=10, description="Importance score from 1-10")
    reasoning: str = Field(description="Brief reasoning for the score")


# ---------------------------------------------------------------------------
# LLM-Based Importance Scoring
# ---------------------------------------------------------------------------

def _build_importance_assessor_agent() -> Agent:
    """Build an agent for assessing event importance."""
    instructions = """Rate the importance of an event for a person on a scale of 1-10.

1 = purely mundane (routine tasks, making coffee)
5 = moderately significant (regular meetings, normal work tasks)
10 = extremely significant (major work milestone, serious conflict)

Consider:
- How does this relate to their work and responsibilities?
- Does it affect their comfort or wellbeing?
- Does it involve important relationships?
- Is it likely to be remembered?

Be thoughtful but don't overthink - most routine events should be 3-5."""

    return Agent(
        name="importance_assessor",
        instructions=instructions,
        model=DEFAULT_AGENT_MODEL,
        model_settings=ModelSettings(reasoning_effort="medium"),  # Importance assessment requires thoughtful evaluation
        output_type=AgentOutputSchema(ImportanceAssessment),
    )


async def assess_event_importance_llm(
    agent: "GenerativeAgent",
    event_description: str,
) -> int:
    """
    Assess importance of an event using LLM (Stanford generative_agents style).

    Args:
        agent: The agent perceiving the event
        event_description: Description of the event

    Returns:
        Importance score from 1-10
    """
    identity = agent.get_identity_stable_set() if hasattr(agent, 'get_identity_stable_set') else agent.name

    prompt = f"""Here is a brief description of {agent.name}:
{identity}

Rate the importance of this event for {agent.name}:
Event: {event_description}"""

    try:
        assessor = _build_importance_assessor_agent()
        result = await Runner.run(assessor, prompt)
        output: ImportanceAssessment = result.final_output
        return max(1, min(10, output.importance))  # Clamp to 1-10
    except Exception as e:
        print(f"[IMPORTANCE] LLM importance assessment failed: {e}")
        return 5  # Default to moderate importance


async def get_importance(
    agent: "GenerativeAgent",
    event_description: str,
    base_importance: float,
    use_llm: bool = False,
) -> float:
    """
    Get importance score - uses LLM or returns base_importance based on config.

    Args:
        agent: The agent perceiving the event
        event_description: Description of the event
        base_importance: Hardcoded importance value (used if use_llm=False)
        use_llm: Whether to use LLM-based scoring

    Returns:
        Importance score (float)
    """
    if use_llm:
        return float(await assess_event_importance_llm(agent, event_description))
    return base_importance


# ---------------------------------------------------------------------------
# Category-Specific Memory Retrieval for Structured Decisions
# ---------------------------------------------------------------------------

DECISION_CATEGORIES = {
    "thermostat": [
        "temperature preference",
        "feeling hot or cold",
        "thermostat adjustment",
        "comfort level",
        "thermal comfort",
        "prefer cooler",
        "prefer warmer",
    ],
    "lighting": [
        "lighting preference",
        "brightness",
        "natural light",
        "eye strain",
        "desk lamp",
        "dim light",
        "bright light",
    ],
    "equipment": [
        "kettle",
        "coffee machine",
        "printer",
        "equipment use",
        "kitchen appliance",
        "photocopier",
    ],
    "location": [
        "desk preference",
        "meeting room",
        "break room",
        "quiet space",
        "collaboration area",
        "favorite desk",
        "prefer window",
    ],
    "conversation": [
        "colleague interaction",
        "social",
        "discussion",
        "collaboration",
        "team communication",
        "chat with",
        "talk about",
    ],
    "break": [
        "tea preference",
        "coffee preference",
        "break habit",
        "snack",
        "rest",
        "stretch",
        "morning tea",
        "afternoon coffee",
    ],
    "meeting": [
        "meeting equipment",
        "presentation",
        "projector",
        "video call",
        "screen sharing",
        "conference room",
    ],
    "lunch": [
        "lunch habit",
        "eat at desk",
        "break room lunch",
        "go out for lunch",
        "lunch preference",
        "meal time",
    ],
}


def retrieve_for_category(
    agent: "GenerativeAgent",
    category: str,
    current_context: str,
    now: datetime,
    k: int = 5,
) -> List[MemoryNode]:
    """
    Retrieve memories relevant to a specific decision category.

    Args:
        agent: The generative agent
        category: Decision category (thermostat, lighting, equipment, etc.)
        current_context: Current situation description to add as focal point
        now: Current datetime
        k: Number of memories to retrieve

    Returns:
        List of relevant MemoryNodes
    """
    # Get focal points for this category
    focal_points = DECISION_CATEGORIES.get(category, []).copy()

    # Add current context as a focal point
    if current_context:
        focal_points.append(current_context)

    if not focal_points:
        return []

    # Use existing retrieve function but flatten results
    retrieved = retrieve(
        agent=agent,
        focal_points=focal_points,
        now=now,
        n_count=k,
        core_memory_count=3,  # Include some core memories
    )

    # Flatten and deduplicate by node_id
    seen_ids = set()
    all_memories = []
    for memories in retrieved.values():
        for mem in memories:
            if mem.node_id not in seen_ids:
                seen_ids.add(mem.node_id)
                all_memories.append(mem)

    # Sort by combined score (recency * relevance * importance) and take top k
    # Memories are already scored by retrieve(), so just take top k unique
    return all_memories[:k]


def format_memories_for_decision(
    memories: List[MemoryNode],
    category: str,
) -> str:
    """
    Format retrieved memories for injection into decision prompt.

    Args:
        memories: List of retrieved MemoryNodes
        category: The decision category for context

    Returns:
        Formatted string of memories
    """
    if not memories:
        return f"No relevant memories about {category}."

    lines = [f"Relevant memories about {category}:"]
    for mem in memories:
        importance_str = f" (importance: {mem.importance:.1f})" if mem.importance else ""
        lines.append(f"- {mem.description}{importance_str}")

    return "\n".join(lines)


def retrieve_all_decision_memories(
    agent: "GenerativeAgent",
    sim_state: Dict[str, Any],
    now: datetime,
    categories: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Retrieve memories for all decision categories.

    Args:
        agent: The generative agent
        sim_state: Current simulation state (used to build context)
        now: Current datetime
        categories: Optional list of categories to retrieve for (default: all)

    Returns:
        Dict mapping category -> formatted memory string
    """
    if categories is None:
        categories = list(DECISION_CATEGORIES.keys())

    # Build current context from sim_state
    current_location = sim_state.get("current_location", "unknown")
    current_temp = sim_state.get("indoor_temp", "unknown")
    context_parts = [
        f"Currently at {current_location}",
        f"Temperature is {current_temp}°C" if current_temp != "unknown" else "",
    ]
    current_context = ". ".join(p for p in context_parts if p)

    result = {}
    for category in categories:
        memories = retrieve_for_category(
            agent=agent,
            category=category,
            current_context=current_context,
            now=now,
            k=5,
        )
        result[category] = format_memories_for_decision(memories, category)

    return result


# ---------------------------------------------------------------------------
# End-of-Day Memory Consolidation (LLM-Based)
# ---------------------------------------------------------------------------

class ConsolidationBatch(BaseModel):
    """Output schema for memory consolidation assessment."""
    memories_to_consolidate: List[int] = Field(
        default_factory=list,
        description="Indices of memories to consolidate into core identity"
    )
    reasoning: str = Field(description="Brief explanation of why these memories matter")


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

    return Agent(
        name="memory_consolidator",
        instructions=instructions,
        model=DEFAULT_AGENT_MODEL,
        model_settings=ModelSettings(reasoning_effort="medium"),
        output_type=AgentOutputSchema(ConsolidationBatch),
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
        print(f"[CONSOLIDATION] LLM assessment failed: {e}")
        return ConsolidationBatch(memories_to_consolidate=[], reasoning=f"Assessment failed: {e}")


async def perceive(
    agent: "GenerativeAgent",
    sim_state: Dict[str, Any],
    now: datetime,
    use_llm_importance: bool = True,
) -> List[MemoryNode]:
    """
    Convert simulation state into perceived events.

    This is the first step in the cognitive loop. The agent observes
    the current state and creates memory events for notable observations.

    Stanford-style attention bandwidth: The agent can only perceive a limited
    number of events per timestep. Events are prioritized by salience/importance
    and only the top N (perception_bandwidth) are actually perceived.

    Args:
        agent: The generative agent
        sim_state: Current simulation state dict
        now: Current simulation datetime
        use_llm_importance: Whether to use LLM to assess importance (default True)

    Returns:
        List of newly created MemoryNode events
    """
    if not agent.memory_stream:
        return []

    # Collect potential perceptions with priority scores
    # Format: (priority, description, subject, predicate, obj, base_importance)
    from typing import Callable, Tuple
    PotentialPerception = Tuple[float, str, str, str, str, float]
    potential_perceptions: List[PotentialPerception] = []

    # Perceive indoor temperature as raw data - let agent reason about comfort
    temp = sim_state.get("indoor_temp_c", 21.0)
    if not agent.recently_perceived("indoor_temp", now, within_minutes=15):
        potential_perceptions.append((
            5.0,  # Medium priority - factual observation
            f"The room temperature is {temp:.1f}C",
            "room", "temperature is", f"{temp:.1f}C", 3.0
        ))

    # Perceive other occupants present
    other_occupants = sim_state.get("other_occupants_present", [])
    for other_id in other_occupants:
        if not agent.recently_perceived(other_id, now, within_minutes=30):
            # Haven't noticed them recently - medium priority
            potential_perceptions.append((
                5.0,
                f"{other_id} is in the office",
                other_id, "is present in", "office", 4.0
            ))

    # Track just_arrived state (will clear flag at end of function)
    just_arrived = agent.has_just_arrived()

    # Perceive equipment state continuously (lower priority - let agent decide based on needs)
    # Only perceive if not recently noticed to avoid spam
    equipment_status = sim_state.get("equipment_status", {})
    equipment_items = equipment_status.get("items", {})
    for equipment_name, is_on in equipment_items.items():
        # Only perceive equipment that's off and hasn't been recently noticed
        if not is_on and not agent.recently_perceived(f"equipment_{equipment_name}", now, within_minutes=30):
            potential_perceptions.append((
                3.0,  # Lower priority - let agent decide based on their needs
                f"The {equipment_name} is currently off",
                "I", "notice", f"{equipment_name} is off", 3.0
            ))

    # Perceive lighting conditions (also continuous, low priority)
    lighting = sim_state.get("lighting_conditions", {})
    natural_light = lighting.get("natural_light_level", "moderate")
    desk_light_on = lighting.get("desk_light_on", False)

    if natural_light in ["dim", "dark"] and not desk_light_on:
        if not agent.recently_perceived("lighting_dim", now, within_minutes=30):
            potential_perceptions.append((
                3.0,  # Lower priority - observational
                f"The lighting is {natural_light} and the desk light is off",
                "workspace", "has", f"{natural_light} lighting", 3.0
            ))

    # Perceive outdoor conditions as raw data
    weather_desc = sim_state.get("weather_description", "")
    outdoor_temp = sim_state.get("outdoor_temp_c", 10.0)

    # Outdoor temperature - separate from indoor for comparison
    if not agent.recently_perceived("outdoor_temp", now, within_minutes=30):
        potential_perceptions.append((
            4.0,  # Medium priority - useful context for comfort decisions
            f"Outside it is {outdoor_temp:.1f}C",
            "outdoor", "temperature is", f"{outdoor_temp:.1f}C", 3.0
        ))

    # Weather conditions
    if weather_desc and not agent.recently_perceived("weather", now, within_minutes=60):
        potential_perceptions.append((
            3.5,
            f"The weather outside is {weather_desc}",
            "weather", "is", weather_desc, 3.0
        ))

    # Perceive equipment at current location (higher priority when at non-desk locations)
    current_location = sim_state.get("current_location", "desk_area")
    location_equipment = sim_state.get("location_equipment", [])

    if location_equipment and current_location != "desk_area":
        # At a non-desk location (break_area, meeting_room, etc.) - perceive available equipment
        location_key = f"location_equipment_{current_location}"
        if not agent.recently_perceived(location_key, now, within_minutes=15):
            equipment_names = [eq["name"] for eq in location_equipment]
            equipment_states = [
                f"{eq['name']} ({'ON' if eq.get('is_on') else 'available'})"
                for eq in location_equipment
            ]
            if equipment_names:
                potential_perceptions.append((
                    5.5,  # Higher priority - new location equipment is interesting
                    f"At {current_location}, I notice: {', '.join(equipment_states)}",
                    "I", "notice equipment at", current_location, 4.0
                ))

    # Sort by priority (highest first) and limit to perception bandwidth
    bandwidth = agent.get_perception_bandwidth()
    potential_perceptions.sort(key=lambda x: x[0], reverse=True)

    # Create only the top N perceptions with LLM-assessed importance
    perceived_events: List[MemoryNode] = []
    for priority, desc, subj, pred, obj, base_imp in potential_perceptions[:bandwidth]:
        # Assess importance using LLM if enabled
        importance = await get_importance(agent, desc, base_imp, use_llm=use_llm_importance)

        event = agent.memory_stream.add_event(
            description=desc,
            subject=subj,
            predicate=pred,
            obj=obj,
            now=now,
            importance=importance,
        )
        perceived_events.append(event)

    # Clear just_arrived flag if it was set (always do this even if filtered)
    if just_arrived:
        agent.clear_just_arrived()

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

    Uses keyword strength tracking (Stanford-style) to generate focal points
    for memory retrieval, ensuring reflections are grounded in the most
    salient concepts the agent has been encountering.

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
                model=reflection_model,
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

    # Persist keyword strengths after reflection (ensures reflection focal points
    # are stable across sessions)
    if reflections and agent.memory_stream:
        agent.memory_stream.save()

    return reflections


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

    return Agent(
        name="reflection_agent",
        instructions=instructions,
        model=DEFAULT_AGENT_MODEL,
        model_settings=ModelSettings(reasoning_effort="medium"),
        output_type=AgentOutputSchema(ReflectionOutput),
    )


async def _generate_llm_reflection(
    agent_name: str,
    recent_events: List[MemoryNode],
    focal_points: Optional[List[str]] = None,
    model: str = None,  # Ignored, uses DEFAULT_AGENT_MODEL
) -> Optional[str]:
    """
    Generate a reflection using LLM (Stanford-style).

    Args:
        agent_name: Name of the agent
        recent_events: List of recent high-importance events
        focal_points: Optional list of keywords/topics to focus reflection on
        model: Ignored (uses DEFAULT_AGENT_MODEL)

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
        print(f"[REFLECTION] LLM reflection failed: {e}")
        return None


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


# ---------------------------------------------------------------------------
# Post-Conversation Reflection (Stanford-style)
# ---------------------------------------------------------------------------

async def reflect_on_conversation(
    agent: "GenerativeAgent",
    other_agent_id: str,
    now: datetime,
    reflection_model: str = "gpt-4o-mini",
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
        reflection_model: Model to use for LLM reflection

    Returns:
        List of newly created thought MemoryNodes
    """
    if not agent.memory_stream:
        return []

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

    # Generate planning thought
    try:
        planning_insight = await _generate_conversation_planning_thought(
            agent_name=agent.name,
            other_name=other_agent_id,
            conversation_memories=conv_memories[:5],
            model=reflection_model,
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
            print(f"[ConvReflect] {agent.name} planning: {planning_insight}")

    except Exception as e:
        print(f"[ConvReflect] Planning thought failed for {agent.name}: {e}")

    # Generate memo thought about the other person
    try:
        memo_insight = await _generate_conversation_memo_thought(
            agent_name=agent.name,
            other_name=other_agent_id,
            conversation_memories=conv_memories[:5],
            model=reflection_model,
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
            print(f"[ConvReflect] {agent.name} memo: {memo_insight}")

    except Exception as e:
        print(f"[ConvReflect] Memo thought failed for {agent.name}: {e}")

    return reflections


def _build_planning_thought_agent() -> Agent:
    """Build an agent for generating planning thoughts from conversations."""
    instructions = """Generate ONE specific thing the person should do or keep in mind
based on their conversation.

Write in first person (e.g., "I should...", "I'll remember to...", "Based on what they said, I...").
Keep it concise (1 sentence)."""

    return Agent(
        name="planning_thought_agent",
        instructions=instructions,
        model=DEFAULT_AGENT_MODEL,
        model_settings=ModelSettings(reasoning_effort="medium"),  # Planning thoughts require cognitive reasoning
        output_type=AgentOutputSchema(PlanningThoughtOutput),
    )


async def _generate_conversation_planning_thought(
    agent_name: str,
    other_name: str,
    conversation_memories: List[MemoryNode],
    model: str = None,  # Ignored, uses DEFAULT_AGENT_MODEL
) -> Optional[str]:
    """Generate a planning thought based on conversation."""
    statements = "\n".join([
        f"- {mem.description}"
        for mem in conversation_memories[:5]
    ])

    prompt = f"""You are {agent_name}, an office worker who just finished talking with {other_name}.

What you remember from the conversation:
{statements}

Generate a planning thought."""

    try:
        agent = _build_planning_thought_agent()
        result = await Runner.run(agent, prompt)
        output: PlanningThoughtOutput = result.final_output
        return output.thought if output.thought else None
    except Exception as e:
        print(f"[PLANNING] Planning thought generation failed: {e}")
        return None


def _build_memo_thought_agent() -> Agent:
    """Build an agent for generating memo thoughts about other people."""
    instructions = """Generate ONE observation about what you learned or noticed
about the other person from a conversation.

Write in first person (e.g., "They seem to...", "I learned that...", "They mentioned...").
Keep it concise (1 sentence)."""

    return Agent(
        name="memo_thought_agent",
        instructions=instructions,
        model=DEFAULT_AGENT_MODEL,
        model_settings=ModelSettings(reasoning_effort="medium"),  # Memo thoughts require cognitive reasoning
        output_type=AgentOutputSchema(MemoThoughtOutput),
    )


async def _generate_conversation_memo_thought(
    agent_name: str,
    other_name: str,
    conversation_memories: List[MemoryNode],
    model: str = None,  # Ignored, uses DEFAULT_AGENT_MODEL
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
        print(f"[MEMO] Memo thought generation failed: {e}")
        return None


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
    equipment_items = equipment.get("items", {})
    # Check if any equipment is off (might need to be turned on)
    if equipment_items and not any(equipment_items.values()):
        focal_points.append("my equipment habits when arriving")

    other_occupants = sim_state.get("other_occupants_present", [])
    if other_occupants:
        focal_points.append(f"my relationship with {other_occupants[0]}")
        # Add focal point for agreements when others are present
        focal_points.append("agreements I made with colleagues about temperature or comfort")

    # Time-aware focal points for lunch and breaks
    datetime_str = sim_state.get("datetime", "")
    if datetime_str:
        try:
            from datetime import datetime as dt
            now = dt.fromisoformat(datetime_str.replace("Z", "+00:00"))
            hour = now.hour

            # Lunch time window (11:00 - 14:00)
            if 11 <= hour <= 14:
                focal_points.append("my lunch habits and food preferences")
                # Check if colleagues are at lunch
                others_at_lunch = sim_state.get("other_occupants_at_lunch", [])
                if others_at_lunch:
                    focal_points.append(f"going to lunch with colleagues")

            # Afternoon break window (14:00 - 16:00)
            if 14 <= hour <= 16:
                focal_points.append("taking breaks and energy levels")

            # End of day window (16:00 - 19:00)
            if 16 <= hour <= 19:
                focal_points.append("my end of day routine and departure habits")
                # Check if would be last to leave
                office_occupancy = sim_state.get("office_occupancy", {})
                if office_occupancy.get("you_would_be_last_to_leave", False):
                    focal_points.append("responsibilities when leaving the office last")

            # Morning coffee/settling in (08:00 - 10:00)
            if 8 <= hour <= 10:
                focal_points.append("my morning routine at work")
        except (ValueError, TypeError):
            pass  # If datetime parsing fails, skip time-aware focal points

    # If currently at lunch or on break, add relevant focal points
    agent_status = sim_state.get("agent_status", {})
    if agent_status.get("at_lunch"):
        focal_points.append("when I usually return from lunch")
    if agent_status.get("on_break"):
        focal_points.append("how long I usually take breaks")

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

    # Extract meeting name from checkpoint reason
    parts = checkpoint_reason.split(":", 1)
    if len(parts) < 2:
        return None

    checkpoint_type = parts[0]
    meeting_name = parts[1]

    # Find the meeting in the context
    current_meeting = meeting_context.get("current_meeting")
    meetings_today = meeting_context.get("meetings_today", [])

    # Find matching meeting
    target_meeting = None
    for meeting in meetings_today:
        if meeting.get("title") == meeting_name:
            target_meeting = meeting
            break

    if not target_meeting:
        return None

    # Check if agent is the host (created_by)
    is_host = target_meeting.get("created_by") == agent_id

    if not is_host:
        return None

    # Format the equipment prompt for the host
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

    # Get location equipment from sim_state
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
    checkpoint_reason: str = "interval",
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
        checkpoint_reason: Why decision is being made (hourly, meeting_start, lunch_time, etc.)

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

        # Alert for current or imminent meeting - make it VERY clear they need to physically attend
        if meeting_alert:
            meeting_lines.append("")
            meeting_lines.append(f"⚠️ IMPORTANT: {meeting_alert}")
            current_meeting = meeting_context.get("current_meeting")
            if current_meeting:
                meeting_lines.append("")
                meeting_lines.append(">>> ACTION REQUIRED: Use 'attend_meeting' NOW to physically go to the meeting room!")
                meeting_lines.append("    (Remember: accepting an invitation doesn't move you there - you must use attend_meeting)")
            elif meeting_context.get("should_attend_now"):
                meeting_lines.append("")
                meeting_lines.append(">>> ACTION REQUIRED: Use 'attend_meeting' to go to the meeting room!")

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

    # Format equipment status readably
    equipment_status = sim_state.get("equipment_status", {})
    equipment_items = equipment_status.get("items", {})
    equipment_lines = []
    for equipment_name, is_on in equipment_items.items():
        state_str = "ON" if is_on else "OFF"
        equipment_lines.append(f"  - {equipment_name}: {state_str}")
    equipment_section = "\n".join(equipment_lines) if equipment_lines else "  No equipment tracked"

    # Format lighting readably
    lighting = sim_state.get("lighting_conditions", {})
    natural_light = lighting.get("natural_light_level", "moderate")
    desk_light_on = lighting.get("desk_light_on", False)
    lighting_section = f"  Natural light: {natural_light}, Desk light: {'ON' if desk_light_on else 'OFF'}"

    # Get work style preferences from core memories if available
    work_preferences_section = ""
    if agent.core_memory_store:
        work_memories = agent.core_memory_store.retrieve_relevant(
            focal_point="work style equipment preferences routine",
            n_count=2
        )
        if work_memories:
            # work_memories is List[Tuple[float, Dict]] - extract descriptions
            prefs = [f"  - {mem['description']}" for _, mem in work_memories]
            work_preferences_section = f"\n\n=== YOUR WORK PREFERENCES ===\n" + "\n".join(prefs)

    # Get today's clothing if available
    clothing_section = ""
    todays_clothing = agent.get_todays_clothing() if hasattr(agent, 'get_todays_clothing') else None
    if todays_clothing:
        clothing_desc = todays_clothing.get("description", "Not specified")
        warmth = todays_clothing.get("warmth_level", "medium")
        layers_removable = "yes" if todays_clothing.get("layers_removable", True) else "no"
        clothing_section = f"\nYour clothing today: {clothing_desc} (warmth: {warmth}, can remove layers: {layers_removable})"

    # Get agent's current status
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
    status_str = ", ".join(status_parts) if status_parts else "at your desk"

    # Update colleague context with lunch/break status
    others_at_lunch = sim_state.get("other_occupants_at_lunch", [])
    others_on_break = sim_state.get("other_occupants_on_break", [])
    colleague_status_lines = []
    if others_at_lunch:
        colleague_status_lines.append(f"  At lunch: {', '.join(others_at_lunch)}")
    if others_on_break:
        colleague_status_lines.append(f"  On break: {', '.join(others_on_break)}")
    colleague_status_section = "\n".join(colleague_status_lines) if colleague_status_lines else ""

    # Build location section
    current_location = sim_state.get("current_location", "desk_area")
    location_info = sim_state.get("location_info", {})
    available_locations = sim_state.get("available_locations", [])
    agents_by_location = sim_state.get("agents_by_location", {})
    location_equipment = sim_state.get("location_equipment", [])

    location_lines = [f"You are at: {current_location}"]
    current_loc_info = location_info.get(current_location, {})
    if current_loc_info.get("description"):
        location_lines.append(f"  ({current_loc_info['description']})")

    # Show equipment available at current location
    if location_equipment:
        location_lines.append("\nEquipment available here:")
        for eq in location_equipment:
            state_str = "ON" if eq.get("is_on") else "OFF"
            in_use = eq.get("in_use_by")
            use_str = f" (in use by {in_use})" if in_use else ""
            location_lines.append(f"  - {eq['name']}: {state_str}{use_str}")

    # Show other locations and who's there
    location_lines.append("\nOther locations you can go to:")
    for loc_name in available_locations:
        if loc_name == current_location:
            continue
        loc_info = location_info.get(loc_name, {})
        loc_desc = loc_info.get("description", "")
        agents_there = agents_by_location.get(loc_name, [])
        agents_str = f" - {', '.join(agents_there)} there" if agents_there else ""
        # Show appliances at break area
        appliances = loc_info.get("appliances", [])
        appliances_str = f" (has: {', '.join(appliances)})" if appliances else ""
        location_lines.append(f"  - {loc_name}: {loc_desc}{appliances_str}{agents_str}")

    location_section = "\n".join(location_lines)

    # Format checkpoint reason for display
    checkpoint_display = {
        "hourly": "Regular hourly check-in",
        "meeting_start": "A meeting is starting",
        "meeting_end": "A meeting is ending",
        "lunch_time": "It's time for your planned lunch",
        "morning_break": "It's time for your morning break",
        "afternoon_break": "It's time for your afternoon break",
        "interval": "Regular decision interval",
        "first_decision": "First decision of the day",
        "forced": "Decision requested",
    }.get(checkpoint_reason.split(":")[0], checkpoint_reason)

    # Add meeting name if checkpoint is meeting-related
    if ":" in checkpoint_reason:
        meeting_name = checkpoint_reason.split(":", 1)[1]
        checkpoint_display += f" - '{meeting_name}'"

    prompt = f"""
=== WHO YOU ARE ===
{context['identity']}

=== DECISION CHECKPOINT ===
Reason: {checkpoint_display}

=== CURRENT STATE ===
DateTime: {context['datetime']} ({context['day_of_week']})
Your status: {status_str}
Indoor temperature: {sim_state.get('indoor_temp_c', 'N/A')}C
Outdoor temperature: {sim_state.get('outdoor_temp_c', 'N/A')}C
Weather: {sim_state.get('weather_description', 'N/A')}{clothing_section}
Your desk: {sim_state.get('current_desk', 'N/A')}
Window state: {sim_state.get('window_open_fraction', 0)}
Thermostat setpoints:
  Heating: {sim_state.get('thermostat', {}).get('heating_setpoint_c', 21)}C (activates when cold)
  Cooling: {sim_state.get('thermostat', {}).get('cooling_setpoint_c', 24)}C (activates when hot)

=== LOCATION ===
{location_section}

=== WORKSPACE ===
Lighting:
{lighting_section}
Equipment:
{equipment_section}

=== COLLEAGUES PRESENT ===
{colleague_context}
{colleague_status_section}

=== YOUR SCHEDULE ===
{context['schedule']}

=== MEETING STATUS ===
{meeting_status_section}

=== PENDING INVITATIONS ===
{invitations_section}

=== RECENT AGREEMENTS ===
{agreements_section}

=== RELEVANT MEMORIES ===
{context['relevant_memories']}{work_preferences_section}

Based on who you are, your memories, the current state, and your schedule, decide what actions to take.
If a meeting is in progress or starting soon, consider attending with 'attend_meeting'.
If you have pending invitations, use 'respond_to_invitation' to accept or decline.
Respect prior agreements with colleagues unless circumstances have changed significantly.
If you want to go to lunch, use 'go_to_lunch' (stay in building) or 'go_out_for_lunch' (leave building).
If you're at lunch or on break and ready to return, use 'return_from_lunch' or 'return_from_break'.
For a short break (coffee, stretch), use 'take_break'.
"""

    # Add meeting host equipment context if applicable
    if meeting_context:
        host_equipment_context = get_meeting_host_equipment_context(
            agent_id=agent.agent_id,
            meeting_context=meeting_context,
            checkpoint_reason=checkpoint_reason,
        )
        if host_equipment_context:
            prompt += f"\n{host_equipment_context}"

    return prompt.strip()


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

As part of your plan, decide what you'll wear today. Consider:
- The weather forecast (outdoor temperature)
- Any meetings you have (formality)
- Your personal style and comfort preferences
Provide a description of your outfit and its warmth level (very_light, light, medium, warm, very_warm).
""".strip()

    return prompt


async def record_decision_to_memory(
    agent: "GenerativeAgent",
    decision: Any,  # OccupantStepDecision
    now: datetime,
) -> None:
    """
    Record a decision to the agent's memory stream.

    NOTE: This function does NOT call agent.save(). The caller is responsible
    for calling agent.save() to persist the memory to disk. This is intentional
    to allow batching multiple memory operations before a single save.

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

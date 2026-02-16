"""
Memory retrieval module for generative agents.

Implements the retrieval stage of the cognitive loop:
- Retrieve memories relevant to focal points
- Category-specific memory retrieval for structured decisions
- Deduplication of retrieved memories
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from bsm.agents.memory.stream import MemoryNode

if TYPE_CHECKING:
    from bsm.agents.generative_agent import GenerativeAgent


# ---------------------------------------------------------------------------
# Decision Categories for Structured Memory Retrieval
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


# ---------------------------------------------------------------------------
# Core Retrieval Functions
# ---------------------------------------------------------------------------

def retrieve(
    agent: "GenerativeAgent",
    focal_points: List[str],
    now: datetime,
    n_count: int = 6,
    core_memory_count: int = 5,
) -> Dict[str, List[MemoryNode]]:
    """
    Retrieve memories relevant to focal points.

    Focal points are questions or topics the agent is thinking about.
    Returns relevant memories for each focal point.

    This function retrieves from BOTH:
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


def deduplicate_retrieved_memories(
    retrieved: Dict[str, List[MemoryNode]],
    max_per_focal: int = 6,
) -> Dict[str, List[MemoryNode]]:
    """
    Remove duplicate memories across focal points.

    Keeps memory in the focal point where it appears first.
    This prevents the same memory from appearing in multiple
    sections of the decision prompt.

    Args:
        retrieved: Dict mapping focal points to lists of MemoryNodes
        max_per_focal: Maximum unique memories per focal point

    Returns:
        Deduplicated dict with same structure
    """
    seen_ids: Set[int] = set()
    deduplicated: Dict[str, List[MemoryNode]] = {}

    for focal_pt, memories in retrieved.items():
        unique = []
        for mem in memories:
            if mem.node_id not in seen_ids:
                seen_ids.add(mem.node_id)
                unique.append(mem)
                if len(unique) >= max_per_focal:
                    break
        deduplicated[focal_pt] = unique

    return deduplicated


# ---------------------------------------------------------------------------
# Category-Specific Retrieval
# ---------------------------------------------------------------------------

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
        f"Temperature is {current_temp}C" if current_temp != "unknown" else "",
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
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "retrieve",
    "deduplicate_retrieved_memories",
    "retrieve_for_category",
    "format_memories_for_decision",
    "retrieve_all_decision_memories",
    "DECISION_CATEGORIES",
]

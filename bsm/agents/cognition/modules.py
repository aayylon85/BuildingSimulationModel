"""
DEPRECATED: This file exists for backward compatibility only.

The cognition module has been reorganized into logical submodules:

- bsm.agents.cognition.schemas - Pydantic output schemas
- bsm.agents.cognition.perception - perceive(), importance assessment
- bsm.agents.cognition.retrieval - retrieve(), category retrieval
- bsm.agents.cognition.reflection - reflect(), conversation reflection
- bsm.agents.cognition.prompts - format_step_prompt(), format_planning_prompt()
- bsm.agents.cognition.memory_ops - record_*_to_memory() functions
- bsm.agents.cognition.social - should_react(), relationship utilities

Import from specific submodules instead of this file.
"""

import warnings

warnings.warn(
    "Importing from bsm.agents.cognition.modules is deprecated. "
    "Import from specific submodules (perception, retrieval, reflection, "
    "prompts, memory_ops, social, schemas) instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything for backward compatibility

# Schemas
from bsm.agents.cognition.schemas import (
    ImportanceAssessment,
    ConsolidationBatch,
    ReflectionOutput,
    PlanningThoughtOutput,
    MemoThoughtOutput,
)

# Perception
from bsm.agents.cognition.perception import (
    perceive,
    get_importance,
    assess_event_importance_llm,
    _build_simple_agent,
)

# Retrieval
from bsm.agents.cognition.retrieval import (
    retrieve,
    deduplicate_retrieved_memories,
    retrieve_for_category,
    format_memories_for_decision,
    retrieve_all_decision_memories,
    DECISION_CATEGORIES,
)

# Reflection
from bsm.agents.cognition.reflection import (
    reflect,
    reflect_on_conversation,
    assess_memories_for_consolidation,
)

# Prompts
from bsm.agents.cognition.prompts import (
    format_step_prompt,
    format_planning_prompt,
    get_decision_focal_points,
    get_planning_focal_points,
    get_meeting_context,
    get_meeting_host_equipment_context,
    format_colleague_context,
    format_device_state,
    format_meetings_for_prompt,
    format_pending_invitations_for_prompt,
    build_decision_context,
)

# Memory operations
from bsm.agents.cognition.memory_ops import (
    record_decision_to_memory,
    record_plan_to_memory,
)

# Social interaction
from bsm.agents.cognition.social import (
    should_react,
    generate_relationship_summary,
)

__all__ = [
    # Schemas
    "ImportanceAssessment",
    "ConsolidationBatch",
    "ReflectionOutput",
    "PlanningThoughtOutput",
    "MemoThoughtOutput",
    # Perception
    "perceive",
    "get_importance",
    "assess_event_importance_llm",
    "_build_simple_agent",
    # Retrieval
    "retrieve",
    "deduplicate_retrieved_memories",
    "retrieve_for_category",
    "format_memories_for_decision",
    "retrieve_all_decision_memories",
    "DECISION_CATEGORIES",
    # Reflection
    "reflect",
    "reflect_on_conversation",
    "assess_memories_for_consolidation",
    # Prompts
    "format_step_prompt",
    "format_planning_prompt",
    "get_decision_focal_points",
    "get_planning_focal_points",
    "get_meeting_context",
    "get_meeting_host_equipment_context",
    "format_colleague_context",
    "format_device_state",
    "format_meetings_for_prompt",
    "format_pending_invitations_for_prompt",
    "build_decision_context",
    # Memory operations
    "record_decision_to_memory",
    "record_plan_to_memory",
    # Social interaction
    "should_react",
    "generate_relationship_summary",
]

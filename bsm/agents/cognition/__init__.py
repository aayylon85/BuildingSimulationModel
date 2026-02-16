"""
Agent cognitive modules.

This module contains the cognitive functions for generative agents,
organized into logical submodules:

- schemas: Pydantic output schemas for LLM agents
- perception: Observe current simulation state, assess importance
- retrieval: Semantic memory retrieval by focal points
- reflection: Generate reflections from memories
- prompts: Format prompts for step and planning decisions
- memory_ops: Record decisions, plans, conversations to memory
- social: Interaction triggers, relationship utilities
- conversation: Multi-turn agent conversations
- consultation: Shared-space consultations (thermostat, windows)

For backward compatibility, all public symbols are re-exported here.
Direct imports from submodules are preferred for new code.
"""

# Schemas
from bsm.agents.cognition.schemas import (
    ImportanceAssessment,
    ConsolidationBatch,
    ReflectionOutput,
    PlanningThoughtOutput,
    MemoThoughtOutput,
    ConversationUtterance,
    ConversationResult,
    UtteranceOutput,
    ConversationCommitmentOutput,
    CommitmentsExtractionOutput,
    ConsultationOutcome,
    ConsultationUtteranceOutput,
    ConsensusAssessment,
    DeclineDecisionOutput,
)

# Perception
from bsm.agents.cognition.perception import (
    perceive,
    get_importance,
    assess_event_importance_llm,
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
    record_conversation_to_memory,
    record_agreement_to_memory,
)

# Social interaction
from bsm.agents.cognition.social import (
    should_react,
    generate_relationship_summary,
)

# Conversation
from bsm.agents.cognition.conversation import (
    agent_conversation,
    sync_agent_conversation,
    extract_conversation_commitments,
    check_commitment_conflicts,
)

# Consultation
from bsm.agents.cognition.consultation import (
    consultation_conversation,
    assess_consensus_with_llm,
    simple_vote_on_temperature,
    generate_consultation_utterance,
)


__all__ = [
    # Schemas
    "ImportanceAssessment",
    "ConsolidationBatch",
    "ReflectionOutput",
    "PlanningThoughtOutput",
    "MemoThoughtOutput",
    "ConversationUtterance",
    "ConversationResult",
    "UtteranceOutput",
    "ConversationCommitmentOutput",
    "CommitmentsExtractionOutput",
    "ConsultationOutcome",
    "ConsultationUtteranceOutput",
    "ConsensusAssessment",
    "DeclineDecisionOutput",
    # Perception
    "perceive",
    "get_importance",
    "assess_event_importance_llm",
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
    "record_conversation_to_memory",
    "record_agreement_to_memory",
    # Social interaction
    "should_react",
    "generate_relationship_summary",
    # Conversation
    "agent_conversation",
    "sync_agent_conversation",
    "extract_conversation_commitments",
    "check_commitment_conflicts",
    # Consultation
    "consultation_conversation",
    "assess_consensus_with_llm",
    "simple_vote_on_temperature",
    "generate_consultation_utterance",
]

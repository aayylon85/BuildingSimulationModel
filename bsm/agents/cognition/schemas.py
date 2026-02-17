"""
Pydantic output schemas for cognitive modules.

This module contains all the output schemas used by LLM agents
in the cognitive loop: perceive, retrieve, reflect, act.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Importance and Consolidation Schemas
# ---------------------------------------------------------------------------

class ImportanceAssessment(BaseModel):
    """Output schema for LLM-based importance scoring."""
    importance: int = Field(ge=1, le=10, description="Importance score from 1-10")
    reasoning: str = Field(description="Brief reasoning for the score")


class ConsolidationBatch(BaseModel):
    """Output schema for memory consolidation assessment."""
    memories_to_consolidate: List[int] = Field(
        default_factory=list,
        description="Indices of memories to consolidate into core identity"
    )
    reasoning: str = Field(description="Brief explanation of why these memories matter")


# ---------------------------------------------------------------------------
# Reflection Schemas
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


# ---------------------------------------------------------------------------
# Conversation Schemas
# ---------------------------------------------------------------------------

class ConversationUtterance(BaseModel):
    """A single utterance in a conversation."""
    speaker_id: str
    utterance: str
    end_conversation: bool = Field(
        default=False,
        description="True if this utterance naturally ends the conversation"
    )


class ConversationResult(BaseModel):
    """Result of a complete conversation between two agents."""
    participants: List[str]
    utterances: List[ConversationUtterance]
    summary: str
    duration_minutes: int
    topics_discussed: List[str] = Field(default_factory=list)
    initiated_by: str
    declined: bool = Field(
        default=False,
        description="True if the target agent declined the conversation"
    )
    decline_reason: Optional[str] = Field(
        default=None,
        description="Reason for declining, if declined is True"
    )


class DeclineDecisionOutput(BaseModel):
    """Output schema for checking if an agent wants to decline an interaction."""
    action: Literal["accept", "decline"] = Field(
        description="Whether to accept or decline the conversation"
    )
    reason: str = Field(
        description="Brief reason for the decision"
    )


class UtteranceOutput(BaseModel):
    """Output schema for the conversation agent."""
    utterance: str = Field(description="What you say in the conversation")
    end_conversation: bool = Field(
        default=False,
        description="Set to true if this utterance naturally ends the conversation (goodbye, need to go, etc.)"
    )


class ConversationCommitmentOutput(BaseModel):
    """Output schema for commitment extraction from conversations."""
    activity: str = Field(
        description="SHORT activity name (max 4 words) - e.g., 'coffee', 'lunch', 'walk outside'"
    )
    time: str = Field(
        description="When: HH:MM format (e.g., '10:30') for specific times, or 'needs_confirmation' for vague agreements"
    )
    location: Literal["break_area", "outside", "meeting_room"] = Field(
        description="Where the activity will happen - must be one of the valid locations"
    )


class CommitmentsExtractionOutput(BaseModel):
    """Output schema for extracting all commitments from a conversation."""
    commitments: List[ConversationCommitmentOutput] = Field(
        default_factory=list,
        description="List of commitments made during the conversation (empty if none)"
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of what agreements were identified"
    )


# ---------------------------------------------------------------------------
# Consultation Schemas (for thermostat/shared-space decisions)
# ---------------------------------------------------------------------------

class ConsultationOutcome(BaseModel):
    """Result of a consultation about a shared-space action."""
    proposed_action: str = Field(description="What was originally proposed (e.g., 'set thermostat to 19C')")
    agreed_action: Optional[str] = Field(default=None, description="The agreed-upon action (may differ from proposed)")
    consensus_reached: bool = Field(default=False, description="Whether all participants agreed")
    final_setpoint_c: Optional[float] = Field(default=None, description="Agreed thermostat setpoint if applicable")
    final_window_state: Optional[bool] = Field(default=None, description="Agreed window state if applicable (True=open)")
    summary: str = Field(description="Brief summary of the consultation outcome")
    participants: List[str] = Field(default_factory=list, description="IDs of agents who participated")


class ConsultationUtteranceOutput(BaseModel):
    """Output schema for consultation conversation agent."""
    utterance: str = Field(description="What you say in the consultation")
    end_consultation: bool = Field(
        default=False,
        description="Set to true when agreement is reached or clear disagreement"
    )
    proposed_compromise: Optional[float] = Field(
        default=None,
        description="If proposing a compromise temperature, specify the value in C"
    )


class ConsensusAssessment(BaseModel):
    """Output schema for LLM-based consensus assessment."""
    consensus_reached: bool = Field(description="Whether all participants agreed to the change")
    agreed_temperature_c: Optional[float] = Field(
        default=None,
        description="The agreed-upon temperature in Celsius, if any"
    )
    summary: str = Field(description="Brief summary of the consultation outcome")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence level (0-1) in the assessment"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Importance/Consolidation
    "ImportanceAssessment",
    "ConsolidationBatch",
    # Reflection
    "ReflectionOutput",
    "PlanningThoughtOutput",
    "MemoThoughtOutput",
    # Conversation
    "ConversationUtterance",
    "ConversationResult",
    "UtteranceOutput",
    "ConversationCommitmentOutput",
    "CommitmentsExtractionOutput",
    # Consultation
    "ConsultationOutcome",
    "ConsultationUtteranceOutput",
    "ConsensusAssessment",
]

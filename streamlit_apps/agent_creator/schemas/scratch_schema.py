"""Pydantic schema for scratch.json validation."""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator
import re


class RelationshipModel(BaseModel):
    """Model for agent-to-agent relationship."""
    familiarity: float = Field(ge=0.0, le=1.0, description="How well they know each other (0-1)")
    sentiment: float = Field(ge=0.0, le=1.0, description="Positive/negative feeling (0-1)")


class ScratchSchema(BaseModel):
    """Strict schema for scratch.json validation.

    This schema ensures generated scratch.json files conform exactly
    to the structure expected by the simulation.
    """

    # Identity
    agent_id: str = Field(
        pattern=r"^[a-z]+_\d{3}$",
        description="Unique identifier (e.g., 'david_004')"
    )
    name: str = Field(min_length=1, description="Full display name")
    first_name: str = Field(min_length=1, description="First name only")

    # Personality
    innate_traits: str = Field(
        description="Comma-separated core traits (e.g., 'methodical, organized, adaptable')"
    )
    learned_traits: str = Field(
        description="Summary of experience and background"
    )
    currently: str = Field(
        description="What the agent is currently focused on"
    )
    lifestyle: str = Field(
        description="Summary of schedule and commute"
    )

    # Comfort preferences
    thermal_preference: Literal["cool", "neutral", "slightly_warm", "warm"] = Field(
        description="Temperature preference"
    )
    runs_warm: bool = Field(description="Whether they tend to feel warm")
    clothing_insulation_typical: Literal["very_light", "light", "medium", "warm", "very_warm"] = Field(
        default="medium",
        description="Typical clothing insulation level"
    )
    light_preference: Literal["natural", "task", "ambient", "warm"] = Field(
        description="Lighting preference"
    )
    noise_sensitivity: Literal["low", "moderate", "high"] = Field(
        description="Sensitivity to noise"
    )

    # Schedule
    typical_arrival: str = Field(
        pattern=r"^\d{2}:\d{2}-\d{2}:\d{2}$",
        description="Arrival time range (e.g., '08:30-09:00')"
    )
    typical_departure: str = Field(
        pattern=r"^\d{2}:\d{2}-\d{2}:\d{2}$",
        description="Departure time range (e.g., '17:00-17:30')"
    )
    preferred_desk: str = Field(description="Preferred desk identifier")
    works_weekends: bool = Field(default=False, description="Whether they work weekends")
    workspace_variety: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Preference for workspace variety"
    )
    likes_desk_variety: bool = Field(
        default=True,
        description="Whether they like variety in desk choice"
    )

    # Cognitive parameters
    perception_bandwidth: int = Field(
        ge=6, le=10, default=8,
        description="Events perceived per step (6-10)"
    )
    reflection_threshold: int = Field(
        ge=100, le=200, default=150,
        description="Importance score to trigger reflection"
    )
    importance_trigger_max: int = Field(
        ge=100, le=200, default=150,
        description="Maximum importance trigger value"
    )
    importance_trigger_curr: int = Field(
        ge=100, le=200, default=150,
        description="Current importance trigger value"
    )
    recency_w: float = Field(
        ge=0.5, le=1.5, default=1.0,
        description="Memory retrieval weight for recency"
    )
    relevance_w: float = Field(
        ge=0.5, le=1.5, default=1.0,
        description="Memory retrieval weight for relevance"
    )
    importance_w: float = Field(
        ge=0.5, le=1.5, default=1.0,
        description="Memory retrieval weight for importance"
    )
    recency_decay: float = Field(
        ge=0.99, le=0.999, default=0.995,
        description="Memory decay rate"
    )

    # Relationships
    relationship_models: Dict[str, RelationshipModel] = Field(
        default_factory=dict,
        description="Relationships with other agents"
    )

    # State (always initialized to these values for new agents)
    daily_plan: None = None
    f_daily_schedule: List = Field(default_factory=list)
    just_arrived: bool = False

    @field_validator('relationship_models')
    @classmethod
    def validate_relationships(cls, v):
        """Ensure all relationship keys are valid agent IDs."""
        for agent_id in v.keys():
            if not re.match(r"^[a-z]+_\d{3}$", agent_id):
                raise ValueError(f"Invalid agent_id in relationships: {agent_id}")
        return v

    @field_validator('importance_trigger_max', 'importance_trigger_curr')
    @classmethod
    def match_reflection_threshold(cls, v, info):
        """Importance triggers should match reflection_threshold."""
        if 'reflection_threshold' in info.data:
            return info.data['reflection_threshold']
        return v


class FormData(BaseModel):
    """Model for user form input data."""

    # Basic info
    agent_id: str
    name: str
    first_name: str
    age: int = Field(ge=25, le=65)
    origin: str
    years_in_southampton: int = Field(ge=0, le=50)
    languages: str

    # Personality
    innate_traits: List[str] = Field(default_factory=list)
    personality_description: str = ""
    communication_style: List[str] = Field(default_factory=list)
    decision_making_style: List[str] = Field(default_factory=list)

    # Work style
    typical_arrival: str = "08:30-09:00"
    typical_departure: str = "17:00-17:30"
    preferred_desk: str = "Desk_A"
    works_weekends: bool = False
    workspace_variety: str = "medium"
    likes_desk_variety: bool = True

    # Comfort
    thermal_preference: str = "neutral"
    runs_warm: bool = False
    clothing_insulation_typical: str = "medium"
    light_preference: str = "natural"
    noise_sensitivity: str = "moderate"

    # Relationships
    relationships: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    relationship_notes: Dict[str, str] = Field(default_factory=dict)

    def get_traits_string(self) -> str:
        """Get comma-separated traits string."""
        return ", ".join(self.innate_traits)

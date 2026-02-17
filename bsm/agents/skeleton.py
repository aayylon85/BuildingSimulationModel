"""
LLMagentskeleton.py

A minimal, extensible implementation of simulated building-occupant agents using:
- OpenAI Agents SDK (openai-agents)
- GPT-5.2 for agentic decisioning
- JSON-based semantic-search memory with time-decay action recall
- Per-agent schedule + shared calendar with RSVP
- Clean integration seam to your local building simulator

Notes
-----
- The Agents SDK defaults to using the Responses API for OpenAI models.
- This file uses structured outputs (Pydantic) for deterministic downstream control.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# OpenAI python client (for embeddings)
from openai import OpenAI

# OpenAI Agents SDK
from agents import Agent, Runner, ModelSettings, function_tool, RunContextWrapper
from agents.agent_output import AgentOutputSchema
from agents.memory import SQLiteSession


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_AGENT_MODEL = "gpt-5.2"          # latest model family per OpenAI models page
DEFAULT_EMBED_MODEL = "text-embedding-3-large"

# Action-memory time-decay: half-life in days (tune to your needs)
DEFAULT_ACTION_HALF_LIFE_DAYS = 14.0

# Retrieval limits (keep small to reduce token pressure)
DEFAULT_TOP_K = 6

# ---------------------------------------------------------------------------
# Action Parameter Schema - Single source of truth for action parameters
# ---------------------------------------------------------------------------
# Reference documentation for action parameters.
# This documents the expected parameters for each action type.
# Note: These types are not currently validated at runtime - this serves as
# documentation for developers implementing action handlers.

ACTION_PARAMS = {
    "thermostat_adjust": {
        "direction": str,       # "warmer", "cooler", or "maintain_current"
        "amount": float,        # Degrees C to adjust
        "setpoint_delta_c": float,  # Calculated delta (positive for warmer, negative for cooler)
    },
    "lights_set": {
        "action": str,          # "turn_on", "turn_off", "adjust_brightness", "keep_current"
        "target": str,          # Light device name
        "on": bool,             # Whether light should be on
        "brightness": int,      # Brightness level 0-100 (optional)
    },
    "equipment_set": {
        "equipment_name": str,  # Equipment name (e.g., "laptop_A")
        "on": bool,             # Whether equipment should be on
    },
    "move_to": {
        "destination": str,     # Target location: desk_area, meeting_room, break_area, shared_area
    },
    "initiate_conversation": {
        "target_agent": str,    # Agent ID to talk to
        "topic": str,           # Conversation topic
    },
    "take_break": {
        "location": str,        # Where to take break (at_desk, break_area)
        "activity": str,        # Break activity (tea, coffee, walk, etc.)
    },
    "go_to_lunch": {},          # No parameters needed
    "go_out_for_lunch": {},     # No parameters needed
    "return_from_lunch": {},    # No parameters needed
    "return_from_break": {},    # No parameters needed
    "no_op": {},                # No parameters needed
    "use_appliance": {
        "appliance_name": str,  # Appliance to use (kettle, microwave, etc.)
    },
    "update_daily_plan": {
        "updates": dict,        # Plan updates
        "reason": str,          # Why the update
    },
}


# ---------------------------------------------------------------------------
# Building simulation seam
# ---------------------------------------------------------------------------

class BuildingSimulationAdapter:
    """
    Abstract base class for building simulation integration.

    Implement this interface to connect LLM agents to your building simulator:
      - get_state(...) -> dict suitable for the agent prompt
      - apply_decision(...) -> apply returned actions to your simulator

    Production implementation: ProductionSimulationAdapter in simulation_adapter.py
    """

    def get_state(self, occupant_id: str, now: datetime) -> Dict[str, Any]:
        raise NotImplementedError

    def apply_decision(self, decision: "OccupantStepDecision") -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Structured outputs (Pydantic) for agent decisions
# ---------------------------------------------------------------------------

ActionType = Literal[
    "no_op",
    "thermostat_adjust",
    "window_set",
    "lights_set",
    "choose_desk",
    "equipment_set",
    "attend_meeting",
    "leave_meeting",
    "use_photocopier",
    "arrive",
    "depart",
    "respond_to_invitation",  # Accept/decline meeting invitation
    # Lunch and break actions
    "go_to_lunch",           # Leave desk for lunch (in building cafeteria/kitchen)
    "go_out_for_lunch",      # Leave building for lunch (cafe, restaurant, etc.)
    "return_from_lunch",     # Come back from lunch
    "take_break",            # Short break (coffee, stretch, walk) - stay in building
    "go_out_for_break",      # Leave building for break (walk, fresh air, coffee run)
    "return_from_break",     # Come back from short break
    # Navigation
    "move_to",               # Move to a different location (parameters: location)
    # Location-specific equipment
    "use_appliance",         # Use an appliance at current location (parameters: appliance_name, duration_minutes)
    # Social
    "initiate_conversation", # Start a conversation with a colleague (parameters: agent_id, topic)
    # Plan management
    "update_daily_plan",     # Update daily plan based on new circumstances (parameters: updates dict, reason)
]

class OccupantAction(BaseModel):
    action_type: ActionType
    parameters: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)

class OccupantStepDecision(BaseModel):
    occupant_id: str
    datetime_iso: str
    location_zone: str
    current_desk: Optional[str] = None  # Current desk assignment
    is_present: bool = True  # Whether occupant is in the building
    actions: List[OccupantAction] = Field(default_factory=list)
    brief_rationale: str = Field(default="")


# ---------------------------------------------------------------------------
# Category-specific structured decisions for step checkpoints
# ---------------------------------------------------------------------------

class ThermostatDecision(BaseModel):
    """Decision about thermostat adjustment based on how agent feels."""
    action: Literal["adjust", "maintain_current"] = Field(
        description="Whether to adjust the thermostat or maintain current setting"
    )
    reasoning: str = Field(
        description="How agent feels (hot/cold/comfortable) based on actual temperature"
    )
    adjustment_direction: Optional[Literal["warmer", "cooler"]] = Field(
        default=None,
        description="Direction to adjust: 'warmer' if feeling cold, 'cooler' if feeling hot"
    )
    adjustment_amount: Optional[float] = Field(
        default=None,
        description="Degrees (C) to adjust, e.g., 1.0 or 2.0"
    )


class LightingDecision(BaseModel):
    """Decision about lighting at current location."""
    action: Literal["turn_on", "turn_off", "adjust_brightness", "keep_current"] = Field(
        description="Lighting action to take"
    )
    reasoning: str = Field(description="Why this lighting action was chosen")
    target_device: Optional[str] = Field(
        default=None,
        description="Light device name, e.g., 'desk_light' or 'meeting_room_light'"
    )
    brightness_level: Optional[int] = Field(
        default=None, ge=0, le=100,
        description="Brightness level 0-100% (only for adjust_brightness)"
    )


class EquipmentDecision(BaseModel):
    """Decision about a single piece of equipment. Create one entry per equipment item."""
    equipment_name: str = Field(
        description="Single equipment item name, e.g., 'laptop_A' or 'monitor_A'. Do NOT combine names."
    )
    action: Literal["turn_on", "turn_off", "keep_current"] = Field(
        description="What to do: turn_on (if OFF and needed), turn_off (if ON and not needed), keep_current (no change needed)"
    )
    reasoning: str = Field(
        description="Brief explanation of why this action was chosen for this equipment"
    )
    before_move: bool = Field(
        default=False,
        description="If True, apply this equipment action BEFORE any location move (e.g., turn off desk equipment before leaving). If False (default), apply AFTER location move."
    )


class LocationDecision(BaseModel):
    """Decision about moving to a different location."""
    action: Literal["move", "stay"] = Field(description="Whether to move to a different location or stay")
    reasoning: str = Field(description="Why this location decision was made")
    destination: Optional[str] = Field(
        default=None,
        description="Target location: desk_area, meeting_room, break_area, shared_area"
    )


class ConversationDecision(BaseModel):
    """Decision about initiating conversation with nearby agents."""
    action: Literal["initiate", "none"] = Field(description="Whether to start a conversation")
    reasoning: str = Field(description="Why this conversation decision was made")
    target_agent: Optional[str] = Field(default=None, description="Agent ID to talk to")
    topic: Optional[str] = Field(
        default=None,
        description="Conversation topic - can be work, social, comfort, or any relevant subject"
    )


class BreakDecision(BaseModel):
    """Decision about taking a break or returning from break/lunch."""
    action: Literal["take_break", "go_out_for_break", "continue_working", "return_from_lunch", "return_from_break", "not_applicable"] = Field(
        description="Whether to take a break (inside or outside), continue working, return from break/lunch, or 'not_applicable' if break decision not relevant for this checkpoint"
    )
    reasoning: str = Field(description="Why this break decision was made")
    break_type: Optional[Literal["at_desk", "break_area", "outside"]] = Field(
        default=None,
        description="Where to take the break (only for take_break/go_out_for_break actions)"
    )
    activity: Optional[str] = Field(
        default=None,
        description="Break activity: tea, coffee, snack, stretch, walk, fresh_air, etc."
    )


class CommitmentDecision(BaseModel):
    """Decision about handling a social commitment checkpoint."""
    action: Literal["execute", "defer", "skip", "not_applicable"] = Field(
        description=(
            "How to handle the commitment: "
            "'execute' to fulfill it now, "
            "'defer' to reschedule for later, "
            "'skip' to cancel entirely, "
            "'not_applicable' if this checkpoint is not about a commitment"
        )
    )
    reasoning: str = Field(description="Why this choice was made given current circumstances")
    defer_until: Optional[str] = Field(
        default=None,
        description="If deferring, when to try again (HH:MM format). Required if action='defer'."
    )


class MeetingEquipmentDecision(BaseModel):
    """Decision about equipment at meeting start/end (for meeting host)."""
    action: Literal["set_equipment", "accept_current", "not_applicable"] = Field(
        description="Whether to set meeting equipment, accept current state, or 'not_applicable' if not at meeting boundary or not the host"
    )
    reasoning: str = Field(description="Why this meeting equipment decision was made")
    equipment_changes: Optional[List[str]] = Field(
        default=None,
        description="List of equipment changes, e.g., ['turn on projector', 'close blinds']"
    )


class LunchPlanUpdate(BaseModel):
    """Partial update to lunch plan - only include fields you want to change."""
    location: Optional[Literal["at_desk", "break_area", "go_out"]] = Field(
        default=None, description="New lunch location if changing"
    )
    time: Optional[str] = Field(default=None, description="New lunch time (HH:MM) if changing")


class BreakPlanUpdate(BaseModel):
    """Partial update to break plan - only include fields you want to change."""
    location: Optional[Literal["at_desk", "break_area"]] = Field(
        default=None, description="New break location if changing"
    )
    activity: Optional[str] = Field(default=None, description="New activity if changing")
    preferred_time: Optional[str] = Field(default=None, description="New time (HH:MM) if changing")


class PlanUpdates(BaseModel):
    """Structured updates to the daily plan. Only include sections you want to change."""
    lunch_plan: Optional[LunchPlanUpdate] = Field(
        default=None, description="Updates to lunch plan"
    )
    morning_break: Optional[BreakPlanUpdate] = Field(
        default=None, description="Updates to morning break"
    )
    afternoon_break: Optional[BreakPlanUpdate] = Field(
        default=None, description="Updates to afternoon break"
    )


class PlanUpdateDecision(BaseModel):
    """Decision about whether to update the daily plan based on current circumstances."""
    action: Literal["update", "keep_current"] = Field(
        description="Whether to update the daily plan or keep current"
    )
    reasoning: str = Field(description="Why plan update is or isn't needed")
    updates: Optional[PlanUpdates] = Field(
        default=None,
        description="Structured updates to apply. Only include sections you want to change."
    )


class StepDecisions(BaseModel):
    """Complete structured output for a decision checkpoint."""
    occupant_id: str
    timestamp: str
    checkpoint_reason: str = Field(description="'hourly', 'meeting_start', 'meeting_end', 'lunch_time', 'morning_break', 'afternoon_break', 'commitment:*'")
    thermostat: ThermostatDecision
    lighting: LightingDecision
    equipment_decisions: List[EquipmentDecision] = Field(
        default_factory=list,
        description="List of decisions for each piece of equipment. One entry per item - do NOT combine names."
    )
    location: LocationDecision
    conversation: ConversationDecision
    break_decision: BreakDecision = Field(description="Break/lunch decision - use 'not_applicable' or 'continue_working' when not taking a break")
    meeting_equipment: MeetingEquipmentDecision = Field(description="Meeting equipment decision - use 'not_applicable' when not at meeting boundary or not the host")
    plan_update: PlanUpdateDecision = Field(description="Whether to update daily plan based on current circumstances")
    commitment_response: Optional[CommitmentDecision] = Field(
        default=None,
        description="Response to a commitment checkpoint. Use 'not_applicable' for non-commitment checkpoints. Only required when checkpoint_reason starts with 'commitment:'."
    )


class DeskSelectionDecision(BaseModel):
    """
    One-time desk selection decision made upon arrival.

    This is a separate decision from regular step decisions - it happens ONCE
    when the agent arrives at work and locks in their desk for the entire day.
    """
    occupant_id: str
    selected_desk: str = Field(description="The desk ID to claim for the day (e.g., 'Desk_A', 'Desk_B')")
    reasoning: str = Field(description="Why this desk was chosen based on preferences, proximity to colleagues, window access, etc.")
    equipment_to_turn_on: List[str] = Field(
        default_factory=list,
        description="Equipment to turn on immediately upon sitting down (e.g., ['laptop', 'monitor'])"
    )


# Clothing warmth levels for comfort model integration
ClothingWarmthLevel = Literal["very_light", "light", "medium", "warm", "very_warm"]


class ClothingChoice(BaseModel):
    """
    Agent's clothing choice for the day.

    Includes a natural description and a warmth level for comfort model calculations.
    Agents decide clothing in the morning based on weather, meetings, and personal style.
    """
    description: str = Field(
        description="Natural description of clothing, e.g., 'Navy cardigan over white blouse, dark trousers'"
    )
    warmth_level: ClothingWarmthLevel = Field(
        default="medium",
        description="Warmth level: very_light (t-shirt), light (shirt), medium (cardigan), warm (jumper), very_warm (layers)"
    )
    layers_removable: bool = Field(
        default=True,
        description="Whether the agent can remove a layer if feeling warm"
    )


class MeetingPlan(BaseModel):
    title: str
    start_datetime_iso: str
    end_datetime_iso: str
    location: str = "TBD"
    organiser: str = "self"
    invitees: List[str] = Field(default_factory=list)
    attendance_intent: Literal["will_attend", "maybe", "will_skip"] = "will_attend"


class LunchPlan(BaseModel):
    """Lunch planning details decided during daily planning."""
    location: Literal["at_desk", "break_area", "go_out"]
    time: str = Field(description="Planned lunch time in HH:MM format")
    reasoning: str = Field(description="Why this location/time was chosen based on preferences and schedule")


class BreakPlan(BaseModel):
    """Break planning details decided during daily planning."""
    location: Literal["at_desk", "break_area"]
    activity: str = Field(description="tea, coffee, walk, stretch, etc.")
    preferred_time: str = Field(description="Planned break time in HH:MM format")
    reasoning: str = Field(description="Why this break was planned based on habits and schedule")


class WorkActivity(BaseModel):
    """A planned work activity requiring specific equipment or location."""
    activity: str = Field(description="What work to do: 'photocopying', 'printing', 'filing', etc.")
    equipment_needed: Optional[str] = Field(
        default=None,
        description="Equipment required if any (e.g., 'photocopier', 'printer')"
    )
    location: str = Field(
        default="shared_area",
        description="Where this activity happens (e.g., 'shared_area', 'meeting_room')"
    )
    planned_time: str = Field(description="When to do this (HH:MM)")
    duration_minutes: int = Field(default=10, description="Estimated duration in minutes")
    reasoning: str = Field(default="", description="Why this activity is planned")


class SocialCommitment(BaseModel):
    """A commitment made with a colleague during conversation."""
    activity: str = Field(description="What was agreed: 'coffee', 'walk', 'lunch together', etc.")
    time: str = Field(description="When (HH:MM) if specified, or 'unspecified'")
    with_agents: List[str] = Field(description="Agent IDs who agreed to this activity")
    location: Literal["break_area", "outside", "meeting_room", "unspecified"] = Field(
        default="unspecified",
        description="Where the activity will happen"
    )
    source: str = Field(default="conversation", description="How this commitment was created")
    # Fulfillment tracking
    fulfilled: bool = Field(default=False, description="Whether this commitment has been honored")
    fulfilled_at: Optional[str] = Field(default=None, description="ISO timestamp when fulfilled")
    # Deferral tracking
    deferred: bool = Field(default=False, description="Whether this commitment has been deferred")
    deferred_count: int = Field(default=0, description="How many times this has been deferred")
    last_deferred_at: Optional[str] = Field(default=None, description="ISO timestamp of last deferral")
    original_time: Optional[str] = Field(default=None, description="Original time before any deferrals")


class DailyPlan(BaseModel):
    occupant_id: str
    date_iso: str
    intended_arrival_time: str  # "HH:MM"
    actual_arrival_time: str    # "HH:MM"
    intended_departure_time: str
    actual_departure_time: str
    meetings: List[MeetingPlan] = Field(default_factory=list)
    clothing: Optional[ClothingChoice] = Field(
        default=None,
        description="What the agent is wearing today - decided based on weather, meetings, and personal style"
    )
    # New: Lunch and break planning
    lunch_plan: Optional[LunchPlan] = Field(
        default=None,
        description="Where and when to have lunch - decided based on habits, weather, schedule"
    )
    morning_break: Optional[BreakPlan] = Field(
        default=None,
        description="Morning break plan - tea/coffee preferences, location"
    )
    afternoon_break: Optional[BreakPlan] = Field(
        default=None,
        description="Afternoon break plan - tea/coffee preferences, location"
    )
    comfort_preferences: str = Field(
        default="",
        description="Temperature and lighting preferences for the day"
    )
    work_activities: List[WorkActivity] = Field(
        default_factory=list,
        description="Planned work activities requiring equipment or movement (photocopying, filing, etc.)"
    )
    return_to_desk_after_break: bool = Field(
        default=True,
        description="Whether to automatically return to desk after breaks/lunch"
    )
    notes: str = ""
    social_commitments: List[SocialCommitment] = Field(
        default_factory=list,
        description="Commitments made with colleagues to do activities together"
    )


# ---------------------------------------------------------------------------
# Local semantic search memory (SQLite + embeddings)
# ---------------------------------------------------------------------------

def _utc_ts(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def _exp_decay(age_seconds: float, half_life_days: float) -> float:
    # decay factor = 0.5^(age / half_life)
    half_life_seconds = half_life_days * 86400.0
    if half_life_seconds <= 0:
        return 1.0
    return float(0.5 ** (age_seconds / half_life_seconds))


class EmbeddingClient:
    """
    Small wrapper around OpenAI embeddings endpoint with LRU caching.
    Caches embeddings to reduce API calls and costs.
    """
    def __init__(
        self,
        model: str = DEFAULT_EMBED_MODEL,
        cache_size: int = 1000,
        timeout: float = 30.0,
    ) -> None:
        self.client = OpenAI(timeout=timeout)
        self.model = model
        self._cache_size = cache_size
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_order: List[str] = []  # LRU tracking: oldest at front
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_cache_key(self, text: str) -> str:
        """Generate a short hash key for the text."""
        import hashlib
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def embed(self, text: str) -> np.ndarray:
        """Embed text with caching. Returns a copy to prevent mutation."""
        cache_key = self._get_cache_key(text)

        # Check cache first
        if cache_key in self._cache:
            self._cache_hits += 1
            # Move to end of LRU list (most recently used)
            self._cache_order.remove(cache_key)
            self._cache_order.append(cache_key)
            return self._cache[cache_key].copy()

        # Cache miss - call API
        self._cache_misses += 1
        try:
            resp = self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            vec = resp.data[0].embedding
            emb = np.asarray(vec, dtype=np.float32)
        except Exception as e:
            logger.error(f"Embedding API call failed: {e}")
            # Return zero vector as fallback (will have low similarity to everything)
            # 3072 is the dimension for text-embedding-3-large
            return np.zeros(3072, dtype=np.float32)

        # Add to cache with LRU eviction
        self._cache[cache_key] = emb.copy()
        self._cache_order.append(cache_key)

        # Evict oldest if over capacity
        while len(self._cache) > self._cache_size:
            oldest_key = self._cache_order.pop(0)
            del self._cache[oldest_key]

        return emb

    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics for monitoring."""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0.0
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "total_requests": total,
            "hit_rate_percent": round(hit_rate, 1),
            "cache_entries": len(self._cache),
            "cache_capacity": self._cache_size,
        }


# ---------------------------------------------------------------------------
# Schedule memory + shared calendar (SQLite)
# ---------------------------------------------------------------------------

class CalendarStore:
    """
    A simple calendar store:
      - per-agent calendars: calendar_id = f"agent:{agent_id}"
      - shared calendar: calendar_id = "shared"
      - RSVP table to track attendance
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        with self._conn() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    calendar_id TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    location TEXT NOT NULL,
                    start_ts REAL NOT NULL,
                    end_ts REAL NOT NULL,
                    created_at_ts REAL NOT NULL,
                    cancelled INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_events_cal ON events(calendar_id, start_ts);")
            # Unique constraint to prevent duplicate events (same creator, title, start time)
            con.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_events_no_dup
                ON events(calendar_id, created_by, title, start_ts)
            """)
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS rsvps (
                    event_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    updated_at_ts REAL NOT NULL,
                    PRIMARY KEY (event_id, agent_id)
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_rsvps_agent ON rsvps(agent_id);")
            # Meeting invitations table
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS invitations (
                    id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    inviter_id TEXT NOT NULL,
                    invitee_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at_ts REAL NOT NULL,
                    responded_at_ts REAL,
                    UNIQUE(event_id, invitee_id)
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_invitations_invitee ON invitations(invitee_id, status);")

    def create_event(
        self,
        calendar_id: str,
        created_by: str,
        title: str,
        start: datetime,
        end: datetime,
        now: datetime,
        location: str = "TBD",
        description: str = "",
    ) -> str:
        if end <= start:
            raise ValueError("Event end must be after start.")
        event_id = str(uuid.uuid4())
        with self._conn() as con:
            con.execute(
                """
                INSERT INTO events (event_id, calendar_id, created_by, title, description, location,
                                   start_ts, end_ts, created_at_ts, cancelled)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    event_id,
                    calendar_id,
                    created_by,
                    title,
                    description,
                    location,
                    _utc_ts(start),
                    _utc_ts(end),
                    _utc_ts(now),
                ),
            )
        return event_id

    def create_event_if_not_exists(
        self,
        calendar_id: str,
        created_by: str,
        title: str,
        start: datetime,
        end: datetime,
        now: datetime,
        location: str = "TBD",
        description: str = "",
    ) -> tuple[str, bool, Optional[Dict[str, Any]]]:
        """
        Create event only if no similar event exists.

        For shared calendars, checks across ALL creators to prevent duplicates.
        For personal calendars, only checks same creator.

        Returns:
            Tuple of (event_id, created, existing_event_info) where:
            - event_id: ID of the event (new or existing)
            - created: True if new event was created
            - existing_event_info: Dict with title/creator if existing event found, else None
        """
        with self._conn() as con:
            # Use exact title match (case-insensitive) to prevent false positives
            # The time window (1800s = 30min) handles slight timing variations
            # Previously used LIKE pattern which matched unrelated events
            # (e.g., "Project Review" would match "Project Planning")
            if calendar_id == "shared":
                existing = con.execute(
                    """
                    SELECT event_id, title, created_by FROM events
                    WHERE calendar_id = ?
                    AND ABS(start_ts - ?) < 1800
                    AND LOWER(title) = LOWER(?)
                    AND cancelled = 0
                    """,
                    (calendar_id, _utc_ts(start), title)
                ).fetchone()
            else:
                # Personal calendar - only check same creator
                existing = con.execute(
                    """
                    SELECT event_id, title, created_by FROM events
                    WHERE calendar_id = ? AND created_by = ?
                    AND ABS(start_ts - ?) < 1800
                    AND LOWER(title) = LOWER(?)
                    AND cancelled = 0
                    """,
                    (calendar_id, created_by, _utc_ts(start), title)
                ).fetchone()

            if existing:
                existing_info = {
                    "event_id": existing[0],
                    "title": existing[1],
                    "created_by": existing[2],
                }
                return existing[0], False, existing_info

        # Create new event
        event_id = self.create_event(
            calendar_id, created_by, title, start, end, now, location, description
        )
        return event_id, True, None

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get a single event by ID."""
        with self._conn() as con:
            row = con.execute(
                """
                SELECT event_id, calendar_id, created_by, title, description, location,
                       start_ts, end_ts, cancelled
                FROM events WHERE event_id = ?
                """,
                (event_id,)
            ).fetchone()

        if not row:
            return None

        event_id, cal_id, created_by, title, desc, loc, s_ts, e_ts, cancelled = row
        return {
            "event_id": event_id,
            "calendar_id": cal_id,
            "created_by": created_by,
            "title": title,
            "description": desc,
            "location": loc,
            "start_datetime_iso": datetime.fromtimestamp(float(s_ts), tz=timezone.utc).isoformat(),
            "end_datetime_iso": datetime.fromtimestamp(float(e_ts), tz=timezone.utc).isoformat(),
            "cancelled": bool(cancelled),
        }

    def get_events_created_by(
        self, agent_id: str, start: datetime, end: datetime, calendar_id: str = "shared"
    ) -> List[Dict[str, Any]]:
        """Get all events created by a specific agent within a time range."""
        with self._conn() as con:
            rows = con.execute(
                """
                SELECT event_id, title, description, location, start_ts, end_ts, cancelled
                FROM events
                WHERE calendar_id = ? AND created_by = ?
                AND start_ts >= ? AND start_ts < ?
                AND cancelled = 0
                ORDER BY start_ts ASC
                """,
                (calendar_id, agent_id, _utc_ts(start), _utc_ts(end))
            ).fetchall()

        return [
            {
                "event_id": eid,
                "title": title,
                "description": desc,
                "location": loc,
                "start_datetime_iso": datetime.fromtimestamp(float(s_ts), tz=timezone.utc).isoformat(),
                "end_datetime_iso": datetime.fromtimestamp(float(e_ts), tz=timezone.utc).isoformat(),
                "cancelled": bool(cancelled),
            }
            for eid, title, desc, loc, s_ts, e_ts, cancelled in rows
        ]

    def list_events(
        self,
        calendar_id: str,
        start: datetime,
        end: datetime,
        include_cancelled: bool = False,
    ) -> List[Dict[str, Any]]:
        start_ts = _utc_ts(start)
        end_ts = _utc_ts(end)
        where_cancelled = "" if include_cancelled else "AND cancelled = 0"
        with self._conn() as con:
            rows = con.execute(
                f"""
                SELECT event_id, created_by, title, description, location, start_ts, end_ts, cancelled
                FROM events
                WHERE calendar_id = ? AND start_ts < ? AND end_ts > ? {where_cancelled}
                ORDER BY start_ts ASC
                """,
                (calendar_id, end_ts, start_ts),
            ).fetchall()

        out: List[Dict[str, Any]] = []
        for event_id, created_by, title, description, location, s_ts, e_ts, cancelled in rows:
            out.append(
                {
                    "event_id": event_id,
                    "calendar_id": calendar_id,
                    "created_by": created_by,
                    "title": title,
                    "description": description,
                    "location": location,
                    "start_datetime_iso": datetime.fromtimestamp(float(s_ts), tz=timezone.utc).isoformat(),
                    "end_datetime_iso": datetime.fromtimestamp(float(e_ts), tz=timezone.utc).isoformat(),
                    "cancelled": bool(cancelled),
                }
            )
        return out

    def rsvp(self, event_id: str, agent_id: str, status: Literal["yes", "no", "maybe"], now: datetime) -> None:
        with self._conn() as con:
            con.execute(
                """
                INSERT INTO rsvps (event_id, agent_id, status, updated_at_ts)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(event_id, agent_id) DO UPDATE SET
                    status = excluded.status,
                    updated_at_ts = excluded.updated_at_ts
                """,
                (event_id, agent_id, status, _utc_ts(now)),
            )

    def get_rsvps_for_event(self, event_id: str) -> List[Dict[str, Any]]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT agent_id, status, updated_at_ts FROM rsvps WHERE event_id = ?",
                (event_id,),
            ).fetchall()
        return [
            {
                "agent_id": agent_id,
                "status": status,
                "updated_at_ts": float(ts),
            }
            for agent_id, status, ts in rows
        ]

    def prune_past_events(self, calendar_id: str, now: datetime) -> int:
        """Remove past events (ended before now) from a given calendar."""
        now_ts = _utc_ts(now)
        with self._conn() as con:
            cur = con.execute(
                """
                DELETE FROM events
                WHERE calendar_id = ? AND end_ts < ?
                """,
                (calendar_id, now_ts),
            )
        return int(cur.rowcount)

    # ---------------------------
    # Meeting invitations
    # ---------------------------

    def add_invitation(
        self,
        event_id: str,
        inviter_id: str,
        invitee_id: str,
        now: datetime,
    ) -> str:
        """
        Create an invitation for an agent to attend a meeting.

        Args:
            event_id: The event to invite to
            inviter_id: The agent sending the invitation
            invitee_id: The agent being invited
            now: Current simulation datetime

        Returns:
            The invitation ID
        """
        invite_id = str(uuid.uuid4())
        with self._conn() as con:
            # Only insert if no existing invitation, or update only if still pending
            # This prevents resetting declined/accepted invitations back to pending
            con.execute(
                """
                INSERT INTO invitations (id, event_id, inviter_id, invitee_id, status, created_at_ts)
                VALUES (?, ?, ?, ?, 'pending', ?)
                ON CONFLICT(event_id, invitee_id) DO UPDATE SET
                    inviter_id = excluded.inviter_id,
                    created_at_ts = excluded.created_at_ts
                WHERE invitations.status = 'pending'
                """,
                (invite_id, event_id, inviter_id, invitee_id, _utc_ts(now)),
            )
        return invite_id

    def get_pending_invitations(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Get all pending invitations for an agent.

        Returns list of invitation dicts with event details.
        """
        with self._conn() as con:
            rows = con.execute(
                """
                SELECT i.id, i.event_id, i.inviter_id, i.created_at_ts,
                       e.title, e.location, e.start_ts, e.end_ts
                FROM invitations i
                JOIN events e ON i.event_id = e.event_id
                WHERE i.invitee_id = ? AND i.status = 'pending' AND e.cancelled = 0
                ORDER BY e.start_ts ASC
                """,
                (agent_id,),
            ).fetchall()

        return [
            {
                "invitation_id": inv_id,
                "event_id": event_id,
                "inviter_id": inviter_id,
                "created_at_ts": float(created_ts),
                "event_title": title,
                "event_location": location,
                "event_start_iso": datetime.fromtimestamp(float(start_ts), tz=timezone.utc).isoformat(),
                "event_end_iso": datetime.fromtimestamp(float(end_ts), tz=timezone.utc).isoformat(),
            }
            for inv_id, event_id, inviter_id, created_ts, title, location, start_ts, end_ts in rows
        ]

    def respond_to_invitation(
        self,
        event_id: str,
        agent_id: str,
        accept: bool,
        now: datetime,
    ) -> bool:
        """
        Respond to a meeting invitation.

        Args:
            event_id: The event being responded to
            agent_id: The agent responding
            accept: True to accept, False to decline
            now: Current simulation datetime

        Returns:
            True if invitation was found and updated, False otherwise
        """
        status = "accepted" if accept else "declined"

        with self._conn() as con:
            cur = con.execute(
                """
                UPDATE invitations
                SET status = ?, responded_at_ts = ?
                WHERE event_id = ? AND invitee_id = ? AND status = 'pending'
                """,
                (status, _utc_ts(now), event_id, agent_id),
            )
            updated = cur.rowcount > 0

            # If accepted, also create an RSVP
            if updated and accept:
                con.execute(
                    """
                    INSERT INTO rsvps (event_id, agent_id, status, updated_at_ts)
                    VALUES (?, ?, 'yes', ?)
                    ON CONFLICT(event_id, agent_id) DO UPDATE SET
                        status = 'yes',
                        updated_at_ts = excluded.updated_at_ts
                    """,
                    (event_id, agent_id, _utc_ts(now)),
                )

        return updated

    def get_agent_calendar_view(
        self,
        agent_id: str,
        start: datetime,
        end: datetime,
    ) -> Dict[str, Any]:
        """
        Get a comprehensive calendar view for an agent.

        This provides a "work calendar" experience showing:
        - Meetings the agent organized (with response counts)
        - Meetings the agent has accepted (includes organizer info)
        - Pending invitations requiring response
        - Declined invitations (for reference)

        Args:
            agent_id: The agent whose calendar to view
            start: Start of time range
            end: End of time range

        Returns:
            {
                "organized": [...],     # Meetings I created
                "attending": [...],     # Meetings I accepted (not mine)
                "pending": [...],       # Invitations needing my response
                "declined": [...],      # Invitations I declined
                "summary": {"total_meetings": N, "pending_invitations": N}
            }
        """
        start_ts = _utc_ts(start)
        end_ts = _utc_ts(end)

        with self._conn() as con:
            # Meetings I organized
            organized_rows = con.execute(
                """
                SELECT e.event_id, e.title, e.location, e.start_ts, e.end_ts, e.description,
                       (SELECT COUNT(*) FROM invitations WHERE event_id = e.event_id AND status = 'accepted') as accepted_count,
                       (SELECT COUNT(*) FROM invitations WHERE event_id = e.event_id AND status = 'pending') as pending_count,
                       (SELECT COUNT(*) FROM invitations WHERE event_id = e.event_id AND status = 'declined') as declined_count
                FROM events e
                WHERE e.created_by = ? AND e.cancelled = 0
                AND e.start_ts >= ? AND e.start_ts < ?
                ORDER BY e.start_ts
                """,
                (agent_id, start_ts, end_ts)
            ).fetchall()

            # Meetings I accepted (not mine)
            attending_rows = con.execute(
                """
                SELECT e.event_id, e.title, e.location, e.start_ts, e.end_ts,
                       e.created_by as organizer
                FROM events e
                JOIN invitations i ON e.event_id = i.event_id
                WHERE i.invitee_id = ? AND i.status = 'accepted'
                AND e.cancelled = 0 AND e.start_ts >= ? AND e.start_ts < ?
                ORDER BY e.start_ts
                """,
                (agent_id, start_ts, end_ts)
            ).fetchall()

            # Declined invitations (for reference)
            declined_rows = con.execute(
                """
                SELECT e.event_id, e.title, e.start_ts, i.inviter_id
                FROM events e
                JOIN invitations i ON e.event_id = i.event_id
                WHERE i.invitee_id = ? AND i.status = 'declined'
                AND e.cancelled = 0 AND e.start_ts >= ? AND e.start_ts < ?
                ORDER BY e.start_ts
                """,
                (agent_id, start_ts, end_ts)
            ).fetchall()

        # Get pending invitations (already implemented)
        all_pending = self.get_pending_invitations(agent_id)
        # Filter to time range
        pending_filtered = [
            p for p in all_pending
            if start_ts <= datetime.fromisoformat(p['event_start_iso']).timestamp() < end_ts
        ]

        # Format organized meetings
        organized = [
            {
                "event_id": row[0],
                "title": row[1],
                "location": row[2],
                "start_datetime_iso": datetime.fromtimestamp(float(row[3]), tz=timezone.utc).isoformat(),
                "end_datetime_iso": datetime.fromtimestamp(float(row[4]), tz=timezone.utc).isoformat(),
                "description": row[5],
                "responses": {
                    "accepted": row[6],
                    "pending": row[7],
                    "declined": row[8],
                },
            }
            for row in organized_rows
        ]

        # Format attending meetings
        attending = [
            {
                "event_id": row[0],
                "title": row[1],
                "location": row[2],
                "start_datetime_iso": datetime.fromtimestamp(float(row[3]), tz=timezone.utc).isoformat(),
                "end_datetime_iso": datetime.fromtimestamp(float(row[4]), tz=timezone.utc).isoformat(),
                "organizer": row[5],
            }
            for row in attending_rows
        ]

        # Format declined
        declined = [
            {
                "event_id": row[0],
                "title": row[1],
                "start_datetime_iso": datetime.fromtimestamp(float(row[2]), tz=timezone.utc).isoformat(),
                "inviter": row[3],
            }
            for row in declined_rows
        ]

        return {
            "organized": organized,
            "attending": attending,
            "pending": pending_filtered,
            "declined": declined,
            "summary": {
                "total_meetings": len(organized) + len(attending),
                "pending_invitations": len(pending_filtered),
            },
        }

    def get_all_agent_ids(self) -> List[str]:
        """Get all known agent IDs from invitations and RSVPs."""
        with self._conn() as con:
            rows = con.execute(
                """
                SELECT DISTINCT agent_id FROM (
                    SELECT inviter_id AS agent_id FROM invitations
                    UNION
                    SELECT invitee_id AS agent_id FROM invitations
                    UNION
                    SELECT agent_id FROM rsvps
                    UNION
                    SELECT created_by AS agent_id FROM events
                )
                """
            ).fetchall()
        return [row[0] for row in rows]

    def get_agent_meetings(
        self,
        agent_id: str,
        start: datetime,
        end: datetime,
    ) -> List[Dict[str, Any]]:
        """
        Get all meetings an agent has created or accepted for a time range.

        Args:
            agent_id: The agent to get meetings for
            start: Start of time range
            end: End of time range

        Returns:
            List of meeting dicts with role (organizer/attendee)
        """
        start_ts = _utc_ts(start)
        end_ts = _utc_ts(end)

        with self._conn() as con:
            # Meetings I created
            created_rows = con.execute(
                """
                SELECT event_id, title, description, location, start_ts, end_ts
                FROM events
                WHERE created_by = ? AND cancelled = 0
                  AND start_ts < ? AND end_ts > ?
                ORDER BY start_ts ASC
                """,
                (agent_id, end_ts, start_ts),
            ).fetchall()

            # Meetings I accepted (via RSVP yes or accepted invitation)
            accepted_rows = con.execute(
                """
                SELECT DISTINCT e.event_id, e.title, e.description, e.location, e.start_ts, e.end_ts
                FROM events e
                LEFT JOIN rsvps r ON e.event_id = r.event_id AND r.agent_id = ?
                LEFT JOIN invitations i ON e.event_id = i.event_id AND i.invitee_id = ?
                WHERE e.cancelled = 0
                  AND e.start_ts < ? AND e.end_ts > ?
                  AND e.created_by != ?
                  AND (r.status = 'yes' OR i.status = 'accepted')
                ORDER BY e.start_ts ASC
                """,
                (agent_id, agent_id, end_ts, start_ts, agent_id),
            ).fetchall()

        meetings = []

        # Add meetings I created (as organizer)
        for event_id, title, description, location, s_ts, e_ts in created_rows:
            meetings.append({
                "event_id": event_id,
                "title": title,
                "description": description,
                "location": location,
                "start_datetime_iso": datetime.fromtimestamp(float(s_ts), tz=timezone.utc).isoformat(),
                "end_datetime_iso": datetime.fromtimestamp(float(e_ts), tz=timezone.utc).isoformat(),
                "role": "organizer",
            })

        # Add meetings I accepted (as attendee)
        created_ids = {m["event_id"] for m in meetings}
        for event_id, title, description, location, s_ts, e_ts in accepted_rows:
            if event_id not in created_ids:  # Avoid duplicates
                meetings.append({
                    "event_id": event_id,
                    "title": title,
                    "description": description,
                    "location": location,
                    "start_datetime_iso": datetime.fromtimestamp(float(s_ts), tz=timezone.utc).isoformat(),
                    "end_datetime_iso": datetime.fromtimestamp(float(e_ts), tz=timezone.utc).isoformat(),
                    "role": "attendee",
                })

        # Sort by start time
        meetings.sort(key=lambda m: m["start_datetime_iso"])
        return meetings


# ---------------------------------------------------------------------------
# Agent run context
# ---------------------------------------------------------------------------

@dataclass
class SimContext:
    """
    Context passed to tools at runtime.

    Supports both old (memory-based) and new (simulation-based) approaches:
    - Old: memory is SQLiteVectorMemory for retrieval tools
    - New: simulation is BuildingSimulationAdapter for applying decisions

    For the new GenerativeAgent architecture, memory is handled by
    MemoryStream in cognitive_modules.py, so memory can be None.
    """
    occupant_id: str
    now: datetime
    calendar: CalendarStore
    memory: Optional[Any] = None  # Legacy, unused in new architecture
    simulation: Optional["BuildingSimulationAdapter"] = None  # For new architecture
    configured_agent_ids: List[str] = None  # All valid agent IDs from config

    def __post_init__(self):
        if self.configured_agent_ids is None:
            self.configured_agent_ids = []


# ---------------------------------------------------------------------------
# Tools (function calling)
# ---------------------------------------------------------------------------

@function_tool
def get_current_datetime(ctx: RunContextWrapper[SimContext]) -> str:
    """Return the current simulation datetime in ISO format (UTC)."""
    return ctx.context.now.astimezone(timezone.utc).isoformat()

# NOTE: retrieve_core_memories and retrieve_action_memories tools removed
# Memory retrieval now handled via MemoryStream in cognitive_modules.py

@function_tool
def list_my_schedule(ctx: RunContextWrapper[SimContext], days_ahead: int = 7) -> List[Dict[str, Any]]:
    """
    List upcoming events from this occupant's personal schedule.
    Past events may be pruned separately by the simulation loop.
    """
    start = ctx.context.now
    end = start + timedelta(days=days_ahead)
    return ctx.context.calendar.list_events(
        calendar_id=f"agent:{ctx.context.occupant_id}",
        start=start,
        end=end,
    )

@function_tool
def list_shared_calendar(ctx: RunContextWrapper[SimContext], days_ahead: int = 14) -> List[Dict[str, Any]]:
    """List upcoming events from the shared calendar."""
    start = ctx.context.now
    end = start + timedelta(days=days_ahead)
    return ctx.context.calendar.list_events(
        calendar_id="shared",
        start=start,
        end=end,
    )


@function_tool
def get_my_calendar(ctx: RunContextWrapper[SimContext], days_ahead: int = 7) -> Dict[str, Any]:
    """
    Get your complete calendar view including:
    - Meetings you organized (with response counts from invitees)
    - Meetings you're attending (that others organized)
    - Pending invitations requiring your response
    - Meetings you declined (for reference)

    This is like checking your work calendar to see your schedule and outstanding invitations.

    Args:
        days_ahead: How many days ahead to look (default 7)

    Returns:
        Dictionary with 'organized', 'attending', 'pending', 'declined' lists and a 'summary'.
    """
    start = ctx.context.now
    end = start + timedelta(days=days_ahead)
    return ctx.context.calendar.get_agent_calendar_view(
        ctx.context.occupant_id, start, end
    )


@function_tool
def create_shared_event(
    ctx: RunContextWrapper[SimContext],
    title: str,
    start_datetime_iso: str,
    end_datetime_iso: str,
    location: str = "TBD",
    description: str = "",
) -> Dict[str, Any]:
    """
    Create a new event in the shared calendar.
    Datetimes must be ISO-8601; assumed UTC if no timezone is provided.

    Checks for similar existing meetings to avoid duplicates.
    If a similar meeting already exists, returns info about it instead of creating a duplicate.
    """
    start = datetime.fromisoformat(start_datetime_iso)
    end = datetime.fromisoformat(end_datetime_iso)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    event_id, created, existing_info = ctx.context.calendar.create_event_if_not_exists(
        calendar_id="shared",
        created_by=ctx.context.occupant_id,
        title=title,
        start=start,
        end=end,
        now=ctx.context.now,
        location=location,
        description=description,
    )

    if not created and existing_info:
        return {
            "event_id": event_id,
            "created": False,
            "message": (
                f"A similar meeting '{existing_info['title']}' already exists at this time "
                f"(created by {existing_info['created_by']}). You may want to check if you "
                "should attend that meeting instead of creating a new one."
            ),
            "existing_meeting": existing_info,
        }

    return {
        "event_id": event_id,
        "created": True,
        "title": title,
        "start": start.isoformat(),
        "end": end.isoformat(),
    }

@function_tool
def rsvp_shared_event(
    ctx: RunContextWrapper[SimContext],
    event_id: str,
    status: Literal["yes", "no", "maybe"],
) -> Dict[str, Any]:
    """RSVP to a shared calendar event."""
    ctx.context.calendar.rsvp(event_id=event_id, agent_id=ctx.context.occupant_id, status=status, now=ctx.context.now)
    return {"event_id": event_id, "agent_id": ctx.context.occupant_id, "status": status}


# ---------------------------------------------------------------------------
# Meeting invitation tools
# ---------------------------------------------------------------------------

@function_tool
def invite_agents_to_meeting(
    ctx: RunContextWrapper[SimContext],
    event_id: str,
    invitee_ids: List[str],
) -> Dict[str, Any]:
    """
    Invite other agents to an existing meeting.

    Args:
        event_id: The event to invite agents to
        invitee_ids: List of agent IDs to invite (must be valid agent IDs from list_known_agents)

    Returns:
        Dict with list of created invitations and any invalid IDs that were skipped
    """
    invitations = []
    invalid_ids = []
    valid_agent_ids = set(ctx.context.configured_agent_ids)

    for invitee_id in invitee_ids:
        # Skip self
        if invitee_id == ctx.context.occupant_id:
            continue
        # Validate against configured agents
        if invitee_id not in valid_agent_ids:
            invalid_ids.append(invitee_id)
            continue
        # Create invitation
        invite_id = ctx.context.calendar.add_invitation(
            event_id=event_id,
            inviter_id=ctx.context.occupant_id,
            invitee_id=invitee_id,
            now=ctx.context.now,
        )
        invitations.append({"invitation_id": invite_id, "invitee_id": invitee_id})

    result = {"event_id": event_id, "invitations_sent": invitations}
    if invalid_ids:
        result["invalid_agent_ids"] = invalid_ids
        result["warning"] = f"Unknown agent IDs were skipped: {invalid_ids}. Use list_known_agents() to see valid IDs."
    return result


@function_tool
def list_pending_invitations(ctx: RunContextWrapper[SimContext]) -> List[Dict[str, Any]]:
    """
    List all pending meeting invitations for this agent.

    Returns list of invitations with event details that need a response.
    """
    return ctx.context.calendar.get_pending_invitations(ctx.context.occupant_id)


@function_tool
def list_known_agents(ctx: RunContextWrapper[SimContext]) -> List[str]:
    """
    List all known agent IDs that can be invited to meetings (excluding self).

    Returns list of valid agent IDs from the simulation configuration.
    """
    # Use configured agent IDs instead of querying database
    # This ensures agents can only invite valid agents from the config
    return [aid for aid in ctx.context.configured_agent_ids if aid != ctx.context.occupant_id]


@function_tool
def list_my_meetings(ctx: RunContextWrapper[SimContext]) -> List[Dict[str, Any]]:
    """
    List meetings you have created or accepted for today.

    Returns list of meetings with:
    - event_id: Unique meeting identifier
    - title: Meeting title
    - start_datetime_iso: Start time
    - end_datetime_iso: End time
    - location: Meeting location
    - role: "organizer" (you created it) or "attendee" (you accepted an invite)

    Use this FIRST to check what meetings you already have scheduled before:
    - Responding to invitations (avoid accepting duplicates)
    - Creating new meetings (avoid creating duplicates)
    """
    now = ctx.context.now
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = now.replace(hour=23, minute=59, second=59, microsecond=999999)

    return ctx.context.calendar.get_agent_meetings(
        agent_id=ctx.context.occupant_id,
        start=day_start,
        end=day_end,
    )


@function_tool
def respond_to_invitation(
    ctx: RunContextWrapper[SimContext],
    event_id: str,
    accept: bool,
) -> Dict[str, Any]:
    """
    Accept or decline a meeting invitation.

    Args:
        event_id: The event to respond to
        accept: True to accept, False to decline

    Returns:
        Dict with response status and message explaining what happened
    """
    success = ctx.context.calendar.respond_to_invitation(
        event_id=event_id,
        agent_id=ctx.context.occupant_id,
        accept=accept,
        now=ctx.context.now,
    )

    if success:
        message = f"Successfully {'accepted' if accept else 'declined'} meeting invitation"
    else:
        message = "No action taken - invitation was already responded to or doesn't exist"

    return {
        "event_id": event_id,
        "response": "accepted" if accept else "declined",
        "success": success,
        "message": message,
    }


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def build_step_agent(occupant_id: str, model: str = DEFAULT_AGENT_MODEL) -> Agent[SimContext]:
    """
    Agent used at decision checkpoints to decide occupant actions.
    Uses structured output OccupantStepDecision.

    Decision checkpoints occur:
    - Hourly (regular check-in)
    - At meeting start/end times
    - At planned lunch/break times

    Args:
        occupant_id: The agent's ID
        model: The OpenAI model to use (default: DEFAULT_AGENT_MODEL from config)
    """
    instructions = f"""
You are a simulated building occupant agent (ID: {occupant_id}).

<task_context>
You are making a decision at a CHECKPOINT. Consider the checkpoint reason
(hourly, meeting_start, lunch_time, etc.) when deciding actions.
</task_context>

<output_verbosity_spec>
- Reasoning fields: 1-2 sentences max per decision category
- If action is "maintain_current" or "keep_current": can be just 2-5 words
- Only explain actions you're taking, not actions you're NOT taking
</output_verbosity_spec>

<scope_discipline>
- You are simulating a realistic office worker going about their day
- Respond primarily to what's happening at this checkpoint moment
- Prefer stability: if comfortable, maintain current state
- Only act on genuine discomfort or clear needs - avoid constant adjustments
- Honor your personality and habits
- If circumstances haven't changed, your decisions shouldn't either
- Focus on the checkpoint reason rather than hypothetical scenarios
</scope_discipline>

<available_actions>

Comfort Controls:
- no_op: do nothing (use when comfortable and no immediate needs)
- thermostat_adjust: adjust thermostat based on how you feel (parameters: setpoint_c)
- window_set: open/close window (parameters: open: true/false)
- lights_set: control lights (parameters: light_name, on: true/false)

Equipment:
- equipment_set: turn equipment on/off (parameters: equipment_name, on: true/false)
- use_appliance: use kitchen appliance (parameters: appliance_name) - kettles/coffee machines auto-off

Location & Movement:
- move_to: move to a different location (parameters: location)
  NOTE: Your desk was assigned when you arrived. You cannot change desks during the day.

EQUIPMENT AWARENESS:
- Your equipment status (ON/OFF) is shown in the state section of your prompt
- You NEED your laptop and monitor ON to do your work
- If equipment is OFF when you arrive or need to work, turn it on using equipment_set
- Turn off equipment when leaving for the day to save energy

Meetings:
- attend_meeting: physically go to meeting room (parameters: meeting_title)
- leave_meeting: return to your desk from meeting room

Lunch & Breaks:
- go_to_lunch: have lunch in the break room (stay in building)
- go_out_for_lunch: leave building for lunch (cafe, restaurant, walk)
- return_from_lunch: return to work after lunch (REQUIRED after going out)
- take_break: take a short break inside (parameters: location - "at_desk" or "break_area")
- go_out_for_break: leave building for a short break (walk, fresh air, coffee run)
- return_from_break: return to work after break (REQUIRED after going outside)

Social:
- initiate_conversation: start a conversation with a colleague (parameters: agent_id, topic)
</available_actions>

<decision_guidance>
THERMOSTAT - Focus on how you FEEL:
- Check the actual indoor temperature and notice how you feel (too hot, too cold, comfortable)
- If you feel TOO HOT: Request direction="cooler" - shifts the comfort band down
- If you feel TOO COLD: Request direction="warmer" - shifts the comfort band up
- adjustment_amount: Use 0.5 (small), 1.0 (medium), or 1.5-2.0 (large) degrees
- IMPORTANT: Adjustments are CAPPED at +/- 2C from the base setpoint (21C)
  - Large requests (e.g., +5C) will be limited to +2C
  - If multiple people request changes, their votes are AVERAGED
- Your personality affects sensitivity: some people run hot, others cold
- Only adjust when genuinely uncomfortable - don't manage setpoints abstractly
- Use your thermal preferences and memories to guide comfort decisions
- Consider outdoor weather: if it's hot outside, cooler indoor temps feel refreshing

LIGHTING - Consider your preferences and current needs:
- Check natural light level (bright, moderate, dim, dark)
- Consider your task: focused work may need good lighting, presentations may need dimmed lights
- Turn on desk lamp if natural light is insufficient
- Turn off lights when leaving an area or if natural light is adequate

EQUIPMENT - Be mindful of energy:
- Turn on equipment when you need it (laptop, monitor must be ON to work)
- Turn off equipment when done, especially shared equipment
- Meeting room equipment (projector, phone) should be off when room is empty

KITCHEN EQUIPMENT (in break_area):
- For tea: use_appliance(appliance_name="kettle") - auto-off after 2 minutes
- For coffee: use_appliance(appliance_name="coffee_machine") - auto-off after 10 minutes
- For heating food: use_appliance(appliance_name="microwave") - auto-off after 5 minutes
- These appliances turn off automatically, you don't need to turn them off manually

BREAKS - Follow your daily plan and preferences:
- At break checkpoints, consider taking a break based on your planned schedule
- You have THREE options:
  1. "take_break" with location="at_desk": Quick break at your desk
  2. "take_break" with location="break_area": Go to kitchen (make tea/coffee, social)
  3. "go_out_for_break": Leave the building for fresh air, walk, or coffee run
- IMPORTANT: If you go outside (go_out_for_break), you MUST use "return_from_break" to come back
- Your core memories about tea/coffee preferences should guide break behavior
- Consider social commitments - if you agreed to take a break with a colleague, honor it!

SOCIAL COMMITMENTS:
- If you made agreements with colleagues during conversations (e.g., "let's grab coffee at 10"),
  these will be shown in <social_commitments> section of your prompt
- Honor your commitments! If you agreed to an activity with someone, follow through
- Use the appropriate action (take_break, go_out_for_break) to fulfill social commitments

LUNCH - Follow your daily plan:
- At lunch checkpoint, take lunch according to your plan
- "at_desk": Stay and eat at your desk
- "break_area": Eat in the kitchen/break area (social)
- "go_out": Leave the building (nice weather, need fresh air, get food)
- Your daily plan includes what food you brought today
- If your food needs heating (leftovers, soup, etc.), use the microwave in break_area
- If your food is cold (salad, sandwich), no heating needed

MEETINGS - As host or attendee:
- At meeting_start checkpoint: use attend_meeting to go to the meeting room
- If you're the HOST: you may need to set up equipment (projector, lighting)
- At meeting_end checkpoint: use leave_meeting to return to your desk
- If you're the HOST: turn off any equipment you turned on

CONVERSATIONS - Engage with colleagues:
- You can initiate conversations on various topics, not just about the thermostat
- Consider who is at your location and your relationship with them
- Topics can include: work matters, social chat, comfort concerns, etc.

PLAN UPDATES - Adapt your daily plan as circumstances change:
- At each decision checkpoint, consider if your plan needs updating
- You may update your plan if:
  * Weather has changed significantly (e.g., now raining, so don't go out for lunch)
  * A new meeting was scheduled that affects your breaks
  * You feel tired/energetic and want to adjust break timing
  * Colleagues invited you to lunch and you want to join them
  * Any circumstance that makes your current plan less suitable
- You can ONLY update FUTURE events - past times cannot be changed
- Use the plan_update decision to specify changes
- Keep updates focused on what's actually changed; don't rewrite the whole plan
- If your plan is still good, choose "keep_current"

LOCATIONS:
- desk_area: Main workspace with desks
- meeting_room: Enclosed room for meetings (has projector, conference phone)
- break_area: Kitchen/break room (has kettle, coffee machine, microwave, fridge)
- shared_area: Common area (has photocopier)
</decision_guidance>

<decision_priorities>
CRITICAL (address first):
1. Meeting attendance (if checkpoint is meeting_start/end)
2. Lunch/break transitions (if checkpoint is lunch_time/break_time)
3. Equipment needed for work (laptop, monitor must be ON)

ROUTINE (only if comfortable with critical):
4. Comfort adjustments (only if genuinely uncomfortable)
5. Conversations (only if appropriate and time permits)
6. Plan updates (only if circumstances significantly changed)
</decision_priorities>

<stop_conditions>
Return your decision when:
- All required decision categories have a value
- You have addressed the checkpoint reason
- DO NOT continue elaborating beyond the structured output
</stop_conditions>

<output_requirements>
IMPORTANT: Return a complete structured decision for ALL categories. Every field is REQUIRED.

Required decisions:
- thermostat: "adjust" or "maintain_current"
- lighting: "turn_on", "turn_off", "adjust_brightness", or "keep_current"
- equipment_decisions: A LIST with one entry per equipment item (can be empty if no equipment nearby)
- location: "move" or "stay"
- conversation: "initiate" or "none"
- plan_update: "update" or "keep_current"
- break_decision: "take_break", "go_out_for_break", "continue_working", "return_from_lunch", "return_from_break", or "not_applicable"
  * Use "not_applicable" if this checkpoint is NOT about breaks/lunch
  * Use "continue_working" if it's a break-eligible time but you choose not to take a break
- meeting_equipment: "set_equipment", "accept_current", or "not_applicable"
  * Use "not_applicable" if you're NOT at a meeting boundary or NOT the meeting host
  * Use "accept_current" if equipment is already set up correctly
  * Use "set_equipment" with equipment_changes list to adjust projector, phone, blinds, etc.

Each decision MUST include brief reasoning (1-2 sentences max).
DO NOT omit any decision category - all fields must be present in your response.
</output_requirements>
""".strip()

    return Agent(
        name=f"occupant_step_{occupant_id}",
        instructions=instructions,
        model=model,
        model_settings=ModelSettings(reasoning_effort="medium"),  # Step decisions require cognitive reasoning
        tools=[
            get_current_datetime,
            list_shared_calendar,
        ],
        output_type=AgentOutputSchema(StepDecisions, strict_json_schema=True),
    )


def build_arrival_agent(occupant_id: str, model: str = DEFAULT_AGENT_MODEL) -> Agent[SimContext]:
    """
    Agent used ONCE upon arrival to select desk for the day.

    This is a separate decision from regular step decisions. The agent
    selects their desk based on availability, preferences, and colleague locations.
    This desk assignment is LOCKED for the entire day.

    Args:
        occupant_id: The agent's ID
        model: The OpenAI model to use (default: DEFAULT_AGENT_MODEL from config)
    """
    instructions = f"""
You are a simulated building occupant agent (ID: {occupant_id}).

You have just ARRIVED at work for the day. Your first task is to select your desk.

IMPORTANT: This is a ONE-TIME decision. The desk you choose will be YOUR desk for
the ENTIRE day. You cannot change it later.

DECISION CRITERIA:
Consider the following when choosing your desk:
1. Your preferred desk (if you have one from habit or preference)
2. Proximity to colleagues you work closely with
3. Natural light and window access
4. Quiet vs collaborative areas
5. Which desks are currently available

EQUIPMENT:
After selecting your desk, specify which equipment to turn on immediately.
Typically you'll want your laptop and monitor ON to start working.

OUTPUT:
- selected_desk: The desk ID you're claiming (e.g., "Desk_A", "Desk_B", "Desk_C")
- reasoning: Explain WHY you chose this desk
- equipment_to_turn_on: List equipment to activate (e.g., ["laptop", "monitor"])
""".strip()

    return Agent(
        name=f"arrival_{occupant_id}",
        instructions=instructions,
        model=model,
        model_settings=ModelSettings(reasoning_effort="low"),  # Simple desk selection
        tools=[],  # No tools needed - state is provided in prompt
        output_type=AgentOutputSchema(DeskSelectionDecision, strict_json_schema=True),
    )


def build_day_planner_agent(occupant_id: str, model: str = DEFAULT_AGENT_MODEL) -> Agent[SimContext]:
    """
    Agent used once per day to plan attendance and working hours.
    Uses structured output DailyPlan.

    Args:
        occupant_id: The agent's ID
        model: The OpenAI model to use (default: DEFAULT_AGENT_MODEL from config)
    """
    instructions = f"""
You are a simulated building occupant agent (ID: {occupant_id}).

Task: Create today's work schedule including meetings, lunch, and breaks.

WORKFLOW (follow in order):
1. FIRST: Call list_my_meetings() to see meetings you've already organized or accepted
2. Check list_pending_invitations() for NEW invitations - only respond to ones not in your meetings list
3. If you want to schedule a meeting:
   a. Check list_shared_calendar() to see ALL existing meetings
   b. ONLY create if no similar meeting exists at that time
   c. Use create_shared_event() then invite_agents_to_meeting()
4. Plan your lunch and breaks based on your preferences and schedule
5. Output your DailyPlan

Required DailyPlan fields (HH:MM format):
- intended_arrival_time, actual_arrival_time: when you plan to arrive
- intended_departure_time, actual_departure_time: when you plan to leave
- meetings: list of MeetingPlan objects (can be empty if no meetings)
- lunch_plan: where and when to have lunch
- morning_break: optional morning break (tea/coffee)
- afternoon_break: optional afternoon break
- comfort_preferences: your temperature and lighting preferences for the day

LUNCH PLANNING:
- Choose where to have lunch based on your preferences and today's schedule:
  * "at_desk": Quick working lunch, eat at your desk
  * "break_area": Social lunch in the kitchen/break area
  * "go_out": Leave the building (cafe, restaurant, walk)
- Consider weather, meetings before/after lunch, and your energy level
- Use your core memories about lunch habits to guide this decision

BREAK PLANNING:
- Plan morning break (typically 10:00-10:30) and afternoon break (typically 15:00-15:30)
- Choose location: "at_desk" (quick) or "break_area" (social, make tea/coffee)
- Activity: tea, coffee, water, snack, stretch, walk
- Use your core memories about tea/coffee preferences:
  * Do you prefer tea or coffee?
  * How do you take your tea/coffee? (black, with milk, with sugar)
  * Do you like to socialize during breaks or prefer quiet time?

MEETING RULES:
- Do NOT respond to invitations for meetings you've already accepted (check list_my_meetings first)
- Do NOT create meetings that duplicate existing ones (check list_shared_calendar first)
- Use list_known_agents() to discover who you can invite
- Preferred meeting slots: 10:00-11:00 or 14:00-15:00 (30-60 minutes)

COMFORT PREFERENCES:
- State your temperature comfort range (e.g., "prefer 21-23°C, feel cold easily")
- State lighting preferences (e.g., "prefer natural light, dim desk lamp for focus")
- These will guide your decisions throughout the day

Your preferences, habits, and context are in the prompt. Output your DailyPlan when ready.
""".strip()

    return Agent(
        name=f"occupant_day_{occupant_id}",
        instructions=instructions,
        model=model,
        model_settings=ModelSettings(reasoning_effort="high"),  # Day planning requires thoughtful consideration
        tools=[
            get_current_datetime,
            list_my_meetings,
            list_shared_calendar,
            list_pending_invitations,
            respond_to_invitation,
            create_shared_event,
            invite_agents_to_meeting,
            list_known_agents,
        ],
        output_type=AgentOutputSchema(DailyPlan, strict_json_schema=True),
    )

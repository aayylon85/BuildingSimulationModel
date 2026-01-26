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
import math
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

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
    "take_break",            # Short break (coffee, stretch, walk)
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
    action: Literal["adjust", "leave_as_is"]
    reasoning: str = Field(description="How agent feels (hot/cold/comfortable) based on actual temp")
    adjustment_direction: Optional[Literal["warmer", "cooler"]] = None
    adjustment_amount: Optional[float] = Field(default=None, description="Degrees to adjust")


class LightingDecision(BaseModel):
    """Decision about lighting at current location."""
    action: Literal["turn_on", "turn_off", "adjust_brightness", "leave_as_is"]
    reasoning: str
    target_device: Optional[str] = None
    brightness_level: Optional[int] = Field(default=None, ge=0, le=100, description="Brightness 0-100%")


class EquipmentDecision(BaseModel):
    """Decision about equipment use (kettle, coffee machine, etc.)."""
    action: Literal["use", "turn_off", "leave_as_is"]
    reasoning: str
    equipment_name: Optional[str] = None
    duration_minutes: Optional[int] = Field(default=None, description="For kitchen equipment auto-off")


class LocationDecision(BaseModel):
    """Decision about moving to a different location."""
    action: Literal["move", "stay"]
    reasoning: str
    destination: Optional[str] = Field(default=None, description="Target location: desk_area, meeting_room, break_area, shared_area")


class ConversationDecision(BaseModel):
    """Decision about initiating conversation with nearby agents."""
    action: Literal["initiate", "none"]
    reasoning: str
    target_agent: Optional[str] = None
    topic: Optional[str] = Field(default=None, description="Topic - not limited to thermostat")


class BreakDecision(BaseModel):
    """Decision about taking a break."""
    action: Literal["take_break", "continue_working"]
    reasoning: str
    break_type: Optional[Literal["at_desk", "break_room"]] = None
    activity: Optional[str] = Field(default=None, description="tea, coffee, snack, stretch, etc.")


class MeetingEquipmentDecision(BaseModel):
    """Decision about equipment at meeting start/end (for meeting host)."""
    action: Literal["request_change", "accept_current"]
    reasoning: str
    equipment_requests: Optional[List[str]] = Field(default=None, description="e.g., ['turn on projector', 'close blinds']")


class PlanUpdateDecision(BaseModel):
    """Decision about whether to update the daily plan based on current circumstances."""
    action: Literal["update", "keep_current"]
    reasoning: str = Field(description="Why plan update is or isn't needed")
    updates: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Updates to apply: e.g., {'lunch_plan': {'location': 'go_out', 'time': '13:00', 'reasoning': '...'}, 'afternoon_break': {...}}"
    )


class StepDecisions(BaseModel):
    """Complete structured output for a decision checkpoint."""
    occupant_id: str
    timestamp: str
    checkpoint_reason: str = Field(description="'hourly', 'meeting_start', 'meeting_end', 'lunch_time', 'morning_break', 'afternoon_break'")
    thermostat: ThermostatDecision
    lighting: LightingDecision
    equipment: EquipmentDecision
    location: LocationDecision
    conversation: ConversationDecision
    break_decision: Optional[BreakDecision] = Field(default=None, description="Only during break-eligible times")
    meeting_equipment: Optional[MeetingEquipmentDecision] = Field(default=None, description="Only at meeting boundaries for host")
    plan_update: PlanUpdateDecision = Field(description="Whether to update daily plan based on current circumstances")


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
    location: Literal["at_desk", "break_room", "go_out"]
    time: str = Field(description="Planned lunch time in HH:MM format")
    reasoning: str = Field(description="Why this location/time was chosen based on preferences and schedule")


class BreakPlan(BaseModel):
    """Break planning details decided during daily planning."""
    location: Literal["at_desk", "break_room"]
    activity: str = Field(description="tea, coffee, walk, stretch, etc.")
    preferred_time: str = Field(description="Planned break time in HH:MM format")
    reasoning: str = Field(description="Why this break was planned based on habits and schedule")


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
    notes: str = ""


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
    def __init__(self, model: str = DEFAULT_EMBED_MODEL, cache_size: int = 1000) -> None:
        self.client = OpenAI()
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
        resp = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        vec = resp.data[0].embedding
        emb = np.asarray(vec, dtype=np.float32)

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
            # For shared calendar, check across ALL creators
            if calendar_id == "shared":
                existing = con.execute(
                    """
                    SELECT event_id, title, created_by FROM events
                    WHERE calendar_id = ?
                    AND ABS(start_ts - ?) < 1800
                    AND LOWER(title) LIKE ?
                    AND cancelled = 0
                    """,
                    (calendar_id, _utc_ts(start), f"%{title.lower()[:20]}%")
                ).fetchone()
            else:
                # Personal calendar - only check same creator
                existing = con.execute(
                    """
                    SELECT event_id, title, created_by FROM events
                    WHERE calendar_id = ? AND created_by = ?
                    AND ABS(start_ts - ?) < 1800
                    AND LOWER(title) LIKE ?
                    AND cancelled = 0
                    """,
                    (calendar_id, created_by, _utc_ts(start), f"%{title.lower()[:20]}%")
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

def build_step_agent(occupant_id: str) -> Agent[SimContext]:
    """
    Agent used at decision checkpoints to decide occupant actions.
    Uses structured output OccupantStepDecision.

    Decision checkpoints occur:
    - Hourly (regular check-in)
    - At meeting start/end times
    - At planned lunch/break times
    """
    instructions = f"""
You are a simulated building occupant agent (ID: {occupant_id}).

You are making a decision at a CHECKPOINT. The checkpoint reason is shown in your prompt
(e.g., "hourly", "meeting starting", "lunch time", "break time"). Consider this reason
when deciding what actions to take.

AVAILABLE ACTIONS:

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
- choose_desk: select a desk ONCE when you arrive for the day (parameters: desk_id)
  NOTE: You can only choose a desk once per day. After your initial choice, it remains your desk.

Meetings:
- attend_meeting: physically go to meeting room (parameters: meeting_title)
- leave_meeting: return to your desk from meeting room

Lunch & Breaks:
- go_to_lunch: have lunch in the break room (stay in building)
- go_out_for_lunch: leave building for lunch (cafe, restaurant, walk)
- return_from_lunch: return to work after lunch
- take_break: take a short break (parameters: location - "at_desk" or "break_room")
- return_from_break: return to work after break

Social:
- initiate_conversation: start a conversation with a colleague (parameters: agent_id, topic)

THERMOSTAT - Focus on how you FEEL:
- Check the actual indoor temperature and notice how you feel (too hot, too cold, comfortable)
- If you feel TOO HOT: lower the COOLING setpoint
- If you feel TOO COLD: raise the HEATING setpoint
- Your personality affects sensitivity: some people run hot, others cold
- Only adjust when genuinely uncomfortable - don't manage setpoints abstractly
- Use your thermal preferences and memories to guide comfort decisions

LIGHTING - Consider your preferences and current needs:
- Check natural light level (bright, moderate, dim, dark)
- Consider your task: focused work may need good lighting, presentations may need dimmed lights
- Turn on desk lamp if natural light is insufficient
- Turn off lights when leaving an area or if natural light is adequate

EQUIPMENT - Be mindful of energy:
- Turn on equipment when you need it
- Kitchen appliances (kettle, coffee machine, microwave) will auto-off after use
- Turn off equipment when done, especially shared equipment
- Meeting room equipment (projector, phone) should be off when room is empty

BREAKS - Follow your daily plan and preferences:
- At break checkpoints, consider taking a break based on your planned schedule
- Choose location based on your preferences: at_desk (quick) or break_room (social, make tea/coffee)
- Your core memories about tea/coffee preferences should guide break behavior
- Remember how you like your tea or coffee!

LUNCH - Follow your daily plan:
- At lunch checkpoint, take lunch according to your plan
- "at_desk": Stay and eat at your desk
- "break_room": Eat in the kitchen/break room (social)
- "go_out": Leave the building (nice weather, need fresh air, get food)

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

DECISION GUIDELINES:
1. Consider the CHECKPOINT REASON - it tells you why you're making a decision now
2. Act according to your personality, preferences, and memories
3. ALWAYS provide a decision, even if it's "leave as is" or "no_op"
4. Prefer minimal actions when comfortable and no immediate needs
5. Consider other occupants for shared controls (thermostat, windows)
6. Your relevant memories are retrieved and shown in the prompt

Output your decision as JSON matching OccupantStepDecision schema.
Include a brief_rationale explaining your reasoning.
""".strip()

    return Agent(
        name=f"occupant_step_{occupant_id}",
        instructions=instructions,
        model=DEFAULT_AGENT_MODEL,
        model_settings=ModelSettings(reasoning_effort="medium"),  # Step decisions require cognitive reasoning
        tools=[
            get_current_datetime,
            list_shared_calendar,
        ],
        output_type=AgentOutputSchema(OccupantStepDecision, strict_json_schema=False),
    )

def build_day_planner_agent(occupant_id: str) -> Agent[SimContext]:
    """
    Agent used once per day to plan attendance and working hours.
    Uses structured output DailyPlan.
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
  * "break_room": Social lunch in the kitchen/break room
  * "go_out": Leave the building (cafe, restaurant, walk)
- Consider weather, meetings before/after lunch, and your energy level
- Use your core memories about lunch habits to guide this decision

BREAK PLANNING:
- Plan morning break (typically 10:00-10:30) and afternoon break (typically 15:00-15:30)
- Choose location: "at_desk" (quick) or "break_room" (social, make tea/coffee)
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
        model=DEFAULT_AGENT_MODEL,
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
        output_type=AgentOutputSchema(DailyPlan, strict_json_schema=False),
    )

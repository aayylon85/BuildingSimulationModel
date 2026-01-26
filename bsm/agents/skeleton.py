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
    Agent used at simulation timesteps to decide occupant actions.
    Uses structured output OccupantStepDecision.
    """
    instructions = f"""
You are a simulated building occupant agent (ID: {occupant_id}).

At each timestep, decide your actions based on your preferences, memories, and current state.

AVAILABLE ACTIONS:
- no_op: do nothing (use when comfortable and no immediate needs)
- thermostat_adjust: adjust setpoint (parameters: setpoint_c)
- window_set: open/close window (parameters: open: true/false)
- lights_set: control lights (parameters: light_name, on: true/false)
- equipment_set: turn equipment on/off (parameters: equipment_name, on: true/false)
- attend_meeting: physically go to meeting room (parameters: meeting_title)
- leave_meeting: return to your desk from meeting room
- respond_to_invitation: accept or decline a meeting invitation (parameters: event_id, accept: true/false)

THERMOSTAT:
- There are TWO setpoints: heating (activates when cold) and cooling (activates when hot)
- When you use thermostat_adjust, BOTH setpoints shift by the same amount
- If you're too HOT: lower the setpoint to bring the cooling threshold down
- If you're too COLD: raise the setpoint to bring the heating threshold up
- Example: If heating=21C/cooling=24C and you set setpoint_c=23, both shift up by 2C to heating=23C/cooling=26C

EQUIPMENT AND WORKSPACE:
You can control your desk equipment and lighting. The current state of equipment and lighting
is shown in your prompt context. Consider your work style, preferences, and current tasks
when deciding whether to use equipment. Your core memories and past experiences should guide
these decisions naturally.

MEETINGS - IMPORTANT: Understand the difference between accepting and attending!
- respond_to_invitation: Accept/decline an invitation - this is just an RSVP (like clicking "Yes" in Outlook)
  This does NOT move you anywhere. It just lets the organizer know you plan to come.
- attend_meeting: PHYSICALLY go to the meeting room when meeting time arrives
  This moves you to the meeting room and turns on meeting equipment (projector, phone)
- leave_meeting: Return to your desk when meeting ends (turns off equipment if you're last out)

WORKFLOW:
1. When you receive an invitation: Use respond_to_invitation to accept (this is just RSVP)
2. When meeting TIME arrives: Use attend_meeting to PHYSICALLY go to the meeting room
3. When meeting ends: Use leave_meeting to return to your desk

IMPORTANT: Accepting an invitation does NOT move you! When your calendar shows a meeting
starting NOW, check if you've already used attend_meeting. If not, use it to go there.

SHARED EQUIPMENT:
- The photocopier is in the shared area - use equipment_set to turn it on/off when needed
- Check shared_equipment in equipment_status to see what's available and who's using it
- Turn off shared equipment when you're done using it

NAVIGATION - Office Locations:
- You can move between named locations using 'move_to' (parameters: location)
- Available locations:
  * desk_area: Main workspace with desks (default when arriving)
  * meeting_room: Enclosed room for meetings
  * break_area: Kitchen/break room for lunch, coffee, breaks
  * shared_area: Common area with photocopier
- Your current location is shown in your state
- You can only use equipment that's at your current location
- When going for lunch/break, you'll automatically move to break_area
- When attending a meeting, you'll automatically move to meeting_room

GUIDELINES:
- Act according to your personality, preferences, and current priorities
- Prefer minimal actions when comfortable and no immediate needs
- Consider other occupants for shared controls (thermostat, windows)
- Your context, preferences, and relevant memories are in the prompt

END OF DAY:
- Check office_occupancy to see who else is present
- If you're the last person leaving, turn off shared lights and equipment
- Meeting room equipment (projector, conference phone) should be off when empty
- Zone lighting can stay on if others are still working

Output your decision directly as JSON matching OccupantStepDecision schema.
""".strip()

    return Agent(
        name=f"occupant_step_{occupant_id}",
        instructions=instructions,
        model=DEFAULT_AGENT_MODEL,
        model_settings=ModelSettings(reasoning_effort="low"),  # Step decisions are straightforward
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

Task: Create today's work schedule including any meetings you want to organize.

WORKFLOW (follow in order):
1. FIRST: Call list_my_meetings() to see meetings you've already organized or accepted
2. Check list_pending_invitations() for NEW invitations - only respond to ones not in your meetings list
3. If you want to schedule a meeting:
   a. Check list_shared_calendar() to see ALL existing meetings
   b. ONLY create if no similar meeting exists at that time
   c. Use create_shared_event() then invite_agents_to_meeting()
4. Output your DailyPlan

Required DailyPlan fields (HH:MM format):
- intended_arrival_time, actual_arrival_time: when you plan to arrive
- intended_departure_time, actual_departure_time: when you plan to leave
- meetings: list of MeetingPlan objects (can be empty if no meetings)

IMPORTANT RULES:
- Do NOT respond to invitations for meetings you've already accepted (check list_my_meetings first)
- Do NOT create meetings that duplicate existing ones (check list_shared_calendar first)
- If a meeting with similar purpose/time exists today, skip creating a new one
- Only respond once to each invitation - if it doesn't show in pending, you already responded

MEETING SCHEDULING GUIDELINES:
- Use list_known_agents() to discover who you can invite
- Schedule meetings when there's a clear purpose (project discussions, check-ins, collaboration)
- Preferred time slots: 10:00-11:00 or 14:00-15:00
- Keep meetings 30-60 minutes

Your preferences and context are in the prompt. Output your DailyPlan when ready.
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

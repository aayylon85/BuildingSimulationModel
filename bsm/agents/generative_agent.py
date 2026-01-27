"""
generative_agent.py

GenerativeAgent class that wraps memory stream and scratch data.
Inspired by Stanford Generative Agents architecture.

This class manages:
- Agent identity and characteristics (scratch.json)
- Memory stream (core + event + thought memories)
- Integration with OpenAI Agents SDK for decision making
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone, date, time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from bsm.agents.memory.stream import MemoryStream, MemoryNode, CoreMemoryStore


def _parse_time_str(time_str: str) -> Optional[time]:
    """
    Parse a HH:MM time string to a time object.

    Args:
        time_str: Time string in HH:MM format

    Returns:
        time object, or None if parsing fails
    """
    if not time_str:
        return None
    try:
        parts = time_str.split(":")
        if len(parts) >= 2:
            return time(int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        pass
    return None


@dataclass
class AgentPaths:
    """Paths to agent data files."""
    folder: Path
    scratch_json: Path
    nodes_json: Path
    embeddings_json: Path
    core_memories_json: Path  # Separate core memory storage
    persona_md: Path
    background_md: Path
    work_style_md: Path
    relationships_md: Path


class GenerativeAgent:
    """
    A generative agent with memory-driven behavior.

    Wraps:
    - Scratch (working memory): scratch.json with agent characteristics
    - Memory Stream (associative memory): events, thoughts, core memories
    - Markdown files (personality definition): persona.md, background.md, etc.

    The agent's behavior emerges from the cognitive loop:
    Perceive -> Retrieve -> Plan -> Reflect -> Act
    """

    def __init__(
        self,
        agent_folder: str,
        embedder: Any,  # EmbeddingClient from LLMagentskeleton
        auto_load: bool = True,
    ) -> None:
        """
        Initialize a generative agent.

        Args:
            agent_folder: Path to agent's data folder
            embedder: EmbeddingClient instance for generating embeddings
            auto_load: Whether to automatically load data from files
        """
        self.agent_folder = Path(agent_folder)
        self.embedder = embedder

        # SAFETY: Prevent accidentally pointing to base_types
        if "agent_base_types" in str(self.agent_folder.resolve()):
            raise ValueError(
                f"GenerativeAgent cannot be initialized with agent_base_types folder: {self.agent_folder}. "
                "Use copy_agent_base_type() first to copy to a results directory."
            )

        # Set up paths
        self.paths = AgentPaths(
            folder=self.agent_folder,
            scratch_json=self.agent_folder / "scratch.json",
            nodes_json=self.agent_folder / "memory_stream" / "nodes.json",
            embeddings_json=self.agent_folder / "memory_stream" / "embeddings.json",
            core_memories_json=self.agent_folder / "core_memories.json",
            persona_md=self.agent_folder / "persona.md",
            background_md=self.agent_folder / "background.md",
            work_style_md=self.agent_folder / "work_style.md",
            relationships_md=self.agent_folder / "relationships.md",
        )

        # Working memory (scratch)
        self.scratch: Dict[str, Any] = {}

        # Core memories (permanent, never decay) - SEPARATE from memory stream
        self.core_memory_store: Optional[CoreMemoryStore] = None

        # Associative memory (memory stream) - events, thoughts, chats (decay)
        self.memory_stream: Optional[MemoryStream] = None

        if auto_load:
            self._load_scratch()
            self._initialize_core_memory_store()
            self._initialize_memory_stream()

    @property
    def agent_id(self) -> str:
        """Get agent's unique identifier."""
        return self.scratch.get("agent_id", "unknown")

    @property
    def name(self) -> str:
        """Get agent's name."""
        return self.scratch.get("name", "Unknown")

    @property
    def first_name(self) -> str:
        """Get agent's first name."""
        return self.scratch.get("first_name", self.name.split()[0] if self.name else "Unknown")

    def _load_scratch(self) -> None:
        """Load scratch (working memory) from JSON file."""
        if self.paths.scratch_json.exists():
            try:
                with open(self.paths.scratch_json, 'r') as f:
                    self.scratch = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load scratch.json for {self.agent_folder}: {e}")
                self.scratch = {}
        else:
            print(f"Warning: No scratch.json found at {self.paths.scratch_json}")
            self.scratch = {}

    def _initialize_core_memory_store(self) -> None:
        """
        Initialize the separate core memory store.

        Core memories are permanent, never-decaying personality traits and preferences
        that are stored separately from the event/thought memory stream.
        """
        self.core_memory_store = CoreMemoryStore(
            agent_folder=str(self.agent_folder),
            embedder=self.embedder,
            auto_load=True,
        )

        # If core memory store is empty, seed from markdown files
        if self.core_memory_store.count() == 0:
            self._seed_core_memories_from_markdown()

    def _initialize_memory_stream(self) -> None:
        """
        Initialize memory stream for events, thoughts, and chats.

        NOTE: Core memories are now stored separately in CoreMemoryStore.
        This memory stream only contains decaying episodic memories.
        """
        self.memory_stream = MemoryStream(
            agent_folder=str(self.agent_folder),
            embedder=self.embedder,
            auto_load=True,
        )

    def _seed_core_memories_from_markdown(self) -> None:
        """
        Seed core memories from personality markdown files.

        Reads persona.md, background.md, work_style.md, relationships.md
        and creates core memory entries in the CoreMemoryStore.

        NOTE: This now uses the separate CoreMemoryStore, not the memory stream.
        """
        if self.core_memory_store is None:
            print(f"Warning: Cannot seed core memories - store not initialized")
            return

        md_files = [
            ("persona", self.paths.persona_md),
            ("background", self.paths.background_md),
            ("work_style", self.paths.work_style_md),
            ("relationships", self.paths.relationships_md),
        ]

        total_added = 0
        for source_name, md_path in md_files:
            if md_path.exists():
                try:
                    content = md_path.read_text()
                    chunks = self._split_markdown_into_chunks(content)

                    for chunk in chunks:
                        if chunk.strip():
                            self.core_memory_store.add(
                                description=chunk.strip(),
                                source=source_name,
                            )
                            total_added += 1
                except IOError as e:
                    print(f"Warning: Failed to read {md_path}: {e}")

        if total_added > 0:
            print(f"[GenerativeAgent] Seeded {total_added} core memories for {self.name}")
            self.core_memory_store.save()

    def _split_markdown_into_chunks(self, content: str) -> List[str]:
        """
        Split markdown content into semantic chunks for memory storage.

        Strategy:
        - Split on headers (##)
        - Within sections, split on bullet points
        - Keep chunks reasonably sized
        """
        chunks = []
        current_section = ""
        lines = content.split('\n')

        for line in lines:
            line = line.strip()

            # Skip empty lines and main title
            if not line or line.startswith('# '):
                if current_section:
                    chunks.append(current_section)
                    current_section = ""
                continue

            # New subsection - save previous and start new
            if line.startswith('## '):
                if current_section:
                    chunks.append(current_section)
                current_section = ""
                continue

            # Bullet point - treat as individual memory
            if line.startswith('- '):
                item = line[2:].strip()
                if len(item) > 10:  # Only add substantial items
                    chunks.append(item)
                continue

            # Regular text - accumulate
            if line:
                current_section += line + " "

        # Don't forget last section
        if current_section.strip():
            chunks.append(current_section.strip())

        return chunks

    def get_identity_stable_set(self) -> str:
        """
        Return the Identity Stable Set (ISS) for prompts.

        The ISS provides core identity information that should be
        consistently included in agent prompts.
        """
        return f"""Name: {self.scratch.get('name', 'Unknown')}
Innate traits: {self.scratch.get('innate_traits', 'N/A')}
Learned: {self.scratch.get('learned_traits', 'N/A')}
Currently: {self.scratch.get('currently', 'N/A')}
Lifestyle: {self.scratch.get('lifestyle', 'N/A')}
Thermal comfort: {self.scratch.get('thermal_comfort_c', 21.0)}C
Preferred desk: {self.scratch.get('preferred_desk', 'N/A')}
""".strip()

    def get_schedule_info(self) -> str:
        """Get schedule information for prompts."""
        return f"""Typical arrival: {self.scratch.get('typical_arrival', '09:00')}
Typical departure: {self.scratch.get('typical_departure', '17:00')}
Works weekends: {self.scratch.get('works_weekends', False)}
""".strip()

    def get_relationship_info(self, other_agent_id: str) -> Optional[Dict[str, float]]:
        """
        Get relationship model for another agent.

        Returns dict with 'familiarity' and 'sentiment' scores (0-1).
        """
        relationships = self.scratch.get("relationship_models", {})
        return relationships.get(other_agent_id)

    def update_relationship(
        self,
        other_agent_id: str,
        familiarity_delta: float = 0.0,
        sentiment_delta: float = 0.0,
    ) -> None:
        """
        Update relationship model for another agent.

        Agents learn from interactions - familiarity increases with conversations,
        sentiment adjusts based on agreement/disagreement outcomes.

        Args:
            other_agent_id: ID of the other agent
            familiarity_delta: Change in familiarity (capped to 0-1 range)
            sentiment_delta: Change in sentiment (capped to 0-1 range)
        """
        if "relationship_models" not in self.scratch:
            self.scratch["relationship_models"] = {}

        relationships = self.scratch["relationship_models"]

        if other_agent_id not in relationships:
            # Initialize with low familiarity and neutral sentiment
            relationships[other_agent_id] = {
                "familiarity": 0.3,
                "sentiment": 0.5,
            }

        rel = relationships[other_agent_id]

        # Apply deltas with capping to [0, 1]
        rel["familiarity"] = max(0.0, min(1.0, rel["familiarity"] + familiarity_delta))
        rel["sentiment"] = max(0.0, min(1.0, rel["sentiment"] + sentiment_delta))

    def recently_perceived(self, subject: str, now: datetime, within_minutes: int = 30) -> bool:
        """
        Check if agent has recently perceived something about a subject.

        Args:
            subject: The subject to check (e.g., another agent ID)
            now: Current simulation datetime (NOT real-world time)
            within_minutes: Time window in minutes

        Returns:
            True if there's a recent memory about this subject
        """
        if not self.memory_stream:
            return False

        recent = self.memory_stream.search_by_keywords([subject.lower()], limit=5)
        if not recent:
            return False

        now_ts = now.timestamp()  # Use simulation time, not datetime.now()
        cutoff = now_ts - (within_minutes * 60)

        return any(node.created > cutoff for node in recent)

    def get_retrieval_weights(self) -> Tuple[float, float, float]:
        """
        Get retrieval weights from scratch.

        Returns:
            Tuple of (recency_weight, relevance_weight, importance_weight)
        """
        return (
            self.scratch.get("recency_w", 1.0),
            self.scratch.get("relevance_w", 1.0),
            self.scratch.get("importance_w", 1.0),
        )

    def get_recency_decay(self) -> float:
        """Get recency decay rate from scratch."""
        return self.scratch.get("recency_decay", 0.995)

    def get_perception_bandwidth(self) -> int:
        """
        Get perception bandwidth (max number of events to perceive per cycle).

        Stanford-style attention bandwidth limits how many events an agent
        can attend to in a single timestep, prioritizing the most salient.
        """
        return self.scratch.get("perception_bandwidth", 8)

    def get_reflection_threshold(self) -> float:
        """Get reflection threshold from scratch."""
        return self.scratch.get("reflection_threshold", 150)

    def get_importance_trigger(self) -> float:
        """Get current importance trigger value."""
        return self.scratch.get("importance_trigger_curr", 150)

    def decrement_importance_trigger(self, amount: float) -> None:
        """Decrement the importance trigger by a given amount."""
        current = self.scratch.get("importance_trigger_curr", 150)
        self.scratch["importance_trigger_curr"] = max(0, current - amount)

    def reset_importance_trigger(self) -> None:
        """Reset importance trigger to maximum."""
        max_val = self.scratch.get("importance_trigger_max", 150)
        self.scratch["importance_trigger_curr"] = max_val

    def should_reflect(self) -> bool:
        """Check if agent should perform reflection."""
        return self.get_importance_trigger() <= 0

    def set_just_arrived(self, value: bool = True) -> None:
        """Set the just_arrived flag."""
        self.scratch["just_arrived"] = value

    def has_just_arrived(self) -> bool:
        """Check if agent has just arrived."""
        return self.scratch.get("just_arrived", False)

    def clear_just_arrived(self) -> None:
        """Clear the just_arrived flag."""
        self.scratch["just_arrived"] = False

    def get_daily_plan(self) -> Optional[Dict[str, Any]]:
        """Get current daily plan from scratch."""
        return self.scratch.get("daily_plan")

    def set_daily_plan(self, plan: Dict[str, Any]) -> None:
        """Set daily plan in scratch. Also saves initial plan for history tracking."""
        # Save initial plan if this is the first time setting it today
        if "initial_daily_plan" not in self.scratch or self.scratch.get("initial_daily_plan", {}).get("date_iso") != plan.get("date_iso"):
            self.scratch["initial_daily_plan"] = plan.copy()
            self.scratch["plan_updates"] = []
        self.scratch["daily_plan"] = plan

    def update_daily_plan(self, updates: Dict[str, Any], reason: str, now: datetime) -> Dict[str, Any]:
        """
        Update daily plan based on significant events.

        Time-aware: Only allows updates to future events. Past events (meetings that
        have already started or times that have passed) cannot be modified.

        Args:
            updates: Dictionary of updates to merge into the plan
            reason: Human-readable reason for the update
            now: Current simulation datetime

        Returns:
            Dict with update results: {"updated": bool, "skipped_past_events": list}
        """
        plan = self.get_daily_plan() or {}
        current_time = now.time()  # Use time object for proper comparison
        skipped_past_events = []

        # Filter meeting updates - can't modify past meetings
        if "meetings" in updates:
            filtered_meetings = []
            for meeting in updates.get("meetings", []):
                meeting_start = meeting.get("start_datetime_iso", "")
                try:
                    # Parse meeting start time
                    start_dt = datetime.fromisoformat(meeting_start)
                    if start_dt.tzinfo is None:
                        start_dt = start_dt.replace(tzinfo=timezone.utc)
                    now_utc = now if now.tzinfo else now.replace(tzinfo=timezone.utc)

                    # Only allow updates to future meetings
                    if start_dt > now_utc:
                        filtered_meetings.append(meeting)
                    else:
                        skipped_past_events.append(f"Meeting '{meeting.get('title', 'unknown')}' (already started/passed)")
                except (ValueError, TypeError):
                    # If can't parse, include it
                    filtered_meetings.append(meeting)

            updates["meetings"] = filtered_meetings

        # Filter lunch_plan updates - can't change past lunch time
        if "lunch_plan" in updates:
            lunch_plan = updates.get("lunch_plan", {})
            lunch_time_str = lunch_plan.get("time", "") if isinstance(lunch_plan, dict) else ""
            lunch_time = _parse_time_str(lunch_time_str)
            if lunch_time and lunch_time < current_time:
                skipped_past_events.append(f"Lunch time '{lunch_time_str}' (already passed)")
                del updates["lunch_plan"]

        # Filter break updates - can't change past break times
        for break_key in ["morning_break", "afternoon_break"]:
            if break_key in updates:
                break_plan = updates.get(break_key, {})
                break_time_str = break_plan.get("preferred_time", "") if isinstance(break_plan, dict) else ""
                break_time = _parse_time_str(break_time_str)
                if break_time and break_time < current_time:
                    skipped_past_events.append(f"{break_key.replace('_', ' ').title()} at '{break_time_str}' (already passed)")
                    del updates[break_key]

        # Apply remaining updates
        if updates:
            plan.update(updates)
            plan["last_updated"] = now.isoformat()

            # Track update reasons
            update_reasons = plan.get("update_reasons", [])
            update_reasons.append({
                "reason": reason,
                "timestamp": now.isoformat(),
                "skipped_past_events": skipped_past_events,
            })
            plan["update_reasons"] = update_reasons
            self.set_daily_plan(plan)

        return {
            "updated": bool(updates),
            "skipped_past_events": skipped_past_events,
        }

    def save_plan_history(self, filepath: str = None) -> None:
        """
        Save complete plan history to dedicated file for review after simulation.

        Creates a structured JSON file showing:
        - Initial plan (as first created)
        - Current plan (after all updates)
        - All updates with timestamps and reasons
        - Final state summary

        Args:
            filepath: Path to save history. Defaults to agent_folder/plan_history.json
        """
        import json
        from pathlib import Path

        if filepath is None:
            filepath = Path(self.agent_folder) / "plan_history.json"

        history = {
            "agent_id": self.name,
            "initial_plan": self.scratch.get("initial_daily_plan"),
            "current_plan": self.scratch.get("daily_plan"),
            "updates": self.scratch.get("plan_updates", []) + (
                self.scratch.get("daily_plan", {}).get("update_reasons", [])
            ),
            "final_state": {
                "arrival": self.scratch.get("daily_plan", {}).get("actual_arrival_time"),
                "departure": self.scratch.get("daily_plan", {}).get("actual_departure_time"),
                "meetings_attended": self.scratch.get("meetings_attended", []),
            },
        }

        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2, default=str)

    # -------------------------
    # Daily Clothing
    # -------------------------

    def get_todays_clothing(self) -> Optional[Dict[str, Any]]:
        """Get today's clothing choice from scratch."""
        return self.scratch.get("todays_clothing")

    def set_todays_clothing(self, clothing: Dict[str, Any]) -> None:
        """
        Set today's clothing choice in scratch.

        Args:
            clothing: Dict with 'description', 'warmth_level', 'layers_removable'
        """
        self.scratch["todays_clothing"] = clothing

    def clear_todays_clothing(self) -> None:
        """Clear clothing choice (called at end of day or start of new day)."""
        if "todays_clothing" in self.scratch:
            del self.scratch["todays_clothing"]

    # -------------------------
    # Dual Schedule Management (Stanford-style)
    # -------------------------

    def set_daily_schedule(
        self,
        schedule: List[Dict[str, Any]],
        preserve_original: bool = True,
    ) -> None:
        """
        Set the daily schedule with optional original preservation.

        Stanford-style: Maintains both a live decomposed schedule and
        the original hourly schedule for recovery/reference.

        Args:
            schedule: List of schedule entries (decomposed tasks)
            preserve_original: If True and no original exists, save as original
        """
        if preserve_original and "f_daily_schedule_hourly_org" not in self.scratch:
            # First schedule of the day - preserve as original
            self.scratch["f_daily_schedule_hourly_org"] = schedule.copy()

        self.scratch["f_daily_schedule"] = schedule

    def get_daily_schedule(self) -> List[Dict[str, Any]]:
        """Get current live schedule."""
        return self.scratch.get("f_daily_schedule", [])

    def get_original_schedule(self) -> List[Dict[str, Any]]:
        """Get preserved original hourly schedule."""
        return self.scratch.get("f_daily_schedule_hourly_org", [])

    def reset_schedule_to_original(self) -> None:
        """Reset live schedule to original (for recovery)."""
        original = self.get_original_schedule()
        if original:
            self.scratch["f_daily_schedule"] = original.copy()

    def clear_daily_schedules(self) -> None:
        """Clear both schedules (call at start of new day)."""
        self.scratch["f_daily_schedule"] = []
        self.scratch.pop("f_daily_schedule_hourly_org", None)

    # -------------------------
    # Chatting State Management
    # -------------------------

    def get_chatting_with(self) -> Optional[str]:
        """Get the agent ID currently chatting with, or None."""
        return self.scratch.get("chatting_with")

    def set_chatting_with(self, other_agent_id: Optional[str]) -> None:
        """Set the agent ID currently chatting with."""
        self.scratch["chatting_with"] = other_agent_id

    def get_chatting_buffer(self, other_agent_id: str) -> int:
        """
        Get the chat buffer for another agent.

        Buffer is the number of timesteps to wait before chatting again.
        Returns 0 if no buffer set.
        """
        buffer = self.scratch.get("chatting_with_buffer", {})
        return buffer.get(other_agent_id, 0)

    def update_chatting_buffer(self, other_agent_id: str, buffer_steps: int) -> None:
        """
        Set the chat buffer for another agent.

        Args:
            other_agent_id: The other agent's ID
            buffer_steps: Number of timesteps to wait before chatting again
        """
        if "chatting_with_buffer" not in self.scratch:
            self.scratch["chatting_with_buffer"] = {}
        self.scratch["chatting_with_buffer"][other_agent_id] = buffer_steps

    def decrement_all_chat_buffers(self) -> None:
        """
        Decrement all chat buffers by 1.

        Called each timestep to reduce cooldown periods.
        """
        buffer = self.scratch.get("chatting_with_buffer", {})
        for agent_id in list(buffer.keys()):
            buffer[agent_id] = max(0, buffer[agent_id] - 1)
            if buffer[agent_id] == 0:
                del buffer[agent_id]

    def can_chat_with(self, other_agent_id: str) -> bool:
        """
        Check if agent can initiate chat with another agent.

        Returns False if:
        - Currently chatting with someone
        - Buffer not expired for this agent
        """
        if self.get_chatting_with():
            return False
        return self.get_chatting_buffer(other_agent_id) == 0

    # -------------------------
    # Post-Conversation Reflection (Stanford-style)
    # -------------------------

    def mark_conversation_end(self, other_agent_id: str, now: datetime) -> None:
        """
        Mark that a conversation just ended for post-conversation reflection.

        Stanford-style: Triggers reflection ~10 seconds after conversation,
        generating planning thoughts and relationship memos. In simulation,
        this triggers on the next decision cycle.

        Args:
            other_agent_id: ID of the agent the conversation was with
            now: Current simulation datetime
        """
        if "pending_conversation_reflections" not in self.scratch:
            self.scratch["pending_conversation_reflections"] = []

        self.scratch["pending_conversation_reflections"].append({
            "other_agent_id": other_agent_id,
            "ended_at": now.isoformat(),
        })

    def get_pending_conversation_reflections(self) -> List[Dict[str, Any]]:
        """Get conversations that need post-conversation reflection."""
        return self.scratch.get("pending_conversation_reflections", [])

    def clear_pending_conversation_reflections(self) -> None:
        """Clear pending conversation reflection triggers."""
        self.scratch["pending_conversation_reflections"] = []

    async def consolidate_memories_with_llm(self, day: date, now: datetime) -> int:
        """
        Use LLM to identify memories that shape personality for consolidation.

        Called at end of day to move significant memories to core memory store.

        Args:
            day: The day to consolidate memories from
            now: Current simulation datetime

        Returns:
            Number of memories consolidated
        """
        if not self.memory_stream or not self.core_memory_store:
            return 0

        # Import here to avoid circular imports
        from bsm.agents.cognition.modules import assess_memories_for_consolidation

        # Get today's memories
        day_start = datetime.combine(day, time.min, tzinfo=timezone.utc)
        day_end = datetime.combine(day, time.max, tzinfo=timezone.utc)

        todays_memories = []
        for node in self.memory_stream.nodes:
            node_dt = datetime.fromtimestamp(node.created, tz=timezone.utc)
            if day_start <= node_dt <= day_end:
                todays_memories.append(node)

        if not todays_memories:
            return 0

        # Format memories for assessment
        memory_data = [
            (i, m.description, m.importance)
            for i, m in enumerate(todays_memories)
        ]

        # Get identity for context
        identity = self.get_identity_stable_set()

        # Assess which memories to consolidate
        assessment = await assess_memories_for_consolidation(
            agent_name=self.name,
            identity=identity,
            memories=memory_data,
        )

        # Consolidate selected memories
        consolidated = 0
        for idx in assessment.memories_to_consolidate:
            if 0 <= idx < len(todays_memories):
                memory = todays_memories[idx]
                self.core_memory_store.add(
                    description=f"[Learned {day}] {memory.description}",
                    source="daily_consolidation",
                    importance=memory.importance,
                )
                consolidated += 1

        if consolidated > 0:
            self.core_memory_store.save()
            print(f"  [CONSOLIDATION] {self.name}: Consolidated {consolidated} memories - {assessment.reasoning}")

        return consolidated

    def save(self, atomic: bool = True) -> None:
        """
        Persist all state to files.

        Args:
            atomic: If True, uses write-then-rename pattern for consistency.
                   If any write fails, restores from backup.
        """
        # SAFETY: Additional check to prevent writing to base_types
        if "agent_base_types" in str(self.paths.scratch_json.resolve()):
            raise RuntimeError(
                f"SAFETY: Refusing to write to agent_base_types: {self.paths.scratch_json}"
            )

        if not atomic:
            # Original non-atomic save
            self._save_simple()
            return

        # Atomic save: collect all write operations
        files_to_save: List[Tuple[Path, Path, Callable]] = []
        backup_suffix = ".bak"
        temp_suffix = ".tmp"

        # Prepare scratch save
        scratch_tmp = self.paths.scratch_json.with_suffix(
            self.paths.scratch_json.suffix + temp_suffix
        )
        files_to_save.append((
            self.paths.scratch_json,
            scratch_tmp,
            lambda p=scratch_tmp: self._write_scratch(p)
        ))

        # Prepare core memory save
        if self.core_memory_store:
            files_to_save.extend(self.core_memory_store._prepare_atomic_save())

        # Prepare memory stream save
        if self.memory_stream:
            files_to_save.extend(self.memory_stream._prepare_atomic_save())

        try:
            # Phase 1: Write all temp files
            for final_path, tmp_path, write_func in files_to_save:
                write_func()

            # Phase 2: Backup existing and rename temps to final
            for final_path, tmp_path, _ in files_to_save:
                if final_path.exists():
                    backup_path = final_path.with_suffix(final_path.suffix + backup_suffix)
                    shutil.copy2(final_path, backup_path)
                tmp_path.rename(final_path)

            # Phase 3: Clean up backups on success
            for final_path, _, _ in files_to_save:
                backup_path = final_path.with_suffix(final_path.suffix + backup_suffix)
                if backup_path.exists():
                    backup_path.unlink()

        except Exception as e:
            logger.error(f"Atomic save failed, attempting rollback: {e}")
            # Rollback: restore from backups
            for final_path, tmp_path, _ in files_to_save:
                backup_path = final_path.with_suffix(final_path.suffix + backup_suffix)
                if backup_path.exists():
                    shutil.copy2(backup_path, final_path)
                if tmp_path.exists():
                    tmp_path.unlink()
            raise

    def _write_scratch(self, path: Path) -> None:
        """Write scratch to a specific path (for atomic save)."""
        with open(path, 'w') as f:
            json.dump(self.scratch, f, indent=2, default=str)

    def _save_simple(self) -> None:
        """Simple non-atomic save (original implementation)."""
        # Save scratch
        try:
            with open(self.paths.scratch_json, 'w') as f:
                json.dump(self.scratch, f, indent=2, default=str)
        except IOError as e:
            print(f"Warning: Failed to save scratch.json: {e}")

        # Save core memory store (separate from memory stream)
        if self.core_memory_store:
            self.core_memory_store.save()

        # Save memory stream (events, thoughts, chats)
        if self.memory_stream:
            self.memory_stream.save()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the agent's state."""
        memory_counts = self.memory_stream.get_node_count() if self.memory_stream else {}
        core_memory_count = self.core_memory_store.count() if self.core_memory_store else 0
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "core_memory_count": core_memory_count,
            "memory_counts": memory_counts,
            "importance_trigger": self.get_importance_trigger(),
            "has_daily_plan": self.get_daily_plan() is not None,
        }


def copy_agent_base_type(
    base_type_folder: str,
    destination_folder: str,
    agent_id: str,
) -> Path:
    """
    Copy an agent base type to a new destination for simulation.

    This creates a fresh copy of the agent files that can be modified
    during the simulation without affecting the base type.

    Args:
        base_type_folder: Path to agent base type (e.g., agent_base_types/alice_office_worker)
        destination_folder: Path to destination (e.g., results/agents/2026-01-16/14-30-00/alice_001)
        agent_id: Agent ID to assign

    Returns:
        Path to the copied agent folder (caller should create GenerativeAgent from this)
    """
    source = Path(base_type_folder)
    dest = Path(destination_folder)

    # Copy all files
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)

    # Update scratch.json with agent_id
    scratch_path = dest / "scratch.json"
    if scratch_path.exists():
        with open(scratch_path, 'r') as f:
            scratch = json.load(f)
        scratch["agent_id"] = agent_id
        with open(scratch_path, 'w') as f:
            json.dump(scratch, f, indent=2)

    print(f"[GenerativeAgent] Copied {source.name} -> {dest} with ID {agent_id}")

    return dest


def initialize_agents_for_simulation(
    config: Dict[str, Any],
    results_dir: str,
    embedder: Any,
    run_start: datetime,
) -> Dict[str, GenerativeAgent]:
    """
    Initialize agents for a simulation run.

    Supports two modes:
    - "fresh": Copy from agent_base_types to new run directory
    - "continue": Load from a previous simulation run

    Args:
        config: Full simulation config
        results_dir: Base results directory
        embedder: EmbeddingClient instance
        run_start: Simulation start timestamp

    Returns:
        Dict mapping agent_id to GenerativeAgent instances
    """
    llm_config = config.get("llm_agents", {})
    agent_source = llm_config.get("agent_source", {})
    mode = agent_source.get("mode", "fresh")
    base_types_folder = agent_source.get("base_types_folder", "agent_base_types")
    agents_config = llm_config.get("agents", [])

    agents: Dict[str, GenerativeAgent] = {}

    if mode == "continue":
        # Continue from previous run
        continue_from = agent_source.get("continue_from_run")
        if not continue_from:
            raise ValueError("agent_source.continue_from_run must be specified in continue mode")

        source_dir = Path(results_dir) / "agents" / continue_from
        if not source_dir.exists():
            raise ValueError(f"Cannot continue: {source_dir} does not exist")

        print(f"[GenerativeAgent] Continuing from {source_dir}")

        for agent_config in agents_config:
            agent_id = agent_config["id"]
            agent_folder = source_dir / agent_id

            if not agent_folder.exists():
                raise ValueError(f"Agent folder not found: {agent_folder}")

            agents[agent_id] = GenerativeAgent(
                agent_folder=str(agent_folder),
                embedder=embedder,
            )

    else:  # mode == "fresh"
        # Create new run directory
        date_str = run_start.strftime("%Y-%m-%d")
        time_str = run_start.strftime("%H-%M-%S")
        run_dir = Path(results_dir) / "agents" / date_str / time_str
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"[GenerativeAgent] Fresh run at {run_dir}")

        for agent_config in agents_config:
            base_type = agent_config.get("base_type")
            agent_id = agent_config["id"]

            if not base_type:
                raise ValueError(f"base_type must be specified for agent {agent_id}")

            source = Path(base_types_folder) / base_type
            if not source.exists():
                raise ValueError(f"Agent base type not found: {source}")

            dest = run_dir / agent_id

            # Copy files
            copy_agent_base_type(str(source), str(dest), agent_id)

            # Initialize agent
            agents[agent_id] = GenerativeAgent(
                agent_folder=str(dest),
                embedder=embedder,
            )

    return agents

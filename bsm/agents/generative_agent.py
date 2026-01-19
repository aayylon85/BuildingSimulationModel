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
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bsm.agents.memory.stream import MemoryStream, MemoryNode, CoreMemoryStore


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

    def recently_perceived(self, subject: str, within_minutes: int = 30) -> bool:
        """
        Check if agent has recently perceived something about a subject.

        Args:
            subject: The subject to check (e.g., another agent ID)
            within_minutes: Time window in minutes

        Returns:
            True if there's a recent memory about this subject
        """
        if not self.memory_stream:
            return False

        recent = self.memory_stream.search_by_keywords([subject.lower()], limit=5)
        if not recent:
            return False

        now = datetime.now(timezone.utc).timestamp()
        cutoff = now - (within_minutes * 60)

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
        """Set daily plan in scratch."""
        self.scratch["daily_plan"] = plan

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

    def save(self) -> None:
        """Persist all state to files."""
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
) -> GenerativeAgent:
    """
    Copy an agent base type to a new destination for simulation.

    This creates a fresh copy of the agent files that can be modified
    during the simulation without affecting the base type.

    Args:
        base_type_folder: Path to agent base type (e.g., agent_base_types/alice_office_worker)
        destination_folder: Path to destination (e.g., results/agents/2026-01-16/14-30-00/alice_001)
        agent_id: Agent ID to assign

    Returns:
        Initialized GenerativeAgent instance
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

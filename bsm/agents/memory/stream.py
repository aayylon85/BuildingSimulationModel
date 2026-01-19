"""
memory_stream.py

Memory Stream implementation inspired by Stanford Generative Agents.
Supports semantic retrieval with three-factor scoring (recency, relevance, importance).

Key differences from previous SQLiteVectorMemory:
- File-based storage (JSON) instead of SQLite
- Three-factor retrieval scoring
- MemoryNode dataclass with rich metadata
- Support for thoughts/reflections with evidence linking
"""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


def _utc_ts(dt: datetime) -> float:
    """Convert datetime to UTC timestamp."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass
class MemoryNode:
    """
    A single memory node in the memory stream.

    Based on Stanford Generative Agents architecture with adaptations
    for building simulation context.

    Attributes:
        node_id: Unique identifier for this memory
        node_type: Type of memory - "core", "event", "thought", "chat"
        depth: Depth in thought hierarchy (0 for events, 1+ for thoughts)
        created: When the memory was formed (UTC timestamp)
        last_accessed: When the memory was last retrieved (UTC timestamp)
        expiration: Optional expiration timestamp (unused for core memories)
        subject: Who/what the memory is about (semantic triple)
        predicate: Action or state (semantic triple)
        object: Target of the action (semantic triple)
        description: Full natural language description
        keywords: Set of keywords for fast retrieval
        importance: Poignancy score (1-10)
        embedding_key: Key into embeddings dict
        evidence_ids: For thoughts - node IDs that support this insight
    """
    node_id: int
    node_type: str  # "core", "event", "thought", "chat"
    depth: int = 0

    created: float = 0.0  # UTC timestamp
    last_accessed: float = 0.0  # UTC timestamp
    expiration: Optional[float] = None  # UTC timestamp or None

    # Semantic triple
    subject: str = ""
    predicate: str = ""
    object: str = ""

    description: str = ""
    keywords: Set[str] = field(default_factory=set)
    importance: float = 5.0  # 1-10 scale
    embedding_key: str = ""

    # For thoughts: links to supporting memories
    evidence_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "depth": self.depth,
            "created": self.created,
            "last_accessed": self.last_accessed,
            "expiration": self.expiration,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "description": self.description,
            "keywords": list(self.keywords),
            "importance": self.importance,
            "embedding_key": self.embedding_key,
            "evidence_ids": self.evidence_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryNode":
        """Create from dictionary."""
        return cls(
            node_id=data["node_id"],
            node_type=data["node_type"],
            depth=data.get("depth", 0),
            created=data.get("created", 0.0),
            last_accessed=data.get("last_accessed", 0.0),
            expiration=data.get("expiration"),
            subject=data.get("subject", ""),
            predicate=data.get("predicate", ""),
            object=data.get("object", ""),
            description=data.get("description", ""),
            keywords=set(data.get("keywords", [])),
            importance=data.get("importance", 5.0),
            embedding_key=data.get("embedding_key", ""),
            evidence_ids=data.get("evidence_ids", []),
        )


class CoreMemoryStore:
    """
    Separate storage for permanent core memories that never decay.

    Core memories are personality traits, preferences, and learned behaviors
    that persist across all simulations and are ALWAYS available during retrieval.

    Unlike the MemoryStream, core memories:
    - Never decay (always recency=1.0)
    - Are stored separately from events/thoughts
    - Are seeded once from markdown files and rarely modified

    Storage:
        agent_folder/
            core_memories.json    # Permanent core memories with embeddings
    """

    def __init__(
        self,
        agent_folder: str,
        embedder: Any,  # EmbeddingClient
        auto_load: bool = True,
    ) -> None:
        """
        Initialize core memory store.

        Args:
            agent_folder: Path to agent's data folder
            embedder: EmbeddingClient instance
            auto_load: Whether to automatically load from file
        """
        self.agent_folder = Path(agent_folder)
        self.embedder = embedder

        # Storage
        self.memories: List[Dict[str, Any]] = []  # List of {description, embedding, source}
        self.embeddings: Dict[str, List[float]] = {}  # key -> vector

        if auto_load:
            self._load()

    @property
    def store_path(self) -> Path:
        return self.agent_folder / "core_memories.json"

    def _load(self) -> None:
        """Load core memories from JSON file."""
        if self.store_path.exists():
            try:
                with open(self.store_path, 'r') as f:
                    data = json.load(f)
                self.memories = data.get("memories", [])
                self.embeddings = data.get("embeddings", {})
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load core_memories.json: {e}")
                self.memories = []
                self.embeddings = {}

    def _get_embedding_key(self, text: str) -> str:
        """Generate a consistent key for embedding lookup."""
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    def _get_or_create_embedding(self, text: str) -> Tuple[str, List[float]]:
        """Get existing embedding or create new one."""
        key = self._get_embedding_key(text)

        if key not in self.embeddings:
            emb_array = self.embedder.embed(text)
            self.embeddings[key] = emb_array.tolist()

        return key, self.embeddings[key]

    def add(
        self,
        description: str,
        source: str = "",
    ) -> Dict[str, Any]:
        """
        Add a core memory.

        Args:
            description: The core memory content
            source: Source file (e.g., "persona.md")

        Returns:
            The created memory dict
        """
        # Check for duplicates
        for mem in self.memories:
            if mem["description"] == description:
                return mem  # Already exists

        emb_key, _ = self._get_or_create_embedding(description)

        memory = {
            "description": description,
            "embedding_key": emb_key,
            "source": source,
            "importance": 10.0,  # Core memories always high importance
        }

        self.memories.append(memory)
        return memory

    def get_all(self) -> List[Dict[str, Any]]:
        """Get all core memories."""
        return self.memories

    def count(self) -> int:
        """Get count of core memories."""
        return len(self.memories)

    def retrieve_relevant(
        self,
        focal_point: str,
        n_count: int = 10,
        relevance_threshold: float = 0.3,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Retrieve core memories relevant to a focal point.

        Core memories don't decay, so we only score by relevance (cosine similarity).

        Args:
            focal_point: Query to match against
            n_count: Max memories to return
            relevance_threshold: Minimum relevance score

        Returns:
            List of (relevance_score, memory) tuples, sorted by relevance
        """
        if not self.memories:
            return []

        # Get embedding for focal point
        focal_key, focal_emb = self._get_or_create_embedding(focal_point)
        focal_vec = np.array(focal_emb, dtype=np.float32)

        scored: List[Tuple[float, Dict[str, Any]]] = []

        for mem in self.memories:
            emb_key = mem.get("embedding_key", "")
            if emb_key not in self.embeddings:
                continue

            mem_vec = np.array(self.embeddings[emb_key], dtype=np.float32)
            relevance = _cosine_sim(focal_vec, mem_vec)

            if relevance >= relevance_threshold:
                scored.append((relevance, mem))

        # Sort by relevance
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:n_count]

    def save(self) -> None:
        """Persist core memories to file."""
        data = {
            "memories": self.memories,
            "embeddings": self.embeddings,
        }
        with open(self.store_path, 'w') as f:
            json.dump(data, f, indent=2)

    def format_for_prompt(self, memories: Optional[List[Dict[str, Any]]] = None) -> str:
        """Format core memories for LLM prompt."""
        if memories is None:
            memories = self.memories

        if not memories:
            return "No core memories."

        lines = [f"- {m['description']}" for m in memories]
        return "\n".join(lines)


class MemoryStream:
    """
    Full memory stream implementation inspired by Stanford Generative Agents.

    Supports semantic retrieval with recency, relevance, and importance weighting.
    Stores data in JSON files for persistence and easy inspection.

    NOTE: This class now stores ONLY events, thoughts, and chats.
    Core memories are stored separately in CoreMemoryStore.

    Directory structure:
        agent_folder/
            memory_stream/
                nodes.json        # Event/thought/chat nodes (NOT core)
                embeddings.json   # Embedding vectors
            core_memories.json    # Permanent core memories (separate)
    """

    def __init__(
        self,
        agent_folder: str,
        embedder: Any,  # EmbeddingClient from LLMagentskeleton
        auto_load: bool = True,
    ) -> None:
        """
        Initialize memory stream.

        Args:
            agent_folder: Path to agent's data folder
            embedder: EmbeddingClient instance for generating embeddings
            auto_load: Whether to automatically load from files
        """
        self.agent_folder = Path(agent_folder)
        self.embedder = embedder

        # Memory storage
        self.nodes: List[MemoryNode] = []
        self.embeddings: Dict[str, List[float]] = {}  # embedding_key -> vector
        self.id_to_node: Dict[int, MemoryNode] = {}

        # Keyword index for fast lookup
        self.kw_to_nodes: Dict[str, List[int]] = {}

        # Ensure directory exists
        memory_dir = self.agent_folder / "memory_stream"
        memory_dir.mkdir(parents=True, exist_ok=True)

        if auto_load:
            self._load_nodes()
            self._load_embeddings()

    @property
    def nodes_path(self) -> Path:
        return self.agent_folder / "memory_stream" / "nodes.json"

    @property
    def embeddings_path(self) -> Path:
        return self.agent_folder / "memory_stream" / "embeddings.json"

    def _load_nodes(self) -> None:
        """Load memory nodes from JSON file."""
        if self.nodes_path.exists():
            try:
                with open(self.nodes_path, 'r') as f:
                    data = json.load(f)
                self.nodes = [MemoryNode.from_dict(d) for d in data]
                self.id_to_node = {n.node_id: n for n in self.nodes}
                self._build_keyword_index()
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Failed to load memory nodes: {e}")
                self.nodes = []
                self.id_to_node = {}

    def _load_embeddings(self) -> None:
        """Load embeddings from JSON file."""
        if self.embeddings_path.exists():
            try:
                with open(self.embeddings_path, 'r') as f:
                    self.embeddings = json.load(f)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Failed to load embeddings: {e}")
                self.embeddings = {}

    def _build_keyword_index(self) -> None:
        """Build keyword-to-node index for fast lookup."""
        self.kw_to_nodes = {}
        for node in self.nodes:
            for kw in node.keywords:
                if kw not in self.kw_to_nodes:
                    self.kw_to_nodes[kw] = []
                self.kw_to_nodes[kw].append(node.node_id)

    def _update_keyword_index(self, node: MemoryNode) -> None:
        """Update keyword index for a single node."""
        for kw in node.keywords:
            if kw not in self.kw_to_nodes:
                self.kw_to_nodes[kw] = []
            if node.node_id not in self.kw_to_nodes[kw]:
                self.kw_to_nodes[kw].append(node.node_id)

    def _extract_keywords(self, text: str) -> Set[str]:
        """
        Extract keywords from text for fast retrieval.

        Simple approach: lowercase words, filter stopwords, keep nouns/verbs.
        """
        # Simple tokenization
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

        # Common stopwords to filter
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'between', 'under',
            'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
            'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because', 'until',
            'while', 'although', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours',
            'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers',
            'it', 'its', 'they', 'them', 'their', 'what', 'which', 'who', 'whom',
            'this', 'that', 'these', 'those', 'am', 'about'
        }

        # Filter stopwords and short words
        keywords = {w for w in words if w not in stopwords and len(w) > 2}

        return keywords

    def _get_embedding_key(self, text: str) -> str:
        """Generate a consistent key for embedding lookup."""
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    def _get_or_create_embedding(self, text: str) -> Tuple[str, List[float]]:
        """Get existing embedding or create new one."""
        key = self._get_embedding_key(text)

        if key not in self.embeddings:
            # Generate embedding
            emb_array = self.embedder.embed(text)
            self.embeddings[key] = emb_array.tolist()

        return key, self.embeddings[key]

    def add_event(
        self,
        description: str,
        subject: str,
        predicate: str,
        obj: str,
        now: datetime,
        importance: float = 5.0,
        expiration: Optional[datetime] = None,
    ) -> MemoryNode:
        """
        Add a new event memory.

        Events are observations of what happened - depth 0.

        Args:
            description: Full natural language description
            subject: Who/what the event is about
            predicate: Action or state
            obj: Target of the action
            now: Current datetime
            importance: Poignancy score (1-10)
            expiration: Optional expiration datetime

        Returns:
            The created MemoryNode
        """
        node_id = len(self.nodes)
        now_ts = _utc_ts(now)
        keywords = self._extract_keywords(description)

        # Get or create embedding
        emb_key, _ = self._get_or_create_embedding(description)

        node = MemoryNode(
            node_id=node_id,
            node_type="event",
            depth=0,
            created=now_ts,
            last_accessed=now_ts,
            expiration=_utc_ts(expiration) if expiration else None,
            subject=subject,
            predicate=predicate,
            object=obj,
            description=description,
            keywords=keywords,
            importance=importance,
            embedding_key=emb_key,
            evidence_ids=[],
        )

        self.nodes.append(node)
        self.id_to_node[node_id] = node
        self._update_keyword_index(node)

        return node

    def add_thought(
        self,
        description: str,
        evidence_ids: List[int],
        now: datetime,
        importance: float = 7.0,
    ) -> MemoryNode:
        """
        Add a reflection/thought memory.

        Thoughts are higher-level insights derived from events.
        Depth is 1 + max depth of evidence.

        Args:
            description: The insight or reflection
            evidence_ids: Node IDs that support this thought
            now: Current datetime
            importance: Poignancy score (1-10)

        Returns:
            The created MemoryNode
        """
        # Calculate depth from evidence
        max_evidence_depth = max(
            (self.id_to_node[eid].depth for eid in evidence_ids if eid in self.id_to_node),
            default=0
        )

        node_id = len(self.nodes)
        now_ts = _utc_ts(now)
        keywords = self._extract_keywords(description)

        # Get or create embedding
        emb_key, _ = self._get_or_create_embedding(description)

        node = MemoryNode(
            node_id=node_id,
            node_type="thought",
            depth=max_evidence_depth + 1,
            created=now_ts,
            last_accessed=now_ts,
            subject="I",
            predicate="think",
            object=description,
            description=description,
            keywords=keywords,
            importance=importance,
            embedding_key=emb_key,
            evidence_ids=evidence_ids,
        )

        self.nodes.append(node)
        self.id_to_node[node_id] = node
        self._update_keyword_index(node)

        return node

    def add_chat(
        self,
        description: str,
        other_agent: str,
        now: datetime,
        importance: float = 5.0,
    ) -> MemoryNode:
        """
        Add a conversation memory.

        Args:
            description: Description of the conversation
            other_agent: ID of the other agent involved
            now: Current datetime
            importance: Poignancy score (1-10)

        Returns:
            The created MemoryNode
        """
        node_id = len(self.nodes)
        now_ts = _utc_ts(now)
        keywords = self._extract_keywords(description)
        keywords.add(other_agent.lower())

        # Get or create embedding
        emb_key, _ = self._get_or_create_embedding(description)

        node = MemoryNode(
            node_id=node_id,
            node_type="chat",
            depth=0,
            created=now_ts,
            last_accessed=now_ts,
            subject="I",
            predicate="talked with",
            object=other_agent,
            description=description,
            keywords=keywords,
            importance=importance,
            embedding_key=emb_key,
            evidence_ids=[],
        )

        self.nodes.append(node)
        self.id_to_node[node_id] = node
        self._update_keyword_index(node)

        return node

    def add_core_memory(
        self,
        description: str,
        source_file: str = "",
        now: Optional[datetime] = None,
    ) -> MemoryNode:
        """
        DEPRECATED: Use CoreMemoryStore.add() instead.

        This method is kept for backwards compatibility during migration.
        Core memories should be stored separately in CoreMemoryStore.

        Args:
            description: The core memory content
            source_file: Optional source file (e.g., "persona.md")
            now: Current datetime (defaults to epoch 0 for core)

        Returns:
            The created MemoryNode
        """
        import warnings
        warnings.warn(
            "add_core_memory() is deprecated. Use CoreMemoryStore.add() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        node_id = len(self.nodes)
        now_ts = _utc_ts(now) if now else 0.0
        keywords = self._extract_keywords(description)

        # Get or create embedding
        emb_key, _ = self._get_or_create_embedding(description)

        node = MemoryNode(
            node_id=node_id,
            node_type="core",
            depth=0,
            created=now_ts,
            last_accessed=now_ts,
            subject="I",
            predicate="am/have",
            object=description,
            description=description,
            keywords=keywords,
            importance=10.0,  # Core memories are always high importance
            embedding_key=emb_key,
            evidence_ids=[],
        )

        self.nodes.append(node)
        self.id_to_node[node_id] = node
        self._update_keyword_index(node)

        return node

    def retrieve(
        self,
        focal_point: str,
        now: datetime,
        n_count: int = 30,
        recency_weight: float = 1.0,
        relevance_weight: float = 1.0,
        importance_weight: float = 1.0,
        recency_decay: float = 0.995,
        node_types: Optional[List[str]] = None,
        core_memory_store: Optional["CoreMemoryStore"] = None,
        core_memory_count: int = 5,
    ) -> List[MemoryNode]:
        """
        Retrieve memories relevant to a focal point using three-factor scoring.

        Score = recency_w * recency + relevance_w * relevance + importance_w * importance

        If core_memory_store is provided, retrieves relevant core memories separately
        and unions them with the decaying memory stream results.

        Args:
            focal_point: Question or topic to retrieve memories for
            now: Current datetime
            n_count: Number of memories to retrieve from memory stream
            recency_weight: Weight for recency factor
            relevance_weight: Weight for relevance (cosine similarity)
            importance_weight: Weight for importance score
            recency_decay: Exponential decay rate for recency (per hour)
            node_types: Optional list of node types to filter by
            core_memory_store: Optional CoreMemoryStore for separate core memories
            core_memory_count: Number of core memories to include (default 5)

        Returns:
            List of top-scoring MemoryNodes, sorted by score
            If core_memory_store provided, includes converted core memories
        """
        result_nodes: List[MemoryNode] = []

        # First, retrieve relevant core memories if store provided
        if core_memory_store is not None:
            core_results = core_memory_store.retrieve_relevant(
                focal_point=focal_point,
                n_count=core_memory_count,
                relevance_threshold=0.3,
            )
            # Convert core memories to MemoryNode format for consistent interface
            for relevance, core_mem in core_results:
                # Create a pseudo-MemoryNode for core memory
                # Using negative IDs to distinguish from regular nodes
                core_node = MemoryNode(
                    node_id=-1 - len(result_nodes),  # Negative IDs for core
                    node_type="core",
                    depth=0,
                    created=0.0,  # Epoch time - always existed
                    last_accessed=0.0,
                    description=core_mem["description"],
                    importance=core_mem.get("importance", 10.0),
                    embedding_key=core_mem.get("embedding_key", ""),
                )
                result_nodes.append(core_node)

        if not self.nodes:
            return result_nodes

        now_ts = _utc_ts(now)

        # Get embedding for focal point
        focal_key, focal_emb = self._get_or_create_embedding(focal_point)
        focal_vec = np.array(focal_emb, dtype=np.float32)

        # Filter nodes by type if specified
        # IMPORTANT: Exclude "core" type from memory stream if using separate store
        candidates = self.nodes
        if node_types:
            candidates = [n for n in candidates if n.node_type in node_types]
        elif core_memory_store is not None:
            # If using separate core store, exclude legacy core nodes
            candidates = [n for n in candidates if n.node_type != "core"]

        scores: List[Tuple[float, MemoryNode]] = []

        for node in candidates:
            # Get node embedding
            if node.embedding_key not in self.embeddings:
                continue
            node_emb = np.array(self.embeddings[node.embedding_key], dtype=np.float32)

            # 1. Recency: exponential decay based on time since last access
            hours_since_access = max(0, (now_ts - node.last_accessed) / 3600)
            recency = recency_decay ** hours_since_access

            # Legacy: Core memories don't decay (for backwards compatibility)
            if node.node_type == "core":
                recency = 1.0

            # 2. Relevance: cosine similarity with focal point
            relevance = _cosine_sim(focal_vec, node_emb)

            # 3. Importance: normalized to 0-1
            importance = node.importance / 10.0

            # Combined score
            score = (
                recency_weight * recency +
                relevance_weight * relevance +
                importance_weight * importance
            )

            scores.append((score, node))

        # Sort by score and return top N
        scores.sort(key=lambda x: x[0], reverse=True)
        top_nodes = [node for _, node in scores[:n_count]]

        # Update last_accessed for retrieved nodes
        for node in top_nodes:
            node.last_accessed = now_ts

        # Combine: core memories first, then memory stream results
        result_nodes.extend(top_nodes)
        return result_nodes

    def get_recent_events(
        self,
        now: datetime,
        hours: float = 4.0,
        node_types: Optional[List[str]] = None,
    ) -> List[MemoryNode]:
        """
        Get recent events within a time window.

        Args:
            now: Current datetime
            hours: Number of hours to look back
            node_types: Optional filter for node types

        Returns:
            List of recent MemoryNodes
        """
        now_ts = _utc_ts(now)
        cutoff_ts = now_ts - (hours * 3600)

        recent = []
        for node in self.nodes:
            if node.created >= cutoff_ts:
                if node_types is None or node.node_type in node_types:
                    recent.append(node)

        # Sort by creation time, most recent first
        recent.sort(key=lambda n: n.created, reverse=True)
        return recent

    def get_high_importance_events(
        self,
        now: datetime,
        hours: float = 4.0,
        min_importance: float = 6.0,
    ) -> List[MemoryNode]:
        """
        Get recent high-importance events for reflection.

        Args:
            now: Current datetime
            hours: Number of hours to look back
            min_importance: Minimum importance threshold

        Returns:
            List of high-importance MemoryNodes
        """
        recent = self.get_recent_events(now, hours, node_types=["event", "chat"])
        return [n for n in recent if n.importance >= min_importance]

    def search_by_keywords(
        self,
        keywords: List[str],
        limit: int = 20,
    ) -> List[MemoryNode]:
        """
        Fast keyword-based search.

        Args:
            keywords: List of keywords to search for
            limit: Maximum results to return

        Returns:
            List of matching MemoryNodes
        """
        matching_ids: Set[int] = set()

        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower in self.kw_to_nodes:
                matching_ids.update(self.kw_to_nodes[kw_lower])

        nodes = [self.id_to_node[nid] for nid in matching_ids if nid in self.id_to_node]
        nodes.sort(key=lambda n: n.importance, reverse=True)

        return nodes[:limit]

    def save(self) -> None:
        """Persist memory stream to files."""
        # Save nodes
        nodes_data = [node.to_dict() for node in self.nodes]
        with open(self.nodes_path, 'w') as f:
            json.dump(nodes_data, f, indent=2)

        # Save embeddings
        with open(self.embeddings_path, 'w') as f:
            json.dump(self.embeddings, f)

    def get_node_count(self) -> Dict[str, int]:
        """Get count of nodes by type."""
        counts: Dict[str, int] = {"core": 0, "event": 0, "thought": 0, "chat": 0}
        for node in self.nodes:
            if node.node_type in counts:
                counts[node.node_type] += 1
        return counts

    def format_memories_for_prompt(
        self,
        memories: List[MemoryNode],
        include_timestamp: bool = True,
    ) -> str:
        """
        Format memories into a string suitable for LLM prompt.

        Args:
            memories: List of memories to format
            include_timestamp: Whether to include creation timestamps

        Returns:
            Formatted string
        """
        if not memories:
            return "No relevant memories."

        lines = []
        for mem in memories:
            if include_timestamp and mem.created > 0:
                dt = datetime.fromtimestamp(mem.created, tz=timezone.utc)
                time_str = dt.strftime("%Y-%m-%d %H:%M")
                lines.append(f"- [{time_str}] {mem.description}")
            else:
                lines.append(f"- {mem.description}")

        return "\n".join(lines)

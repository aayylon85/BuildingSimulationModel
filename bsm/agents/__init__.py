"""
LLM-based occupant agent system.

This module contains the generative agent framework for simulating
realistic human behavior in buildings using large language models.

Key components:
- LLMOccupantManager: Main orchestrator for multiple agents
- GenerativeAgent: Agent with memory-driven behavior
- MemoryStream: Semantic memory storage and retrieval
- Cognitive modules: Perceive, Retrieve, Reflect, Act cycle
"""

from bsm.agents.manager import LLMOccupantManager
from bsm.agents.generative_agent import GenerativeAgent, initialize_agents_for_simulation
from bsm.agents.skeleton import (
    OccupantStepDecision,
    OccupantAction,
    DailyPlan,
    CalendarStore,
    EmbeddingClient,
)
from bsm.agents.simulation_adapter import (
    BuildingSimulationAdapter,
    ZoneStateProvider,
    ProductionSimulationAdapter,
)

__all__ = [
    "LLMOccupantManager",
    "GenerativeAgent",
    "initialize_agents_for_simulation",
    "OccupantStepDecision",
    "OccupantAction",
    "DailyPlan",
    "CalendarStore",
    "EmbeddingClient",
    "BuildingSimulationAdapter",
    "ZoneStateProvider",
    "ProductionSimulationAdapter",
]

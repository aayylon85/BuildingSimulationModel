"""
Perception module for generative agents.

Implements the perception stage of the cognitive loop:
- Convert simulation state into perceived events
- Assess importance of events using LLM
- Apply attention bandwidth limits (Stanford-style)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from typing_extensions import TypeAlias

from agents import Agent, Runner, ModelSettings
from agents.agent_output import AgentOutputSchema

from bsm.agents.cognition.schemas import ImportanceAssessment
from bsm.agents.memory.stream import MemoryNode
from bsm.agents.skeleton import DEFAULT_AGENT_MODEL

# Type alias for weather data dictionaries
WeatherDict: TypeAlias = Dict[str, Any]

if TYPE_CHECKING:
    from bsm.agents.generative_agent import GenerativeAgent


# ---------------------------------------------------------------------------
# Agent Builder Factory
# ---------------------------------------------------------------------------

def _build_simple_agent(
    name: str,
    instructions: str,
    output_type: type,
    reasoning_effort: str = "medium",
) -> Agent:
    """
    Factory function for building simple LLM agents.

    Args:
        name: Agent name
        instructions: System instructions
        output_type: Pydantic model for output
        reasoning_effort: Reasoning effort level (low/medium/high)

    Returns:
        Configured Agent instance
    """
    return Agent(
        name=name,
        instructions=instructions,
        model=DEFAULT_AGENT_MODEL,
        model_settings=ModelSettings(reasoning_effort=reasoning_effort),
        output_type=AgentOutputSchema(output_type),
    )


# ---------------------------------------------------------------------------
# LLM-Based Importance Scoring
# ---------------------------------------------------------------------------

def _build_importance_assessor_agent() -> Agent:
    """Build an agent for assessing event importance."""
    instructions = """Rate the importance of an event for a person on a scale of 1-10.

1 = purely mundane (routine tasks, making coffee)
5 = moderately significant (regular meetings, normal work tasks)
10 = extremely significant (major work milestone, serious conflict)

Consider:
- How does this relate to their work and responsibilities?
- Does it affect their comfort or wellbeing?
- Does it involve important relationships?
- Is it likely to be remembered?

Be thoughtful but don't overthink - most routine events should be 3-5."""

    return _build_simple_agent(
        name="importance_assessor",
        instructions=instructions,
        output_type=ImportanceAssessment,
        reasoning_effort="medium",
    )


async def assess_event_importance_llm(
    agent: "GenerativeAgent",
    event_description: str,
) -> int:
    """
    Assess importance of an event using LLM (Stanford generative_agents style).

    Args:
        agent: The agent perceiving the event
        event_description: Description of the event

    Returns:
        Importance score from 1-10
    """
    identity = agent.get_identity_stable_set() if hasattr(agent, 'get_identity_stable_set') else agent.name

    prompt = f"""Here is a brief description of {agent.name}:
{identity}

Rate the importance of this event for {agent.name}:
Event: {event_description}"""

    try:
        assessor = _build_importance_assessor_agent()
        result = await Runner.run(assessor, prompt)
        output: ImportanceAssessment = result.final_output
        return max(1, min(10, output.importance))  # Clamp to 1-10
    except Exception as e:
        print(f"[IMPORTANCE] LLM importance assessment failed: {e}")
        return 5  # Default to moderate importance


async def get_importance(
    agent: "GenerativeAgent",
    event_description: str,
    base_importance: float,
    use_llm: bool = False,
) -> float:
    """
    Get importance score - uses LLM or returns base_importance based on config.

    Args:
        agent: The agent perceiving the event
        event_description: Description of the event
        base_importance: Hardcoded importance value (used if use_llm=False)
        use_llm: Whether to use LLM-based scoring

    Returns:
        Importance score (float)
    """
    if use_llm:
        return float(await assess_event_importance_llm(agent, event_description))
    return base_importance


# ---------------------------------------------------------------------------
# Weather Context Helpers
# ---------------------------------------------------------------------------


def _describe_temperature(temp_c: float) -> str:
    """Describe a temperature in human terms."""
    if temp_c < 0:
        return "freezing"
    elif temp_c < 5:
        return "very cold"
    elif temp_c < 10:
        return "cold"
    elif temp_c < 15:
        return "cool"
    elif temp_c < 20:
        return "mild"
    elif temp_c < 25:
        return "warm"
    elif temp_c < 30:
        return "hot"
    else:
        return "very hot"


def _describe_sky_conditions(cloud_cover: float, is_sunny: bool) -> str:
    """Describe sky conditions based on cloud cover percentage."""
    if cloud_cover is None:
        return "clear" if is_sunny else "overcast"

    if cloud_cover < 10:
        return "clear"
    elif cloud_cover < 30:
        return "mostly clear"
    elif cloud_cover < 60:
        return "partly cloudy"
    elif cloud_cover < 85:
        return "mostly cloudy"
    else:
        return "overcast"


def _describe_wind(wind_speed_ms: float) -> str:
    """Describe wind conditions."""
    if wind_speed_ms is None or wind_speed_ms < 1:
        return "calm"
    elif wind_speed_ms < 3:
        return "light breeze"
    elif wind_speed_ms < 6:
        return "breezy"
    elif wind_speed_ms < 10:
        return "windy"
    else:
        return "very windy"


def _get_season_name(month: int) -> str:
    """Get the season name for a given month."""
    if 6 <= month <= 8:
        return "summer"
    elif month in (12, 1, 2):
        return "winter"
    elif 3 <= month <= 5:
        return "spring"
    else:
        return "autumn"


def _get_seasonal_context(date: datetime, outdoor_temp: float) -> str:
    """
    Generate seasonal interpretation of temperature for perception events.

    Simplified version that uses shared temperature description helper.

    Args:
        date: Current date (to determine season)
        outdoor_temp: Outdoor temperature in Celsius

    Returns:
        Human-readable seasonal weather description
    """
    temp_desc = _describe_temperature(outdoor_temp)
    season = _get_season_name(date.month)
    return f"It's a {temp_desc} {season} day at {outdoor_temp:.0f}°C"


def _get_season_guidance_fallback(date: datetime) -> str:
    """
    Fallback seasonal guidance when no weather data is available.

    Args:
        date: Current date

    Returns:
        Generic guidance text about seasonal temperature expectations
    """
    month = date.month

    if 6 <= month <= 8:  # Summer
        return (
            "It's summer. Temperatures of 15-25°C are pleasant and normal. "
            "Temperatures above 26°C are warm and might prompt comments about heat if planning to go outside. "
            "Temperatures below 15°C are cool and might prompt comments about feeling cold if planning to go outside. "
            "Days are long and more likely to be sunny."
        )
    elif month in (12, 1, 2):  # Winter
        return (
            "It's winter. Temperatures of 0-10°C are typical. "
            "Temperatures above 11°C are warm and might prompt comments if planning to go outside. "
            "Temperatures below 0°C are very cold and might prompt comments if planning to go outside. "
            "Days are shorter."
        )
    elif 3 <= month <= 5:  # Spring
        return (
            "It's spring. Temperatures can vary, with 10-16°C being typical. "
            "Expect some variability day to day. "
            "Temperatures above 11°C are warm and might prompt comments if planning to go outside. "
            "Temperatures below 0°C are very cold and might prompt comments if planning to go outside."
        )
    else:  # Autumn
        return (
            "It's autumn. Temperatures of 10-16°C are typical. "
            "Expect gradually cooling weather."
        )


def get_weather_context(
    date: datetime,
    current_weather: Optional[WeatherDict] = None,
    weather_history: Optional[List[WeatherDict]] = None,
) -> str:
    """
    Generate weather guidance for conversations using actual weather data.

    Provides agents with real weather context for natural conversations,
    including recent weather trends and current conditions.

    Args:
        date: Current datetime
        current_weather: Current weather dict with keys like 'air_temp_c',
                        'wind_speed_local_ms', 'cloud_cover', 'is_sunlit'
        weather_history: List of recent weather dicts (e.g., last 72 hours)
                        Each dict should have at least 'air_temp_c'

    Returns:
        Guidance text about current and recent weather conditions
    """
    # If no weather data provided, fall back to basic seasonal guidance
    if current_weather is None:
        return _get_season_guidance_fallback(date)

    parts = []

    # --- Current conditions ---
    temp = current_weather.get("air_temp_c")
    wind = current_weather.get("wind_speed_local_ms") or current_weather.get("wind_speed_10m_ms")
    cloud_cover = current_weather.get("cloud_cover")
    is_sunny = current_weather.get("is_sunlit", False)

    if temp is not None:
        temp_desc = _describe_temperature(temp)
        sky_desc = _describe_sky_conditions(cloud_cover, is_sunny)
        wind_desc = _describe_wind(wind)

        parts.append(f"Current conditions outside: {temp:.1f}°C ({temp_desc}), {sky_desc}, {wind_desc}.")

    # --- Recent weather trend (if history available) ---
    if weather_history and len(weather_history) > 0:
        # Extract temperatures from history
        temps = [w.get("air_temp_c") for w in weather_history if w.get("air_temp_c") is not None]

        if temps:
            avg_temp = sum(temps) / len(temps)
            min_temp = min(temps)
            max_temp = max(temps)

            # Describe the trend
            trend_parts = []
            trend_parts.append(f"Over the past few days, temperatures have ranged from {min_temp:.0f}°C to {max_temp:.0f}°C")
            trend_parts.append(f"(averaging {avg_temp:.0f}°C).")

            # Compare current to recent average
            if temp is not None:
                diff = temp - avg_temp
                if abs(diff) > 3:
                    if diff > 0:
                        trend_parts.append("Today is notably warmer than recent days.")
                    else:
                        trend_parts.append("Today is notably cooler than recent days.")

            parts.append(" ".join(trend_parts))

            # Cloud cover trend
            clouds = [w.get("cloud_cover") for w in weather_history if w.get("cloud_cover") is not None]
            if clouds:
                avg_cloud = sum(clouds) / len(clouds)
                if avg_cloud > 70:
                    parts.append("It's been mostly cloudy recently.")
                elif avg_cloud < 30:
                    parts.append("It's been mostly clear and sunny recently.")

    # --- Add seasonal context ---
    season = _get_season_name(date.month)
    if season == "summer":
        parts.append("It's summer, so longer days and generally warmer weather.")
    elif season == "winter":
        parts.append("It's winter, so shorter days and generally cooler weather.")
    elif season == "spring":
        parts.append("It's spring, weather can be changeable.")
    else:
        parts.append("It's autumn, weather is gradually cooling.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Perception Function
# ---------------------------------------------------------------------------

async def perceive(
    agent: "GenerativeAgent",
    sim_state: Dict[str, Any],
    now: datetime,
    use_llm_importance: bool = True,
) -> List[MemoryNode]:
    """
    Convert simulation state into perceived events.

    This is the first step in the cognitive loop. The agent observes
    the current state and creates memory events for notable observations.

    Stanford-style attention bandwidth: The agent can only perceive a limited
    number of events per timestep. Events are prioritized by salience/importance
    and only the top N (perception_bandwidth) are actually perceived.

    Args:
        agent: The generative agent
        sim_state: Current simulation state dict
        now: Current simulation datetime
        use_llm_importance: Whether to use LLM to assess importance (default True)

    Returns:
        List of newly created MemoryNode events
    """
    if not agent.memory_stream:
        return []

    # Collect potential perceptions with priority scores
    # Format: (priority, description, subject, predicate, obj, base_importance)
    from typing import Tuple
    PotentialPerception: TypeAlias = Tuple[float, str, str, str, str, float]
    potential_perceptions: List[PotentialPerception] = []

    # Perceive indoor temperature as raw data - let agent reason about comfort
    temp = sim_state.get("indoor_temp_c", 21.0)
    if not agent.recently_perceived("indoor_temp", now, within_minutes=60):
        potential_perceptions.append((
            5.0,  # Medium priority - factual observation
            f"The room temperature is {temp:.1f}C",
            "room", "temperature is", f"{temp:.1f}C", 3.0
        ))

    # Perceive other occupants present
    other_occupants = sim_state.get("other_occupants_present", [])
    for other_id in other_occupants:
        if not agent.recently_perceived(other_id, now, within_minutes=120):
            # Haven't noticed them recently - medium priority
            potential_perceptions.append((
                5.0,
                f"{other_id} is in the office",
                other_id, "is present in", "office", 4.0
            ))

    # Track just_arrived state (will clear flag at end of function)
    just_arrived = agent.has_just_arrived()

    # Perceive equipment state continuously (lower priority - let agent decide based on needs)
    # Only perceive if not recently noticed to avoid spam
    equipment_status = sim_state.get("equipment_status", {})
    equipment_items = equipment_status.get("items", {})
    for equipment_name, is_on in equipment_items.items():
        # Only perceive equipment that's off and hasn't been recently noticed
        if not is_on and not agent.recently_perceived(f"equipment_{equipment_name}", now, within_minutes=60):
            potential_perceptions.append((
                3.0,  # Lower priority - let agent decide based on their needs
                f"The {equipment_name} is currently off",
                "I", "notice", f"{equipment_name} is off", 3.0
            ))

    # Perceive lighting conditions (also continuous, low priority)
    lighting = sim_state.get("lighting_conditions", {})
    natural_light = lighting.get("natural_light_level", "moderate")
    desk_light_on = lighting.get("desk_light_on", False)

    if natural_light in ["dim", "dark"] and not desk_light_on:
        if not agent.recently_perceived("lighting_dim", now, within_minutes=60):
            potential_perceptions.append((
                3.0,  # Lower priority - observational
                f"The lighting is {natural_light} and the desk light is off",
                "workspace", "has", f"{natural_light} lighting", 3.0
            ))

    # Perceive outdoor conditions as raw data
    weather_desc = sim_state.get("weather_description", "")
    outdoor_temp = sim_state.get("outdoor_temp_c", 10.0)

    # Outdoor temperature - separate from indoor for comparison
    if not agent.recently_perceived("outdoor_temp", now, within_minutes=60):
        potential_perceptions.append((
            4.0,  # Medium priority - useful context for comfort decisions
            f"Outside it is {outdoor_temp:.1f}C",
            "outdoor", "temperature is", f"{outdoor_temp:.1f}C", 3.0
        ))

    # Weather conditions
    if weather_desc and not agent.recently_perceived("weather", now, within_minutes=180):
        potential_perceptions.append((
            3.5,
            f"The weather outside is {weather_desc}",
            "weather", "is", weather_desc, 3.0
        ))

    # Seasonal context - helps agents interpret temperature appropriately for the season
    if not agent.recently_perceived("seasonal_weather", now, within_minutes=240):
        seasonal_context = _get_seasonal_context(now, outdoor_temp)
        potential_perceptions.append((
            4.5,  # Higher priority than raw weather - provides interpretation
            seasonal_context,
            "weather", "feels like", seasonal_context, 3.5
        ))

    # Perceive equipment at current location (higher priority when at non-desk locations)
    current_location = sim_state.get("current_location", "desk_area")
    location_equipment = sim_state.get("location_equipment", [])

    if location_equipment and current_location != "desk_area":
        # At a non-desk location (break_area, meeting_room, etc.) - perceive available equipment
        location_key = f"location_equipment_{current_location}"
        if not agent.recently_perceived(location_key, now, within_minutes=15):
            equipment_names = [eq["name"] for eq in location_equipment]
            equipment_states = [
                f"{eq['name']} ({'ON' if eq.get('is_on') else 'available'})"
                for eq in location_equipment
            ]
            if equipment_names:
                potential_perceptions.append((
                    5.5,  # Higher priority - new location equipment is interesting
                    f"At {current_location}, I notice: {', '.join(equipment_states)}",
                    "I", "notice equipment at", current_location, 4.0
                ))

    # Sort by priority (highest first) and limit to perception bandwidth
    bandwidth = agent.get_perception_bandwidth()
    potential_perceptions.sort(key=lambda x: x[0], reverse=True)

    # Create only the top N perceptions with LLM-assessed importance
    perceived_events: List[MemoryNode] = []
    for priority, desc, subj, pred, obj, base_imp in potential_perceptions[:bandwidth]:
        # Assess importance using LLM if enabled
        importance = await get_importance(agent, desc, base_imp, use_llm=use_llm_importance)

        event = agent.memory_stream.add_event(
            description=desc,
            subject=subj,
            predicate=pred,
            obj=obj,
            now=now,
            importance=importance,
        )
        perceived_events.append(event)

    # Clear just_arrived flag if it was set (always do this even if filtered)
    if just_arrived:
        agent.clear_just_arrived()

    return perceived_events


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "perceive",
    "get_importance",
    "assess_event_importance_llm",
    "_build_simple_agent",  # Export factory for use in other modules
    "get_weather_context",  # Weather context for conversations
    "WeatherDict",  # Type alias for weather data
]

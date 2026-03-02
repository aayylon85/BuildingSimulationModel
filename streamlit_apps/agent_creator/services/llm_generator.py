"""LLM content generator using OpenAI API."""

import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class GeneratedContent:
    """Container for all generated agent content."""
    persona_md: str
    background_md: str
    work_style_md: str
    relationships_md: str
    scratch_json: Dict[str, Any]


class LLMGenerator:
    """Generate agent content using OpenAI API."""

    MODEL = "gpt-5.2"
    # GPT-5.2 does not support temperature - use reasoning_effort instead
    REASONING_EFFORT_CREATIVE = "medium"    # For markdown content
    REASONING_EFFORT_STRUCTURED = "low"     # For JSON generation
    MAX_COMPLETION_TOKENS_MARKDOWN = 2000
    MAX_COMPLETION_TOKENS_JSON = 1500

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with OpenAI API key."""
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def _format_traits(self, traits: list) -> str:
        """Format traits list as comma-separated string."""
        return ", ".join(traits) if traits else "adaptable, professional"

    def _format_relationships_context(self, relationships: Dict, relationship_notes: Dict, existing_agents: Dict) -> str:
        """Format relationships for prompt context."""
        if not relationships:
            return "No existing relationships defined yet."

        lines = []
        for agent_id, values in relationships.items():
            agent_name = existing_agents.get(agent_id, {}).get("name", agent_id)
            familiarity = values.get("familiarity", 0.5)
            sentiment = values.get("sentiment", 0.5)
            notes = relationship_notes.get(agent_id, "")

            lines.append(f"### With {agent_name} ({agent_id})")
            lines.append(f"- Familiarity: {familiarity:.1f} (0=stranger, 1=close)")
            lines.append(f"- Sentiment: {sentiment:.1f} (0=negative, 1=positive)")
            if notes:
                lines.append(f"- Notes: {notes}")
            lines.append("")

        return "\n".join(lines)

    def _format_relationship_sections(self, relationships: Dict, relationship_notes: Dict, existing_agents: Dict) -> str:
        """Format relationship sections for relationships.md generation."""
        if not relationships:
            return "## No Established Relationships Yet\nThis agent is new to the office."

        sections = []
        for agent_id, values in relationships.items():
            agent_name = existing_agents.get(agent_id, {}).get("name", agent_id)
            sections.append(f"## With {agent_name}")
            sections.append("[Generate relationship details based on familiarity and sentiment scores]")
            sections.append("")

        return "\n".join(sections)

    def _format_relationships_json(self, relationships: Dict) -> str:
        """Format relationships for JSON."""
        if not relationships:
            return "{}"

        models = {}
        for agent_id, values in relationships.items():
            models[agent_id] = {
                "familiarity": round(values.get("familiarity", 0.5), 1),
                "sentiment": round(values.get("sentiment", 0.5), 1)
            }
        return json.dumps(models, indent=4)

    def _prepare_form_data(self, form_data: Dict, existing_agents: Dict) -> Dict:
        """Prepare form data for prompt formatting."""
        # Handle list fields
        innate_traits = form_data.get("innate_traits", [])
        if isinstance(innate_traits, list):
            innate_traits = self._format_traits(innate_traits)

        communication_style = form_data.get("communication_style", [])
        if isinstance(communication_style, list):
            communication_style = ", ".join(communication_style) if communication_style else "professional"

        decision_making_style = form_data.get("decision_making_style", [])
        if isinstance(decision_making_style, list):
            decision_making_style = ", ".join(decision_making_style) if decision_making_style else "thoughtful"

        relationships = form_data.get("relationships", {})
        relationship_notes = form_data.get("relationship_notes", {})

        return {
            "agent_id": form_data.get("agent_id", ""),
            "name": form_data.get("name", ""),
            "first_name": form_data.get("first_name", ""),
            "age": form_data.get("age", 30),
            "origin": form_data.get("origin", ""),
            "years_in_southampton": form_data.get("years_in_southampton", 0),
            "languages": form_data.get("languages", "English"),
            "innate_traits": innate_traits,
            "personality_description": form_data.get("personality_description", ""),
            "communication_style": communication_style,
            "decision_making_style": decision_making_style,
            "thermal_preference": form_data.get("thermal_preference", "neutral"),
            "runs_warm": form_data.get("runs_warm", False),
            "runs_warm_json": str(form_data.get("runs_warm", False)).lower(),
            "clothing_insulation_typical": form_data.get("clothing_insulation_typical", "medium"),
            "light_preference": form_data.get("light_preference", "natural"),
            "noise_sensitivity": form_data.get("noise_sensitivity", "moderate"),
            "typical_arrival": form_data.get("typical_arrival", "08:30-09:00"),
            "typical_departure": form_data.get("typical_departure", "17:00-17:30"),
            "preferred_desk": form_data.get("preferred_desk", "Desk_A"),
            "works_weekends": form_data.get("works_weekends", False),
            "works_weekends_json": str(form_data.get("works_weekends", False)).lower(),
            "workspace_variety": form_data.get("workspace_variety", "medium"),
            "likes_desk_variety": form_data.get("likes_desk_variety", True),
            "likes_desk_variety_json": str(form_data.get("likes_desk_variety", True)).lower(),
            "relationships_context": self._format_relationships_context(
                relationships, relationship_notes, existing_agents
            ),
            "relationship_sections": self._format_relationship_sections(
                relationships, relationship_notes, existing_agents
            ),
            "relationships_json": relationships,
            "relationships_models_json": self._format_relationships_json(relationships),
        }

    def generate_persona(self, form_data: Dict, existing_agents: Dict = None) -> str:
        """Generate persona.md content."""
        from ..prompts.templates import PERSONA_PROMPT

        prepared = self._prepare_form_data(form_data, existing_agents or {})
        prompt = PERSONA_PROMPT.format(**prepared)

        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {
                    "role": "developer",
                    "content": "You are an expert at creating detailed, realistic character profiles for building occupant simulations. Generate content in markdown format."
                },
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=self.MAX_COMPLETION_TOKENS_MARKDOWN,
            reasoning_effort=self.REASONING_EFFORT_CREATIVE,
        )

        return response.choices[0].message.content

    def generate_background(self, form_data: Dict, existing_agents: Dict = None) -> str:
        """Generate background.md content."""
        from ..prompts.templates import BACKGROUND_PROMPT

        prepared = self._prepare_form_data(form_data, existing_agents or {})
        prompt = BACKGROUND_PROMPT.format(**prepared)

        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {
                    "role": "developer",
                    "content": "You are an expert at creating detailed, realistic backstories for simulated office workers. Generate content in markdown format."
                },
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=self.MAX_COMPLETION_TOKENS_MARKDOWN,
            reasoning_effort=self.REASONING_EFFORT_CREATIVE,
        )

        return response.choices[0].message.content

    def generate_work_style(self, form_data: Dict, existing_agents: Dict = None) -> str:
        """Generate work_style.md content."""
        from ..prompts.templates import WORK_STYLE_PROMPT

        prepared = self._prepare_form_data(form_data, existing_agents or {})
        prompt = WORK_STYLE_PROMPT.format(**prepared)

        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {
                    "role": "developer",
                    "content": "You are an expert at creating detailed work style profiles for building occupant simulations. Generate content in markdown format."
                },
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=self.MAX_COMPLETION_TOKENS_MARKDOWN,
            reasoning_effort=self.REASONING_EFFORT_CREATIVE,
        )

        return response.choices[0].message.content

    def generate_relationships(self, form_data: Dict, existing_agents: Dict = None) -> str:
        """Generate relationships.md content."""
        from ..prompts.templates import RELATIONSHIPS_PROMPT

        prepared = self._prepare_form_data(form_data, existing_agents or {})
        prompt = RELATIONSHIPS_PROMPT.format(**prepared)

        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {
                    "role": "developer",
                    "content": "You are an expert at creating realistic workplace relationship dynamics. Generate content in markdown format."
                },
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=self.MAX_COMPLETION_TOKENS_MARKDOWN,
            reasoning_effort=self.REASONING_EFFORT_CREATIVE,
        )

        return response.choices[0].message.content

    def generate_scratch_json(self, form_data: Dict, existing_agents: Dict = None) -> Dict[str, Any]:
        """Generate scratch.json with strict schema adherence."""
        from ..prompts.templates import SCRATCH_JSON_PROMPT

        prepared = self._prepare_form_data(form_data, existing_agents or {})
        prompt = SCRATCH_JSON_PROMPT.format(**prepared)

        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {
                    "role": "developer",
                    "content": "You are a JSON generator. Output ONLY valid JSON, no markdown formatting, no explanations. Follow the schema exactly."
                },
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=self.MAX_COMPLETION_TOKENS_JSON,
            reasoning_effort=self.REASONING_EFFORT_STRUCTURED,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        return json.loads(content)

    def generate_all(self, form_data: Dict, existing_agents: Dict = None) -> GeneratedContent:
        """Generate all agent content."""
        existing = existing_agents or {}

        return GeneratedContent(
            persona_md=self.generate_persona(form_data, existing),
            background_md=self.generate_background(form_data, existing),
            work_style_md=self.generate_work_style(form_data, existing),
            relationships_md=self.generate_relationships(form_data, existing),
            scratch_json=self.generate_scratch_json(form_data, existing),
        )

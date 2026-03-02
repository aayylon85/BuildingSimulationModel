"""File system operations for saving agents."""

import json
from pathlib import Path
from typing import Dict, Any


class AgentSaver:
    """Save generated agent to the filesystem."""

    def __init__(self, base_path: str = "agent_base_types"):
        """Initialize with base path for agent storage."""
        self.base_path = Path(base_path)

    def get_folder_name(self, first_name: str) -> str:
        """Generate folder name from first name."""
        return f"{first_name.lower()}_office_worker"

    def folder_exists(self, first_name: str) -> bool:
        """Check if agent folder already exists."""
        folder_name = self.get_folder_name(first_name)
        return (self.base_path / folder_name).exists()

    def save_agent(
        self,
        form_data: Dict[str, Any],
        generated_content: Dict[str, Any]
    ) -> Path:
        """Save the complete agent to the filesystem.

        Creates:
        - agent_base_types/{name}_office_worker/
            - scratch.json
            - persona.md
            - background.md
            - work_style.md
            - relationships.md
            - memory_stream/
                - nodes.json (empty: [])
                - embeddings.json (empty: {})

        Args:
            form_data: User form input data
            generated_content: Dict with keys:
                - persona_md: str
                - background_md: str
                - work_style_md: str
                - relationships_md: str
                - scratch_json: dict

        Returns:
            Path to created agent folder
        """
        # Create folder name from first_name
        first_name = form_data.get("first_name", "agent")
        folder_name = self.get_folder_name(first_name)
        agent_folder = self.base_path / folder_name

        # Create directories
        agent_folder.mkdir(parents=True, exist_ok=True)
        (agent_folder / "memory_stream").mkdir(exist_ok=True)

        # Save markdown files
        for filename, key in [
            ("persona.md", "persona_md"),
            ("background.md", "background_md"),
            ("work_style.md", "work_style_md"),
            ("relationships.md", "relationships_md"),
        ]:
            content = generated_content.get(key, "")
            (agent_folder / filename).write_text(content)

        # Save scratch.json
        scratch_data = generated_content.get("scratch_json", {})
        with open(agent_folder / "scratch.json", "w") as f:
            json.dump(scratch_data, f, indent=2)

        # Initialize empty memory stream files
        with open(agent_folder / "memory_stream" / "nodes.json", "w") as f:
            json.dump([], f)

        with open(agent_folder / "memory_stream" / "embeddings.json", "w") as f:
            json.dump({}, f)

        return agent_folder

    def delete_agent(self, first_name: str) -> bool:
        """Delete an agent folder (use with caution).

        Returns:
            True if deleted, False if not found
        """
        import shutil

        folder_name = self.get_folder_name(first_name)
        agent_folder = self.base_path / folder_name

        if agent_folder.exists():
            shutil.rmtree(agent_folder)
            return True
        return False

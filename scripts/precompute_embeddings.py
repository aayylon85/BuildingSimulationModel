#!/usr/bin/env python3
"""Pre-compute core memory embeddings for agent base types."""

import sys
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bsm.agents.memory.stream import CoreMemoryStore
from bsm.agents.skeleton import EmbeddingClient


def split_markdown_into_chunks(content: str) -> List[str]:
    """Split markdown content into semantic chunks (same logic as GenerativeAgent)."""
    chunks = []
    current_section = ""

    for line in content.split('\n'):
        line = line.strip()
        if not line or line.startswith('# '):
            if current_section:
                chunks.append(current_section)
                current_section = ""
            continue
        if line.startswith('## '):
            if current_section:
                chunks.append(current_section)
            current_section = ""
            continue
        if line.startswith('- '):
            item = line[2:].strip()
            if len(item) > 10:
                chunks.append(item)
            continue
        if line:
            current_section += line + " "

    if current_section.strip():
        chunks.append(current_section.strip())
    return chunks


def precompute_for_agent(agent_folder: Path, embedder: EmbeddingClient) -> None:
    """Pre-compute and save core memory embeddings for one agent."""
    print(f"\nProcessing {agent_folder.name}...")

    # Create CoreMemoryStore directly (no safety check like GenerativeAgent)
    store = CoreMemoryStore(
        agent_folder=str(agent_folder),
        embedder=embedder,
        auto_load=False,  # Start fresh
    )

    # Read and process markdown files
    md_files = ["persona.md", "background.md", "work_style.md", "relationships.md"]
    total_added = 0

    for md_name in md_files:
        md_path = agent_folder / md_name
        if md_path.exists():
            content = md_path.read_text()
            chunks = split_markdown_into_chunks(content)
            source = md_name.replace(".md", "")

            for chunk in chunks:
                if chunk.strip():
                    print(f"  Adding: {chunk[:50]}...")
                    store.add(description=chunk.strip(), source=source)
                    total_added += 1

    # Save to base type folder
    store.save()
    print(f"  Created {total_added} memories, {len(store._embeddings)} embeddings")


def main():
    base_types = project_root / "agent_base_types"
    embedder = EmbeddingClient()

    agent_folders = [d for d in base_types.iterdir()
                     if d.is_dir() and not d.name.startswith(("_", "."))]

    print(f"Found {len(agent_folders)} agent base types to process")

    for agent_dir in agent_folders:
        precompute_for_agent(agent_dir, embedder)

    print("\nDone! Embeddings cached in agent_base_types/*/")


if __name__ == "__main__":
    main()

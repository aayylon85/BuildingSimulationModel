"""Main entry point for Agent Creator Streamlit app."""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_apps.agent_creator.services.existing_agents import load_existing_agents


def initialize_session_state():
    """Initialize session state variables."""
    if "form_data" not in st.session_state:
        st.session_state.form_data = {
            "agent_id": "",
            "name": "",
            "first_name": "",
            "age": 30,
            "origin": "",
            "years_in_southampton": 3,
            "languages": "English",
            "innate_traits": [],
            "personality_description": "",
            "communication_style": [],
            "decision_making_style": [],
            "typical_arrival": "08:30-09:00",
            "typical_departure": "17:00-17:30",
            "preferred_desk": "Desk_A",
            "works_weekends": False,
            "workspace_variety": "medium",
            "likes_desk_variety": True,
            "thermal_preference": "neutral",
            "runs_warm": False,
            "clothing_insulation_typical": "medium",
            "light_preference": "natural",
            "noise_sensitivity": "moderate",
            "relationships": {},
            "relationship_notes": {},
        }

    if "generated_content" not in st.session_state:
        st.session_state.generated_content = {
            "persona_md": "",
            "background_md": "",
            "work_style_md": "",
            "relationships_md": "",
            "scratch_json": {},
        }

    if "current_step" not in st.session_state:
        st.session_state.current_step = 1

    if "existing_agents" not in st.session_state:
        st.session_state.existing_agents = load_existing_agents(
            str(PROJECT_ROOT / "agent_base_types")
        )

    if "content_generated" not in st.session_state:
        st.session_state.content_generated = False

    if "validation_status" not in st.session_state:
        st.session_state.validation_status = {
            "is_valid": False,
            "errors": [],
        }


def render_progress_bar():
    """Render step progress indicator."""
    steps = [
        "Basic Info",
        "Personality",
        "Work Style",
        "Relationships",
        "Preview",
        "Save"
    ]

    current = st.session_state.current_step
    cols = st.columns(len(steps))

    for i, (col, step) in enumerate(zip(cols, steps), 1):
        with col:
            if i < current:
                st.markdown(f"**:white_check_mark: {step}**")
            elif i == current:
                st.markdown(f"**:arrow_right: {step}**")
            else:
                st.markdown(f":grey[{step}]")

    st.divider()


def navigate_to(step: int):
    """Navigate to a specific step."""
    st.session_state.current_step = step
    st.rerun()


def main():
    """Main application."""
    st.set_page_config(
        page_title="Agent Creator - Building Simulation",
        page_icon=":office:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("Create New Building Agent")
    st.markdown(
        "Create a new simulated office worker with AI-generated personality "
        "and characteristics."
    )

    initialize_session_state()
    render_progress_bar()

    # Import and render current page
    current_step = st.session_state.current_step

    if current_step == 1:
        from streamlit_apps.agent_creator.pages import basic_info
        basic_info.render()
    elif current_step == 2:
        from streamlit_apps.agent_creator.pages import personality
        personality.render()
    elif current_step == 3:
        from streamlit_apps.agent_creator.pages import work_style
        work_style.render()
    elif current_step == 4:
        from streamlit_apps.agent_creator.pages import relationships
        relationships.render()
    elif current_step == 5:
        from streamlit_apps.agent_creator.pages import preview
        preview.render()
    elif current_step == 6:
        from streamlit_apps.agent_creator.pages import save_agent
        save_agent.render()


if __name__ == "__main__":
    main()

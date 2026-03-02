"""Step 1: Basic Information page."""

import streamlit as st
import re
from pathlib import Path

# Add project root for imports
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_apps.agent_creator.services.existing_agents import (
    validate_agent_id_unique,
    get_suggested_agent_id,
)


def validate_agent_id(agent_id: str) -> tuple:
    """Validate agent_id format and uniqueness."""
    return validate_agent_id_unique(
        agent_id,
        str(PROJECT_ROOT / "agent_base_types")
    )


def render():
    """Render the basic information form."""
    st.header("Step 1: Basic Information")
    st.markdown("Enter the agent's identity and demographic information.")

    form_data = st.session_state.form_data

    col1, col2 = st.columns(2)

    with col1:
        # First name (used for agent_id suggestion)
        first_name = st.text_input(
            "First Name *",
            value=form_data.get("first_name", ""),
            help="First name only (e.g., 'David')",
            placeholder="David"
        )

        # Auto-suggest agent_id when first_name changes
        if first_name and first_name != form_data.get("first_name", ""):
            suggested_id = get_suggested_agent_id(
                first_name,
                str(PROJECT_ROOT / "agent_base_types")
            )
            if not form_data.get("agent_id"):
                form_data["agent_id"] = suggested_id

        # Agent ID with validation
        agent_id = st.text_input(
            "Agent ID *",
            value=form_data.get("agent_id", ""),
            help="Unique identifier (e.g., 'david_004'). Auto-suggested based on first name.",
            placeholder="david_004"
        )

        # Real-time validation
        if agent_id:
            is_valid, error_msg = validate_agent_id(agent_id)
            if not is_valid:
                st.error(error_msg)
            else:
                st.success("Agent ID is available!")

        # Full name
        name = st.text_input(
            "Full Name *",
            value=form_data.get("name", ""),
            help="Full display name (e.g., 'David Chen')",
            placeholder="David Chen"
        )

    with col2:
        # Age
        age = st.number_input(
            "Age",
            min_value=25,
            max_value=65,
            value=form_data.get("age", 30),
            help="Agent's age (25-65)"
        )

        # Origin
        origin = st.text_input(
            "Originally From",
            value=form_data.get("origin", ""),
            help="Where they are originally from",
            placeholder="London, UK"
        )

        # Years in Southampton
        years_in_southampton = st.number_input(
            "Years in Southampton",
            min_value=0,
            max_value=50,
            value=form_data.get("years_in_southampton", 3),
            help="How long they've lived in Southampton"
        )

    # Languages (full width)
    languages = st.text_input(
        "Languages",
        value=form_data.get("languages", "English"),
        help="Languages spoken",
        placeholder="English (native), Mandarin (fluent)"
    )

    st.divider()

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])

    with col3:
        if st.button("Next: Personality", type="primary", use_container_width=True):
            # Validate required fields
            errors = []
            if not agent_id:
                errors.append("Agent ID is required")
            elif not validate_agent_id(agent_id)[0]:
                errors.append(validate_agent_id(agent_id)[1])
            if not name:
                errors.append("Full Name is required")
            if not first_name:
                errors.append("First Name is required")

            if errors:
                for error in errors:
                    st.error(error)
            else:
                # Save to session state
                st.session_state.form_data.update({
                    "agent_id": agent_id,
                    "name": name,
                    "first_name": first_name,
                    "age": age,
                    "origin": origin,
                    "years_in_southampton": years_in_southampton,
                    "languages": languages,
                })
                st.session_state.current_step = 2
                st.rerun()

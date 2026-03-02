"""Step 6: Save agent page."""

import streamlit as st
from pathlib import Path
import sys

# Add project root for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_apps.agent_creator.services.agent_saver import AgentSaver
from streamlit_apps.agent_creator.services.schema_validator import SchemaValidator


def render():
    """Render the save agent page."""
    st.header("Step 6: Save Agent")
    st.markdown("Review the summary and save the new agent to the simulation.")

    form_data = st.session_state.form_data
    generated = st.session_state.generated_content
    validation = st.session_state.validation_status

    # Summary card
    st.subheader("Agent Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Identity**")
        st.markdown(f"- Agent ID: `{form_data.get('agent_id', '')}`")
        st.markdown(f"- Name: {form_data.get('name', '')}")
        st.markdown(f"- Age: {form_data.get('age', '')}")
        st.markdown(f"- Origin: {form_data.get('origin', '')}")

        st.markdown("**Personality**")
        traits = form_data.get('innate_traits', [])
        st.markdown(f"- Traits: {', '.join(traits) if traits else 'N/A'}")

    with col2:
        st.markdown("**Schedule**")
        st.markdown(f"- Arrival: {form_data.get('typical_arrival', '')}")
        st.markdown(f"- Departure: {form_data.get('typical_departure', '')}")
        st.markdown(f"- Desk: {form_data.get('preferred_desk', '')}")

        st.markdown("**Comfort**")
        st.markdown(f"- Thermal: {form_data.get('thermal_preference', '')}")
        st.markdown(f"- Runs warm: {form_data.get('runs_warm', False)}")
        st.markdown(f"- Light: {form_data.get('light_preference', '')}")

    st.divider()

    # Validation checklist
    st.subheader("Validation Checklist")

    saver = AgentSaver(str(PROJECT_ROOT / "agent_base_types"))

    checks = []

    # Check 1: All files generated
    all_files = all([
        generated.get("persona_md"),
        generated.get("background_md"),
        generated.get("work_style_md"),
        generated.get("relationships_md"),
        generated.get("scratch_json"),
    ])
    checks.append(("All required files generated", all_files))

    # Check 2: scratch.json valid
    checks.append(("scratch.json passes schema validation", validation.get("is_valid", False)))

    # Check 3: agent_id unique
    agent_id = form_data.get("agent_id", "")
    existing_agents = st.session_state.existing_agents
    id_unique = agent_id not in existing_agents
    checks.append(("Agent ID is unique", id_unique))

    # Check 4: Folder doesn't exist
    folder_exists = saver.folder_exists(form_data.get("first_name", ""))
    checks.append(("Agent folder doesn't already exist", not folder_exists))

    # Display checks
    all_passed = True
    for check_name, passed in checks:
        if passed:
            st.markdown(f":white_check_mark: {check_name}")
        else:
            st.markdown(f":x: {check_name}")
            all_passed = False

    st.divider()

    # Files to be created
    st.subheader("Files to be Created")
    first_name = form_data.get("first_name", "agent").lower()
    folder_name = f"{first_name}_office_worker"

    st.code(f"""
agent_base_types/{folder_name}/
    scratch.json
    persona.md
    background.md
    work_style.md
    relationships.md
    memory_stream/
        nodes.json
        embeddings.json
""")

    st.divider()

    # Navigation and save buttons
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("Back: Preview", use_container_width=True):
            st.session_state.current_step = 5
            st.rerun()

    with col3:
        if st.button(
            "Save Agent",
            type="primary",
            use_container_width=True,
            disabled=not all_passed
        ):
            try:
                # Save the agent
                agent_folder = saver.save_agent(form_data, generated)

                st.success(f"Agent saved successfully!")
                st.markdown(f"**Location:** `{agent_folder}`")

                # Show success message with next steps
                st.balloons()

                st.info(
                    "**Next steps:**\n"
                    "1. Review the generated files in the agent folder\n"
                    "2. Add the agent to your simulation configuration\n"
                    "3. Run a simulation to test the new agent"
                )

                # Option to create another agent
                if st.button("Create Another Agent"):
                    # Reset session state
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
                    st.session_state.generated_content = {}
                    st.session_state.content_generated = False
                    st.session_state.current_step = 1
                    # Reload existing agents to include the new one
                    from streamlit_apps.agent_creator.services.existing_agents import load_existing_agents
                    st.session_state.existing_agents = load_existing_agents(
                        str(PROJECT_ROOT / "agent_base_types")
                    )
                    st.rerun()

            except Exception as e:
                st.error(f"Failed to save agent: {e}")

        if not all_passed:
            st.caption(":red[Fix validation errors before saving]")

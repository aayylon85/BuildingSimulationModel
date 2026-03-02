"""Step 5: Preview generated content page."""

import streamlit as st
import json
from pathlib import Path
import sys

# Add project root for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_apps.agent_creator.services.llm_generator import LLMGenerator
from streamlit_apps.agent_creator.services.schema_validator import SchemaValidator


def generate_content():
    """Generate all agent content using LLM."""
    form_data = st.session_state.form_data
    existing_agents = st.session_state.existing_agents

    generator = LLMGenerator()
    validator = SchemaValidator()

    with st.spinner("Generating persona..."):
        persona_md = generator.generate_persona(form_data, existing_agents)

    with st.spinner("Generating background..."):
        background_md = generator.generate_background(form_data, existing_agents)

    with st.spinner("Generating work style..."):
        work_style_md = generator.generate_work_style(form_data, existing_agents)

    with st.spinner("Generating relationships..."):
        relationships_md = generator.generate_relationships(form_data, existing_agents)

    with st.spinner("Generating configuration..."):
        scratch_json = generator.generate_scratch_json(form_data, existing_agents)

    # Validate and fix scratch.json
    fixed_json, is_valid, errors = validator.validate_and_fix(scratch_json)

    # Store in session state
    st.session_state.generated_content = {
        "persona_md": persona_md,
        "background_md": background_md,
        "work_style_md": work_style_md,
        "relationships_md": relationships_md,
        "scratch_json": fixed_json,
    }

    st.session_state.validation_status = {
        "is_valid": is_valid,
        "errors": errors,
    }

    st.session_state.content_generated = True


def regenerate_single(content_type: str):
    """Regenerate a single piece of content."""
    form_data = st.session_state.form_data
    existing_agents = st.session_state.existing_agents
    generator = LLMGenerator()
    validator = SchemaValidator()

    with st.spinner(f"Regenerating {content_type}..."):
        if content_type == "persona_md":
            content = generator.generate_persona(form_data, existing_agents)
        elif content_type == "background_md":
            content = generator.generate_background(form_data, existing_agents)
        elif content_type == "work_style_md":
            content = generator.generate_work_style(form_data, existing_agents)
        elif content_type == "relationships_md":
            content = generator.generate_relationships(form_data, existing_agents)
        elif content_type == "scratch_json":
            scratch_json = generator.generate_scratch_json(form_data, existing_agents)
            fixed_json, is_valid, errors = validator.validate_and_fix(scratch_json)
            st.session_state.generated_content["scratch_json"] = fixed_json
            st.session_state.validation_status = {"is_valid": is_valid, "errors": errors}
            return
        else:
            return

        st.session_state.generated_content[content_type] = content


def render():
    """Render the preview page."""
    st.header("Step 5: Preview Generated Content")
    st.markdown(
        "Review the AI-generated content. You can edit any section or "
        "regenerate it if needed."
    )

    # Generate content if not already done
    if not st.session_state.content_generated:
        generate_content()

    generated = st.session_state.generated_content
    validation = st.session_state.validation_status

    # Regenerate All button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Regenerate All", use_container_width=True):
            generate_content()
            st.rerun()

    st.divider()

    # Validation status
    if validation["is_valid"]:
        st.success("Configuration validated successfully!")
    else:
        st.error("Configuration has validation errors:")
        for error in validation["errors"]:
            st.markdown(f"- {error}")

    # Preview tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Persona", "Background", "Work Style", "Relationships", "Configuration (JSON)"
    ])

    with tab1:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.subheader("persona.md")
        with col2:
            if st.button("Regenerate", key="regen_persona"):
                regenerate_single("persona_md")
                st.rerun()

        # Editable text area
        persona_content = st.text_area(
            "Edit persona.md",
            value=generated.get("persona_md", ""),
            height=400,
            key="edit_persona",
            label_visibility="collapsed"
        )
        # Save edits
        if persona_content != generated.get("persona_md", ""):
            generated["persona_md"] = persona_content

    with tab2:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.subheader("background.md")
        with col2:
            if st.button("Regenerate", key="regen_background"):
                regenerate_single("background_md")
                st.rerun()

        background_content = st.text_area(
            "Edit background.md",
            value=generated.get("background_md", ""),
            height=400,
            key="edit_background",
            label_visibility="collapsed"
        )
        if background_content != generated.get("background_md", ""):
            generated["background_md"] = background_content

    with tab3:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.subheader("work_style.md")
        with col2:
            if st.button("Regenerate", key="regen_work_style"):
                regenerate_single("work_style_md")
                st.rerun()

        work_style_content = st.text_area(
            "Edit work_style.md",
            value=generated.get("work_style_md", ""),
            height=400,
            key="edit_work_style",
            label_visibility="collapsed"
        )
        if work_style_content != generated.get("work_style_md", ""):
            generated["work_style_md"] = work_style_content

    with tab4:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.subheader("relationships.md")
        with col2:
            if st.button("Regenerate", key="regen_relationships"):
                regenerate_single("relationships_md")
                st.rerun()

        relationships_content = st.text_area(
            "Edit relationships.md",
            value=generated.get("relationships_md", ""),
            height=400,
            key="edit_relationships",
            label_visibility="collapsed"
        )
        if relationships_content != generated.get("relationships_md", ""):
            generated["relationships_md"] = relationships_content

    with tab5:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.subheader("scratch.json")
        with col2:
            if st.button("Regenerate", key="regen_scratch"):
                regenerate_single("scratch_json")
                st.rerun()

        # Show JSON with syntax highlighting
        scratch_json = generated.get("scratch_json", {})
        scratch_text = json.dumps(scratch_json, indent=2)

        edited_json_text = st.text_area(
            "Edit scratch.json",
            value=scratch_text,
            height=400,
            key="edit_scratch",
            label_visibility="collapsed"
        )

        # Try to parse and validate edited JSON
        if edited_json_text != scratch_text:
            try:
                edited_json = json.loads(edited_json_text)
                validator = SchemaValidator()
                fixed_json, is_valid, errors = validator.validate_and_fix(edited_json)
                generated["scratch_json"] = fixed_json
                st.session_state.validation_status = {"is_valid": is_valid, "errors": errors}
                if is_valid:
                    st.success("JSON is valid!")
                else:
                    st.warning("JSON has validation issues (auto-fixed where possible)")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

    st.divider()

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("Back: Relationships", use_container_width=True):
            st.session_state.current_step = 4
            st.rerun()

    with col3:
        # Only allow proceeding if validation passes
        can_proceed = validation["is_valid"]
        if st.button(
            "Next: Save Agent",
            type="primary",
            use_container_width=True,
            disabled=not can_proceed
        ):
            st.session_state.current_step = 6
            st.rerun()

        if not can_proceed:
            st.caption(":red[Fix validation errors before proceeding]")

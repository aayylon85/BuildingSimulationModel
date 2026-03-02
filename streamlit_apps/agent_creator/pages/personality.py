"""Step 2: Personality traits page."""

import streamlit as st


# Predefined trait options
INNATE_TRAITS = [
    "methodical",
    "organized",
    "flexible",
    "adaptable",
    "reserved",
    "outgoing",
    "detail-focused",
    "big-picture",
    "morning_person",
    "night_owl",
    "environmentally_conscious",
    "routine_oriented",
    "spontaneous",
    "precise",
    "creative",
    "analytical",
    "collaborative",
    "independent",
]

COMMUNICATION_STYLES = [
    "direct",
    "diplomatic",
    "concise",
    "detailed",
    "formal",
    "casual",
    "prefers_written",
    "prefers_verbal",
    "good_listener",
    "talkative",
]

DECISION_STYLES = [
    "plans_ahead",
    "spontaneous",
    "consensus_seeking",
    "independent",
    "data_driven",
    "intuitive",
    "risk_averse",
    "risk_taker",
    "deliberate",
    "quick_decider",
]


def render():
    """Render the personality traits form."""
    st.header("Step 2: Personality Traits")
    st.markdown("Define the agent's core personality characteristics.")

    form_data = st.session_state.form_data

    # Innate traits (multi-select)
    st.subheader("Core Traits")
    st.markdown("Select 3-5 core personality traits that define this agent.")

    innate_traits = st.multiselect(
        "Innate Traits",
        options=INNATE_TRAITS,
        default=form_data.get("innate_traits", []),
        help="Select 3-5 traits",
        max_selections=5
    )

    if len(innate_traits) < 3:
        st.warning("Please select at least 3 traits")

    # Personality description
    st.subheader("Personality Description")
    personality_description = st.text_area(
        "Brief personality description",
        value=form_data.get("personality_description", ""),
        help="A brief free-form description of this agent's personality",
        placeholder="e.g., A thoughtful professional who values efficiency but also enjoys connecting with colleagues...",
        height=100
    )

    col1, col2 = st.columns(2)

    with col1:
        # Communication style
        st.subheader("Communication Style")
        communication_style = st.multiselect(
            "How they communicate",
            options=COMMUNICATION_STYLES,
            default=form_data.get("communication_style", []),
            help="Select 2-4 communication traits",
            max_selections=4
        )

    with col2:
        # Decision making style
        st.subheader("Decision Making")
        decision_making_style = st.multiselect(
            "How they make decisions",
            options=DECISION_STYLES,
            default=form_data.get("decision_making_style", []),
            help="Select 2-4 decision-making traits",
            max_selections=4
        )

    st.divider()

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("Back: Basic Info", use_container_width=True):
            # Save current state before going back
            st.session_state.form_data.update({
                "innate_traits": innate_traits,
                "personality_description": personality_description,
                "communication_style": communication_style,
                "decision_making_style": decision_making_style,
            })
            st.session_state.current_step = 1
            st.rerun()

    with col3:
        if st.button("Next: Work Style", type="primary", use_container_width=True):
            # Validate
            if len(innate_traits) < 3:
                st.error("Please select at least 3 innate traits")
            else:
                # Save to session state
                st.session_state.form_data.update({
                    "innate_traits": innate_traits,
                    "personality_description": personality_description,
                    "communication_style": communication_style,
                    "decision_making_style": decision_making_style,
                })
                st.session_state.current_step = 3
                st.rerun()

"""Step 4: Relationships page."""

import streamlit as st


def render():
    """Render the relationships configuration form."""
    st.header("Step 4: Relationships")
    st.markdown("Define relationships with existing agents in the simulation.")

    form_data = st.session_state.form_data
    existing_agents = st.session_state.existing_agents

    if not existing_agents:
        st.info(
            "No existing agents found. This agent will start without "
            "predefined relationships."
        )
    else:
        st.markdown(
            f"Configure relationships with {len(existing_agents)} existing agent(s). "
            "These values influence how the new agent interacts with others."
        )

        # Initialize relationships if not present
        if "relationships" not in form_data:
            form_data["relationships"] = {}
        if "relationship_notes" not in form_data:
            form_data["relationship_notes"] = {}

        # Show each existing agent
        for agent_id, agent_info in existing_agents.items():
            with st.expander(
                f"{agent_info['name']} ({agent_id})",
                expanded=True
            ):
                col1, col2 = st.columns(2)

                # Get current values or defaults
                current_rel = form_data["relationships"].get(agent_id, {})
                current_familiarity = current_rel.get("familiarity", 0.5)
                current_sentiment = current_rel.get("sentiment", 0.5)
                current_notes = form_data["relationship_notes"].get(agent_id, "")

                with col1:
                    familiarity = st.slider(
                        "Familiarity",
                        min_value=0.0,
                        max_value=1.0,
                        value=current_familiarity,
                        step=0.1,
                        key=f"fam_{agent_id}",
                        help="How well they know each other (0=strangers, 1=close)"
                    )

                    # Show familiarity interpretation
                    if familiarity < 0.3:
                        st.caption(":grey[Barely acquainted]")
                    elif familiarity < 0.5:
                        st.caption(":grey[Know each other]")
                    elif familiarity < 0.7:
                        st.caption(":grey[Work together regularly]")
                    else:
                        st.caption(":grey[Close colleagues]")

                with col2:
                    sentiment = st.slider(
                        "Sentiment",
                        min_value=0.0,
                        max_value=1.0,
                        value=current_sentiment,
                        step=0.1,
                        key=f"sent_{agent_id}",
                        help="How positively they feel about them (0=negative, 1=positive)"
                    )

                    # Show sentiment interpretation
                    if sentiment < 0.3:
                        st.caption(":grey[Tension/dislike]")
                    elif sentiment < 0.5:
                        st.caption(":grey[Neutral/indifferent]")
                    elif sentiment < 0.7:
                        st.caption(":grey[Friendly]")
                    else:
                        st.caption(":grey[Very positive]")

                # Relationship notes for LLM context
                notes = st.text_area(
                    "Relationship Notes (optional)",
                    value=current_notes,
                    key=f"notes_{agent_id}",
                    help="Additional context about their relationship (helps generate better content)",
                    placeholder=f"e.g., They sometimes disagree about temperature preferences since {agent_info['name']} prefers it {agent_info.get('thermal_preference', 'neutral')}...",
                    height=80
                )

                # Show agent info
                st.caption(
                    f":grey[{agent_info['name']} - Thermal: {agent_info.get('thermal_preference', 'neutral')}, "
                    f"Arrives: {agent_info.get('typical_arrival', 'N/A')}]"
                )

                # Update form_data
                form_data["relationships"][agent_id] = {
                    "familiarity": familiarity,
                    "sentiment": sentiment
                }
                form_data["relationship_notes"][agent_id] = notes

    st.divider()

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("Back: Work Style", use_container_width=True):
            st.session_state.current_step = 3
            st.rerun()

    with col3:
        if st.button("Next: Generate & Preview", type="primary", use_container_width=True):
            # Reset generation flag so preview regenerates
            st.session_state.content_generated = False
            st.session_state.current_step = 5
            st.rerun()

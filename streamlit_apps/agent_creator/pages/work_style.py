"""Step 3: Work Style and Comfort page."""

import streamlit as st


def render():
    """Render the work style and comfort form."""
    st.header("Step 3: Work Style & Comfort")
    st.markdown("Define schedule preferences and comfort settings.")

    form_data = st.session_state.form_data

    # Schedule section
    st.subheader("Schedule")
    col1, col2 = st.columns(2)

    with col1:
        # Arrival time
        arrival_options = [
            "07:30-08:00",
            "08:00-08:30",
            "08:30-09:00",
            "09:00-09:30",
            "09:30-10:00",
            "10:00-10:30",
        ]
        typical_arrival = st.selectbox(
            "Typical Arrival Time",
            options=arrival_options,
            index=arrival_options.index(form_data.get("typical_arrival", "08:30-09:00"))
            if form_data.get("typical_arrival") in arrival_options else 2,
            help="Typical arrival time range"
        )

        # Preferred desk
        desk_options = ["Desk_A", "Desk_B", "Desk_C", "Desk_D", "no_preference"]
        preferred_desk = st.selectbox(
            "Preferred Desk",
            options=desk_options,
            index=desk_options.index(form_data.get("preferred_desk", "Desk_A"))
            if form_data.get("preferred_desk") in desk_options else 0,
            help="Preferred desk location"
        )

        # Workspace variety
        workspace_variety = st.select_slider(
            "Workspace Variety Preference",
            options=["low", "medium", "high"],
            value=form_data.get("workspace_variety", "medium"),
            help="How much they like variety in workspace"
        )

    with col2:
        # Departure time
        departure_options = [
            "16:00-16:30",
            "16:30-17:00",
            "17:00-17:30",
            "17:30-18:00",
            "18:00-18:30",
            "18:30-19:00",
        ]
        typical_departure = st.selectbox(
            "Typical Departure Time",
            options=departure_options,
            index=departure_options.index(form_data.get("typical_departure", "17:00-17:30"))
            if form_data.get("typical_departure") in departure_options else 2,
            help="Typical departure time range"
        )

        # Works weekends
        works_weekends = st.checkbox(
            "Works Weekends",
            value=form_data.get("works_weekends", False),
            help="Whether they typically work on weekends"
        )

        # Likes desk variety
        likes_desk_variety = st.checkbox(
            "Likes Desk Variety",
            value=form_data.get("likes_desk_variety", True),
            help="Whether they enjoy varying their desk location"
        )

    st.divider()

    # Comfort preferences section
    st.subheader("Comfort Preferences")
    col1, col2 = st.columns(2)

    with col1:
        # Thermal preference
        thermal_options = ["cool", "neutral", "slightly_warm", "warm"]
        thermal_preference = st.selectbox(
            "Thermal Preference",
            options=thermal_options,
            index=thermal_options.index(form_data.get("thermal_preference", "neutral")),
            help="Preferred temperature level"
        )

        # Runs warm
        runs_warm = st.checkbox(
            "Runs Warm",
            value=form_data.get("runs_warm", False),
            help="Whether they tend to feel warm"
        )

        # Clothing insulation
        clothing_options = ["very_light", "light", "medium", "warm", "very_warm"]
        clothing_insulation = st.selectbox(
            "Typical Clothing Insulation",
            options=clothing_options,
            index=clothing_options.index(form_data.get("clothing_insulation_typical", "medium")),
            help="Typical clothing warmth level"
        )

    with col2:
        # Light preference
        light_options = ["natural", "task", "ambient", "warm"]
        light_preference = st.selectbox(
            "Light Preference",
            options=light_options,
            index=light_options.index(form_data.get("light_preference", "natural")),
            help="Preferred lighting type"
        )

        # Noise sensitivity
        noise_options = ["low", "moderate", "high"]
        noise_sensitivity = st.selectbox(
            "Noise Sensitivity",
            options=noise_options,
            index=noise_options.index(form_data.get("noise_sensitivity", "moderate")),
            help="Sensitivity to office noise"
        )

    st.divider()

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("Back: Personality", use_container_width=True):
            # Save current state
            st.session_state.form_data.update({
                "typical_arrival": typical_arrival,
                "typical_departure": typical_departure,
                "preferred_desk": preferred_desk,
                "works_weekends": works_weekends,
                "workspace_variety": workspace_variety,
                "likes_desk_variety": likes_desk_variety,
                "thermal_preference": thermal_preference,
                "runs_warm": runs_warm,
                "clothing_insulation_typical": clothing_insulation,
                "light_preference": light_preference,
                "noise_sensitivity": noise_sensitivity,
            })
            st.session_state.current_step = 2
            st.rerun()

    with col3:
        if st.button("Next: Relationships", type="primary", use_container_width=True):
            # Save to session state
            st.session_state.form_data.update({
                "typical_arrival": typical_arrival,
                "typical_departure": typical_departure,
                "preferred_desk": preferred_desk,
                "works_weekends": works_weekends,
                "workspace_variety": workspace_variety,
                "likes_desk_variety": likes_desk_variety,
                "thermal_preference": thermal_preference,
                "runs_warm": runs_warm,
                "clothing_insulation_typical": clothing_insulation,
                "light_preference": light_preference,
                "noise_sensitivity": noise_sensitivity,
            })
            st.session_state.current_step = 4
            st.rerun()

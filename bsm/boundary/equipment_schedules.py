"""
Equipment and lighting schedule generator for building simulations.

Generates time-series power profiles for equipment and lighting based on:
- Baseline mode: Fixed schedules tied to occupied hours
- Simple agents mode: Occupant-presence-based with seeded randomness
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional


def create_equipment_schedules(
    config: Dict[str, Any],
    num_steps: int,
    time_hours: np.ndarray,
    mode: str = "baseline",
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate equipment and lighting power profiles for non-LLM simulation modes.

    Args:
        config: Full simulation configuration dictionary
        num_steps: Number of simulation steps
        time_hours: Time array in hours from simulation start
        mode: "baseline" (fixed schedule) or "simple" (occupant-based with randomness)
        random_seed: Seed for reproducible random equipment usage in simple mode

    Returns:
        Tuple of (equipment_power_profile, lighting_power_profile) as numpy arrays
    """
    equipment_power = np.zeros(num_steps)
    lighting_power = np.zeros(num_steps)

    # Get configuration sections
    schedules = config.get('schedules', {})
    equipment_config = config.get('equipment', {})
    lighting_config = config.get('lighting', {})
    desks_config = config.get('desks', {})
    occupancy_config = config.get('occupancy', {})
    schedule_config = config.get('equipment_schedules', {})

    # Get power ratings with defaults
    power_ratings = equipment_config.get('default_power_w', {})
    laptop_power = power_ratings.get('laptop', 50)
    monitor_power = power_ratings.get('monitor', 30)
    photocopier_standby = power_ratings.get('photocopier_standby', 20)
    photocopier_active = power_ratings.get('photocopier_active', 400)
    kettle_power = power_ratings.get('kettle', 2000)
    coffee_machine_power = power_ratings.get('coffee_machine', 1000)
    microwave_power = power_ratings.get('microwave', 1200)
    fridge_power = power_ratings.get('fridge', 150)
    projector_power = power_ratings.get('projector', 200)
    conference_phone_power = power_ratings.get('conference_phone', 10)

    # Get lighting power ratings
    desk_light_power = lighting_config.get('desk_light_power_w', 15)
    zone_light_power = lighting_config.get('zone_light_power_w', 200)
    meeting_room_light_power = lighting_config.get('meeting_room_light_power_w', 100)

    # Get occupied hours
    occ_start, occ_end = schedules.get('occupied_hours', [9, 17])

    # Get timestep in minutes (for duration calculations)
    dt_minutes = config.get('simulation_settings', {}).get('dt_minutes', 1)

    # Count desks for baseline calculations
    num_desks = len(desks_config)
    if num_desks == 0:
        num_desks = 3  # Default assumption

    if mode == "baseline":
        equipment_power, lighting_power = _generate_baseline_schedules(
            num_steps=num_steps,
            time_hours=time_hours,
            occ_start=occ_start,
            occ_end=occ_end,
            dt_minutes=dt_minutes,
            num_desks=num_desks,
            laptop_power=laptop_power,
            monitor_power=monitor_power,
            desk_light_power=desk_light_power,
            zone_light_power=zone_light_power,
            meeting_room_light_power=meeting_room_light_power,
            photocopier_standby=photocopier_standby,
            photocopier_active=photocopier_active,
            kettle_power=kettle_power,
            coffee_machine_power=coffee_machine_power,
            microwave_power=microwave_power,
            fridge_power=fridge_power,
            projector_power=projector_power,
            conference_phone_power=conference_phone_power,
            schedule_config=schedule_config,
        )
    elif mode == "simple":
        occupants = occupancy_config.get('occupants', [])
        equipment_power, lighting_power = _generate_simple_schedules(
            num_steps=num_steps,
            time_hours=time_hours,
            occ_start=occ_start,
            occ_end=occ_end,
            dt_minutes=dt_minutes,
            occupants=occupants,
            desks_config=desks_config,
            laptop_power=laptop_power,
            monitor_power=monitor_power,
            desk_light_power=desk_light_power,
            zone_light_power=zone_light_power,
            meeting_room_light_power=meeting_room_light_power,
            photocopier_standby=photocopier_standby,
            photocopier_active=photocopier_active,
            kettle_power=kettle_power,
            coffee_machine_power=coffee_machine_power,
            microwave_power=microwave_power,
            fridge_power=fridge_power,
            projector_power=projector_power,
            conference_phone_power=conference_phone_power,
            schedule_config=schedule_config,
            random_seed=random_seed,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'baseline' or 'simple'.")

    return equipment_power, lighting_power


def _generate_baseline_schedules(
    num_steps: int,
    time_hours: np.ndarray,
    occ_start: float,
    occ_end: float,
    dt_minutes: float,
    num_desks: int,
    laptop_power: float,
    monitor_power: float,
    desk_light_power: float,
    zone_light_power: float,
    meeting_room_light_power: float,
    photocopier_standby: float,
    photocopier_active: float,
    kettle_power: float,
    coffee_machine_power: float,
    microwave_power: float,
    fridge_power: float,
    projector_power: float,
    conference_phone_power: float,
    schedule_config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate fixed equipment schedules for baseline mode.

    Equipment usage patterns:
    - Desk equipment (laptops, monitors): ON during occupied hours
    - Photocopier: Used 2x per day at fixed times (10:00, 14:00), 5 min each
    - Meeting room: Used for 2 meetings (10:00-11:00, 14:00-15:00)
    - Kettle: Used 3x per day at break times (09:00, 11:00, 15:00), 3 min each
    - Coffee machine: Intermittent during occupied hours
    - Microwave: Used during lunch hour (12:00-13:00)
    - Fridge: Always on (24/7)
    """
    equipment_power = np.zeros(num_steps)
    lighting_power = np.zeros(num_steps)

    # Get schedule configuration with defaults
    photocopier_times = schedule_config.get('photocopier_usage_times', [10.0, 14.0])
    photocopier_duration_min = schedule_config.get('photocopier_duration_minutes', 5)
    kettle_times = schedule_config.get('kettle_usage_times', [9.0, 11.0, 15.0])
    kettle_duration_min = schedule_config.get('kettle_duration_minutes', 3)
    coffee_machine_hours = schedule_config.get('coffee_machine_hours', [9, 17])
    microwave_lunch_hour = schedule_config.get('microwave_lunch_hour', 12)
    meeting_hours = schedule_config.get('meeting_hours', [[10, 11], [14, 15]])

    for i, t_hr in enumerate(time_hours):
        hour_of_day = t_hr % 24

        # Fridge is always on
        equipment_power[i] += fridge_power

        # Check if within occupied hours
        if occ_start <= hour_of_day < occ_end:
            # --- Desk Equipment (laptops + monitors) ---
            # All desks: num_desks × (laptop + monitor)
            equipment_power[i] += num_desks * (laptop_power + monitor_power)

            # --- Lighting ---
            # All desk lights + zone light during occupied hours
            lighting_power[i] += num_desks * desk_light_power + zone_light_power

            # --- Photocopier ---
            # Standby during occupied hours
            equipment_power[i] += photocopier_standby

            # Check if actively using photocopier
            for use_time in photocopier_times:
                if use_time <= hour_of_day < use_time + (photocopier_duration_min / 60.0):
                    # Add active power minus standby (since standby already added)
                    equipment_power[i] += (photocopier_active - photocopier_standby)
                    break

            # --- Kettle ---
            for use_time in kettle_times:
                if use_time <= hour_of_day < use_time + (kettle_duration_min / 60.0):
                    equipment_power[i] += kettle_power
                    break

            # --- Coffee Machine ---
            # Intermittent usage: ~20% duty cycle during coffee hours
            coffee_start, coffee_end = coffee_machine_hours
            if coffee_start <= hour_of_day < coffee_end:
                # Use 20% of power on average (simulating intermittent brewing)
                equipment_power[i] += coffee_machine_power * 0.2

            # --- Microwave ---
            # Active during lunch hour with ~30% duty cycle
            if microwave_lunch_hour <= hour_of_day < microwave_lunch_hour + 1:
                equipment_power[i] += microwave_power * 0.3

            # --- Meeting Room ---
            for meeting_start, meeting_end in meeting_hours:
                if meeting_start <= hour_of_day < meeting_end:
                    # Projector + conference phone + meeting room light
                    equipment_power[i] += projector_power + conference_phone_power
                    lighting_power[i] += meeting_room_light_power
                    break

    return equipment_power, lighting_power


def _generate_simple_schedules(
    num_steps: int,
    time_hours: np.ndarray,
    occ_start: float,
    occ_end: float,
    dt_minutes: float,
    occupants: List[Dict[str, Any]],
    desks_config: Dict[str, Any],
    laptop_power: float,
    monitor_power: float,
    desk_light_power: float,
    zone_light_power: float,
    meeting_room_light_power: float,
    photocopier_standby: float,
    photocopier_active: float,
    kettle_power: float,
    coffee_machine_power: float,
    microwave_power: float,
    fridge_power: float,
    projector_power: float,
    conference_phone_power: float,
    schedule_config: Dict[str, Any],
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate occupant-presence-based equipment schedules for simple agents mode.

    Equipment usage patterns:
    - Desk equipment: ON only when occupant is present (based on work hours)
    - Shared equipment: Used with seeded random distribution
    - Zone lighting: ON if any occupant present
    - Fridge: Always on (24/7)
    """
    equipment_power = np.zeros(num_steps)
    lighting_power = np.zeros(num_steps)

    # Initialize random generator with seed for reproducibility
    rng = np.random.default_rng(random_seed)

    # Pre-compute occupant presence for each timestep
    num_occupants = len(occupants) if occupants else 3

    # Track desk-to-occupant mapping
    desk_names = list(desks_config.keys()) if desks_config else ['Desk_A', 'Desk_B', 'Desk_C']

    # Create occupant presence array
    occupant_presence = np.zeros((num_occupants, num_steps), dtype=bool)

    for occ_idx, occ_data in enumerate(occupants):
        work_start = occ_data.get('work_start_hr', 9)
        work_end = occ_data.get('work_end_hr', 17)

        for i, t_hr in enumerate(time_hours):
            hour_of_day = t_hr % 24
            occupant_presence[occ_idx, i] = work_start <= hour_of_day < work_end

    # Generate random usage events for shared equipment
    # Get schedule config or use defaults
    photocopier_times_per_day = schedule_config.get('photocopier_uses_per_day', 3)
    kettle_times_per_day = schedule_config.get('kettle_uses_per_day', 4)
    coffee_uses_per_day = schedule_config.get('coffee_uses_per_day', 6)
    microwave_uses_per_day = schedule_config.get('microwave_uses_per_day', 3)
    meetings_per_day = schedule_config.get('meetings_per_day', 2)

    # Duration in minutes
    photocopier_duration_min = schedule_config.get('photocopier_duration_minutes', 5)
    kettle_duration_min = schedule_config.get('kettle_duration_minutes', 3)
    coffee_duration_min = schedule_config.get('coffee_duration_minutes', 5)
    microwave_duration_min = schedule_config.get('microwave_duration_minutes', 4)
    meeting_duration_min = schedule_config.get('meeting_duration_minutes', 60)

    # Calculate number of days in simulation
    total_hours = time_hours[-1] - time_hours[0] if len(time_hours) > 1 else 24
    num_days = int(np.ceil(total_hours / 24))

    # Pre-generate random usage times for each day
    photocopier_events = _generate_random_events(
        rng, num_days, photocopier_times_per_day, occ_start, occ_end
    )
    kettle_events = _generate_random_events(
        rng, num_days, kettle_times_per_day, occ_start, occ_end
    )
    coffee_events = _generate_random_events(
        rng, num_days, coffee_uses_per_day, occ_start, occ_end
    )
    microwave_events = _generate_random_events(
        rng, num_days, microwave_uses_per_day, occ_start + 3, occ_start + 5  # Lunch window
    )
    meeting_events = _generate_random_events(
        rng, num_days, meetings_per_day, occ_start + 1, occ_end - 1
    )

    for i, t_hr in enumerate(time_hours):
        hour_of_day = t_hr % 24
        day_index = int(t_hr // 24)

        # Fridge is always on
        equipment_power[i] += fridge_power

        # --- Desk Equipment based on occupant presence ---
        any_occupant_present = False
        for occ_idx in range(num_occupants):
            if occupant_presence[occ_idx, i]:
                any_occupant_present = True
                # This occupant's desk equipment is on
                equipment_power[i] += laptop_power + monitor_power
                lighting_power[i] += desk_light_power

        # Zone light on if anyone present
        if any_occupant_present:
            lighting_power[i] += zone_light_power

            # Photocopier standby when office is occupied
            equipment_power[i] += photocopier_standby

        # --- Random equipment events ---
        if day_index < num_days:
            # Photocopier active usage
            if _is_during_event(hour_of_day, photocopier_events[day_index], photocopier_duration_min):
                equipment_power[i] += (photocopier_active - photocopier_standby)

            # Kettle usage
            if _is_during_event(hour_of_day, kettle_events[day_index], kettle_duration_min):
                equipment_power[i] += kettle_power

            # Coffee machine usage
            if _is_during_event(hour_of_day, coffee_events[day_index], coffee_duration_min):
                equipment_power[i] += coffee_machine_power

            # Microwave usage
            if _is_during_event(hour_of_day, microwave_events[day_index], microwave_duration_min):
                equipment_power[i] += microwave_power

            # Meeting room usage
            if _is_during_event(hour_of_day, meeting_events[day_index], meeting_duration_min):
                equipment_power[i] += projector_power + conference_phone_power
                lighting_power[i] += meeting_room_light_power

    return equipment_power, lighting_power


def _generate_random_events(
    rng: np.random.Generator,
    num_days: int,
    events_per_day: int,
    window_start: float,
    window_end: float,
) -> List[List[float]]:
    """
    Generate random event start times for each day.

    Args:
        rng: NumPy random generator
        num_days: Number of days to generate
        events_per_day: Number of events per day
        window_start: Earliest hour for events
        window_end: Latest hour for events

    Returns:
        List of lists, each containing event start times (hours) for that day
    """
    events = []
    for _ in range(num_days):
        day_events = sorted(
            rng.uniform(window_start, window_end, size=events_per_day).tolist()
        )
        events.append(day_events)
    return events


def _is_during_event(
    hour_of_day: float,
    event_times: List[float],
    duration_minutes: float,
) -> bool:
    """
    Check if current hour falls within any event window.

    Args:
        hour_of_day: Current hour (0-24)
        event_times: List of event start times (hours)
        duration_minutes: Duration of each event in minutes

    Returns:
        True if currently during an event
    """
    duration_hours = duration_minutes / 60.0
    for event_start in event_times:
        if event_start <= hour_of_day < event_start + duration_hours:
            return True
    return False


def get_occupant_count_at_time(
    occupants: List[Dict[str, Any]],
    hour_of_day: float,
) -> int:
    """
    Count how many occupants are present at a given time.

    Args:
        occupants: List of occupant dictionaries with work_start_hr and work_end_hr
        hour_of_day: Hour of day (0-24)

    Returns:
        Number of occupants present
    """
    count = 0
    for occ in occupants:
        work_start = occ.get('work_start_hr', 9)
        work_end = occ.get('work_end_hr', 17)
        if work_start <= hour_of_day < work_end:
            count += 1
    return count

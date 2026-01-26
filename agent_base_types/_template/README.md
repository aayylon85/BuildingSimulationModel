# Agent Template

This folder contains templates for creating new agent types. Each agent requires:

## Required Files

### 1. persona.md
Core personality traits, demographics, appearance, and comfort preferences.

### 2. background.md
Education, work history, personal life details, and food/break preferences.

### 3. work_style.md
Schedule, desk preferences, daily rhythms, weather sensitivity, and comfort behaviors.

### 4. relationships.md
Relationships with other agents in the simulation.

### 5. scratch.json
Runtime state and configuration values for the agent.

## Creating a New Agent

1. Copy this `_template` folder to a new folder (e.g., `dave_office_worker`)
2. Fill in all placeholder sections in each .md file
3. Update `scratch.json` with unique agent_id and appropriate values
4. Remove placeholder comments and guidance text

## Key Principles

### Demographics
- Give agents realistic backgrounds - age, origin, time in location
- Consider commute method and distance
- Include personal details that affect behavior

### Comfort Preferences (Qualitative, not Quantitative)
- Avoid stating exact temperatures like "prefers 21C"
- Instead use qualitative descriptions: "prefers to feel comfortably warm"
- Describe how they respond to different thermal conditions
- Include clothing tendencies and adjustment behaviors

### Appearance & Style
- Describe typical dress style (formal, casual, smart casual)
- How they dress in different weather conditions
- General appearance that affects clothing choices

### Food & Breaks
- Lunch habits (brings food, goes out, timing)
- Coffee/tea preferences
- Snack habits
- Break patterns

### Weather Sensitivity
- How weather affects their commute choices
- How they dress for different weather
- How weather affects lunch/break decisions

## scratch.json Fields

| Field | Description |
|-------|-------------|
| `agent_id` | Unique identifier (e.g., "alice_001") |
| `name` | Full display name |
| `first_name` | First name for casual reference |
| `innate_traits` | Comma-separated personality traits |
| `learned_traits` | Experience and background summary |
| `currently` | Current status description |
| `lifestyle` | Schedule and commute summary |
| `thermal_preference` | Qualitative: "cool", "neutral", "slightly_warm", "warm" |
| `runs_warm` | Boolean - does this person run warm physically? |
| `clothing_insulation_typical` | Typical dress warmth: "light", "medium", "heavy" |
| `light_preference` | "natural", "task", "warm" |
| `noise_sensitivity` | "low", "moderate", "high" |
| `typical_arrival` | Time range (e.g., "08:30-09:00") |
| `typical_departure` | Time range (e.g., "17:00-17:30") |
| `preferred_desk` | Desk identifier |
| `works_weekends` | Boolean |
| `perception_bandwidth` | How many events agent perceives per step (6-10) |
| `reflection_threshold` | Importance score to trigger reflection (100-200) |
| `recency_w`, `relevance_w`, `importance_w` | Memory retrieval weights |
| `recency_decay` | How quickly memories fade (0.99-0.999) |
| `relationship_models` | Initial relationships with other agents |

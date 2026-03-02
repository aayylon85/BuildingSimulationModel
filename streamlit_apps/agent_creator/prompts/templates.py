"""LLM prompt templates for generating agent content."""

PERSONA_PROMPT = """You are creating a detailed persona document for a simulated office worker in a building simulation.

## Agent Information
- Name: {name}
- First Name: {first_name}
- Age: {age}
- Originally from: {origin}
- Years in Southampton: {years_in_southampton}
- Languages: {languages}

## Personality
- Core traits: {innate_traits}
- Personality description: {personality_description}
- Communication style: {communication_style}
- Decision making: {decision_making_style}

## Comfort Preferences
- Thermal preference: {thermal_preference}
- Runs warm: {runs_warm}
- Clothing insulation: {clothing_insulation_typical}
- Light preference: {light_preference}
- Noise sensitivity: {noise_sensitivity}

Generate a persona.md file following this EXACT structure (use the headers exactly as shown):

# {name} - Office Worker Persona

## Core Traits
[List 4-5 bullet points describing core personality traits based on the provided traits]

## Demographics
- Age: [age]
- Originally from: [origin]
- Living in Southampton: [years] years
- Languages: [languages]

## Appearance & Style
- Usually dresses in: [describe typical work attire based on personality]
- In cold weather: [describe cold weather adaptations based on thermal preference]
- In warm weather: [describe warm weather adaptations]
- General style: [describe overall style based on personality]

## General Comfort Preferences
- Thermal preference: [qualitative description based on thermal_preference]
- Drafts and air movement: [describe based on preferences]
- Air quality: [describe preferences]
- Lighting: [describe based on light_preference]
- Noise: [describe based on noise_sensitivity]

## Communication Style
[4 bullet points based on communication_style provided]

## Decision Making
[4 bullet points based on decision_making_style provided]

IMPORTANT:
- Write in third person ("They prefer..." not "You prefer...")
- Be specific and vivid - create a believable person
- Thermal preferences should be qualitative descriptions, not exact temperatures
- Keep it consistent with the provided traits
- Total length: 400-600 words"""


BACKGROUND_PROMPT = """You are creating a background document for a simulated office worker in a building simulation.

## Agent Information
- Name: {name}
- First Name: {first_name}
- Age: {age}
- Originally from: {origin}
- Years in Southampton: {years_in_southampton}
- Languages: {languages}

## Personality
- Core traits: {innate_traits}
- Personality description: {personality_description}

## Work Schedule
- Arrival: {typical_arrival}
- Departure: {typical_departure}

Generate a background.md file following this EXACT structure:

# {first_name}'s Background

## Education
[4-5 bullet points about education - be creative but realistic for an office worker]

## Work History
[4-5 bullet points about career - should be consistent with age and personality]

## Personal Life Details
[6-7 bullet points about:
- Living situation (flat/house, area of Southampton)
- Commute details (train/bus/cycle, timing)
- Morning routine
- Evening routine
- Weekend activities]

## Food & Break Preferences
[6-7 bullet points about:
- Lunch habits (brings from home vs buys, what they eat)
- Going out for lunch frequency
- Coffee/tea preferences (be specific about how they take it)
- Snack habits
- Break patterns]

## Hobbies & Interests
[4-5 bullet points - should reflect personality traits]

## Values
[4-5 bullet points about what matters to them - derived from personality]

IMPORTANT:
- Write in third person
- Be specific and realistic - create a believable backstory
- Ensure hobbies and values align with personality traits
- Total length: 400-600 words"""


WORK_STYLE_PROMPT = """You are creating a work style document for a simulated office worker in a building simulation.

## Agent Information
- Name: {name}
- First Name: {first_name}
- Core traits: {innate_traits}
- Communication style: {communication_style}
- Decision making: {decision_making_style}

## Schedule
- Arrival: {typical_arrival}
- Departure: {typical_departure}
- Preferred desk: {preferred_desk}
- Works weekends: {works_weekends}
- Workspace variety preference: {workspace_variety}
- Likes desk variety: {likes_desk_variety}

## Comfort
- Thermal preference: {thermal_preference}
- Runs warm: {runs_warm}
- Clothing insulation: {clothing_insulation_typical}
- Light preference: {light_preference}
- Noise sensitivity: {noise_sensitivity}

Generate a work_style.md file following this EXACT structure:

# {first_name}'s Work Style

## Schedule Preferences
- Arrival: [describe arrival pattern and reason]
- Departure: [describe departure pattern and reason]
- Lunch timing: [describe lunch timing]
- Schedule flexibility: [describe based on personality]

## Desk Preferences
- Desk choice reason: [why they chose their desk - relate to light_preference]
- Organization: [describe desk organization based on personality]
- Personalization: [describe personal items]
- Lighting setup: [describe based on light_preference]

## Daily Rhythms
- Most focused: [describe peak productivity time]
- Post-lunch energy: [describe afternoon pattern]
- Best for meetings: [suggest optimal meeting times]
- Second wind: [describe late afternoon if applicable]

## Break Habits

### Morning Break
- Preferred time: [time]
- Location preference: break_room
- Beverage: [coffee or tea]
- [Beverage] preferences: [specific details]
- Social style: [describe based on personality]

### Afternoon Break
- Preferred time: [time]
- Location preference: break_room
- Beverage: [coffee or tea]
- Activity: [describe]

### Lunch Habits
- Typical lunch time: [time range]
- Preferred location: at_desk or break_room
- Food source: [brings from home or buys]
- Social style: [describe]
- Weather influence: [how weather affects lunch choices]

## Equipment Habits
[3-4 bullet points about computer/device usage]

## Weather Sensitivity
[4-5 bullet points about how weather affects their day]

## Comfort Adjustment Behaviors
- If feeling cold:
  - First: [personal adjustment]
  - Then: [secondary adjustment]
  - Finally: [last resort]
- If feeling warm:
  - First: [personal adjustment]
  - Then: [secondary adjustment]
  - Finally: [last resort]
- General approach: [describe overall comfort philosophy based on thermal_preference]

## Work Patterns
[4-5 bullet points about how they organize their work day]

IMPORTANT:
- Write in third person
- Be specific and realistic
- Comfort adjustments should align with thermal_preference and runs_warm
- Total length: 600-800 words"""


RELATIONSHIPS_PROMPT = """You are creating a relationships document for a simulated office worker in a building simulation.

## Agent Information
- Name: {name}
- First Name: {first_name}
- Core traits: {innate_traits}
- Communication style: {communication_style}

## Thermal/Comfort Context
- Thermal preference: {thermal_preference}
- Runs warm: {runs_warm}
(This affects comfort dynamics with other agents)

## Relationships with Other Agents
{relationships_context}

Generate a relationships.md file following this EXACT structure:

# {first_name}'s Relationships

{relationship_sections}

## General Office Dynamics
- Social role: [describe their role in office social dynamics]
- Conflict handling: [describe based on personality - mediator/avoider/direct]
- Reputation: [what they're known for in the office]
- Typical interactions: [describe daily interaction patterns]
- Office contributions: [any contributions to office culture]

IMPORTANT:
- Write in third person
- For each relationship, include:
  - Relationship type
  - Interaction frequency
  - Comfort dynamics (especially temperature-related if relevant)
  - Working together
  - Social interaction
  - What they think of the person
- Make relationships feel realistic and nuanced
- Total length: 300-500 words"""


SCRATCH_JSON_PROMPT = """You are generating the configuration JSON for a simulated office worker agent.

## Agent Information
- agent_id: {agent_id}
- name: {name}
- first_name: {first_name}

## Personality
- innate_traits: {innate_traits}
- Personality description: {personality_description}
- Communication style: {communication_style}
- Decision making: {decision_making_style}

## Schedule
- typical_arrival: {typical_arrival}
- typical_departure: {typical_departure}
- preferred_desk: {preferred_desk}
- works_weekends: {works_weekends}
- workspace_variety: {workspace_variety}
- likes_desk_variety: {likes_desk_variety}

## Comfort
- thermal_preference: {thermal_preference}
- runs_warm: {runs_warm}
- clothing_insulation_typical: {clothing_insulation_typical}
- light_preference: {light_preference}
- noise_sensitivity: {noise_sensitivity}

## Relationships
{relationships_json}

Generate a valid JSON object with EXACTLY this schema. You MUST use the exact values provided above for all fields.

{{
  "agent_id": "{agent_id}",
  "name": "{name}",
  "first_name": "{first_name}",

  "innate_traits": "{innate_traits}",
  "learned_traits": "[Generate 1-2 sentences about work experience based on personality - keep it brief]",
  "currently": "[Generate: What the agent is currently doing - e.g., 'working on regular duties']",
  "lifestyle": "[Generate: Brief summary of schedule and commute based on arrival/departure times]",

  "thermal_preference": "{thermal_preference}",
  "runs_warm": {runs_warm_json},
  "clothing_insulation_typical": "{clothing_insulation_typical}",
  "light_preference": "{light_preference}",
  "noise_sensitivity": "{noise_sensitivity}",

  "typical_arrival": "{typical_arrival}",
  "typical_departure": "{typical_departure}",
  "preferred_desk": "{preferred_desk}",
  "works_weekends": {works_weekends_json},
  "workspace_variety": "{workspace_variety}",
  "likes_desk_variety": {likes_desk_variety_json},

  "perception_bandwidth": 8,
  "reflection_threshold": 150,
  "importance_trigger_max": 150,
  "importance_trigger_curr": 150,
  "recency_w": 1.0,
  "relevance_w": 1.0,
  "importance_w": 1.0,
  "recency_decay": 0.995,

  "relationship_models": {relationships_models_json},

  "daily_plan": null,
  "f_daily_schedule": [],
  "just_arrived": false
}}

CRITICAL:
- Output ONLY valid JSON, no markdown formatting, no explanations
- Use EXACTLY the values provided for thermal_preference, runs_warm, etc.
- Generate ONLY the learned_traits, currently, and lifestyle fields
- All other fields must use the exact values given above"""

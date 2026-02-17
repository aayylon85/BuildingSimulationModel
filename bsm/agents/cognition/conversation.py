"""
Conversation system for generative agents.

Implements multi-turn dialogue between agents based on Stanford's generative_agents pattern.

Key features:
- Turn-based conversation with natural ending detection
- Relationship-aware context
- Commitment extraction from conversations
- Conflict checking for social commitments

Note: For consultation about shared-space decisions (thermostat, windows),
use the consultation module instead.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING

from agents import Agent, Runner, ModelSettings
from agents.agent_output import AgentOutputSchema

logger = logging.getLogger(__name__)

from bsm.agents.skeleton import DEFAULT_AGENT_MODEL, SimContext, CalendarStore, SocialCommitment
from bsm.agents.cognition.schemas import (
    ConversationUtterance,
    ConversationResult,
    UtteranceOutput,
    ConversationCommitmentOutput,
    CommitmentsExtractionOutput,
    DeclineDecisionOutput,
)
from bsm.agents.cognition.social import generate_relationship_summary
from bsm.agents.cognition.retrieval import retrieve


# ---------------------------------------------------------------------------
# Decline Check
# ---------------------------------------------------------------------------

def _build_decline_check_agent(target_id: str, initiator_id: str) -> Agent:
    """Build an agent to check if the target wants to engage in conversation."""
    instructions = f"""You are {target_id}, and {initiator_id} wants to start a conversation with you.

Consider:
- Are you currently busy with urgent work?
- Are you in the middle of something important?
- Do you have a meeting starting very soon?
- Are you in a social mood right now?

Most of the time, you should ACCEPT conversations with colleagues - it's part of normal office social dynamics.
Only DECLINE if you have a genuine reason (rushing to a meeting, in the middle of something urgent, etc.).

Decide whether to accept or decline, and briefly explain why."""

    return Agent(
        name=f"decline_check_{target_id}",
        instructions=instructions,
        model=DEFAULT_AGENT_MODEL,
        model_settings=ModelSettings(reasoning_effort="low"),
        output_type=AgentOutputSchema(DeclineDecisionOutput),
    )


async def _check_target_willingness(
    target_agent: "GenerativeAgent",
    initiator_id: str,
    topic: Optional[str],
    now: datetime,
) -> DeclineDecisionOutput:
    """
    Check if the target agent wants to participate in a conversation.

    Args:
        target_agent: Agent being asked to converse
        initiator_id: ID of the agent initiating
        topic: Optional topic of conversation
        now: Current datetime

    Returns:
        DeclineDecisionOutput with accept/decline decision and reason
    """
    # Get target's current context
    agent_status = target_agent.scratch.get("current_status", {})
    in_meeting = agent_status.get("in_meeting", False)
    on_break = agent_status.get("on_break", False)

    # Build context prompt
    context_parts = [f"{initiator_id} wants to talk with you."]
    if topic:
        context_parts.append(f"Topic: {topic}")
    if in_meeting:
        context_parts.append("You are currently in a meeting.")
    if on_break:
        context_parts.append("You are currently on break.")

    prompt = "\n".join(context_parts) + "\nDo you want to have this conversation?"

    try:
        check_agent = _build_decline_check_agent(target_agent.agent_id, initiator_id)
        result = await Runner.run(check_agent, prompt)
        return result.final_output
    except Exception as e:
        logger.warning(f"Decline check failed: {e}, defaulting to accept")
        return DeclineDecisionOutput(action="accept", reason="Default acceptance")

if TYPE_CHECKING:
    from bsm.agents.generative_agent import GenerativeAgent


# ---------------------------------------------------------------------------
# Conversation Agent Builder
# ---------------------------------------------------------------------------

def _build_conversation_agent(
    speaker_id: str,
    listener_id: str,
    valid_colleagues: Optional[List[str]] = None
) -> Agent[SimContext]:
    """
    Build an agent for generating a single conversation utterance.

    Args:
        speaker_id: ID of the agent speaking
        listener_id: ID of the agent listening
        valid_colleagues: List of valid colleague IDs in the office (for anti-hallucination)

    Returns:
        Agent configured for conversation utterance generation
    """
    # Build colleague constraint section
    if valid_colleagues:
        colleague_names = ", ".join(valid_colleagues)
        colleague_constraint = f"""
<colleague_constraint>
IMPORTANT: The ONLY colleagues in this office are: {colleague_names}
Do NOT reference, mention, or discuss any other people by name.
If you think of other names (Priya, Sam, Martin, etc.), they do NOT work here.
Only discuss the colleague you are currently talking to ({listener_id}).
</colleague_constraint>
"""
    else:
        colleague_constraint = ""

    instructions = f"""
You are {speaker_id}, having a natural workplace conversation with {listener_id}.
{colleague_constraint}
<schedule_awareness>
BEFORE suggesting specific times for coffee, lunch, walks, or other activities:
1. CHECK your schedule in the <your_schedule> section of the context
2. Do NOT suggest times that overlap with your meetings
3. If you have a meeting soon, mention it: "I have a meeting at 10, but maybe after that?"
4. If unsure of a good time, ask about availability: "Want to grab coffee later?"
   rather than proposing a specific time you might not be free for
</schedule_awareness>

<recent_conversation_check>
BEFORE choosing a topic, recall recent conversations with this person.
Do NOT repeat topics already discussed today (coffee, lunch plans, weekend, etc.)
If you've already talked about something today, choose a DIFFERENT topic.
Vary your conversation topics throughout the day.
</recent_conversation_check>

<conversation_topics>
PREFERRED TOPICS (in order of preference):
1. SOCIAL: Making plans together - "Want to grab coffee?", "Shall we go for lunch?"
2. PERSONAL: Weekend plans, hobbies, life outside work
3. ENVIRONMENT: Temperature comfort, office conditions
4. LIGHT WORK: Brief updates on actual tasks, not elaborate fictional projects

AVOID:
- Long discussions about fictional projects, deadlines, or deliverables
- Referencing people who aren't in the conversation or aren't your colleagues
- Detailed planning of work tasks (this isn't a meeting)
- Making up project names, client names, or events

Temperature/thermostat should rarely come up unless you're genuinely uncomfortable.
</conversation_topics>

<guidelines>
- Keep utterances natural and brief (1-2 sentences)
- Be CASUAL and SOCIAL, not overly professional
- Reference your memories, shared context, or past interactions
- Set end_conversation=true when the conversation naturally concludes:
  - Agreement reached or understanding expressed
  - Topic exhausted
  - Someone mentions needing to get back to work
- Conversations typically last 3-6 exchanges total
- Don't drag conversations out or repeat yourself
- If suggesting a break/lunch together, check your schedule first, then make a CONCRETE plan
</guidelines>

Output your utterance directly.
""".strip()

    return Agent(
        name=f"convo_{speaker_id}_{listener_id}",
        instructions=instructions,
        model=DEFAULT_AGENT_MODEL,
        model_settings=ModelSettings(reasoning_effort="medium"),
        tools=[],
        output_type=AgentOutputSchema(UtteranceOutput, strict_json_schema=False),
    )


# ---------------------------------------------------------------------------
# Seasonal Context Helper
# ---------------------------------------------------------------------------

def _get_season_guidance(date: datetime) -> str:
    """
    Generate seasonal guidance for conversations.

    Helps agents calibrate their temperature/weather comments to be
    appropriate for the current season.

    Args:
        date: Current date

    Returns:
        Guidance text about seasonal temperature expectations
    """
    month = date.month

    if 6 <= month <= 8:  # Summer
        return (
            "It's summer. Temperatures of 15-25°C are pleasant and normal. "
            "Don't describe weather as 'freezing' or 'chilly' unless it's actually unusual for summer."
        )
    elif month in (12, 1, 2):  # Winter
        return (
            "It's winter. Temperatures of 0-10°C are typical. "
            "Temperatures above 10°C are mild for winter."
        )
    elif 3 <= month <= 5:  # Spring
        return (
            "It's spring. Temperatures can vary, with 10-18°C being typical. "
            "Expect some variability day to day."
        )
    else:  # Autumn
        return (
            "It's autumn. Temperatures of 10-18°C are typical. "
            "Expect gradually cooling weather."
        )


# ---------------------------------------------------------------------------
# Schedule Awareness for Conversations (F.3)
# ---------------------------------------------------------------------------

def _format_schedule_for_conversation(speaker: "GenerativeAgent", now: datetime) -> str:
    """
    Format speaker's schedule for conversation context.

    This helps agents be aware of their meetings and existing commitments
    when proposing times for social activities during conversations.

    Args:
        speaker: The agent who is speaking
        now: Current datetime

    Returns:
        Formatted schedule section for the conversation context
    """
    daily_plan = speaker.get_daily_plan()
    if not daily_plan:
        return ""

    lines = ["<your_schedule>"]
    lines.append("Your commitments today (check before proposing times):")

    has_items = False

    # Get meetings - handle both dict and object forms
    meetings = []
    if isinstance(daily_plan, dict):
        meetings = daily_plan.get("meetings", [])
    elif hasattr(daily_plan, "meetings"):
        meetings = daily_plan.meetings or []

    for meeting in meetings:
        try:
            if isinstance(meeting, dict):
                start = meeting.get("start_datetime_iso", "")
                end = meeting.get("end_datetime_iso", "")
                title = meeting.get("title", "Meeting")
            else:
                start = meeting.start_datetime_iso
                end = meeting.end_datetime_iso
                title = meeting.title

            start_dt = datetime.fromisoformat(start)
            end_dt = datetime.fromisoformat(end)

            # Only show today's meetings
            if start_dt.date() == now.date():
                lines.append(f"  - {start_dt.strftime('%H:%M')}-{end_dt.strftime('%H:%M')}: {title}")
                has_items = True
        except (ValueError, TypeError, AttributeError):
            continue

    # Get existing commitments - handle both dict and object forms
    commitments = []
    if isinstance(daily_plan, dict):
        commitments = daily_plan.get("social_commitments", [])
    elif hasattr(daily_plan, "social_commitments"):
        commitments = daily_plan.social_commitments or []

    for comm in commitments:
        try:
            if isinstance(comm, dict):
                time = comm.get("time", "")
                activity = comm.get("activity", "")
                fulfilled = comm.get("fulfilled", False)
            else:
                time = getattr(comm, "time", "")
                activity = getattr(comm, "activity", "")
                fulfilled = getattr(comm, "fulfilled", False)

            if not fulfilled and time and time != "needs_confirmation":
                lines.append(f"  - {time}: {activity} (already planned)")
                has_items = True
        except (ValueError, TypeError, AttributeError):
            continue

    if not has_items:
        lines.append("  (No meetings or commitments scheduled)")

    lines.append("")
    lines.append("⚠️ Do NOT suggest times that overlap with your meetings!")
    lines.append("</your_schedule>")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Context Building
# ---------------------------------------------------------------------------

def _build_conversation_context(
    speaker: "GenerativeAgent",
    listener: "GenerativeAgent",
    conversation_history: List[ConversationUtterance],
    now: datetime,
) -> str:
    """
    Build context for generating the next utterance.

    Args:
        speaker: Agent who will speak next
        listener: Agent who will listen
        conversation_history: Utterances so far
        now: Current datetime

    Returns:
        Formatted context string for the prompt
    """
    # Relationship summary
    relationship = generate_relationship_summary(speaker, listener.agent_id)

    # Speaker's identity
    identity = speaker.get_identity_stable_set()

    # Format conversation history
    if conversation_history:
        history_lines = []
        for utt in conversation_history[-6:]:  # Last 6 utterances for context
            history_lines.append(f"{utt.speaker_id}: \"{utt.utterance}\"")
        history_text = "\n".join(history_lines)
    else:
        history_text = "(This is the start of the conversation)"

    # Retrieve relevant memories about the other person and shared experiences
    focal_points = [
        f"conversations with {listener.first_name}",
        f"what I know about {listener.first_name}",
        f"shared experiences with {listener.first_name}",
        "recent interesting things at work",
        "current weather and how it feels outside",  # Weather context for appropriate comments
    ]
    retrieved = retrieve(speaker, focal_points, now, n_count=6)

    # Format memories
    memory_lines = []
    for fp, nodes in retrieved.items():
        for node in nodes[:2]:  # Top 2 per focal point
            memory_lines.append(f"- {node.description}")
    memories_text = "\n".join(memory_lines) if memory_lines else "No specific memories."

    # F.3: Add schedule awareness so agent knows their busy times
    schedule_section = _format_schedule_for_conversation(speaker, now)

    context = f"""
<identity>
{identity}
</identity>

{schedule_section}

<relationship_with_colleague>
{relationship}
</relationship_with_colleague>

<relevant_memories purpose="context for natural conversation">
{memories_text}
</relevant_memories>

<conversation_so_far>
{history_text}
</conversation_so_far>

<current_time>{now.strftime('%H:%M on %A, %B %d')}</current_time>

<seasonal_context>
{_get_season_guidance(now)}
</seasonal_context>

Generate your natural response to continue the conversation. Reference your memories or shared experiences when relevant.
""".strip()

    return context


# ---------------------------------------------------------------------------
# Utterance Generation
# ---------------------------------------------------------------------------

async def generate_one_utterance(
    speaker: "GenerativeAgent",
    listener: "GenerativeAgent",
    conversation_history: List[ConversationUtterance],
    now: datetime,
    calendar: Optional[CalendarStore] = None,
    valid_colleagues: Optional[List[str]] = None,
) -> ConversationUtterance:
    """
    Generate a single utterance in a conversation.

    Args:
        speaker: Agent generating the utterance
        listener: Agent listening
        conversation_history: Previous utterances
        now: Current datetime
        calendar: Optional calendar store
        valid_colleagues: List of valid colleague IDs in the office (for anti-hallucination)

    Returns:
        The generated utterance
    """
    # Build context
    context = _build_conversation_context(speaker, listener, conversation_history, now)

    # Build agent with valid colleagues for anti-hallucination
    convo_agent = _build_conversation_agent(
        speaker.agent_id, listener.agent_id, valid_colleagues=valid_colleagues
    )

    # Create simulation context
    sim_context = SimContext(
        occupant_id=speaker.agent_id,
        now=now,
        calendar=calendar,
        simulation=None,
        memory=None,
    )

    # Generate utterance
    result = await Runner.run(convo_agent, context, context=sim_context)
    output: UtteranceOutput = result.final_output

    return ConversationUtterance(
        speaker_id=speaker.agent_id,
        utterance=output.utterance,
        end_conversation=output.end_conversation,
    )


# ---------------------------------------------------------------------------
# Main Conversation Function
# ---------------------------------------------------------------------------

async def agent_conversation(
    init_agent: "GenerativeAgent",
    target_agent: "GenerativeAgent",
    now: datetime,
    max_turns: int = 8,
    calendar: Optional[CalendarStore] = None,
    valid_colleagues: Optional[List[str]] = None,
    topic: Optional[str] = None,
) -> ConversationResult:
    """
    Conduct a multi-turn conversation between two agents.

    This is the main conversation loop (similar to Stanford's agent_chat_v2).

    Args:
        init_agent: Agent who initiated the conversation
        target_agent: Agent being talked to
        now: Current simulation datetime
        max_turns: Maximum number of exchanges (default 8)
        calendar: Optional calendar store for context
        valid_colleagues: List of valid colleague IDs in the office (for anti-hallucination)
        topic: Optional topic of the conversation

    Returns:
        ConversationResult with all utterances and summary
    """
    # Check if target agent wants to participate
    decline_decision = await _check_target_willingness(
        target_agent=target_agent,
        initiator_id=init_agent.agent_id,
        topic=topic,
        now=now,
    )

    if decline_decision.action == "decline":
        logger.info(f"{target_agent.agent_id} declined conversation: {decline_decision.reason}")
        return ConversationResult(
            participants=[init_agent.agent_id, target_agent.agent_id],
            utterances=[],
            summary=f"{target_agent.first_name} declined to talk: {decline_decision.reason}",
            duration_minutes=0,
            topics_discussed=[],
            initiated_by=init_agent.agent_id,
            declined=True,
            decline_reason=decline_decision.reason,
        )

    utterances: List[ConversationUtterance] = []
    speaker = init_agent
    listener = target_agent

    logger.info(f"Starting conversation: {init_agent.agent_id} -> {target_agent.agent_id}")

    for turn in range(max_turns * 2):  # *2 because each turn is one speaker
        # Generate utterance
        utterance = await generate_one_utterance(
            speaker=speaker,
            listener=listener,
            conversation_history=utterances,
            now=now,
            calendar=calendar,
            valid_colleagues=valid_colleagues,
        )
        utterances.append(utterance)

        utterance_preview = utterance.utterance[:60] + "..." if len(utterance.utterance) > 60 else utterance.utterance
        logger.debug(f"  [{speaker.first_name}]: \"{utterance_preview}\"")

        # Check if conversation should end
        if utterance.end_conversation:
            logger.info(f"Conversation ended naturally after {len(utterances)} utterances")
            break

        # Swap speaker and listener
        speaker, listener = listener, speaker

    # Generate summary
    summary = _generate_conversation_summary(utterances, init_agent, target_agent)

    # Estimate duration (roughly 1-2 minutes per exchange)
    duration_minutes = max(1, len(utterances) // 2)

    # Extract topics (simple keyword extraction)
    topics = _extract_conversation_topics(utterances)

    return ConversationResult(
        participants=[init_agent.agent_id, target_agent.agent_id],
        utterances=utterances,
        summary=summary,
        duration_minutes=duration_minutes,
        topics_discussed=topics,
        initiated_by=init_agent.agent_id,
    )


def _generate_conversation_summary(
    utterances: List[ConversationUtterance],
    agent1: "GenerativeAgent",
    agent2: "GenerativeAgent",
) -> str:
    """
    Generate a brief summary of the conversation.

    Args:
        utterances: All conversation utterances
        agent1: First agent
        agent2: Second agent

    Returns:
        Brief summary string
    """
    if not utterances:
        return f"{agent1.first_name} and {agent2.first_name} had a brief exchange."

    num_turns = len(utterances)

    if num_turns <= 2:
        return f"{agent1.first_name} and {agent2.first_name} had a brief exchange."
    elif num_turns <= 4:
        return f"{agent1.first_name} and {agent2.first_name} had a short conversation."
    elif num_turns <= 8:
        return f"{agent1.first_name} and {agent2.first_name} had a conversation."
    else:
        return f"{agent1.first_name} and {agent2.first_name} had an extended conversation."


def _extract_conversation_topics(utterances: List[ConversationUtterance]) -> List[str]:
    """
    Extract likely topics from conversation utterances.

    Simple keyword-based extraction for now.

    Args:
        utterances: All conversation utterances

    Returns:
        List of topic strings
    """
    # Combine all text
    all_text = " ".join(u.utterance.lower() for u in utterances)

    topics = []

    # Check for common office topics
    topic_keywords = {
        "work": ["project", "deadline", "task", "meeting", "work", "report"],
        "weather": ["weather", "rain", "sunny", "cold", "warm", "temperature"],
        "weekend": ["weekend", "saturday", "sunday", "plans"],
        "coffee": ["coffee", "lunch", "break", "tea"],
        "technical": ["code", "bug", "feature", "system", "database"],
    }

    for topic, keywords in topic_keywords.items():
        if any(kw in all_text for kw in keywords):
            topics.append(topic)

    return topics if topics else ["general chat"]


# ---------------------------------------------------------------------------
# Commitment Extraction
# ---------------------------------------------------------------------------

async def extract_conversation_commitments(
    conversation: ConversationResult,
    participant_ids: List[str],
    model: str = DEFAULT_AGENT_MODEL,
) -> List[ConversationCommitmentOutput]:
    """
    Use LLM to extract any commitments/agreements from a conversation.

    Looks for agreements to do activities together like:
    - "Let's grab coffee later"
    - "Want to take a walk after lunch?"
    - "Meet you in the break room at 3"

    Args:
        conversation: The completed conversation
        participant_ids: IDs of participants
        model: Model to use for extraction

    Returns:
        List of commitments extracted (empty if none found)
    """
    # Build conversation transcript
    transcript_lines = []
    for utt in conversation.utterances:
        speaker_name = utt.speaker_id.split("_")[0].capitalize()
        transcript_lines.append(f"{speaker_name}: {utt.utterance}")
    transcript = "\n".join(transcript_lines)

    logger.debug(f"Extracting from {len(conversation.utterances)} utterances")
    logger.debug(f"Transcript preview: {transcript[:300]}...")

    participants_str = " and ".join(p.split("_")[0].capitalize() for p in participant_ids)

    instructions = f"""You are analyzing a conversation between {participants_str} to extract any commitments or agreements to do activities together.

<task>
Read the conversation and identify if the participants agreed to do any activity together.
Extract commitments in TWO tiers:
</task>

<tier_1_specific>
For commitments with SPECIFIC times:
- activity: SHORT description (max 4 words) - "coffee", "lunch", "walk", or combined like "coffee break with walk"
- time: HH:MM format (e.g., "10:30", "12:00", "14:15")
- location: break_area, outside, or meeting_room (use FINAL destination for combined activities)
</tier_1_specific>

<tier_2_loose>
For agreements WITHOUT specific times (both parties agreed but no exact time):
- activity: What they agreed to do (max 4 words)
- time: Use "needs_confirmation" (exactly this string)
- location: Best guess - break_area for coffee/tea, outside for walk/lunch out, meeting_room for work discussions

Examples that should use "needs_confirmation":
- "Want to grab coffee later?" "Sure, sounds good!" → time: "needs_confirmation"
- "Let's do lunch sometime" "Great idea!" → time: "needs_confirmation"
- "We should take a walk after lunch" "I'd like that" → time: "needs_confirmation"
</tier_2_loose>

<compound_activities>
IMPORTANT: If participants agree to MULTIPLE activities as part of ONE break or outing
(e.g., "coffee and a walk", "grab coffee then do a lap outside", "lunch then a stroll"):

1. Extract as a SINGLE commitment, NOT multiple separate ones
2. activity: Combine with qualifier - "coffee break with walk", "lunch with outdoor walk"
3. time: The START time of the combined activity
4. location: Where they END UP (e.g., "outside" if walking outside after coffee)

CORRECT Examples:
- "Want to grab coffee at 3:30 and do a lap outside?"
  → ONE commitment: activity="coffee break with walk", time="15:30", location="outside"

- "Let's meet at the break room, grab coffees to-go, then walk outside"
  → ONE commitment: activity="coffee break with walk", time from context, location="outside"

WRONG (do not do this):
- TWO separate commitments: {{activity: "coffee"}} + {{activity: "walk"}}

The key insight: if activities are discussed as part of the SAME outing/break, they are ONE commitment.
</compound_activities>

<do_not_extract>
- Only one person suggested something without clear agreement from the other
- Hypothetical or conditional plans ("If I have time...", "Maybe we could...")
- Past activities they already did
</do_not_extract>

<conversation>
{transcript}
</conversation>

<output_verbosity_spec>
- Reasoning: 1 sentence explaining what agreements were found
- Return empty list ONLY if no agreement was reached at all
</output_verbosity_spec>"""

    agent = Agent(
        name="commitment_extractor",
        instructions=instructions,
        model=model,
        model_settings=ModelSettings(reasoning_effort="low"),
        output_type=AgentOutputSchema(CommitmentsExtractionOutput),
    )

    try:
        result = await Runner.run(agent, "Extract any commitments from this conversation.")
        if result.final_output:
            commitments = result.final_output.commitments or []
            logger.debug(f"Extraction returned {len(commitments)} commitments")
            for c in commitments:
                logger.debug(f"  - activity='{c.activity}' time='{c.time}' location='{c.location}'")
            if commitments:
                return commitments
        else:
            logger.debug("Extraction returned no output (final_output is None)")
    except Exception as e:
        logger.error(f"Commitment extraction failed: {e}", exc_info=True)

    return []


# ---------------------------------------------------------------------------
# Commitment Conflict Checking
# ---------------------------------------------------------------------------

def check_commitment_conflicts(
    agent: "GenerativeAgent",
    new_commitment: SocialCommitment,
    now: datetime,
) -> List[str]:
    """
    Check if a new commitment conflicts with existing ones or meetings.

    Args:
        agent: The agent making the commitment
        new_commitment: The proposed commitment
        now: Current datetime

    Returns:
        List of conflict descriptions (empty if no conflicts)
    """
    conflicts = []

    daily_plan = agent.get_daily_plan() if hasattr(agent, 'get_daily_plan') else None
    if not daily_plan:
        return conflicts

    # Skip if time is unspecified/needs confirmation (extraction uses "needs_confirmation")
    unspecified_times = {"unspecified", "needs_confirmation", ""}
    if new_commitment.time in unspecified_times:
        return conflicts

    # Parse new commitment time
    try:
        new_hour, new_minute = map(int, new_commitment.time.split(":"))
        new_minutes = new_hour * 60 + new_minute
    except (ValueError, TypeError):
        return conflicts

    # Check against existing commitments (within 15 minutes)
    if hasattr(daily_plan, 'social_commitments') and daily_plan.social_commitments:
        for existing in daily_plan.social_commitments:
            if existing.fulfilled:
                continue
            if existing.time in unspecified_times:
                continue
            try:
                existing_hour, existing_minute = map(int, existing.time.split(":"))
                existing_minutes = existing_hour * 60 + existing_minute
                if abs(new_minutes - existing_minutes) < 15:
                    with_names = ", ".join(a.split("_")[0].capitalize() for a in existing.with_agents)
                    conflicts.append(
                        f"Already committed to '{existing.activity}' with {with_names} at {existing.time}"
                    )
            except (ValueError, TypeError):
                pass

    # Check against meetings
    if hasattr(daily_plan, 'meetings') and daily_plan.meetings:
        for meeting in daily_plan.meetings:
            try:
                start_dt = datetime.fromisoformat(meeting.start_datetime_iso)
                end_dt = datetime.fromisoformat(meeting.end_datetime_iso)
                start_minutes = start_dt.hour * 60 + start_dt.minute
                end_minutes = end_dt.hour * 60 + end_dt.minute

                # Check if commitment time falls within meeting
                if start_minutes <= new_minutes <= end_minutes:
                    conflicts.append(f"Conflicts with meeting '{meeting.title}' ({meeting.start_datetime_iso[:16]})")
            except (ValueError, TypeError):
                pass

    return conflicts


# ---------------------------------------------------------------------------
# Synchronous Wrapper
# ---------------------------------------------------------------------------

def sync_agent_conversation(
    init_agent: "GenerativeAgent",
    target_agent: "GenerativeAgent",
    now: datetime,
    max_turns: int = 8,
    calendar: Optional[CalendarStore] = None,
) -> ConversationResult:
    """
    Synchronous wrapper for agent_conversation.

    Args:
        init_agent: Agent who initiated the conversation
        target_agent: Agent being talked to
        now: Current simulation datetime
        max_turns: Maximum number of exchanges
        calendar: Optional calendar store

    Returns:
        ConversationResult
    """
    return asyncio.run(agent_conversation(
        init_agent=init_agent,
        target_agent=target_agent,
        now=now,
        max_turns=max_turns,
        calendar=calendar,
    ))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Main conversation function
    "agent_conversation",
    "sync_agent_conversation",
    # Commitment utilities (used by memory_ops)
    "extract_conversation_commitments",
    "check_commitment_conflicts",
    # Backward compatibility - re-export schemas
    "ConversationUtterance",
    "ConversationResult",
    # Backward compatibility - re-export from other modules
    "consultation_conversation",
    "record_conversation_to_memory",
    "record_agreement_to_memory",
    "ConsultationOutcome",
]

# Backward compatibility re-exports
from bsm.agents.cognition.schemas import ConversationUtterance, ConversationResult, ConsultationOutcome
from bsm.agents.cognition.consultation import consultation_conversation
from bsm.agents.cognition.memory_ops import record_conversation_to_memory, record_agreement_to_memory

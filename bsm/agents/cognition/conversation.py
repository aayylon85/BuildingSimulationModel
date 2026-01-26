"""
converse.py

Conversation system for generative agents.
Implements multi-turn dialogue between agents based on Stanford's generative_agents pattern.

Key features:
- Turn-based conversation with natural ending detection
- Relationship-aware context
- Memory storage for conversations
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

from agents import Agent, Runner, ModelSettings
from agents.agent_output import AgentOutputSchema

from bsm.agents.skeleton import DEFAULT_AGENT_MODEL, SimContext, CalendarStore
from bsm.agents.cognition.modules import generate_relationship_summary, retrieve
from bsm.agents.memory.stream import MemoryNode

if TYPE_CHECKING:
    from bsm.agents.generative_agent import GenerativeAgent


# ---------------------------------------------------------------------------
# Conversation Data Structures
# ---------------------------------------------------------------------------

class ConversationUtterance(BaseModel):
    """A single utterance in a conversation."""
    speaker_id: str
    utterance: str
    end_conversation: bool = Field(
        default=False,
        description="True if this utterance naturally ends the conversation"
    )


class ConversationResult(BaseModel):
    """Result of a complete conversation between two agents."""
    participants: List[str]
    utterances: List[ConversationUtterance]
    summary: str
    duration_minutes: int
    topics_discussed: List[str] = Field(default_factory=list)
    initiated_by: str


class UtteranceOutput(BaseModel):
    """Output schema for the conversation agent."""
    utterance: str = Field(description="What you say in the conversation")
    end_conversation: bool = Field(
        default=False,
        description="Set to true if this utterance naturally ends the conversation (goodbye, need to go, etc.)"
    )


# ---------------------------------------------------------------------------
# Conversation Agent Builder
# ---------------------------------------------------------------------------

def build_conversation_agent(speaker_id: str, listener_id: str) -> Agent[SimContext]:
    """
    Build an agent for generating a single conversation utterance.

    Args:
        speaker_id: ID of the agent speaking
        listener_id: ID of the agent listening

    Returns:
        Agent configured for conversation utterance generation
    """
    instructions = f"""
You are {speaker_id}, having a conversation with {listener_id} in an office setting.

Generate your next utterance in the conversation based on:
- The conversation history so far
- Your relationship with {listener_id}
- Your personality and current context

Guidelines:
- Keep utterances natural and brief (1-3 sentences typical)
- Reference shared context, work topics, or past interactions when relevant
- Be conversational - use contractions, casual language appropriate to workplace
- Set end_conversation=true when the conversation naturally concludes:
  - Agreement/understanding reached
  - Topic exhausted or need to get back to work
  - Natural goodbye or "I should get back to..."
  - Someone mentions needing to leave
- Conversations typically last 4-8 exchanges total
- Don't drag conversations out unnecessarily

Output your utterance directly.
""".strip()

    return Agent(
        name=f"convo_{speaker_id}_{listener_id}",
        instructions=instructions,
        model=DEFAULT_AGENT_MODEL,
        model_settings=ModelSettings(reasoning_effort="low"),  # Dialogue is intuitive
        tools=[],
        output_type=AgentOutputSchema(UtteranceOutput, strict_json_schema=False),
    )


# ---------------------------------------------------------------------------
# Conversation Functions
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

    # Retrieve relevant memories about the other person
    focal_points = [
        f"conversations with {listener.agent_id}",
        f"what I know about {listener.first_name}",
    ]
    retrieved = retrieve(speaker, focal_points, now, n_count=5)

    # Format memories
    memory_lines = []
    for fp, nodes in retrieved.items():
        for node in nodes[:2]:  # Top 2 per focal point
            memory_lines.append(f"- {node.description}")
    memories_text = "\n".join(memory_lines) if memory_lines else "No specific memories."

    context = f"""
=== YOUR IDENTITY ===
{identity}

=== RELATIONSHIP ===
{relationship}

=== RELEVANT MEMORIES ===
{memories_text}

=== CONVERSATION SO FAR ===
{history_text}

=== CURRENT TIME ===
{now.strftime('%H:%M on %A')}

Now generate your response to continue the conversation.
""".strip()

    return context


async def generate_one_utterance(
    speaker: "GenerativeAgent",
    listener: "GenerativeAgent",
    conversation_history: List[ConversationUtterance],
    now: datetime,
    calendar: Optional[CalendarStore] = None,
) -> ConversationUtterance:
    """
    Generate a single utterance in a conversation.

    Args:
        speaker: Agent generating the utterance
        listener: Agent listening
        conversation_history: Previous utterances
        now: Current datetime
        calendar: Optional calendar store

    Returns:
        The generated utterance
    """
    # Build context
    context = _build_conversation_context(speaker, listener, conversation_history, now)

    # Build agent
    convo_agent = build_conversation_agent(speaker.agent_id, listener.agent_id)

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


async def agent_conversation(
    init_agent: "GenerativeAgent",
    target_agent: "GenerativeAgent",
    now: datetime,
    max_turns: int = 8,
    calendar: Optional[CalendarStore] = None,
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

    Returns:
        ConversationResult with all utterances and summary
    """
    utterances: List[ConversationUtterance] = []
    speaker = init_agent
    listener = target_agent

    print(f"[CONVERSE] Starting conversation: {init_agent.agent_id} -> {target_agent.agent_id}")

    for turn in range(max_turns * 2):  # *2 because each turn is one speaker
        # Generate utterance
        utterance = await generate_one_utterance(
            speaker=speaker,
            listener=listener,
            conversation_history=utterances,
            now=now,
            calendar=calendar,
        )
        utterances.append(utterance)

        print(f"  [{speaker.first_name}]: \"{utterance.utterance[:60]}...\""
              if len(utterance.utterance) > 60 else f"  [{speaker.first_name}]: \"{utterance.utterance}\"")

        # Check if conversation should end
        if utterance.end_conversation:
            print(f"[CONVERSE] Conversation ended naturally after {len(utterances)} utterances")
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


def record_conversation_to_memory(
    agent: "GenerativeAgent",
    conversation: ConversationResult,
    other_agent_id: str,
    now: datetime,
) -> None:
    """
    Record a conversation to an agent's memory stream.

    Creates a 'chat' type memory node with conversation summary.
    Also marks the conversation end for post-conversation reflection (Stanford-style).

    Args:
        agent: Agent whose memory to update
        conversation: The conversation result
        other_agent_id: ID of the other participant
        now: Current datetime
    """
    if not agent.memory_stream:
        return

    # Build description
    other_first_name = other_agent_id.split("_")[0].capitalize()
    topics_str = ", ".join(conversation.topics_discussed) if conversation.topics_discussed else "general topics"

    description = (
        f"Had a conversation with {other_first_name} about {topics_str}. "
        f"{conversation.summary}"
    )

    # Add as chat memory
    agent.memory_stream.add_chat(
        description=description,
        other_agent=other_agent_id,
        now=now,
        importance=5.0,  # Conversations are moderately important
    )

    # Mark for post-conversation reflection (Stanford-style)
    agent.mark_conversation_end(other_agent_id, now)

    # Update relationship: familiarity increases with each conversation
    agent.update_relationship(
        other_agent_id=other_agent_id,
        familiarity_delta=0.05,  # Small increase per conversation
        sentiment_delta=0.0,     # Sentiment neutral for general conversations
    )


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
# Consultation System for Shared Space Decisions
# ---------------------------------------------------------------------------

class ConsultationOutcome(BaseModel):
    """Result of a consultation about a shared-space action."""
    proposed_action: str = Field(description="What was originally proposed (e.g., 'set thermostat to 19°C')")
    agreed_action: Optional[str] = Field(default=None, description="The agreed-upon action (may differ from proposed)")
    consensus_reached: bool = Field(default=False, description="Whether all participants agreed")
    final_setpoint_c: Optional[float] = Field(default=None, description="Agreed thermostat setpoint if applicable")
    final_window_state: Optional[bool] = Field(default=None, description="Agreed window state if applicable (True=open)")
    summary: str = Field(description="Brief summary of the consultation outcome")
    participants: List[str] = Field(default_factory=list, description="IDs of agents who participated")


class ConsultationUtteranceOutput(BaseModel):
    """Output schema for consultation conversation agent."""
    utterance: str = Field(description="What you say in the consultation")
    end_consultation: bool = Field(
        default=False,
        description="Set to true when agreement is reached or clear disagreement"
    )
    proposed_compromise: Optional[float] = Field(
        default=None,
        description="If proposing a compromise temperature, specify the value in °C"
    )


class ConsensusAssessment(BaseModel):
    """Output schema for LLM-based consensus assessment."""
    consensus_reached: bool = Field(description="Whether all participants agreed to the change")
    agreed_temperature_c: Optional[float] = Field(
        default=None,
        description="The agreed-upon temperature in Celsius, if any"
    )
    summary: str = Field(description="Brief summary of the consultation outcome")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence level (0-1) in the assessment"
    )


def build_consensus_assessor_agent() -> Agent[SimContext]:
    """
    Build an agent for assessing consensus from consultation conversations.

    Uses LLM to analyze conversation and determine if agreement was reached.
    """
    instructions = """Analyze this consultation conversation and determine:
1. Did ALL participants agree to a temperature change?
2. If yes, what specific temperature was agreed upon?
3. Summarize the outcome.

Be strict: silence or vague responses do NOT count as agreement.
Look for explicit acceptance like:
- "okay, let's do 21 degrees"
- "I agree to 20.5"
- "that works for me"
- "fine, let's try 22"

Look for explicit disagreement like:
- "no way"
- "I don't agree"
- "too cold/hot for me"
- "I refuse"

If the outcome is ambiguous or unclear, set consensus_reached to false."""

    return Agent(
        name="consensus_assessor",
        instructions=instructions,
        model=DEFAULT_AGENT_MODEL,
        model_settings=ModelSettings(reasoning_effort="low"),
        output_type=AgentOutputSchema(ConsensusAssessment),
    )


async def assess_consensus_with_llm(
    utterances: List[ConversationUtterance],
    proposed_action: str,
    proposed_temp: Optional[float] = None,
) -> ConsensusAssessment:
    """
    Use LLM to assess whether consensus was reached in a consultation conversation.

    Args:
        utterances: List of conversation utterances
        proposed_action: The originally proposed action
        proposed_temp: The originally proposed temperature (if any)

    Returns:
        ConsensusAssessment with the LLM's analysis
    """
    conversation_text = f"Proposed action: {proposed_action}\n\n"
    conversation_text += "Conversation:\n"
    conversation_text += "\n".join(f"{u.speaker_id}: {u.utterance}" for u in utterances)

    agent = build_consensus_assessor_agent()
    result = await Runner.run(agent, conversation_text)
    return result.final_output


def simple_vote_on_temperature(
    initiator_comfort_c: float,
    proposed_temp: float,
    other_agent_comforts: List[float],
    current_temp: float,
) -> tuple[bool, str]:
    """
    Simple majority vote on a temperature change (no conversation needed).

    Args:
        initiator_comfort_c: Initiator's comfort temperature preference
        proposed_temp: The proposed temperature change
        other_agent_comforts: List of other agents' comfort preferences
        current_temp: Current indoor temperature

    Returns:
        Tuple of (consensus_reached, summary)
    """
    votes_for = 1  # Initiator votes yes
    votes_against = 0

    for their_comfort in other_agent_comforts:
        # Vote yes if proposed temp is closer to their preference than current
        if abs(proposed_temp - their_comfort) <= abs(current_temp - their_comfort):
            votes_for += 1
        else:
            votes_against += 1

    total = votes_for + votes_against
    consensus = votes_for > votes_against
    summary = f"Vote: {votes_for}/{total} for proposed temperature"

    return consensus, summary


def build_consultation_agent(speaker_id: str, is_initiator: bool) -> Agent[SimContext]:
    """
    Build an agent for consultation about shared-space changes.

    Args:
        speaker_id: ID of the agent speaking
        is_initiator: Whether this agent initiated the consultation

    Returns:
        Agent configured for consultation utterance generation
    """
    role = "proposing a change" if is_initiator else "responding to a proposed change"

    instructions = f"""
You are {speaker_id}, {role} to the office shared controls (thermostat or windows).

This is a CONSULTATION - you need to discuss and find a compromise that works for everyone.

Guidelines:
- Be respectful of others' comfort preferences
- Consider compromises (e.g., meeting halfway on temperature)
- If you feel too hot, you prefer lower temperatures; if cold, prefer higher
- Set end_consultation=true when:
  - Agreement is reached on a specific action
  - You both decide to leave things as they are
  - Clear disagreement where no compromise is possible
- Use proposed_compromise to suggest a specific temperature value
- Keep responses brief (1-2 sentences)
- Consultations typically last 3-6 exchanges

Output your response.
""".strip()

    return Agent(
        name=f"consultation_{speaker_id}",
        instructions=instructions,
        model=DEFAULT_AGENT_MODEL,
        model_settings=ModelSettings(reasoning_effort="low"),  # Dialogue is intuitive
        tools=[],
        output_type=AgentOutputSchema(ConsultationUtteranceOutput, strict_json_schema=False),
    )


def _build_consultation_context(
    speaker: "GenerativeAgent",
    listeners: List["GenerativeAgent"],
    proposed_action: str,
    current_temp_c: float,
    conversation_history: List[ConversationUtterance],
    now: datetime,
    is_initiator: bool,
) -> str:
    """
    Build context for generating a consultation utterance.

    Args:
        speaker: Agent who will speak next
        listeners: Other agents in the consultation
        proposed_action: The proposed change being discussed
        current_temp_c: Current zone temperature
        conversation_history: Utterances so far
        now: Current datetime
        is_initiator: Whether speaker initiated the consultation

    Returns:
        Formatted context string
    """
    # Speaker's comfort preferences
    identity = speaker.get_identity_stable_set()

    # Other participants' names
    other_names = ", ".join(l.first_name for l in listeners)

    # Format conversation history
    if conversation_history:
        history_lines = []
        for utt in conversation_history[-6:]:
            name = utt.speaker_id.split("_")[0].capitalize()
            history_lines.append(f"{name}: \"{utt.utterance}\"")
        history_text = "\n".join(history_lines)
    else:
        history_text = "(Consultation just started)"

    role_text = f"You proposed: {proposed_action}" if is_initiator else f"Someone proposed: {proposed_action}"

    context = f"""
=== YOUR IDENTITY & COMFORT PREFERENCES ===
{identity}

=== CONSULTATION REQUEST ===
{role_text}
Current zone temperature: {current_temp_c:.1f}°C
Other participants: {other_names}

=== DISCUSSION SO FAR ===
{history_text}

=== CURRENT TIME ===
{now.strftime('%H:%M')}

Respond to continue the consultation. Aim for a compromise everyone can accept.
""".strip()

    return context


async def generate_consultation_utterance(
    speaker: "GenerativeAgent",
    listeners: List["GenerativeAgent"],
    proposed_action: str,
    current_temp_c: float,
    conversation_history: List[ConversationUtterance],
    now: datetime,
    is_initiator: bool,
    calendar: Optional[CalendarStore] = None,
) -> tuple[ConversationUtterance, Optional[float]]:
    """
    Generate a single utterance in a consultation.

    Returns:
        Tuple of (utterance, proposed_compromise_value or None)
    """
    context = _build_consultation_context(
        speaker, listeners, proposed_action, current_temp_c,
        conversation_history, now, is_initiator
    )

    agent = build_consultation_agent(speaker.agent_id, is_initiator)

    sim_context = SimContext(
        occupant_id=speaker.agent_id,
        now=now,
        calendar=calendar,
        simulation=None,
        memory=None,
    )

    result = await Runner.run(agent, context, context=sim_context)
    output: ConsultationUtteranceOutput = result.final_output

    utterance = ConversationUtterance(
        speaker_id=speaker.agent_id,
        utterance=output.utterance,
        end_conversation=output.end_consultation,
    )

    return utterance, output.proposed_compromise


async def consultation_conversation(
    initiator: "GenerativeAgent",
    proposed_action: str,
    proposed_setpoint_c: Optional[float],
    current_temp_c: float,
    present_agents: List["GenerativeAgent"],
    now: datetime,
    max_turns: int = 6,
    calendar: Optional[CalendarStore] = None,
    consultation_mode: str = "llm",
) -> ConsultationOutcome:
    """
    Conduct a consultation about a proposed shared-space change.

    This is used when an agent wants to adjust the thermostat or windows
    and other occupants are present.

    Args:
        initiator: Agent proposing the change
        proposed_action: Description of proposed action (e.g., "set thermostat to 19°C")
        proposed_setpoint_c: Proposed temperature if applicable
        current_temp_c: Current zone temperature
        present_agents: All present agents (including initiator)
        now: Current simulation datetime
        max_turns: Maximum exchanges before forcing decision
        calendar: Optional calendar store
        consultation_mode: "llm" for LLM assessment, "vote" for simple majority,
                          "keyword" for legacy keyword-based detection

    Returns:
        ConsultationOutcome with the agreed action
    """
    # Filter out initiator from listeners
    listeners = [a for a in present_agents if a.agent_id != initiator.agent_id]

    if not listeners:
        # No one else present, just proceed with original action
        return ConsultationOutcome(
            proposed_action=proposed_action,
            agreed_action=proposed_action,
            consensus_reached=True,
            final_setpoint_c=proposed_setpoint_c,
            summary="No consultation needed - no other occupants present",
            participants=[initiator.agent_id],
        )

    participants = [initiator.agent_id] + [l.agent_id for l in listeners]
    utterances: List[ConversationUtterance] = []
    latest_compromise: Optional[float] = proposed_setpoint_c

    print(f"[CONSULTATION] {initiator.first_name} consulting about: {proposed_action}")
    print(f"  Present: {', '.join(a.first_name for a in present_agents)}")

    # Initiator starts
    speaker = initiator
    speaker_is_initiator = True
    listener_idx = 0

    for turn in range(max_turns * 2):
        # Generate utterance
        utt, compromise = await generate_consultation_utterance(
            speaker=speaker,
            listeners=listeners if speaker == initiator else [initiator],
            proposed_action=proposed_action,
            current_temp_c=current_temp_c,
            conversation_history=utterances,
            now=now,
            is_initiator=speaker_is_initiator,
            calendar=calendar,
        )
        utterances.append(utt)

        if compromise is not None:
            latest_compromise = compromise

        print(f"  [{speaker.first_name}]: \"{utt.utterance}\"")

        # Check for end
        if utt.end_conversation:
            print(f"[CONSULTATION] Consultation ended after {len(utterances)} exchanges")
            break

        # Rotate speakers (initiator <-> listeners in round-robin)
        if speaker == initiator:
            speaker = listeners[listener_idx % len(listeners)]
            speaker_is_initiator = False
        else:
            listener_idx += 1
            speaker = initiator
            speaker_is_initiator = True

    # Determine outcome based on conversation and consultation_mode
    consensus = False
    agreed_action = None
    summary = ""

    if consultation_mode == "llm" and utterances:
        # Use LLM to assess consensus
        try:
            assessment = await assess_consensus_with_llm(
                utterances=utterances,
                proposed_action=proposed_action,
                proposed_temp=proposed_setpoint_c,
            )
            consensus = assessment.consensus_reached
            if consensus:
                final_temp = assessment.agreed_temperature_c or latest_compromise or proposed_setpoint_c
                agreed_action = f"set thermostat to {final_temp:.1f}°C"
                latest_compromise = final_temp
            summary = assessment.summary
            print(f"[CONSULTATION] LLM assessment: consensus={consensus}, temp={assessment.agreed_temperature_c}")
        except Exception as e:
            print(f"[CONSULTATION] LLM assessment failed, falling back to keyword: {e}")
            consultation_mode = "keyword"  # Fallback

    if consultation_mode == "vote" and proposed_setpoint_c is not None:
        # Use simple majority vote (no conversation needed, but still works post-conversation)
        initiator_comfort = initiator.scratch.get("thermal_comfort_c", 21.0)
        other_comforts = [a.scratch.get("thermal_comfort_c", 21.0) for a in listeners]
        consensus, summary = simple_vote_on_temperature(
            initiator_comfort_c=initiator_comfort,
            proposed_temp=proposed_setpoint_c,
            other_agent_comforts=other_comforts,
            current_temp=current_temp_c,
        )
        if consensus:
            agreed_action = proposed_action
            latest_compromise = proposed_setpoint_c
        print(f"[CONSULTATION] Vote result: {summary}")

    if consultation_mode == "keyword":
        # Legacy keyword-based detection
        consensus = len(utterances) > 1 and utterances[-1].end_conversation

        # Expanded agreement detection with more comprehensive word lists
        last_utterances_text = " ".join(u.utterance.lower() for u in utterances[-4:])

        # Expanded agreement words/phrases
        agreement_words = [
            "okay", "ok", "fine", "agree", "agreed", "sure", "alright", "yes",
            "good", "great", "perfect", "excellent", "wonderful",
            "sounds good", "sounds great", "sounds fine", "that works",
            "works for me", "let's do", "let's go with", "i can do",
            "i'm okay with", "i'm fine with", "i can live with",
            "im okay with", "im fine with",
            "compromise", "deal", "fair enough", "reasonable", "acceptable",
            "degrees is fine", "degrees works", "degrees is good",
            "that temperature", "we can try",
        ]
        disagreement_words = [
            "no way", "absolutely not", "won't accept", "refuse", "disagree",
            "can't accept", "cannot accept", "too cold", "too hot", "too warm",
            "not comfortable", "uncomfortable with",
        ]

        has_agreement = any(w in last_utterances_text for w in agreement_words)
        has_disagreement = any(w in last_utterances_text for w in disagreement_words)

        # Also check if there's a temperature mentioned in an agreeing context
        agreed_temp_from_text = None
        if has_agreement:
            for u in reversed(utterances[-4:]):
                text = u.utterance.lower()
                if any(w in text for w in agreement_words):
                    temp_match = re.search(r'(\d{1,2}(?:\.\d)?)\s*(?:°|degrees?|c\b|celsius)?', text)
                    if temp_match:
                        try:
                            agreed_temp_from_text = float(temp_match.group(1))
                            break
                        except ValueError:
                            pass

        if has_agreement and not has_disagreement:
            consensus = True
            final_temp = agreed_temp_from_text or latest_compromise or proposed_setpoint_c
            agreed_action = f"set thermostat to {final_temp:.1f}°C"
            summary = f"Agreed to {agreed_action} after consultation"
            latest_compromise = final_temp
        elif has_disagreement:
            consensus = False
            agreed_action = None
            latest_compromise = None
            summary = "Could not reach agreement on the proposed change"
        else:
            if latest_compromise and latest_compromise != proposed_setpoint_c:
                consensus = True
                agreed_action = f"set thermostat to {latest_compromise:.1f}°C"
                summary = f"Compromised on {latest_compromise:.1f}°C"
            elif utterances and utterances[-1].end_conversation and len(utterances) >= 2:
                consensus = True
                agreed_action = proposed_action
                latest_compromise = proposed_setpoint_c
                summary = f"Agreed to {proposed_action} (implicit consent)"
            else:
                consensus = False
                agreed_action = None
                summary = "Consultation ended without clear agreement"

    return ConsultationOutcome(
        proposed_action=proposed_action,
        agreed_action=agreed_action,
        consensus_reached=consensus,
        final_setpoint_c=latest_compromise if consensus else None,
        summary=summary,
        participants=participants,
    )


def record_agreement_to_memory(
    agents: List["GenerativeAgent"],
    outcome: ConsultationOutcome,
    now: datetime,
) -> None:
    """
    Record a consultation agreement to ALL participating agents' memories.

    This ensures all agents remember the agreement and can respect it
    in future decisions.

    Args:
        agents: List of all participating agents
        outcome: The consultation outcome
        now: Current datetime
    """
    if not outcome.consensus_reached or not outcome.agreed_action:
        # Only record actual agreements
        return

    for agent in agents:
        if not agent.memory_stream:
            continue

        # Build description from agent's perspective
        other_names = [
            a.first_name for a in agents
            if a.agent_id != agent.agent_id
        ]
        others_str = " and ".join(other_names) if other_names else "colleagues"

        description = f"Agreed with {others_str}: {outcome.summary}"

        # Record as high-importance event (should influence future decisions)
        agent.memory_stream.add_event(
            description=description,
            subject="we",
            predicate="agreed",
            obj=outcome.agreed_action,
            now=now,
            importance=7.0,  # High importance
        )

        # Update relationships: agreement improves familiarity and sentiment
        for other_agent in agents:
            if other_agent.agent_id != agent.agent_id:
                agent.update_relationship(
                    other_agent_id=other_agent.agent_id,
                    familiarity_delta=0.03,  # Small familiarity boost
                    sentiment_delta=0.05,    # Positive sentiment from agreement
                )

        print(f"  [MEMORY] Recorded agreement for {agent.first_name}")

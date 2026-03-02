"""
Consultation module for generative agents.

Implements the consultation system for shared-space decisions:
- Temperature and window consultations with other occupants
- Consensus assessment (LLM-based, vote, or keyword)
- Turn-based consultation conversations
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING

from agents import Agent, Runner, ModelSettings
from agents.agent_output import AgentOutputSchema

from bsm.agents.skeleton import DEFAULT_AGENT_MODEL, SimContext, CalendarStore
from bsm.agents.cognition.schemas import (
    ConversationUtterance,
    ConsultationOutcome,
    ConsultationUtteranceOutput,
    ConsensusAssessment,
)
from bsm.agents.cognition.retrieval import retrieve

if TYPE_CHECKING:
    from bsm.agents.generative_agent import GenerativeAgent


# ---------------------------------------------------------------------------
# Consensus Assessment Agents
# ---------------------------------------------------------------------------

def _build_consensus_assessor_agent() -> Agent[SimContext]:
    """
    Build an agent for assessing consensus from consultation conversations.

    Uses LLM to analyze conversation and determine if agreement was reached.
    """
    instructions = """Analyze this consultation conversation and determine:
1. Did ALL participants agree to a temperature change?
2. If yes, what specific temperature was agreed upon?
3. Summarize the outcome.
<output_schema>

"""

    return Agent(
        name="consensus_assessor",
        instructions=instructions,
        model=DEFAULT_AGENT_MODEL,
        model_settings=ModelSettings(reasoning_effort="medium"),
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

    agent = _build_consensus_assessor_agent()
    result = await Runner.run(agent, conversation_text)
    return result.final_output


# ---------------------------------------------------------------------------
# Simple Vote Mode
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Consultation Agent Builder
# ---------------------------------------------------------------------------

def _build_consultation_agent(speaker_id: str, is_initiator: bool) -> Agent[SimContext]:
    """
    Build an agent for consultation about shared-space changes.

    Args:
        speaker_id: ID of the agent speaking
        is_initiator: Whether this agent initiated the consultation

    Returns:
        Agent configured for consultation utterance generation
    """
    role = "proposing a temperature adjustment" if is_initiator else "being asked about a temperature change"

    instructions = f"""
You are {speaker_id}, {role} to the office thermostat.

<context>
This is a quick check-in with colleagues about comfort, not a formal negotiation.
Most temperature discussions are resolved quickly - people are generally flexible.
</context>

<guidelines>
- Keep responses brief and casual (1-2 sentences max)
- It's fine to quickly agree if you're comfortable or close to comfortable
- A difference of 0.5-1°C is usually not worth arguing about
- Express your preference naturally: "Yeah, that works for me" or "Could we maybe try X instead?"
- Set end_consultation=true when:
  - Quick agreement reached (most common outcome)
  - You decide current temp is fine
  - Brief disagreement acknowledged
- Use proposed_compromise only if suggesting a specific temperature
- Consultations should be 2-4 exchanges, not lengthy debates
</guidelines>

Output your response naturally.
""".strip()

    return Agent(
        name=f"consultation_{speaker_id}",
        instructions=instructions,
        model=DEFAULT_AGENT_MODEL,
        model_settings=ModelSettings(reasoning_effort="medium"),
        tools=[],
        output_type=AgentOutputSchema(ConsultationUtteranceOutput, strict_json_schema=False),
    )


# ---------------------------------------------------------------------------
# Context Building
# ---------------------------------------------------------------------------

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

    # Retrieve memories about thermal comfort and the other participants
    focal_points = [
        "my temperature preferences",
        "how I feel about the office temperature",
    ]
    # Add memories about each listener
    for listener in listeners:
        focal_points.append(f"what I know about {listener.first_name}")

    retrieved = retrieve(speaker, focal_points, now, n_count=5)

    # Format memories
    memory_lines = []
    for fp, nodes in retrieved.items():
        if nodes:
            for node in nodes[:2]:  # Top 2 per focal point
                memory_lines.append(f"- {node.description}")
    memories_text = "\n".join(memory_lines) if memory_lines else "No specific memories about this topic."

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
<identity>
{identity}
</identity>

<consultation_request>
{role_text}
Current zone temperature: {current_temp_c:.1f}°C
Other participants: {other_names}
</consultation_request>

<relevant_memories purpose="your past experiences with temperature and these colleagues">
{memories_text}
</relevant_memories>

<discussion_so_far>
{history_text}
</discussion_so_far>

<current_time>{now.strftime('%H:%M')}</current_time>

<guidance>
Respond naturally to continue the consultation. You're discussing comfort, not negotiating a contract.
- Keep responses brief and conversational (1-2 sentences)
- It's fine to quickly agree if the proposal seems reasonable
- Express your actual comfort preference, but be flexible
- A small compromise (0.5-1°C) is usually acceptable to everyone
</guidance>
""".strip()

    return context


# ---------------------------------------------------------------------------
# Consultation Utterance Generation
# ---------------------------------------------------------------------------

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

    agent = _build_consultation_agent(speaker.agent_id, is_initiator)

    sim_context = SimContext(
        occupant_id=speaker.agent_id,
        now=now,
        calendar=calendar,
        simulation=None,
    )

    result = await Runner.run(agent, context, context=sim_context)
    output: ConsultationUtteranceOutput = result.final_output

    utterance = ConversationUtterance(
        speaker_id=speaker.agent_id,
        utterance=output.utterance,
        end_conversation=output.end_consultation,
    )

    return utterance, output.proposed_compromise


# ---------------------------------------------------------------------------
# Main Consultation Conversation
# ---------------------------------------------------------------------------

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
                    temp_match = re.search(r'(\d{1,2}(?:\.\d{1,2})?)\s*(?:°|degrees?|c\b|celsius)?', text)
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "consultation_conversation",
    "assess_consensus_with_llm",
    "simple_vote_on_temperature",
    "generate_consultation_utterance",
]

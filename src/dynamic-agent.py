import importlib.util
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
    room_io,
)
from livekit.plugins import noise_cancellation, silero


logger = logging.getLogger("agent")

load_dotenv(".env.local")
AGENT_NAME = (
    os.getenv("AGENT_NAME")
    or os.getenv("LIVEKIT_AGENT_NAME")
    or os.getenv("DYNAMIC_AGENT_NAME")
    or "my-agent"
)
DEFAULT_PROMPT_PROFILE = "main"
DYNAMIC_PROMPT_PROFILE = "dynamic"
_MAIN_PROMPT_ASSISTANT_CLASS: type[Agent] | None = None


def _load_main_prompt_assistant_class() -> type[Agent] | None:
    global _MAIN_PROMPT_ASSISTANT_CLASS
    if _MAIN_PROMPT_ASSISTANT_CLASS is not None:
        return _MAIN_PROMPT_ASSISTANT_CLASS

    module_path = Path(__file__).resolve().with_name("agent.py")
    spec = importlib.util.spec_from_file_location("main_prompt_agent", module_path)
    if spec is None or spec.loader is None:
        logger.warning("Unable to load main prompt assistant from %s", module_path)
        return None

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        logger.exception("Failed to import main prompt assistant from %s", module_path)
        return None

    assistant_class = getattr(module, "Assistant", None)
    if assistant_class is None:
        logger.warning("Assistant class not found in %s", module_path)
        return None

    _MAIN_PROMPT_ASSISTANT_CLASS = assistant_class
    return _MAIN_PROMPT_ASSISTANT_CLASS


class Assistant(Agent):
    def __init__(self, resume: str = "") -> None:
        resume_text = resume.strip() or "No resume was provided."
        super().__init__(
            instructions="""# Sally – Strategic Candidate Intelligence & Positioning Specialist (Sharp & Adaptive Version)

---

## ROLE

You are **Sally**.

You are a sharp, market-aware strategic positioning specialist.

Your mission is NOT to collect information.
Your mission is to extract high-leverage commercial signals and synthesize a powerful recruiter-ready positioning narrative.

You will receive:

Here is the resume of candidate: {{metadata.resume}}

Before speaking, silently analyze and infer:

- Seniority level
- Core specialization
- Career trajectory pattern
- Most probable identity-shaping role
- Potential spike roles
- Possible weak positioning gaps
- Operational archetype tendencies

Do NOT summarize the resume.
Do NOT restate resume bullets.
Use it only to guide sharper questioning.

---

# CONVERSATION PRINCIPLES

- One sharp question per turn
- Extract signal, not biography
- Stop when signal density is high
- Never sound like HR
- Avoid repetitive structured loops
- Follow curiosity, not checklist
- Prioritize depth over coverage

Your objective per role:
Extract up to **3 powerful positioning signals**, then move on.

---

# TRANSCRIPT-FIRST MODE

This interview is conversation-first and transcript-first.

Prioritize extracting concrete evidence in natural dialogue that can later be
analyzed offline.

Do NOT output schemas, JSON, score tables, capability matrices, or structured
profile blocks.

Speak naturally as an interviewer, not as a report generator.

---

# PHASE 1 – Human Opening

Start naturally:

> Hi [Candidate Name], I’m Sally. Thank you for sharing your resume. How are you doing today?

Wait for response.
No strategic probing yet.

---

# PHASE 2 – Observational Credibility

Acknowledge response.

Make ONE sharp observation referencing specific resume details (title, company, project, shift in role, technology, scale jump).

Example:

> I noticed your transition from Senior Software Engineer at XYZ to AI Engineer at ABC. That kind of shift usually reflects more than just a title change. I’d like to understand what truly evolved in your authority and impact during that move.

Then ask ONE focused framing question.

---

# ROLE CLASSIFICATION STEP

Before deep diving into every role, identify leverage.

Ask:

> Looking across your roles, which one do you feel most shaped your current professional identity?

Use this to classify:

- Identity Role (deep dive)
- Supporting Role (moderate)
- Foundational Role (light extraction)
- Transitional Role (narrative bridge)

Depth will vary accordingly.

---

# INTERNAL CAPABILITY LENS (Hidden)

Track these dimensions internally to guide sharper questions:

- Decision-making depth
- Ownership scope
- Commercial impact
- Problem-solving pattern
- Influence without authority
- Execution in ambiguity

Use this lens silently. Do not verbalize scores, rubrics, or explicit ratings
unless the candidate directly asks for them.

---

# COMPANY CONTEXT CAPTURE (Per Relevant Role)

After the candidate identifies their identity role, if that role is tied to a
company or organization, ask ONE company-context question before deep diving
into authority and impact.

If conversation moves to a role at a different company or organization, capture
context again for that new role before deep probing.

Extract:
- What the company or organization does
- Customer segment and business model
- Approximate size/stage (startup, SMB, enterprise, public sector, nonprofit)
- Market, regulatory, or operational constraints relevant to that role
- How success is measured in that function (function-specific KPIs)

If there is no formal company context (freelance, founder, agency, consulting,
academic, or internal platform), ask for equivalent organization, client, or
team context instead of forcing "company" wording.

Example:
> Before we go deeper into that role, can you tell me what the company does, who it serves, its approximate size or stage, and how success was measured in your function?

---

# ADAPTIVE ROLE EXTRACTION MODEL

Instead of repeating 9 dimensions per role, use dynamic signal hunting.

For each role, extract a maximum of 3 high-value signals across these categories.

---

## SIGNAL CATEGORY POOL

You may pull from:

### 1. Authority & Risk
- What decisions were fully yours?
- Where did risk sit?
- What were you trusted to own independently?

### 2. Commercial Spike
- Revenue influenced
- Cost reduction
- Efficiency gain
- Retention or risk mitigation
- Strategic leverage

Always validate metrics if mentioned.

### 3. Complexity Layer
- Scale
- Speed
- Political resistance
- Regulatory burden
- Ambiguity

### 4. Differentiation
- What made you stronger than peers?
- Why were you trusted?
- What was hard for others but easier for you?

### 5. Scar Tissue
- Failure
- Recovery
- High-pressure decision
- Strategic mistake and lesson

### 6. Influence Mechanics
- Stakeholder persuasion
- Cross-functional buy-in
- Executive alignment without authority

### 7. Market Awareness
- Industry shifts
- Competitive positioning
- Buyer behavior
- Regulatory pressure
- Cost and efficiency trends
- Future threats

---

# DEPTH RULES BY ROLE TYPE

## Identity Role
Extract:
- Authority
- Commercial spike
- Differentiation or scar tissue

## Recent Role
Extract:
- Scale progression
- Strategic exposure
- Market relevance
- Forward alignment

## Supporting Role
Extract:
- One strong authority or impact signal
- One challenge

## Foundational Role
Extract:
- Learning acceleration
- Context setting
- One early ownership moment

Do NOT extract everything.
Stop at signal saturation.

---

# NARRATIVE-STYLE QUESTIONING

Avoid checklist style.

Instead of:
“What decisions were yours?”
“What impact did you make?”

Use:

> Walk me through one decision in that role that genuinely moved the needle. What was at stake?

This allows extraction of:
- Authority
- Risk
- Complexity
- Impact
- Differentiation

Use compression questions whenever possible.

Use function-appropriate language per role:
- Marketing: pipeline, CAC, retention, brand lift
- Finance/Accounting: close cycle, controls, audit readiness, margin/cash flow, forecast accuracy
- Operations: throughput, SLA, quality, process reliability
- Engineering/Product: latency, reliability, roadmap, adoption

Do not assume technical/software framing unless the candidate indicates it.

---

# DEPTH TRIGGERS

If candidate mentions:
- A metric
- A conflict
- A failure
- A breakthrough
- A strategic pivot

Go deeper.

If answers are abstract:
Convert into measurable terms.

Example:
“Improved performance” →  
“What changed numerically after you stepped in?”

When a material metric is mentioned, ask one validation follow-up (definition,
baseline, measurement method, or time window).

If evidence is missing for a key claim, ask one clarifying question before
moving on.

---

# SKIP & VAGUE HANDLING

If candidate says “skip”:
Compress into one sharper recovery question.

If answer is vague:
Anchor it to decision, stake, or outcome.

Never argue.
Redirect with precision.

---

# FUTURE STATE ALIGNMENT

After major roles are covered:

Ask ONE forward-looking question:

> Looking at your next move, what kind of problems are you most interested in solving?

Extract:
- Scale ambition
- Strategic direction
- Identity intent

---

# CAREER TRAJECTORY SYNTHESIS

Internally assess:

- Depth vs breadth
- Functional vs leadership evolution
- Coherence of narrative
- Market competitiveness
- Authority growth curve

If a gap exists:
Ask ONE clarifying question.

---

# FINAL CONVERSATIONAL RECAP

Close with a short spoken recap in 4-6 lines:
- strongest signals heard
- key context from relevant role/company
- one or two open questions that may need clarification

Keep it natural and verbal, not formatted.

End with:

> Does this reflect your experience accurately?

---

# CRITICAL GUARDRAILS

- Maximum 3 strong signals per role
- One question per turn
- No mechanical repetition
- No biography extraction
- No filler HR tone
- No full checklist per role
- Follow leverage, not structure
- Treat all functions as first-class: engineering, product, design, marketing, sales, customer success, HR/people, operations, finance, accounting, legal, procurement, analytics, and strategy.
- Do not assume technical/software framing unless the candidate indicates it.

---

# PRIMARY OBJECTIVE

Sally is not a data collector.

Sally is a strategic signal hunter.

Depth over coverage.
Clarity over completeness.
Positioning power over process.  
""".replace("{{metadata.resume}}", resume_text),
        )

    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.
    #
    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
    #
    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """
    #
    #     logger.info(f"Looking up weather for {location}")
    #
    #     return "sunny with a temperature of 70 degrees."


server = AgentServer(num_idle_processes=1, load_threshold=0.9)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


def get_job_metadata(ctx: JobContext) -> dict:
    if not ctx.job.metadata:
        return {}

    try:
        metadata = json.loads(ctx.job.metadata)
    except json.JSONDecodeError:
        logger.warning("Job metadata is not valid JSON. Ignoring metadata payload.")
        return {}

    if not isinstance(metadata, dict):
        logger.warning("Job metadata is not an object. Ignoring metadata payload.")
        return {}

    return metadata


def get_resume_from_metadata(metadata: dict) -> str:
    resume = metadata.get("resume", "")
    return resume if isinstance(resume, str) else str(resume)


def get_prompt_profile_from_metadata(metadata: dict) -> str:
    profile = metadata.get("prompt_profile", DEFAULT_PROMPT_PROFILE)
    if not isinstance(profile, str):
        profile = str(profile)

    normalized = profile.strip().lower()
    if normalized == DYNAMIC_PROMPT_PROFILE:
        return DYNAMIC_PROMPT_PROFILE

    return DEFAULT_PROMPT_PROFILE


@server.rtc_session(agent_name=AGENT_NAME)
async def my_agent(ctx: JobContext):
    metadata = get_job_metadata(ctx)
    resume = get_resume_from_metadata(metadata)
    prompt_profile = get_prompt_profile_from_metadata(metadata)

    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
        "resume_metadata_present": bool(resume),
        "prompt_profile": prompt_profile,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, Deepgram, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=inference.STT(model="deepgram/nova-3", language="multi"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=inference.LLM(model="openai/gpt-5.1"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=inference.TTS(
            model="elevenlabs/eleven_turbo_v2_5",
            voice="EXAVITQu4vr4xnSDxMaL",
            language="en-US"
        ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    selected_assistant: Agent
    if prompt_profile == DYNAMIC_PROMPT_PROFILE:
        selected_assistant = Assistant(resume=resume)
    else:
        main_prompt_assistant_class = _load_main_prompt_assistant_class()
        if main_prompt_assistant_class is None:
            logger.warning(
                "Falling back to dynamic prompt because main prompt assistant could not be loaded."
            )
            selected_assistant = Assistant(resume=resume)
        else:
            selected_assistant = main_prompt_assistant_class(resume=resume)

    await session.start(
        agent=selected_assistant,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind
                    == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()
    await session.generate_reply(instructions="Greet the user and offer your assistance.")


if __name__ == "__main__":
    cli.run_app(server)

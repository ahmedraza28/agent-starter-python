import json
import logging
import os
import sys

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
AGENT_NAME = os.getenv("AGENT_NAME") or os.getenv("LIVEKIT_AGENT_NAME") or "my-agent"
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _is_env_enabled(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUE_VALUES


def _get_turn_detector_mode() -> str:
    mode = os.getenv("TURN_DETECTOR_MODE")
    if mode:
        normalized = mode.strip().lower()
        if normalized in {"off", "none", "disabled"}:
            return "off"
        if normalized in {"english", "en"}:
            return "english"
        if normalized in {"multilingual", "multi"}:
            return "multilingual"
        if normalized == "stt":
            return "stt"
        logger.warning(
            "Unknown TURN_DETECTOR_MODE=%s. Falling back to ENABLE_TURN_DETECTOR behavior.",
            mode,
        )

    return "multilingual" if _is_env_enabled("ENABLE_TURN_DETECTOR", default=False) else "off"


TURN_DETECTOR_MODE = _get_turn_detector_mode()


def _register_turn_detector_plugins_for_download() -> None:
    # Ensure download-files command includes turn-detector model assets in the image.
    if "download-files" not in sys.argv:
        return

    from livekit.plugins.turn_detector.english import EnglishModel
    from livekit.plugins.turn_detector.multilingual import MultilingualModel

    _ = (EnglishModel, MultilingualModel)


_register_turn_detector_plugins_for_download()


class Assistant(Agent):
    def __init__(self, resume: str = "") -> None:
        resume_text = resume.strip() or "No resume was provided."
        super().__init__(
            instructions="""# Sally – Strategic Candidate Intelligence & Commercial Positioning Specialist

---

## ROLE

You are **Sally**, a market-aware strategic talent positioning specialist.  
Your mission is to extract **deep commercial signals** from candidates and produce a **recruiter-ready narrative** that maximizes market value.

You will receive:  
Here is the resume of candidate: {{metadata.resume}}

Before speaking, silently analyze the resume and infer:

- Seniority level  
- Core domain / specialization  
- Career trajectory pattern  
- Possible differentiators vs peers  
- Potential weak positioning areas  
- Likely operational archetype (wartime vs peacetime vs disruptor vs maintainer)  

**Rules:**

- Do **not** repeat resume details  
- Do **not** summarize  
- Use it only to guide sharper questioning

---

## CONVERSATION ARCHITECTURE

The conversation follows a **multi-phase opening** + **multi-role sequential extraction** + **market-aware signal synthesis**.

---

### PHASE 1 – Human Opening & Self-Introduction

- Introduce yourself by name:  
> “Hi Ahmed, I am Sally, thank you for sharing your resume. How are you doing today?”  

- Establish **light professional rapport**  
- **STOP** after asking one question; wait for response  
- Do not analyze or probe strategically yet

---

### PHASE 2 – Observational Credibility + Resume Framing

- Acknowledge their reply  
- Make **one sharp observation referencing exact resume details** (titles, projects, technologies, achievements)  
- Frame the purpose: **dig into authority, commercial impact, differentiators**

Example:  
> “I see from your resume that you led AI-driven agent deployment projects at Nixortech Solutions. That work usually involves decisions and complexity not visible on paper.  
> I’d like to explore your real decision authority, commercial impact, and unique value so your positioning reflects your true market strength.”

**Do not ask multiple questions here. Keep it tight.**

---

### PHASE 3 – Candidate-Sourced Company Context

Ask candidate to provide **company context referencing resume details**:

1. Company size / scale  
2. Industry / business model  
3. Regulatory or operational complexity  
4. Market positioning  

Example:  
> “Can you describe your company [from your resume] — its industry, approximate size, primary business focus, and competitive positioning?”  

Take answers **one at a time**.

---

## CORE SIGNAL EXTRACTION – Multi-Layer Sequential Flow

Repeat the following for **each role** listed on the candidate’s resume:

### 1. Operational Archetype

Classify environment:

- Wartime operator: high urgency, crisis-driven  
- Peacetime optimizer: efficiency-focused  
- Disruptor: innovation, risk-taking  
- Maintainer: process/system focus  

Example:  
> “Would you describe your environment as driving change under pressure, maintaining stability, or actively innovating?”

---

### 2. Authority & Ownership

Ask **decision-focused questions**:  

- What decisions were fully yours?  
- Where did risk sit?  
- What initiatives were you trusted to own independently?  

Explore **one authority dimension per turn**.

---

### 3. Complexity & Challenge

Identify dominant difficulty:  

- Technical scale / problem size  
- Speed or time pressure  
- Stakeholder politics / influence  
- Regulatory / compliance burden  
- Ambiguity  

Ask **one dimension per turn**, then move to impact.

---

### 4. Commercial Impact & Economic Spike

Convert effort → outcome:  

- Revenue generated or influenced  
- Cost reduction  
- Efficiency gains / time saved  
- Risk mitigation / retention improvements  

Follow-up on metrics if claimed:  
> “You mentioned improving client satisfaction by 30%. How was this measured?”  

Crystallize **one spike per role**.

---

### 5. Differentiation

Ask:  
> “What made you stronger than others at your level in that situation?”  

---

### 6. Scar Tissue & Resilience

Extract failure or recovery stories:  

- Toughest decision  
- Mistakes & lessons  
- Recovery under pressure  

Example:  
> “Can you describe a time a project didn’t go as planned, and how you recovered?”

---

### 7. Influence Mapping

Ask how they achieved outcomes without direct authority:

- Stakeholder persuasion  
- Cross-team buy-in  
- Executive alignment  

---

### 8. Education & Certifications

- Explore **only if relevant for positioning, domain shift, or analytical capability**  
- Example:  
> “I see you studied computer science—how did that prepare you for AI solution delivery?”  
> “Do you hold certifications that reinforce your technical authority or leadership credibility?”

---

### 9. Future-State / Target Alignment

Ask **one reflective question**:  

- Target role / organization type  
- Problems they want to solve next  
- Desired scale / responsibility  

Example:  
> “Looking at your next role, what challenges or problems would you like to tackle?”

---

### 10. Career Trajectory Synthesis

After all roles are covered:

- Assess **depth vs breadth, technical vs leadership, coherent narrative**  
- Connect **past roles → current expertise → future target**  
- Ask one clarification if gaps appear

---

### HANDLING VAGUE OR SKIPPED ANSWERS

- If candidate says **“skip”** → compress into **one sharp recovery question**  
- Abstract answers → convert into **concrete outcomes**  

Example:  
> “Improved speed” → “What changed in delivery timelines after you joined?”

---

### PACING & GUARDRAILS

- Stop after **3 high-value signals per role** → pivot to next role  
- Extract **depth over breadth**, one strong commercial signal at a time

---

### MARKET-AWARE SIGNALS

- Ask candidate to reflect on **industry trends, competitor positioning, emerging technologies**  
- Example:  
> “Considering the rapid evolution in AI voice agents, how have you ensured your solutions remain ahead of competitors?”  

---

### FINAL POSITIONING SYNTHESIS

Produce **recruiter-ready narrative**:

1. Core professional identity  
2. Environment / operational archetype  
3. Authority & ownership  
4. Differentiator / spike  
5. Market value angle  
6. Multi-role, trend-aware context  

Confirm with candidate:  
> “Does this align with how you want to be positioned?”

---

### CRITICAL RULES

- Never sound like HR or form-filler  
- Ask **only one question per turn** (max two if clarification needed)  
- Always convert duties into **commercial signals**  
- Extract **competitive advantage**  
- Maintain authority without interrogation tone  
- Reference **exact resume details** in observations & questions  
- Apply pacing guardrails per role  
- Include **archetype, spike, resilience, influence, education, certifications, future alignment, market awareness**  
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


server = AgentServer(num_idle_processes=0)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


def get_resume_from_job_metadata(ctx: JobContext) -> str:
    if not ctx.job.metadata:
        return ""

    try:
        metadata = json.loads(ctx.job.metadata)
    except json.JSONDecodeError:
        logger.warning("Job metadata is not valid JSON. Ignoring resume payload.")
        return ""

    if not isinstance(metadata, dict):
        logger.warning("Job metadata is not an object. Ignoring resume payload.")
        return ""

    resume = metadata.get("resume", "")
    return resume if isinstance(resume, str) else str(resume)


def _build_turn_detection(mode: str):
    if mode == "english":
        try:
            from livekit.plugins.turn_detector.english import EnglishModel

            return EnglishModel()
        except RuntimeError:
            logger.exception(
                "English turn detector model files unavailable. Falling back to STT turn detection."
            )
            return "stt"
    if mode == "multilingual":
        try:
            from livekit.plugins.turn_detector.multilingual import MultilingualModel

            return MultilingualModel()
        except RuntimeError:
            logger.exception(
                "Multilingual turn detector model files unavailable. Falling back to English turn detection."
            )
            return _build_turn_detection("english")
    if mode == "stt":
        return "stt"
    return None


@server.rtc_session(agent_name=AGENT_NAME)
async def my_agent(ctx: JobContext):
    resume = get_resume_from_job_metadata(ctx)

    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
        "resume_metadata_present": bool(resume),
        "turn_detector_mode": TURN_DETECTOR_MODE,
    }

    # Set up a voice AI pipeline using OpenAI, ElevenLabs, and Deepgram.
    session_kwargs = {
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        "stt": inference.STT(model="deepgram/nova-3", language="multi"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        "llm": inference.LLM(model="openai/gpt-5.1"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        "tts": inference.TTS(
            model="elevenlabs/eleven_turbo_v2_5",
            voice="EXAVITQu4vr4xnSDxMaL",
            language="en-US",
        ),
        # VAD is used to determine when the user is speaking.
        "vad": ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        "preemptive_generation": True,
    }
    turn_detection = _build_turn_detection(TURN_DETECTOR_MODE)
    if turn_detection is not None:
        session_kwargs["turn_detection"] = turn_detection

    session = AgentSession(**session_kwargs)

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
    await session.start(
        agent=Assistant(resume=resume),
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

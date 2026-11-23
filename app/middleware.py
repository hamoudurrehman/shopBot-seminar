"""The whole safety and reliability stack, as one middleware list.

Same hooks as notebooks 02-04, with one difference: the LangGraph server runs agents
asynchronously, so our own wrap_* hooks and the prompt are `async def` and `await` the handler.
"""
import re
from datetime import datetime
from typing import NotRequired
from uuid import uuid4

from langchain.agents import AgentState
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware, ModelCallLimitMiddleware, ModelFallbackMiddleware, PIIMiddleware,
    SummarizationMiddleware, ToolCallLimitMiddleware, ToolRetryMiddleware,
    before_model, dynamic_prompt, wrap_model_call, wrap_tool_call,
)
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage

from app.context import FALLBACK_MODEL, MODEL
from app.tools import is_urgent

SYSTEM_PROMPT = """You are a care assistant for patients with long-term conditions.
You help patients log readings and symptoms and understand their care plan.

Never do these:
- Never diagnose.
- Never give or change medicine doses. Send dose questions to the doctor or pharmacist.
- If something sounds urgent, tell the patient to contact their care team.

Rules:
- When the patient tells you a NEW reading, log it with log_reading, even a very high one.
  Never re-log readings you looked up. Never use escalate_to_nurse for a reading:
  log_reading checks thresholds and alerts the nurse itself.
- Call escalate_to_nurse for worrying symptoms (fainting, chest tightness) or when the patient asks for a nurse.
- For how-to questions, or what the doctor noted or planned at past check-ins,
  use search_documents and cite the [source].
- If the documents don't contain the answer, say "That's not in my documents." Don't guess.
- Before booking, ask for date of birth and call verify_identity.
- Use remember to save lasting facts (ongoing problems, preferences). If you remember an ongoing problem, ask how it is going."""


# ── notebook 03: dose guard ──
DOSE_QUESTION = re.compile(r"\b(dose|dosage|doses|\d+\s?mg|double|stop taking)\b", re.I)   # CUSTOM: safety policy
DOSE_REFUSAL = ("I can't advise on medicine doses. Please ask your doctor or pharmacist. "
                "I've noted your question for your care team.")


@before_model(can_jump_to=["end"])
def dose_guard(state, runtime):
    last = state["messages"][-1]
    if isinstance(last, HumanMessage) and DOSE_QUESTION.search(last.content):
        return {"messages": [AIMessage(DOSE_REFUSAL)], "jump_to": "end"}
    return None


# ── notebook 03: urgent-reading guard ──
# The interrupt lives inside log_reading, so it only fires if the model CALLS log_reading.
# The model can skip it ("I've already logged that"). This guard makes the call itself, in code.
BP_IN_TEXT = re.compile(r"\b(\d{2,3})\s*/\s*(\d{2,3})\b")                           # CUSTOM: e.g. 190/120
GLUCOSE_IN_TEXT = re.compile(r"\b(?:glucose|sugar)\b\D{0,20}?(\d{2,3})\b", re.I)    # CUSTOM: e.g. sugar is 320


def find_reading(text: str):
    """Return (kind, value) if the message contains a reading, else None."""
    bp = BP_IN_TEXT.search(text)
    if bp:
        return "blood_pressure", f"{bp.group(1)}/{bp.group(2)}"
    glucose = GLUCOSE_IN_TEXT.search(text)
    if glucose:
        return "glucose", glucose.group(1)
    return None


@before_model(can_jump_to=["tools"])
def urgent_reading_guard(state, runtime):
    last = state["messages"][-1]
    if runtime.context.role != "patient" or not isinstance(last, HumanMessage):
        return None
    reading = find_reading(last.text)
    if not reading:
        return None
    kind, value = reading
    if not is_urgent(runtime.context.patient_token, kind, value):
        return None   # normal readings: the model logs them as usual
    # Write the tool call ourselves and skip the model: log_reading runs, and its interrupt() pauses for a nurse
    call = {"name": "log_reading", "args": {"kind": kind, "value": value}, "id": f"guard-{uuid4().hex[:8]}"}
    return {"messages": [AIMessage(content="", tool_calls=[call])], "jump_to": "tools"}


# ── notebook 02: PII redaction + audit ──
def pii_middleware():
    # CUSTOM: our identifier formats. The redaction itself is LangChain's.
    return [PIIMiddleware(name, detector=pattern, strategy="redact",
                          apply_to_input=True, apply_to_output=True, apply_to_tool_results=True)
            for name, pattern in [("mrn", r"MRN-\d{6}"), ("phone", r"\b555-\d{4}\b")]]


AUDIT_LOG = []   # CUSTOM: in production, write to an append-only store


@wrap_tool_call
async def audit_log(request, handler):
    ctx = request.runtime.context
    AUDIT_LOG.append({"time": datetime.now().isoformat(timespec="seconds"), "user": ctx.user_id,
                      "role": ctx.role, "patient": ctx.patient_token, "tool": request.tool_call["name"]})
    return await handler(request)


# ── notebook 02: dynamic tools, model and prompt ──
class CareState(AgentState):
    identity_verified: NotRequired[bool]


DOCTOR_ONLY = {"patients_needing_attention"}               # CUSTOM: access policy
NEEDS_IDENTITY = {"list_free_slots", "book_appointment"}   # CUSTOM: access policy
_strong_model = None


@wrap_model_call(state_schema=CareState)
async def pick_tools_and_model(request, handler):
    global _strong_model
    role = request.runtime.context.role
    tools = request.tools
    if role != "doctor":
        tools = [t for t in tools if t.name not in DOCTOR_ONLY]
    if not request.state.get("identity_verified"):
        tools = [t for t in tools if t.name not in NEEDS_IDENTITY]
    model = request.model
    if role == "doctor":
        _strong_model = _strong_model or init_chat_model(FALLBACK_MODEL)
        model = _strong_model
    return await handler(request.override(tools=tools, model=model))


@dynamic_prompt
async def care_prompt(request):
    """Prompt by role and language, plus what we remember about this patient (notebook 06)."""
    ctx = request.runtime.context
    tone = ("You are talking to a doctor. Be brief and clinical."
            if ctx.role == "doctor" else "You are talking to a patient. Use plain, warm language.")
    store = request.runtime.store
    items = await store.asearch(("memories", ctx.patient_token), limit=10) if store else []
    facts = "\n".join(f"- {i.value['fact']}" for i in items) or "- nothing yet"
    return (f"{SYSTEM_PROMPT}\n{tone}\nAlways reply in {ctx.language}.\n\n"
            f"What you remember about this patient:\n{facts}")


def build_middleware():
    """Runs top to bottom on the way in. Redaction sits above summarization on purpose."""
    return [
        dose_guard,
        urgent_reading_guard,
        *pii_middleware(),
        SummarizationMiddleware(MODEL, trigger=("messages", 40), keep=("messages", 20)),
        care_prompt,
        pick_tools_and_model,
        ModelFallbackMiddleware(FALLBACK_MODEL),
        ModelCallLimitMiddleware(run_limit=10),
        ToolCallLimitMiddleware(run_limit=10),
        ToolRetryMiddleware(max_retries=2),
        audit_log,
        HumanInTheLoopMiddleware(interrupt_on={"escalate_to_nurse": True, "forget_me": True}),
    ]

"""
Finished pieces from earlier notebooks.

Each notebook writes ONE new concept out in full, then that code is copied here
so the next notebook can just `from common import ...`.
Sections are labelled with the notebook that built them.
"""
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NotRequired

from langchain.agents import AgentState
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware, ModelCallLimitMiddleware, ModelFallbackMiddleware,
    PIIMiddleware, SummarizationMiddleware, ToolCallLimitMiddleware, ToolRetryMiddleware,
    before_model, dynamic_prompt, wrap_model_call, wrap_tool_call,
)
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.types import Command, interrupt
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")   # so importing common always has the API keys

MODEL = "openai:gpt-4o-mini"
FALLBACK_MODEL = "openai:gpt-4.1-mini"
DATA = Path(__file__).resolve().parent.parent / "data"
PATIENTS = json.loads((DATA / "patients.json").read_text())   # SYNTHETIC patients only


# ───────────────────────── helpers (used by every notebook) ─────────────────────────

def ask(agent, text, thread="demo", context=None):
    """Send one message and print the reply (or the pause, if a human must step in)."""
    print(f"🧑 {text}")
    config = {"configurable": {"thread_id": thread}}
    result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config, context=context or CareContext())
    _print_result(result)


def resume(agent, decision, thread="demo", context=None):
    """Answer a pause with a human decision, then let the agent carry on."""
    config = {"configurable": {"thread_id": thread}}
    result = agent.invoke(Command(resume=decision), config, context=context or CareContext())
    _print_result(result)


def _print_result(result):
    if "__interrupt__" in result:
        print("⏸️  PAUSED, waiting for a human:", result["__interrupt__"][0].value)
    else:
        print(f"🤖 {result['messages'][-1].content}\n")


def show_state(agent, thread="demo"):
    """Receipt: print every message the checkpointer saved for this thread."""
    state = agent.get_state({"configurable": {"thread_id": thread}})
    for m in state.values["messages"]:
        calls = [f"{c['name']}({c['args']})" for c in getattr(m, "tool_calls", [])]
        print(f"[{m.type:>5}] {m.content[:100]!r} {' '.join(calls)}")


def chat(agent, thread="live", context=None):
    """Type live instead of running the scripted ask() cells. Type 'quit' to stop."""
    while (text := input("You: ")) != "quit":
        ask(agent, text, thread, context)


def talk(agent, thread="live", context=None):
    """Live chat for the exercises: also prints every tool call, so you see what the model sent.
    Press Enter on an empty line (or type 'quit') to stop."""
    config = {"configurable": {"thread_id": thread}}
    while (text := input("You: ").strip()) not in ("", "quit"):
        print(f"🧑 {text}")
        result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config,
                              context=context or CareContext())
        last_human = max(i for i, m in enumerate(result["messages"]) if m.type == "human")
        for m in result["messages"][last_human + 1:]:
            for c in getattr(m, "tool_calls", None) or []:
                args = ", ".join(f"{k}={v!r}" for k, v in c["args"].items())
                print(f"   🔧 {c['name']}({args})")
        _print_result(result)


# ───────────────────────── 01_agent ─────────────────────────

@dataclass
class CareContext:
    """Who is talking. Set by the app, never by the chat, so the model can't change it."""
    user_id: str = "sara"
    role: str = "patient"          # "patient" or "doctor"
    patient_token: str = "p-001"   # which patient's data this conversation may touch
    language: str = "English"


SYSTEM_PROMPT = """You are a care assistant for patients with long-term conditions.
You help patients log readings and symptoms and understand their care plan.

Never do these:
- Never diagnose.
- Never give or change medicine doses. Send dose questions to the doctor or pharmacist.
- If something sounds urgent, tell the patient to contact their care team."""

# 03 (safety) adds the escalation rules
CARE_PROMPT = SYSTEM_PROMPT + """
- When the patient tells you a NEW reading, log it with log_reading, even a very high one.
  Never re-log readings you looked up. Never use escalate_to_nurse for a reading:
  log_reading checks thresholds and alerts the nurse itself.
- Call escalate_to_nurse for worrying symptoms (fainting, chest tightness) or when the patient asks for a nurse."""

# A fake database so the demo has some history to show
READINGS = [
    {"patient": "p-001", "kind": "blood_pressure", "value": "142/90", "time": "2026-11-18 08:00"},
    {"patient": "p-001", "kind": "blood_pressure", "value": "136/86", "time": "2026-11-20 08:00"},
    {"patient": "p-002", "kind": "glucose", "value": "138", "time": "2026-11-20 07:30"},
    {"patient": "p-002", "kind": "glucose", "value": "320", "time": "2026-11-21 07:30"},
]
SYMPTOMS = []


@tool
def get_care_plan(runtime: ToolRuntime[CareContext]) -> str:
    """Get the patient's care plan: condition, target and medicines."""
    p = PATIENTS[runtime.context.patient_token]
    return f"Condition: {p['condition']}. Target: {p['target']}. Medicines: {', '.join(p['medications'])}."


@tool
def log_symptom(symptom: str, runtime: ToolRuntime[CareContext]) -> str:
    """Log a symptom the patient reports, e.g. 'headache since this morning'."""
    SYMPTOMS.append({"patient": runtime.context.patient_token, "symptom": symptom, "time": _now()})
    return f"Logged symptom: {symptom}"


@tool
def get_recent_readings(runtime: ToolRuntime[CareContext], limit: int = 5) -> str:
    """Get the patient's most recent home readings."""
    mine = [r for r in READINGS if r["patient"] == runtime.context.patient_token][-limit:]
    return "\n".join(f"{r['time']}  {r['kind']}: {r['value']}" for r in mine) or "No readings yet."


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


# ───────────────────────── 02_middleware ─────────────────────────

APPOINTMENTS = []
FREE_SLOTS = ["2026-11-25 09:00", "2026-11-25 14:30", "2026-11-27 11:00"]


@tool
def verify_identity(date_of_birth: str, runtime: ToolRuntime[CareContext]) -> Command:
    """Verify the patient's identity with their date of birth (YYYY-MM-DD). Required before booking."""
    ok = PATIENTS[runtime.context.patient_token]["date_of_birth"] == date_of_birth
    reply = "Identity verified." if ok else "That date of birth doesn't match our records."
    # A tool can update agent state, not just return text:
    return Command(update={"identity_verified": ok,
                           "messages": [ToolMessage(reply, tool_call_id=runtime.tool_call_id)]})


@tool
def list_free_slots() -> str:
    """List free appointment slots with the patient's doctor."""
    return "\n".join(FREE_SLOTS)


@tool
def book_appointment(slot: str, runtime: ToolRuntime[CareContext]) -> str:
    """Book one of the free slots (exact text from list_free_slots)."""
    if slot not in FREE_SLOTS:
        return f"{slot} is not free. Free slots: {', '.join(FREE_SLOTS)}"
    FREE_SLOTS.remove(slot)
    APPOINTMENTS.append({"patient": runtime.context.patient_token, "slot": slot})
    return f"Booked {slot}."


@tool
def patients_needing_attention() -> str:
    """For doctors: list patients whose latest reading breached their threshold."""
    latest = {r["patient"]: r for r in READINGS}
    flagged = [f"{pid} ({PATIENTS[pid]['condition']}): {r['kind']} {r['value']} at {r['time']}"
               for pid, r in latest.items() if is_urgent(pid, r["kind"], r["value"])]
    return "\n".join(flagged) or "No patients above threshold right now."


class CareState(AgentState):
    identity_verified: NotRequired[bool]


DOCTOR_ONLY = {"patients_needing_attention"}               # CUSTOM: access policy
NEEDS_IDENTITY = {"list_free_slots", "book_appointment"}   # CUSTOM: access policy
strong_model = init_chat_model(FALLBACK_MODEL)


@wrap_model_call(state_schema=CareState)
def pick_tools_and_model(request, handler):
    """Rebuild the tool list (and model) on every call, from who is talking and what's verified."""
    role = request.runtime.context.role
    tools = request.tools
    if role != "doctor":
        tools = [t for t in tools if t.name not in DOCTOR_ONLY]
    if not request.state.get("identity_verified"):
        tools = [t for t in tools if t.name not in NEEDS_IDENTITY]
    model = strong_model if role == "doctor" else request.model
    return handler(request.override(tools=tools, model=model))


def make_care_prompt(base_prompt):
    """02's @dynamic_prompt, for any base prompt (later notebooks pass in a longer one)."""
    @dynamic_prompt
    def care_prompt(request):
        """The system prompt is written fresh on every call, from the runtime context."""
        ctx = request.runtime.context
        tone = ("You are talking to a doctor. Be brief and clinical."
                if ctx.role == "doctor" else "You are talking to a patient. Use plain, warm language.")
        return (f"{base_prompt}\n- Before booking, ask for date of birth and call verify_identity.\n"
                f"{tone}\nAlways reply in {ctx.language}.")
    return care_prompt


def pii_middleware():
    # CUSTOM: our identifier formats. The redaction itself is LangChain's PIIMiddleware.
    return [
        PIIMiddleware("mrn", detector=r"MRN-\d{6}", strategy="redact",
                      apply_to_input=True, apply_to_output=True, apply_to_tool_results=True),
        PIIMiddleware("phone", detector=r"\b555-\d{4}\b", strategy="redact",
                      apply_to_input=True, apply_to_output=True, apply_to_tool_results=True),
    ]


AUDIT_LOG = []


@wrap_tool_call
def audit_log(request, handler):
    """Record who used which tool on which patient, and when."""
    ctx = request.runtime.context
    AUDIT_LOG.append({"time": _now(), "user": ctx.user_id, "role": ctx.role,
                      "patient": ctx.patient_token, "tool": request.tool_call["name"]})
    return handler(request)


# ───────────────────────── 03_safety ─────────────────────────

def is_urgent(patient_token: str, kind: str, value: str) -> bool:
    # CUSTOM: clinical thresholds are policy, set per patient by a doctor
    limits = PATIENTS[patient_token]["thresholds"]
    numbers = [int(n) for n in re.findall(r"\d+", value)]
    if kind == "blood_pressure" and len(numbers) >= 2:
        return numbers[0] >= limits.get("systolic", 999) or numbers[1] >= limits.get("diastolic", 999)
    if kind == "glucose" and numbers:
        return numbers[0] >= limits.get("glucose_high", 9999) or numbers[0] <= limits.get("glucose_low", 0)
    return False


@tool
def log_reading(kind: str, value: str, runtime: ToolRuntime[CareContext]) -> str:
    """Log a home reading. kind is 'blood_pressure' (value like '128/82') or 'glucose' (value in mg/dL like '140')."""
    patient = runtime.context.patient_token
    note = ""
    if is_urgent(patient, kind, value):
        # Code decides, not the model: interrupt() pauses the whole agent until a nurse answers
        nurse_reply = interrupt({"alert": "reading above threshold", "patient": patient, "kind": kind, "value": value})
        note = f" URGENT, reviewed by nurse: {nurse_reply}"
    READINGS.append({"patient": patient, "kind": kind, "value": value, "time": _now()})
    return f"Logged {kind} {value}.{note}"


@tool
def escalate_to_nurse(reason: str, runtime: ToolRuntime[CareContext]) -> str:
    """Alert the nurse about a worrying SYMPTOM. Never use this for readings: log_reading alerts the nurse itself."""
    return f"Nurse alerted about {runtime.context.patient_token}: {reason}"


DOSE_QUESTION = re.compile(r"\b(dose|dosage|doses|\d+\s?mg|double|stop taking)\b", re.I)   # CUSTOM: safety policy
DOSE_REFUSAL = ("I can't advise on medicine doses. Please ask your doctor or pharmacist. "
                "I've noted your question for your care team.")


@before_model(can_jump_to=["end"])
def dose_guard(state, runtime):
    """Refuse dose questions with a fixed reply, before the model ever sees them."""
    last = state["messages"][-1]
    if isinstance(last, HumanMessage) and DOSE_QUESTION.search(last.content):
        return {"messages": [AIMessage(DOSE_REFUSAL)], "jump_to": "end"}
    return None


# ───────────────────────── 04_rag ─────────────────────────

def build_leaflet_search():
    """Index the leaflets (public) and check-in notes (one patient each), return a search tool."""
    docs = [Document(p.read_text(), metadata={"source": p.name, "scope": "public"})
            for p in sorted((DATA / "leaflets").glob("*.md"))]
    docs += [Document(p.read_text(), metadata={"source": p.name, "scope": "private", "patient_token": p.stem})
             for p in sorted((DATA / "notes").glob("*.md"))]
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    vector_store = InMemoryVectorStore.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-small"))

    @tool(response_format="content_and_artifact")
    def search_documents(query: str, runtime: ToolRuntime[CareContext]):
        """Search approved patient leaflets and this patient's own check-in notes."""
        me = runtime.context.patient_token

        def allowed(doc):   # the filter comes from context, never from the model
            return doc.metadata["scope"] == "public" or doc.metadata.get("patient_token") == me

        found = vector_store.similarity_search(query, k=3, filter=allowed)
        text = "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in found)
        return text or "Nothing found in the documents.", [d.metadata for d in found]

    return search_documents


RAG_PROMPT = CARE_PROMPT + """
- For how-to questions, or what the doctor noted or planned at past check-ins,
  use search_documents and cite the [source].
- If the documents don't contain the answer, say "That's not in my documents." Don't guess."""

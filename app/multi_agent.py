"""Notebook 05: safety node -> supervisor -> specialist agents. langgraph.json serves `care_team`."""
import re

from langchain.agents import create_agent
from langchain.messages import AIMessage
from langchain.tools import ToolRuntime, tool
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from app import tools as t
from app.context import MODEL, CareContext
from app.middleware import SYSTEM_PROMPT, dose_guard
from app.rag import search_documents

records_agent = create_agent(
    MODEL, tools=[t.get_care_plan, t.log_reading, t.log_symptom, t.get_recent_readings, search_documents,
                  t.patients_needing_attention],
    system_prompt=SYSTEM_PROMPT + "\nYou handle care plans, readings, symptoms and documents.",
    context_schema=CareContext)

booking_agent = create_agent(
    MODEL, tools=[t.list_free_slots, t.book_appointment],
    system_prompt="You book appointments. If no exact slot is given, list the free slots.",
    context_schema=CareContext)


class PreVisitBrief(BaseModel):
    condition: str
    latest_readings: list[str]
    trend: str = Field(description="improving, stable or worsening")
    concerns: list[str]
    questions_for_visit: list[str]


brief_agent = create_agent(MODEL, tools=[t.get_care_plan, t.get_recent_readings, search_documents],
                           system_prompt="Prepare a factual pre-visit brief for a doctor. No diagnosis.",
                           response_format=PreVisitBrief, context_schema=CareContext)


def _run(agent, request, runtime):
    result = agent.invoke({"messages": [{"role": "user", "content": request}]}, context=runtime.context)
    return result["messages"][-1].content


@tool
def ask_records_agent(request: str, runtime: ToolRuntime[CareContext]) -> str:
    """Delegate anything about care plans, readings, symptoms, leaflets or check-in notes."""
    return _run(records_agent, request, runtime)


@tool
def ask_booking_agent(request: str, runtime: ToolRuntime[CareContext]) -> str:
    """Delegate anything about appointment slots and booking."""
    return _run(booking_agent, request, runtime)


@tool
def write_previsit_brief(runtime: ToolRuntime[CareContext]) -> str:
    """For doctors: a structured brief of this patient before their visit."""
    result = brief_agent.invoke({"messages": [{"role": "user", "content": "Brief me on this patient."}]},
                                context=runtime.context)
    return result["structured_response"].model_dump_json(indent=2)


SUPERVISOR_PROMPT = """You are the front desk of a care team. You know NOTHING about the patient
or our documents yourself, so never answer from your own knowledge.
For every request, call a specialist and pass on the patient's words exactly, including any numbers:
- ask_records_agent: care plans, targets, medicines, readings (including new ones to log), symptoms,
  how-to questions and check-in notes
- ask_booking_agent: appointment slots and booking
- write_previsit_brief: a doctor's brief before a visit
Then relay the specialist's answer. Never give or change medicine doses."""

supervisor = create_agent(
    MODEL, tools=[ask_records_agent, ask_booking_agent, write_previsit_brief],
    system_prompt=SUPERVISOR_PROMPT,
    middleware=[dose_guard],   # same code-level guard as notebook 03
    context_schema=CareContext)

EMERGENCY = re.compile(r"chest pain|can'?t breathe|unconscious|stroke|suicid", re.I)   # CUSTOM: triage policy
EMERGENCY_REPLY = "This could be an emergency. Call emergency services now. I have alerted your care team."


def safety(state: MessagesState):
    if EMERGENCY.search(state["messages"][-1].content):
        return {"messages": [AIMessage(EMERGENCY_REPLY)]}
    return {}


def after_safety(state: MessagesState):
    return END if isinstance(state["messages"][-1], AIMessage) else "supervisor"


def build_care_team(checkpointer=None):
    graph = StateGraph(MessagesState, context_schema=CareContext)
    graph.add_node("safety", safety)
    graph.add_node("supervisor", supervisor)
    graph.add_edge(START, "safety")
    graph.add_conditional_edges("safety", after_safety, ["supervisor", END])
    graph.add_edge("supervisor", END)
    return graph.compile(checkpointer=checkpointer)


care_team = build_care_team()

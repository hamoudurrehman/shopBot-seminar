"""Everything the UI needs from the agent server, in one place.

The UI has NO agent logic. It talks to `langgraph dev` over HTTP with the LangGraph SDK,
and these few calls are the whole frontend <-> backend contract:

    threads.create / threads.search   -> one saved conversation per user
    runs.stream(..., context=...)     -> send a message, watch each step happen
    runs.stream(..., command=resume)  -> continue a paused run (nurse or patient decision)
    threads.get_state                 -> messages + pending interrupts
    store.search_items                -> long-term memories
"""
import functools
import json
import os

import httpx
from langgraph_sdk import get_sync_client

SERVER_URL = os.getenv("LANGGRAPH_URL", "http://127.0.0.1:2024")
STUDIO_URL = f"https://smith.langchain.com/studio/?baseUrl={SERVER_URL}"
GRAPH = "agent"   # the graph name from langgraph.json

client = get_sync_client(url=SERVER_URL)


def retry_once(fn):
    """The dev server closes idle connections, so a reused one can fail. Safe for read-only calls."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except httpx.TransportError:
            return fn(*args, **kwargs)
    return wrapper


def server_ok() -> bool:
    try:
        client.assistants.search(limit=1)
        return True
    except Exception:
        return False


# ── threads: one saved conversation per user ──

@retry_once
def latest_thread(user_id: str) -> str | None:
    found = client.threads.search(metadata={"user_id": user_id}, limit=1,
                                  sort_by="updated_at", sort_order="desc")
    return found[0]["thread_id"] if found else None


def new_thread(context: dict, name: str) -> str:
    # We store who the thread belongs to, so the nurse inbox knows which patient it is
    return client.threads.create(metadata={**context, "name": name})["thread_id"]


@retry_once
def get_state(thread_id: str) -> dict:
    return client.threads.get_state(thread_id)


@retry_once
def thread_status(thread_id: str) -> str:
    return client.threads.get(thread_id)["status"]   # idle | busy | interrupted | error


# ── runs: send a message, or resume a paused run ──

def stream_message(thread_id: str, text: str, context: dict):
    """Yield one (node, update) pair per step: middleware, model, tools, interrupts."""
    yield from _stream(thread_id, context, input={"messages": [{"role": "user", "content": text}]})


def stream_resume(thread_id: str, resume, context: dict):
    yield from _stream(thread_id, context, command={"resume": resume})


def _stream(thread_id, context, **kwargs):
    for attempt in range(2):
        started = False
        try:
            for part in client.runs.stream(thread_id, GRAPH, context=context, stream_mode="updates", **kwargs):
                started = True
                if part.event == "updates":
                    yield from part.data.items()
                elif part.event == "error":
                    raise RuntimeError(part.data)
            return
        except httpx.TransportError:
            if started or attempt:   # only retry if the request never got through
                raise


# ── interrupts: what is the run waiting for? ──

def pending_interrupts(state: dict) -> list[dict]:
    return state.get("interrupts") or []


def describe_interrupt(value: dict) -> dict:
    """Turn an interrupt payload into something a person can act on.

    Two kinds exist in our agent:
      - interrupt() inside log_reading  -> {"alert", "patient", "kind", "value"}   resume with any text
      - HumanInTheLoopMiddleware        -> {"action_requests": [...]}             resume with decisions
    """
    if "action_requests" in value:
        action = value["action_requests"][0]
        return {"kind": "approval", "tool": action["name"], "args": action["args"],
                "count": len(value["action_requests"])}
    return {"kind": "reading", "reading": value.get("kind", ""), "value": value.get("value", ""),
            "patient": value.get("patient", "")}


def decisions(approve: bool, count: int = 1, message: str = "") -> dict:
    """Resume value for HumanInTheLoopMiddleware: one decision per paused tool call."""
    one = {"type": "approve"} if approve else {"type": "reject", "message": message or "Declined."}
    return {"decisions": [one] * count}


@retry_once
def nurse_inbox() -> list[dict]:
    """Every patient conversation paused on a nurse decision."""
    items = []
    for thread in client.threads.search(status="interrupted", limit=50, sort_by="updated_at", sort_order="desc"):
        meta = thread.get("metadata") or {}
        if meta.get("role") != "patient":
            continue
        for task_interrupts in (thread.get("interrupts") or {}).values():
            for intr in task_interrupts:
                info = describe_interrupt(intr["value"])
                if info.get("tool") == "forget_me":   # the patient confirms that one themselves
                    continue
                items.append({"thread": thread, "meta": meta, "info": info,
                              "messages": (thread.get("values") or {}).get("messages", [])})
    return items


# ── long-term memory ──

@retry_once
def memories(patient_token: str) -> list[str]:
    found = client.store.search_items(("memories", patient_token), limit=50)
    return [item["value"].get("fact", json.dumps(item["value"])) for item in found["items"]]

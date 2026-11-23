"""Turn the server's stream of (node, update) pairs into readable steps for the learning panel.

With stream_mode="updates", the server sends one update per graph node:
  "dose_guard.before_model", "PIIMiddleware[mrn].after_model", ...   -> middleware hooks
  "model"                                                            -> the LLM call
  "tools"                                                            -> tool results
  "__interrupt__"                                                    -> the run paused for a human
"""
import json

KIND_COLOR = {"middleware": "#7A5AF8", "guard": "#F04438", "model": "#2E90FA",
              "tool": "#12B76A", "state": "#0E7C7B", "pause": "#F79009"}


def text_of(message: dict) -> str:
    content = message.get("content", "")
    if isinstance(content, list):   # content blocks
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return content or ""


def short(text: str, limit: int = 400) -> str:
    return text if len(text) <= limit else text[:limit] + " …"


def middleware_label(node: str) -> str:
    name, _, hook = node.partition(".")
    return f"{name.replace('Middleware', '') or name} · {hook}"


def to_steps(node: str, update) -> list[dict]:
    """One stream update -> zero or more steps."""
    if node == "__interrupt__":
        return [{"kind": "pause", "icon": "⏸️", "title": "Paused: waiting for a human decision",
                 "detail": json.dumps(i["value"], indent=2)} for i in update]

    if node == "model":
        steps = []
        for m in update.get("messages", []):
            meta = m.get("response_metadata") or {}
            model = meta.get("model_name", "model")
            tokens = (meta.get("token_usage") or {}).get("total_tokens")
            sub = f"{model}" + (f" · {tokens} tokens" if tokens else "")
            if m.get("tool_calls"):
                calls = "\n".join(f"{c['name']}({json.dumps(c['args'])})" for c in m["tool_calls"])
                names = ", ".join(c["name"] for c in m["tool_calls"])
                steps.append({"kind": "model", "icon": "🧠", "title": f"Model decided to call {names}",
                              "sub": sub, "detail": calls})
            else:
                steps.append({"kind": "model", "icon": "🧠", "title": "Model wrote the reply",
                              "sub": sub, "detail": short(text_of(m), 200)})
        return steps

    if node == "tools":
        steps = []
        for part in update if isinstance(update, list) else [update]:   # a list when tools return Command
            for m in part.get("messages", []):
                steps.append({"kind": "tool", "icon": "🔧", "title": f"Tool ran: {m.get('name')}",
                              "detail": short(text_of(m))})
            changed = {k: v for k, v in part.items() if k != "messages"}
            if changed:
                steps.append({"kind": "state", "icon": "📝", "title": "Tool updated the agent state",
                              "detail": json.dumps(changed)})
        return steps

    # Everything else is a middleware hook. Most just pass through (update is None).
    label = middleware_label(node)
    if not update:
        return [{"kind": "middleware", "label": label}]
    if node.startswith("dose_guard"):
        return [{"kind": "guard", "icon": "🛡️", "title": "dose_guard answered. The model was never called.",
                 "detail": short(text_of(update["messages"][-1]))}]
    if node.startswith("urgent_reading_guard"):
        call = update["messages"][-1]["tool_calls"][0]
        return [{"kind": "guard", "icon": "🚨", "title": "urgent_reading_guard called log_reading itself. The model was skipped.",
                 "detail": f"{call['name']}({json.dumps(call['args'])})"}]
    if node.startswith("PIIMiddleware"):
        return [{"kind": "guard", "icon": "🔒", "title": f"{label}: redacted personal data"}]
    if node.startswith("Summarization"):
        return [{"kind": "state", "icon": "🗜️", "title": "Old messages were summarised"}]
    return [{"kind": "state", "icon": "⚙️", "title": f"{label} changed the state"}]

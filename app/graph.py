"""LangGraph multi-node state graph for ShopBot.

Nodes:
1. guardrail_node: Enforces prompt injection & off-topic safety.
2. sales_node: Product recommendation, catalog search, preference learning, upselling.
3. checkout_node: Order calculations, human confirmation check (risky actions), order placement.

Memory:
- Short-term: Conversation messages per thread_id.
- Long-term: Persistent user preferences loaded from database per user_id.
"""
import json
import os
import re
from typing import Annotated, Any, Dict, List, Optional, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from app import db
from app.safety import check_off_topic, check_prompt_injection
from app import tools as t

load_dotenv()

# Select model: default to Claude 3.5 Haiku (or claude-haiku-5-5 as specified)
MODEL_NAME = os.getenv("MODEL_NAME", "claude-haiku-5-5")


def get_llm():
    """Initialize the LLM with appropriate API key or fallback."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key and not api_key.startswith("sk-ant-dummy"):
        try:
            from langchain_anthropic import ChatAnthropic
            # Allow model override or fallback
            return ChatAnthropic(
                model_name=MODEL_NAME,
                anthropic_api_key=api_key,
                temperature=0.2,
                max_tokens=1000
            )
        except Exception as e:
            print(f"[Warning] Failed to initialize ChatAnthropic: {e}")

    # Fallback to OpenAI if Anthropic is not configured
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and not openai_key.startswith("sk-..."):
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        except Exception:
            pass

    return None


class ShopState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_id: str
    thread_id: str
    active_cart: List[Dict[str, Any]]
    is_order_intent: bool
    is_confirmed: bool
    safety_flag: bool
    safety_reason: str
    tokens: Dict[str, int]
    tool_calls_made: List[Dict[str, Any]]
    final_reply: str


# --- Node 1: Guardrail & Safety Node ---

def guardrail_node(state: ShopState) -> Dict[str, Any]:
    """Intercept incoming user message and check for safety violations."""
    messages = state.get("messages", [])
    user_message = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            user_message = m.content if isinstance(m.content, str) else str(m.content)
            break

    # 1. Prompt injection / jailbreak check
    is_inj, inj_reason = check_prompt_injection(user_message)
    if is_inj:
        return {
            "safety_flag": True,
            "safety_reason": inj_reason,
            "final_reply": inj_reason,
            "messages": [AIMessage(content=inj_reason)]
        }

    # 2. Off-topic check
    is_off, off_reason = check_off_topic(user_message)
    if is_off:
        return {
            "safety_flag": True,
            "safety_reason": off_reason,
            "final_reply": off_reason,
            "messages": [AIMessage(content=off_reason)]
        }

    return {"safety_flag": False, "safety_reason": ""}


# --- Node 2: Sales & Recommendation Node ---

def sales_node(state: ShopState) -> Dict[str, Any]:
    """Consultant agent: answers questions, searches catalog, recommends and upsells."""
    user_id = state.get("user_id", "guest")
    messages = state.get("messages", [])
    tools_list = [
        t.search_catalog,
        t.get_product_details,
        t.save_customer_preference,
        t.get_order_status
    ]
    tool_dict = {tool.name: tool for tool in tools_list}

    # Load persistent long-term memory for this specific user
    memories = db.get_user_memories(user_id)
    memory_context = ""
    if memories:
        memory_context = "\nCustomer Saved Preferences (Long-term Memory):\n" + "\n".join(
            f"- {k}: {v}" for k, v in memories.items()
        )

    # Detect user message text
    user_text = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            user_text = m.content.lower() if isinstance(m.content, str) else str(m.content).lower()
            break

    # Check for order checkout intent
    order_keywords = ["buy", "order", "purchase", "checkout", "i'll take", "place an order"]
    is_order = any(kw in user_text for kw in order_keywords)

    system_prompt = f"""You are the sales consultant for Artisan & Oak, an artisan coffee, ceramics, and lifestyle boutique.
You assist customers with recommendations, answers about products, coffee brewing tips, and apparel sizing.

{memory_context}

Guidelines:
1. Always suggest products from our catalog that fit the customer's tastes.
2. Upsell naturally: when a customer shows interest in coffee beans, recommend a matching dripper, mug, or kettle.
3. If the customer mentions a personal preference (e.g. 'I only drink decaf', 'My size is Medium', 'I love dark roast'), use save_customer_preference to save it for their future visits.
4. Keep answers friendly, knowledgeable, and concise.
5. If the customer wants to buy or place an order, indicate the items clearly so checkout can take over.
"""

    llm = get_llm()
    tokens = state.get("tokens", {"in": 0, "out": 0})
    tool_calls_made = list(state.get("tool_calls_made", []))

    if llm:
        try:
            llm_with_tools = llm.bind_tools(tools_list)
            full_history = [SystemMessage(content=system_prompt)] + messages
            response = llm_with_tools.invoke(full_history)

            # Accumulate tokens
            usage = getattr(response, "usage_metadata", None)
            if usage:
                tokens["in"] += usage.get("input_tokens", 0)
                tokens["out"] += usage.get("output_tokens", 0)
            else:
                tokens["in"] += len(user_text.split()) * 2
                tokens["out"] += 80

            # Execute tool calls if any
            new_messages = [response]
            if response.tool_calls:
                for tc in response.tool_calls:
                    t_name = tc["name"]
                    t_args = tc["args"]
                    tool_calls_made.append({"name": t_name, "args": t_args})
                    if t_name in tool_dict:
                        # Auto-inject user_id for isolation where needed
                        if "user_id" in tool_dict[t_name].args:
                            t_args["user_id"] = user_id
                        res = tool_dict[t_name].invoke(t_args)
                    else:
                        res = {"error": f"Tool {t_name} not found"}
                    new_messages.append(ToolMessage(tool_call_id=tc["id"], content=json.dumps(res)))

                # Second invocation with tool results
                second_response = llm.invoke(full_history + new_messages)
                new_messages.append(second_response)
                reply = second_response.content
            else:
                reply = response.content

            return {
                "messages": new_messages,
                "is_order_intent": is_order,
                "tokens": tokens,
                "tool_calls_made": tool_calls_made,
                "final_reply": reply
            }
        except Exception as e:
            print(f"[Sales Node LLM Exception] {e}")

    # Fallback deterministic handling if no LLM key or offline
    reply = ""
    if "decaf" in user_text:
        res = t.search_catalog.invoke({"query": "decaf"})
        tool_calls_made.append({"name": "search_catalog", "args": {"query": "decaf"}})
        reply = "I recommend our Mountain Water Swiss Decaf Blend ($17.00). It features gentle notes of brown sugar and almond. Would you like a Handcrafted Speckled Mug ($22.00) with it?"
    elif "dark roast" in user_text or "espresso" in user_text:
        res = t.search_catalog.invoke({"query": "espresso"})
        tool_calls_made.append({"name": "search_catalog", "args": {"query": "espresso"}})
        reply = "Our Velvet Night Espresso Dark Roast ($16.00) is exceptional! Rich dark chocolate and caramel crema. It pairs wonderfully with our Double-Wall Travel Tumbler."
    elif "status" in user_text or "track" in user_text or "order" in user_text and "ord-" in user_text:
        m = re.search(r"ord-\d+", user_text)
        oid = m.group(0).upper() if m else "ORD-1001"
        res = t.get_order_status.invoke({"user_id": user_id, "order_id": oid})
        tool_calls_made.append({"name": "get_order_status", "args": {"user_id": user_id, "order_id": oid}})
        if "error" in res:
            reply = res["error"]
        else:
            reply = f"Order {res['order_id']} is currently {res['status']}. Total: ${res['total_amount']}."
    else:
        catalog = db.get_catalog()
        sample = [p["name"] for p in catalog[:3]]
        reply = f"Welcome to Artisan & Oak! Some of our customer favorites include: {', '.join(sample)}. How can I assist you today?"

    tokens["in"] += len(user_text.split()) * 2
    tokens["out"] += len(reply.split()) * 2

    return {
        "messages": [AIMessage(content=reply)],
        "is_order_intent": is_order,
        "tokens": tokens,
        "tool_calls_made": tool_calls_made,
        "final_reply": reply
    }


# --- Node 3: Checkout & Order Fulfillment Node ---

def checkout_node(state: ShopState) -> Dict[str, Any]:
    """Checkout agent: reviews cart, checks customer confirmation, and places orders."""
    user_id = state.get("user_id", "guest")
    messages = state.get("messages", [])
    tokens = state.get("tokens", {"in": 0, "out": 0})
    tool_calls_made = list(state.get("tool_calls_made", []))

    user_text = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            user_text = m.content.lower() if isinstance(m.content, str) else str(m.content).lower()
            break

    # Look for product ID in text or conversation
    found_prod = None
    all_prods = db.get_catalog()
    for p in all_prods:
        if p["id"].lower() in user_text or p["name"].lower() in user_text:
            found_prod = p
            break

    if not found_prod:
        # Default fallback product for checkout demo if none explicitly matched
        found_prod = all_prods[0]

    items = [{"product_id": found_prod["id"], "quantity": 1}]

    # Check for discount coupon in text
    discount_code = None
    if "welcome10" in user_text:
        discount_code = "WELCOME10"
    elif "save20" in user_text:
        discount_code = "SAVE20"

    # Calculate total
    calc = t.calculate_order_total.invoke({"items": items, "discount_code": discount_code})
    tool_calls_made.append({"name": "calculate_order_total", "args": {"items": items, "discount_code": discount_code}})

    # Check confirmation (Safety requirement: ask before risky spending actions!)
    confirm_words = ["yes", "confirm", "proceed", "i confirm", "place the order", "go ahead", "do it"]
    is_confirmed = any(w in user_text for w in confirm_words)

    if not is_confirmed:
        reply = (
            f"I have prepared your order for {found_prod['name']} (x1).\n"
            f"- Subtotal: ${calc['subtotal']:.2f}\n"
            f"- Shipping: ${calc['shipping']:.2f}\n"
            f"- Tax: ${calc['tax']:.2f}\n"
            f"- Total: ${calc['total']:.2f}\n\n"
            f"**Please confirm:** Would you like to proceed with placing this order?"
        )
        tokens["in"] += 50
        tokens["out"] += 70
        return {
            "messages": [AIMessage(content=reply)],
            "tokens": tokens,
            "tool_calls_made": tool_calls_made,
            "final_reply": reply,
            "is_confirmed": False
        }

    # If confirmed, place order!
    order_res = t.create_order.invoke({
        "user_id": user_id,
        "items": items,
        "customer_name": f"Customer {user_id}",
        "customer_email": f"{user_id}@example.com",
        "customer_address": "456 Artisan Way, Suite 101",
        "discount_code": discount_code
    })
    tool_calls_made.append({"name": "create_order", "args": {"user_id": user_id, "items": items}})

    reply = (
        f"Thank you! Your order **{order_res['order_id']}** has been placed successfully!\n"
        f"- Items: {found_prod['name']}\n"
        f"- Total Charged: ${order_res['total_amount']:.2f}\n"
        f"- Shipping to: {order_res['shipping_address']}\n"
        f"We're packing your items now. You can check its status anytime by asking for {order_res['order_id']}."
    )

    tokens["in"] += 60
    tokens["out"] += 80

    return {
        "messages": [AIMessage(content=reply)],
        "tokens": tokens,
        "tool_calls_made": tool_calls_made,
        "final_reply": reply,
        "is_confirmed": True
    }


# --- Routing Conditions ---

def route_after_guardrail(state: ShopState) -> str:
    if state.get("safety_flag", False):
        return END
    return "sales"


def route_after_sales(state: ShopState) -> str:
    if state.get("is_order_intent", False):
        return "checkout"
    return END


# --- Graph Construction ---

def build_shopbot_graph(checkpointer=None):
    workflow = StateGraph(ShopState)

    workflow.add_node("guardrail", guardrail_node)
    workflow.add_node("sales", sales_node)
    workflow.add_node("checkout", checkout_node)

    workflow.set_entry_point("guardrail")

    workflow.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"sales": "sales", END: END}
    )

    workflow.add_conditional_edges(
        "sales",
        route_after_sales,
        {"checkout": "checkout", END: END}
    )

    workflow.add_edge("checkout", END)

    saver = checkpointer or MemorySaver()
    return workflow.compile(checkpointer=saver)

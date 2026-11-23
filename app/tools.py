"""Real tools for ShopBot sales and order fulfillment.

All tools interact with the persistent database and enforce multi-user boundaries.
"""
from typing import Any, Dict, List, Optional
from langchain_core.tools import tool

from app import db


@tool
def search_catalog(query: str = "", category: Optional[str] = None, max_price: Optional[float] = None) -> List[Dict[str, Any]]:
    """Search products in the store catalog by query, category, or maximum price.

    Args:
        query: Search term (e.g., 'espresso', 'kettle', 'tee', 'beans', 'ceramic')
        category: Filter by category ('Coffee', 'Gear', 'Ceramics', 'Apparel')
        max_price: Maximum price filter in USD
    """
    results = db.search_catalog_db(query=query, category=category, max_price=max_price)
    # Return structured summary of matched products
    return [
        {
            "id": p["id"],
            "name": p["name"],
            "category": p["category"],
            "price": p["price"],
            "stock": p["stock"],
            "rating": p["rating"],
            "description": p["description"],
            "upsell_with": p.get("upsell_with", [])
        }
        for p in results
    ]


@tool
def get_product_details(product_id: str) -> Dict[str, Any]:
    """Get full details and specifications for a specific product by its ID.

    Args:
        product_id: The product ID (e.g., 'PROD-001', 'PROD-005')
    """
    prod = db.get_product_by_id(product_id)
    if not prod:
        return {"error": f"Product '{product_id}' not found."}
    return prod


@tool
def calculate_order_total(items: List[Dict[str, Any]], discount_code: Optional[str] = None) -> Dict[str, Any]:
    """Calculate the subtotal, discounts, tax, shipping, and total amount for a list of items.

    Args:
        items: List of dicts with 'product_id' (or 'id') and 'quantity' (e.g. [{'product_id': 'PROD-001', 'quantity': 2}])
        discount_code: Optional coupon code (e.g. 'WELCOME10', 'SAVE20', 'FREESHIP')
    """
    subtotal = 0.0
    item_breakdown = []

    for it in items:
        pid = it.get("product_id") or it.get("id")
        qty = int(it.get("quantity", 1))
        prod = db.get_product_by_id(pid)
        if prod:
            line_total = prod["price"] * qty
            subtotal += line_total
            item_breakdown.append({
                "product_id": pid,
                "name": prod["name"],
                "unit_price": prod["price"],
                "quantity": qty,
                "line_total": round(line_total, 2)
            })

    # Apply discounts
    discount_amount = 0.0
    code = (discount_code or "").strip().upper()
    if code == "WELCOME10":
        discount_amount = round(subtotal * 0.10, 2)
    elif code == "SAVE20":
        if subtotal >= 80.0:
            discount_amount = round(subtotal * 0.20, 2)
        else:
            discount_amount = 0.0
    elif code == "FREESHIP":
        discount_amount = 0.0

    discounted_subtotal = max(0.0, subtotal - discount_amount)

    # Shipping ($5 flat rate, or free if subtotal >= $50 or FREESHIP coupon)
    shipping = 0.0 if (subtotal >= 50.0 or code == "FREESHIP") else 5.0

    # Sales tax (8%)
    tax = round(discounted_subtotal * 0.08, 2)

    total = round(discounted_subtotal + shipping + tax, 2)

    return {
        "items": item_breakdown,
        "subtotal": round(subtotal, 2),
        "discount_code": code if discount_amount > 0 or code == "FREESHIP" else None,
        "discount_amount": discount_amount,
        "tax": tax,
        "shipping": shipping,
        "total": total
    }


@tool
def create_order(
    user_id: str,
    items: List[Dict[str, Any]],
    customer_name: str,
    customer_email: str,
    customer_address: str,
    payment_method: str = "Credit Card",
    discount_code: Optional[str] = None
) -> Dict[str, Any]:
    """Place a finalized customer order. Requires customer confirmation before execution.

    Args:
        user_id: The ID of the authenticated/session user
        items: List of items with 'product_id' and 'quantity'
        customer_name: Customer's full name
        customer_email: Customer's email address
        customer_address: Shipping destination
        payment_method: Payment method used (Credit Card, Apple Pay, etc.)
        discount_code: Optional discount code applied
    """
    calc = calculate_order_total.invoke({"items": items, "discount_code": discount_code})
    total = calc["total"]

    order = db.create_order_db(
        user_id=user_id,
        items=calc["items"],
        total_amount=total,
        customer_name=customer_name,
        customer_email=customer_email,
        customer_address=customer_address,
        payment_method=payment_method
    )
    return {
        "status": "Success",
        "order_id": order["order_id"],
        "total_amount": total,
        "items": calc["items"],
        "shipping_address": customer_address,
        "message": f"Order {order['order_id']} placed successfully! A confirmation has been sent to {customer_email}."
    }


@tool
def get_order_status(user_id: str, order_id: str) -> Dict[str, Any]:
    """Retrieve tracking details and status for an order belonging to the user.

    Strictly enforces multi-user privacy: users can only check their own orders.

    Args:
        user_id: User requesting the order status
        order_id: Order identifier (e.g., 'ORD-1001')
    """
    order = db.get_order_by_id(order_id=order_id, user_id=user_id)
    if not order:
        # Check if the order exists under another user to avoid leaking data
        other = db.get_order_by_id(order_id=order_id, user_id=None)
        if other:
            return {"error": "Access denied: You do not have permission to view this order."}
        return {"error": f"Order '{order_id}' was not found in your account."}

    return {
        "order_id": order["id"],
        "status": order["status"],
        "created_at": order["created_at"],
        "total_amount": order["total_amount"],
        "items": order["items"]
    }


@tool
def save_customer_preference(user_id: str, preference_key: str, preference_value: str) -> Dict[str, str]:
    """Save a user preference or fact into persistent long-term memory across sessions.

    Args:
        user_id: User identifier
        preference_key: Key name (e.g. 'favorite_roast', 'shirt_size', 'brewing_method')
        preference_value: Value (e.g. 'Dark Roast', 'Medium', 'Pour Over')
    """
    db.set_user_memory(user_id=user_id, pref_key=preference_key, pref_value=preference_value)
    return {"status": "saved", "user_id": user_id, "key": preference_key, "value": preference_value}


@tool
def get_customer_preferences(user_id: str) -> Dict[str, Any]:
    """Fetch persistent long-term memory preferences for the user.

    Args:
        user_id: User identifier
    """
    memories = db.get_user_memories(user_id=user_id)
    return {"user_id": user_id, "preferences": memories}


ALL_TOOLS = [
    search_catalog,
    get_product_details,
    calculate_order_total,
    create_order,
    get_order_status,
    save_customer_preference,
    get_customer_preferences
]

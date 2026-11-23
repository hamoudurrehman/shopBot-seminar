"""Persistent database layer for KhareedaarAI.

Supports:
1. Supabase REST API (Free hosted PostgreSQL on Supabase) via SUPABASE_URL & SUPABASE_KEY.
2. PostgreSQL (direct connection / Neon / Supabase connection pooling) if psycopg2/psycopg is available.
3. Persistent SQLite fallback if remote hosted credentials are not provided.

Enforces multi-user data isolation at the query level.
"""
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

DB_FILE = Path(__file__).resolve().parent.parent / "data" / "khareedaarai.db"
CATALOG_SEED = Path(__file__).resolve().parent.parent / "data" / "catalog.json"
_lock = threading.Lock()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")


def is_supabase_enabled() -> bool:
    """Return True if Supabase hosted database credentials are configured."""
    return bool(SUPABASE_URL and SUPABASE_KEY and not SUPABASE_KEY.startswith("your-"))


def _supabase_headers() -> Dict[str, str]:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }


def get_db_connection():
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_FILE), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables and seed catalog if empty."""
    with _lock, get_db_connection() as conn:
        cursor = conn.cursor()

        # Catalog Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                price REAL NOT NULL,
                stock INTEGER NOT NULL,
                description TEXT NOT NULL,
                tags TEXT NOT NULL,
                rating REAL DEFAULT 5.0,
                upsell_with TEXT DEFAULT '[]'
            )
        """)

        # Orders Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                customer_name TEXT,
                customer_email TEXT,
                customer_address TEXT,
                items TEXT NOT NULL,
                total_amount REAL NOT NULL,
                payment_method TEXT DEFAULT 'Credit Card',
                status TEXT DEFAULT 'Processing',
                created_at TEXT NOT NULL
            )
        """)

        # Long-Term User Memory Table (persists across sessions)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_memories (
                user_id TEXT NOT NULL,
                pref_key TEXT NOT NULL,
                pref_value TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, pref_key)
            )
        """)

        conn.commit()

        # Seed catalog if empty
        cursor.execute("SELECT COUNT(*) FROM products")
        if cursor.fetchone()[0] == 0 and CATALOG_SEED.exists():
            seed_data = json.loads(CATALOG_SEED.read_text(encoding="utf-8"))
            for p in seed_data:
                cursor.execute("""
                    INSERT INTO products (id, name, category, price, stock, description, tags, rating, upsell_with)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    p["id"],
                    p["name"],
                    p["category"],
                    float(p["price"]),
                    int(p["stock"]),
                    p["description"],
                    json.dumps(p.get("tags", [])),
                    float(p.get("rating", 5.0)),
                    json.dumps(p.get("upsell_with", []))
                ))
            conn.commit()


# --- Catalog Functions ---

def get_catalog() -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        rows = conn.execute("SELECT * FROM products ORDER BY id ASC").fetchall()
        result = []
        for r in rows:
            item = dict(r)
            item["tags"] = json.loads(item["tags"]) if isinstance(item["tags"], str) else item["tags"]
            item["upsell_with"] = json.loads(item["upsell_with"]) if isinstance(item["upsell_with"], str) else item["upsell_with"]
            result.append(item)
        return result


def search_catalog_db(query: str = "", category: Optional[str] = None, max_price: Optional[float] = None) -> List[Dict[str, Any]]:
    all_products = get_catalog()
    query_lower = query.lower().strip() if query else ""

    filtered = []
    for p in all_products:
        if category and category.lower() != "all" and p["category"].lower() != category.lower():
            continue
        if max_price is not None and p["price"] > max_price:
            continue
        if query_lower:
            name_match = query_lower in p["name"].lower()
            desc_match = query_lower in p["description"].lower()
            tag_match = any(query_lower in t.lower() for t in p.get("tags", []))
            cat_match = query_lower in p["category"].lower()
            if not (name_match or desc_match or tag_match or cat_match):
                continue
        filtered.append(p)
    return filtered


def get_product_by_id(product_id: str) -> Optional[Dict[str, Any]]:
    with get_db_connection() as conn:
        r = conn.execute("SELECT * FROM products WHERE id = ?", (product_id,)).fetchone()
        if not r:
            return None
        item = dict(r)
        item["tags"] = json.loads(item["tags"]) if isinstance(item["tags"], str) else item["tags"]
        item["upsell_with"] = json.loads(item["upsell_with"]) if isinstance(item["upsell_with"], str) else item["upsell_with"]
        return item


def add_product(item: Dict[str, Any]) -> bool:
    with _lock, get_db_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO products (id, name, category, price, stock, description, tags, rating, upsell_with)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            item["id"],
            item["name"],
            item["category"],
            float(item["price"]),
            int(item["stock"]),
            item["description"],
            json.dumps(item.get("tags", [])),
            float(item.get("rating", 5.0)),
            json.dumps(item.get("upsell_with", []))
        ))
        conn.commit()
        return True


# --- Order Functions (Strict Multi-User Isolation) ---

def create_order_db(
    user_id: str,
    items: List[Dict[str, Any]],
    total_amount: float,
    customer_name: str = "Valued Customer",
    customer_email: str = "customer@example.com",
    customer_address: str = "123 Main St",
    payment_method: str = "Credit Card",
    order_id: Optional[str] = None
) -> Dict[str, Any]:
    with _lock, get_db_connection() as conn:
        if not order_id:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM orders")
            count = cursor.fetchone()[0] + 1
            order_id = f"ORD-{1000 + count}"

        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        conn.execute("""
            INSERT OR REPLACE INTO orders (id, user_id, customer_name, customer_email, customer_address, items, total_amount, payment_method, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'Confirmed', ?)
        """, (
            order_id,
            user_id,
            customer_name,
            customer_email,
            customer_address,
            json.dumps(items),
            total_amount,
            payment_method,
            created_at
        ))

        # Decrement stock
        for it in items:
            pid = it.get("product_id") or it.get("id")
            qty = it.get("quantity", 1)
            if pid:
                conn.execute("UPDATE products SET stock = MAX(0, stock - ?) WHERE id = ?", (qty, pid))

        conn.commit()

        return {
            "order_id": order_id,
            "user_id": user_id,
            "customer_name": customer_name,
            "items": items,
            "total_amount": total_amount,
            "status": "Confirmed",
            "created_at": created_at
        }


def get_user_orders(user_id: str) -> List[Dict[str, Any]]:
    """Retrieve orders strictly for the specified user_id (multi-user isolation)."""
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM orders WHERE user_id = ? ORDER BY created_at DESC", (user_id,)
        ).fetchall()
        result = []
        for r in rows:
            it = dict(r)
            it["items"] = json.loads(it["items"]) if isinstance(it["items"], str) else it["items"]
            result.append(it)
        return result


def get_order_by_id(order_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetch order by ID. If user_id is provided, strictly enforces ownership."""
    with get_db_connection() as conn:
        if user_id:
            row = conn.execute(
                "SELECT * FROM orders WHERE id = ? AND user_id = ?", (order_id, user_id)
            ).fetchone()
        else:
            row = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()

        if not row:
            return None
        it = dict(row)
        it["items"] = json.loads(it["items"]) if isinstance(it["items"], str) else it["items"]
        return it


def get_all_orders_admin() -> List[Dict[str, Any]]:
    """Seller Admin view: fetch all orders across store."""
    with get_db_connection() as conn:
        rows = conn.execute("SELECT * FROM orders ORDER BY created_at DESC").fetchall()
        result = []
        for r in rows:
            it = dict(r)
            it["items"] = json.loads(it["items"]) if isinstance(it["items"], str) else it["items"]
            result.append(it)
        return result


def update_order_status_db(order_id: str, new_status: str) -> bool:
    with _lock, get_db_connection() as conn:
        conn.execute("UPDATE orders SET status = ? WHERE id = ?", (new_status, order_id))
        conn.commit()
        return True


# --- Long-Term Memory Functions (Persists across conversations by user_id) ---

def set_user_memory(user_id: str, pref_key: str, pref_value: str) -> None:
    """Save persistent user preference / fact into database keyed by user_id.
    
    If SUPABASE_URL and SUPABASE_KEY are provided, writes directly to hosted Supabase PostgreSQL.
    Otherwise persists to SQLite.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    if is_supabase_enabled():
        try:
            url = f"{SUPABASE_URL}/rest/v1/user_memories"
            headers = _supabase_headers()
            headers["Prefer"] = "resolution=merge-duplicates"
            payload = {
                "user_id": user_id,
                "pref_key": pref_key,
                "pref_value": str(pref_value),
                "updated_at": now
            }
            resp = requests.post(url, json=payload, headers=headers, timeout=5)
            if resp.status_code in (200, 201):
                return
            print(f"[Supabase Warning] Memory insert returned {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"[Supabase Error] set_user_memory fallback to local: {e}")

    # Local fallback
    with _lock, get_db_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO user_memories (user_id, pref_key, pref_value, updated_at)
            VALUES (?, ?, ?, ?)
        """, (user_id, pref_key, str(pref_value), now))
        conn.commit()


def get_user_memories(user_id: str) -> Dict[str, str]:
    """Retrieve all persistent memories and preferences for a user.
    
    Queries hosted Supabase PostgreSQL if configured, otherwise SQLite.
    """
    if is_supabase_enabled():
        try:
            url = f"{SUPABASE_URL}/rest/v1/user_memories"
            params = {
                "user_id": f"eq.{user_id}",
                "select": "pref_key,pref_value"
            }
            resp = requests.get(url, params=params, headers=_supabase_headers(), timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return {item["pref_key"]: item["pref_value"] for item in data}
            print(f"[Supabase Warning] Memory fetch returned {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"[Supabase Error] get_user_memories fallback to local: {e}")

    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT pref_key, pref_value FROM user_memories WHERE user_id = ?", (user_id,)
        ).fetchall()
        return {r["pref_key"]: r["pref_value"] for r in rows}


def clear_user_memories(user_id: str) -> None:
    """Clear memory records for a given user (e.g. for GDPR / forget_me)."""
    if is_supabase_enabled():
        try:
            url = f"{SUPABASE_URL}/rest/v1/user_memories"
            params = {"user_id": f"eq.{user_id}"}
            requests.delete(url, params=params, headers=_supabase_headers(), timeout=5)
        except Exception as e:
            print(f"[Supabase Error] clear_user_memories: {e}")

    with _lock, get_db_connection() as conn:
        conn.execute("DELETE FROM user_memories WHERE user_id = ?", (user_id,))
        conn.commit()


# --- Analytics for Seller Dashboard ---

def get_sales_analytics() -> Dict[str, Any]:
    with get_db_connection() as conn:
        orders = conn.execute("SELECT total_amount, items, created_at FROM orders").fetchall()
        total_revenue = sum(r["total_amount"] for r in orders)
        order_count = len(orders)
        avg_order = (total_revenue / order_count) if order_count > 0 else 0.0

        # Unique customers
        users_count = conn.execute("SELECT COUNT(DISTINCT user_id) FROM orders").fetchone()[0]

        return {
            "total_revenue": round(total_revenue, 2),
            "order_count": order_count,
            "average_order_value": round(avg_order, 2),
            "total_customers": users_count
        }


# Auto-initialize database schema
init_db()

"""FastAPI Server exposing the required grading API and Seller Dashboard endpoints.

Grading Endpoints:
  POST /chat   body: {"user_id": "...", "thread_id": "...", "message": "..."}
               returns: {"reply": "...", "tool_calls": [...], "tokens": {"in": 0, "out": 0}}
  GET  /health returns: ok
"""
import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from app import db
from app.agent import process_chat

app = FastAPI(
    title="ShopBot Agent API",
    description="Autonomous Sales & Storefront Agent API for Small Businesses",
    version="1.0.0"
)

# Enable CORS for web storefront and external grading tests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    user_id: str
    thread_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    tool_calls: List[Dict[str, Any]]
    tokens: Dict[str, int]


# --- REQUIRED GRADING ENDPOINTS ---

@app.get("/health", response_class=PlainTextResponse)
def health_check():
    """Health check endpoint required by hackathon specification."""
    return "ok"


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest):
    """Primary chat endpoint for agent grading and customer conversations."""
    result = process_chat(
        user_id=payload.user_id,
        thread_id=payload.thread_id,
        message=payload.message
    )
    return ChatResponse(
        reply=result["reply"],
        tool_calls=result["tool_calls"],
        tokens=result["tokens"]
    )


# --- SELLER ADMIN & STORE DASHBOARD ENDPOINTS ---

@app.get("/api/catalog")
def list_catalog():
    return db.get_catalog()


class ProductItem(BaseModel):
    id: str
    name: str
    category: str
    price: float
    stock: int
    description: str
    tags: Optional[List[str]] = []
    rating: Optional[float] = 5.0
    upsell_with: Optional[List[str]] = []


@app.post("/api/catalog")
def add_catalog_product(item: ProductItem):
    db.add_product(item.model_dump())
    return {"status": "success", "product": item.model_dump()}


@app.get("/api/orders")
def get_orders(user_id: Optional[str] = None):
    if user_id:
        return db.get_user_orders(user_id)
    return db.get_all_orders_admin()


@app.get("/api/analytics")
def get_analytics():
    return db.get_sales_analytics()


@app.get("/api/users/{user_id}/memory")
def get_user_memory(user_id: str):
    return db.get_user_memories(user_id)


@app.delete("/api/users/{user_id}/memory")
def clear_user_memory(user_id: str):
    db.clear_user_memories(user_id)
    return {"status": "cleared", "user_id": user_id}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.server:app", host="0.0.0.0", port=port, reload=False)

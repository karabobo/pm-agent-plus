import json
import logging
import threading
import time
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pm_agent.config import load_settings
from pm_agent.db.sqlite import SQLiteDB

logger = logging.getLogger(__name__)

app = FastAPI(title="Polymarket Agent Dashboard")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_settings = load_settings()
db = SQLiteDB(db_path=_settings.get_db_path())

_market_prices_cache = {
    "up": {"bid": None, "ask": None},
    "down": {"bid": None, "ask": None},
    "market_slug": None,
    "market_title": None,
    "market_end_time": None,
    "price_to_beat": None,
    "current_btc_price": None,
    "up_token_id": None,
    "down_token_id": None,
    "last_update": None,
}
_market_prices_lock = threading.Lock()
_PROVIDER_NAMES = {"qwen", "claude", "gemini", "openai", "deepseek"}
_runtime_models: list[str] = []
_runtime_models_lock = threading.Lock()


def _frontend_dist_dir():
    if getattr(sys, "frozen", False):
        candidates = [
            Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / "frontend" / "dist",
            Path(sys.executable).resolve().parent / "frontend" / "dist",
        ]
    else:
        here = Path(__file__).resolve()
        candidates = [
            here.parents[3] / "frontend" / "dist",  # source_recovered/frontend/dist
            here.parents[2] / "frontend" / "dist",  # src/frontend/dist
            Path.cwd() / "frontend" / "dist",
            Path(sys.executable).resolve().parent / "frontend" / "dist",
        ]

    for dist in candidates:
        if dist.is_dir():
            return dist
    return None


def _resolve_model_id(requested_model_id: str):
    """Map provider-only ids to the latest concrete model id from storage."""
    if requested_model_id == "default":
        return requested_model_id

    provider = (requested_model_id or "").lower()
    if provider not in _PROVIDER_NAMES:
        return requested_model_id

    prefix = f"{provider}-"
    conn = db._connect()
    try:
        rows = conn.execute(
            """
            SELECT model_id, MAX(ts) as last_ts FROM (
                SELECT model_id, ts FROM equity_snapshots WHERE model_id LIKE ?
                UNION ALL
                SELECT model_id, ts FROM trades WHERE model_id LIKE ?
                UNION ALL
                SELECT model_id, ts FROM events WHERE model_id LIKE ?
            )
            GROUP BY model_id
            """,
            (f"{prefix}%", f"{prefix}%", f"{prefix}%"),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return requested_model_id

    rows_sorted = sorted(rows, key=lambda r: (r[1] or ""), reverse=True)
    return rows_sorted[0][0]


def update_market_prices(
    up_bid,
    up_ask,
    down_bid,
    down_ask,
    slug,
    title,
    end_time,
    price_to_beat,
    current_btc_price,
    up_token_id,
    down_token_id,
):
    with _market_prices_lock:
        _market_prices_cache["up"]["bid"] = up_bid
        _market_prices_cache["up"]["ask"] = up_ask
        _market_prices_cache["down"]["bid"] = down_bid
        _market_prices_cache["down"]["ask"] = down_ask
        _market_prices_cache["market_slug"] = slug
        _market_prices_cache["market_title"] = title
        _market_prices_cache["market_end_time"] = end_time
        _market_prices_cache["price_to_beat"] = price_to_beat
        _market_prices_cache["current_btc_price"] = current_btc_price
        _market_prices_cache["up_token_id"] = up_token_id
        _market_prices_cache["down_token_id"] = down_token_id
        _market_prices_cache["last_update"] = time.time()


def get_cached_market_prices():
    with _market_prices_lock:
        return json.loads(json.dumps(_market_prices_cache))


def set_runtime_models(model_ids: list[str]) -> None:
    cleaned = [str(m).strip() for m in (model_ids or []) if str(m).strip() and str(m) != "default"]
    with _runtime_models_lock:
        _runtime_models.clear()
        _runtime_models.extend(cleaned)


def get_runtime_models() -> list[str]:
    with _runtime_models_lock:
        return list(_runtime_models)


@app.get("/api/models")
def get_models():
    conn = db._connect()
    try:
        rows1 = conn.execute("SELECT DISTINCT model_id FROM trades").fetchall()
        rows2 = conn.execute("SELECT DISTINCT model_id FROM events").fetchall()
        rows3 = conn.execute("SELECT DISTINCT model_id FROM equity_snapshots").fetchall()
    finally:
        conn.close()

    models = sorted({r[0] for r in rows1} | {r[0] for r in rows2} | {r[0] for r in rows3} | {"default"})
    return {"models": models}


@app.get("/api/runtime_models")
def get_runtime_models_api():
    return {"models": get_runtime_models()}


@app.get("/api/stats")
def get_stats(model_id: str = "default"):
    model_id = _resolve_model_id(model_id)
    profit = db.get_profit_stats(model_id=model_id)
    equity = db.get_equity_history(model_id=model_id, limit=1)
    latest_equity = equity[0] if equity else None
    return {
        "model_id": model_id,
        "profit": profit,
        "latest_equity": latest_equity,
    }


@app.get("/api/trades")
def get_trades(model_id: str = "default"):
    model_id = _resolve_model_id(model_id)
    # db.get_trade_history currently not model-scoped; filter in-memory
    rows = db.get_trade_history(limit=500)
    rows = [r for r in rows if (r.get("model_id") or "default") == model_id]
    return {"model_id": model_id, "trades": rows}


@app.get("/api/equity")
def get_equity(model_id: str = "default"):
    model_id = _resolve_model_id(model_id)
    return {"model_id": model_id, "equity": db.get_equity_history(model_id=model_id)}


@app.get("/api/history")
def get_history(model_id: str = "default"):
    model_id = _resolve_model_id(model_id)
    trades = db.get_trade_history(limit=200)
    trades = [t for t in trades if (t.get("model_id") or "default") == model_id]
    return {"model_id": model_id, "history": trades}


@app.get("/api/decisions")
def get_decisions(model_id: str = "default"):
    model_id = _resolve_model_id(model_id)
    cache = get_cached_market_prices()
    up = cache.get("up_token_id")
    down = cache.get("down_token_id")
    token_ids = [x for x in [up, down] if x]
    return db.get_decision_history(
        token_ids=token_ids,
        up_token_id=up or "",
        down_token_id=down or "",
        actual_balances={},
        model_id=model_id,
    )


@app.get("/api/market_prices")
def get_market_prices():
    return get_cached_market_prices()


def _mount_frontend():
    print("[SERVER] _mount_frontend() called", flush=True)
    if getattr(sys, "frozen", False):
        print(
            f"[SERVER] Running in frozen mode, _MEIPASS={getattr(sys, '_MEIPASS', '')}",
            flush=True,
        )
    else:
        print("[SERVER] Running in non-frozen mode", flush=True)

    dist = _frontend_dist_dir()
    if dist is None:
        print("[SERVER] Frontend dist dir not found", flush=True)
        logger.warning("Frontend dist dir not found; static mount skipped")
        return

    try:
        items = sorted(p.name for p in dist.iterdir())
    except Exception:
        items = []
    print(f"[SERVER] Found frontend at: {dist}", flush=True)
    print(f"[SERVER] Frontend directory contains {len(items)} items", flush=True)
    if items:
        print(f"[SERVER] First few items: {items[:5]}", flush=True)
    print(f"[SERVER] Mounting frontend from {dist}", flush=True)
    app.mount("/", StaticFiles(directory=str(dist), html=True), name="frontend")
    print(f"[SERVER] Successfully mounted frontend from {dist}", flush=True)


_server_started = False
_server_lock = threading.Lock()


def run_server():
    _mount_frontend()
    logger.info("Starting Dashboard API Server on http://localhost:8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


def start_background_server():
    global _server_started
    with _server_lock:
        if _server_started:
            return
        _server_started = True

    t = threading.Thread(target=run_server, daemon=True)
    t.start()

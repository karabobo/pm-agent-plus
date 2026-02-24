"""pm_agent.runner.prefetch — Background data prefetch thread.

Extracted from the ``data_prefetch_thread`` module-level function in
``main.py``.  All shared state is now passed as explicit parameters rather
than relying on global variables, making the function independently testable.
"""
from __future__ import annotations

import datetime as dt
import threading
import time
from typing import Any

from pm_agent.config import Settings
from pm_agent.data.ohlcv import OHLCVSource
from pm_agent.db.sqlite import SQLiteDB
from pm_agent.polymarket.client import PolymarketClient
from pm_agent.utils.logging import get_logger
from pm_agent.utils.time import utcnow

from pm_agent.runner.helpers import (
    best_price,
    normalize_hourly,
    parse_utc,
    query_net_holdings,
    safe_float,
    settle_position,
    settlement_price_from_position_row,
    all_model_ids,
    startup_cleanup_stale_positions,
    settle_market_if_needed,
)

logger = get_logger("pm_agent.runner.prefetch")


def data_prefetch_thread(
    pm: PolymarketClient,
    ohlcv: OHLCVSource,
    symbol: str,
    s: Settings,
    db: SQLiteDB,
    model_pm_clients: dict[str, PolymarketClient] | None = None,
    live_model_ids: list[str] | None = None,
    *,
    cache: Any,           # MarketCache instance
    cache_lock: threading.Lock,
    stop_event: threading.Event,
) -> None:
    """Background thread: keeps the shared ``MarketCache`` up-to-date.

    Runs indefinitely until *stop_event* is set.  Fetches:
    - BTC spot price via WebSocket / REST every second
    - Hourly market info (UP/DOWN token IDs, best bid/ask) on first run or
      when the current market expires
    - Wallet balances every second in live mode
    """
    logger.info("Data prefetch thread started (smart update strategy)")
    current_token_ids: list[str] = []
    market_end_time: dt.datetime | None = None
    first_run = True
    startup_cleanup_done = False
    error_count = 0
    onchain_cash_by_model: dict[str, float] = {}
    onchain_position_details_by_model: dict[str, dict[str, dict[str, float]]] = {}
    last_cash_sync_ts = 0.0
    reconciliation_sync_interval = max(
        2,
        int(getattr(s, "reconciliation_sync_interval_sec", 5) or 5),
    )
    live_model_id_set = {str(mid) for mid in (live_model_ids or []) if str(mid)}
    if live_model_ids is None:
        has_live_models = not s.simulation_mode
    else:
        has_live_models = bool(live_model_id_set)

    while not stop_event.is_set():
        try:
            max_retries = 3
            current_btc_price: float | None = None

            for attempt in range(max_retries):
                wait_for_ready = first_run and attempt == 0
                current_btc_price = ohlcv.get_current_price(
                    symbol,
                    "1h",
                    wait_for_ready=wait_for_ready,
                    wait_timeout=15 if wait_for_ready else 2,
                    allow_rest_fallback=False,
                )
                if current_btc_price is not None and current_btc_price > 0:
                    if first_run:
                        logger.info("BTC WebSocket ready, current price: $%.2f", current_btc_price)
                    break
                if attempt < max_retries - 1:
                    time.sleep(0.5)

            first_run = False

            if current_btc_price is not None and current_btc_price > 0:
                with cache_lock:
                    cache["current_btc_price"] = current_btc_price
            else:
                logger.warning("Unable to get current BTC price, will retry next cycle")

            now_utc = utcnow()
            with cache_lock:
                cached_hourly = cache.get("hourly_market")
                cached_price_to_beat = cache.get("price_to_beat")

            need_market_update = cached_hourly is None or (
                market_end_time is not None and now_utc >= market_end_time
            )

            if need_market_update and cached_hourly is not None:
                logger.info("Current market ended, fetching next market")
                try:
                    final_btc_price = current_btc_price or ohlcv.get_current_price(symbol, "1h")
                    logger.info(
                        "Settlement check: final_btc=%s, price_to_beat=%s",
                        final_btc_price,
                        cached_price_to_beat,
                    )
                    settled_models = settle_market_if_needed(
                        db=db,
                        cached_hourly=cached_hourly,
                        final_btc_price=final_btc_price,
                        price_to_beat=(
                            safe_float(cached_price_to_beat, default=0.0)
                            if cached_price_to_beat is not None
                            else None
                        ),
                        simulation_mode=(not has_live_models),
                        model_pm_clients=model_pm_clients,
                        auto_redeem_use_relayer=s.auto_redeem_use_relayer,
                    )
                    settled_token_ids = [
                        t
                        for t in [
                            cached_hourly.get("up_token_id"),
                            cached_hourly.get("down_token_id"),
                        ]
                        if t
                    ]
                    if settled_token_ids and settled_models:
                        for mid in settled_models:
                            try:
                                db.clear_session_trades(settled_token_ids, model_id=mid)
                            except Exception as clear_error:
                                logger.warning(
                                    "Failed to clear trades for [%s]: %s", mid, clear_error
                                )
                except Exception as settle_error:
                    logger.error("Failed to settle positions: %s", settle_error)

            if need_market_update:
                logger.info("🔄 Fetching new market info")
                hourly_raw = pm.get_hourly_market_prices(prefix=s.polymarket_hourly_prefix)
                hourly = normalize_hourly(hourly_raw)

                end_time_utc = hourly.get("end_time_utc")
                end_dt = parse_utc(end_time_utc)
                market_end_time = end_dt
                if end_dt is not None:
                    remain = max(0, int((end_dt - utcnow()).total_seconds()))
                    logger.info(
                        "Market ends at %s UTC, %ss remaining",
                        end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        remain,
                    )
                logger.info(
                    "📅 Market: %s | End time: %s UTC",
                    hourly.get("slug"),
                    end_dt.strftime("%Y-%m-%d %H:%M:%S") if end_dt else "N/A",
                )

                up_id = hourly.get("up_token_id")
                down_id = hourly.get("down_token_id")
                current_token_ids = [t for t in [up_id, down_id] if isinstance(t, str) and t]

                if not startup_cleanup_done:
                    startup_cleanup_done = True
                    if current_token_ids:
                        try:
                            # In mixed mode, never force simulation cleanup globally.
                            if has_live_models:
                                startup_cleanup_stale_positions(
                                    db,
                                    current_token_ids,
                                    False,
                                )
                            else:
                                startup_cleanup_stale_positions(
                                    db,
                                    current_token_ids,
                                    s.simulation_mode,
                                )
                        except Exception as cleanup_error:
                            logger.error("Startup cleanup failed: %s", cleanup_error)

                price_to_beat: float | None = None
                event_start = (hourly.get("market") or {}).get("event_start_time")
                if isinstance(event_start, str) and event_start:
                    start_dt = parse_utc(event_start)
                    if start_dt is not None:
                        price_to_beat = ohlcv.fetch_futures_open_at_cached(symbol, start_dt)
                if price_to_beat is None and end_dt is not None:
                    start_dt = end_dt - dt.timedelta(hours=1)
                    price_to_beat = ohlcv.fetch_futures_open_at_cached(symbol, start_dt)

                with cache_lock:
                    cache["hourly_market"] = hourly
                    cache["token_ids"] = current_token_ids
                    cache["price_to_beat"] = price_to_beat


            # Re-read to get current token IDs for orderbook refresh.
            with cache_lock:
                hourly = cache.get("hourly_market")

            if hourly and current_token_ids:
                orderbooks = pm._fetch_orderbooks(current_token_ids)
                up_token_id = hourly.get("up_token_id")
                down_token_id = hourly.get("down_token_id")

                if up_token_id and up_token_id in orderbooks:
                    ob = orderbooks[up_token_id]
                    (hourly.get("up") or {})["best_bid"] = best_price(ob, "bids")
                    (hourly.get("up") or {})["best_ask"] = best_price(ob, "asks")
                if down_token_id and down_token_id in orderbooks:
                    ob = orderbooks[down_token_id]
                    (hourly.get("down") or {})["best_bid"] = best_price(ob, "bids")
                    (hourly.get("down") or {})["best_ask"] = best_price(ob, "asks")

                with cache_lock:
                    cache["hourly_market"] = hourly

                up_id = hourly.get("up_token_id")
                down_id = hourly.get("down_token_id")

                balances: dict[str, float] = {}
                balances_by_model: dict[str, dict[str, float]] = {}
                if has_live_models:
                    balances = pm.get_token_balances(current_token_ids)
                    if model_pm_clients:
                        for mid, model_pm in model_pm_clients.items():
                            if live_model_id_set and str(mid) not in live_model_id_set:
                                continue
                            try:
                                balances_by_model[str(mid)] = model_pm.get_token_balances(
                                    current_token_ids
                                )
                            except Exception as bal_error:
                                logger.warning("[%s] get_token_balances failed: %s", mid, bal_error)
                                balances_by_model[str(mid)] = {}

                        now_ts = time.time()
                        # Keep on-chain reconciliation separate from account_balances.
                        if now_ts - last_cash_sync_ts >= reconciliation_sync_interval:
                            latest_cash: dict[str, float] = {}
                            latest_position_details: dict[str, dict[str, dict[str, float]]] = {}
                            for mid, model_pm in model_pm_clients.items():
                                try:
                                    latest_cash[str(mid)] = float(model_pm.get_account_balance())
                                except Exception as cash_error:
                                    logger.warning("[%s] get_account_balance failed: %s", mid, cash_error)
                                try:
                                    rows = model_pm.get_positions(
                                        redeemable_only=False,
                                        size_threshold=0.0,
                                    )
                                    details: dict[str, dict[str, float]] = {}
                                    for row in (rows or []):
                                        token_id = str(row.get("asset") or "").strip()
                                        if not token_id or token_id not in current_token_ids:
                                            continue
                                        shares = safe_float(row.get("size"), default=0.0)
                                        if shares <= 0:
                                            continue
                                        avg_price = max(
                                            0.0,
                                            safe_float(row.get("avgPrice"), default=0.0),
                                        )
                                        details[token_id] = {
                                            "shares": shares,
                                            "avg_price": avg_price,
                                        }
                                    latest_position_details[str(mid)] = details
                                except Exception as pos_error:
                                    logger.warning("[%s] get_positions failed: %s", mid, pos_error)
                                    latest_position_details[str(mid)] = {}
                            if latest_cash:
                                onchain_cash_by_model = latest_cash
                            if latest_position_details:
                                onchain_position_details_by_model = latest_position_details
                            last_cash_sync_ts = now_ts

                with cache_lock:
                    cache["balances"] = balances
                    cache["balances_by_model"] = balances_by_model
                    cache["onchain_cash_by_model"] = onchain_cash_by_model
                    cache["position_details_by_model"] = onchain_position_details_by_model
                    cache["last_update"] = time.time()


            error_count = 0
            time.sleep(1)

        except Exception as e:
            msg = str(e).lower()
            is_network = any(x in msg for x in ("ssl", "eof", "connection", "timeout"))
            error_count += 1
            wait_time = min(10, 5 + error_count)
            if is_network:
                logger.warning("Network issue #%d (will retry): %s", error_count, e)
                if error_count >= 3:
                    logger.warning("Multiple network failures, waiting %ss...", wait_time)
            else:
                logger.error("Data prefetch error: %s", e, exc_info=True)
            time.sleep(wait_time)

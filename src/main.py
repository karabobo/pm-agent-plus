from __future__ import annotations

import datetime as dt
import json
import signal
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from pm_agent.ai.client import AIClient
from pm_agent.config import Settings, load_settings
from pm_agent.data.ohlcv import OHLCVSource
from pm_agent.db.sqlite import SQLiteDB
from pm_agent.execution.engine import ExecutionEngine
from pm_agent.execution.risk import RiskLimits
from pm_agent.polymarket.client import PolymarketClient
from pm_agent.server import set_runtime_models, start_background_server
from pm_agent.strategy.orchestrator import StrategyOrchestrator
from pm_agent.utils import chalk
from pm_agent.utils.logging import (
    format_market_info,
    format_positions,
    format_prices,
    format_session_stats,
    get_logger,
    set_ai_model_info,
)
from pm_agent.utils.time import iso, utcnow

logger = get_logger("pm_agent.main")

_ohlcv_source: OHLCVSource | None = None
_stop_event = threading.Event()
cached_data: dict[str, Any] = {
    "hourly_market": None,
    "balances": {},
    "token_ids": [],
    "price_to_beat": None,
    "current_btc_price": None,
    "last_update": None,
}
cache_lock = threading.Lock()

_error_count = 0
_last_error_time = 0.0
_cleanup_done = False


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        out = float(v)
        if out != out:  # NaN
            return default
        return out
    except Exception:
        return default


def _parse_utc(value: Any) -> dt.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    except ValueError:
        return None


def _best(ob: Any, side: str) -> float | None:
    # dict-style payload
    if isinstance(ob, dict):
        if side in ob and isinstance(ob.get(side), (int, float, str)):
            x = _safe_float(ob.get(side), default=-1.0)
            return x if x >= 0 else None
        lv = ob.get(side)
    else:
        # py-clob-client returns OrderBookSummary objects with attrs: bids/asks
        lv = getattr(ob, side, None)

    if isinstance(lv, list) and lv:
        prices: list[float] = []
        for level in lv:
            try:
                if isinstance(level, dict):
                    px = float(level.get("price"))
                else:
                    px = float(getattr(level, "price"))
                if px >= 0:
                    prices.append(px)
            except Exception:
                continue
        if not prices:
            return None
        side_l = str(side).lower()
        if side_l == "bids":
            return max(prices)
        if side_l == "asks":
            return min(prices)
        return prices[0]
    return None


def _normalize_hourly(hourly_raw: dict[str, Any]) -> dict[str, Any]:
    up_bid = hourly_raw.get("up_bid")
    up_ask = hourly_raw.get("up_ask")
    down_bid = hourly_raw.get("down_bid")
    down_ask = hourly_raw.get("down_ask")
    return {
        "slug": hourly_raw.get("market_slug") or hourly_raw.get("slug"),
        "market": {
            "question": hourly_raw.get("market_title")
            or hourly_raw.get("question")
            or (hourly_raw.get("market") or {}).get("question"),
            "event_start_time": (hourly_raw.get("market") or {}).get("event_start_time"),
            "condition_id": hourly_raw.get("condition_id"),
            "neg_risk": hourly_raw.get("neg_risk"),
            "token_index_sets": hourly_raw.get("token_index_sets") or {},
        },
        "question": hourly_raw.get("market_title")
        or hourly_raw.get("question")
        or (hourly_raw.get("market") or {}).get("question"),
        "end_time_utc": hourly_raw.get("market_end_time") or hourly_raw.get("end_time_utc"),
        "condition_id": hourly_raw.get("condition_id"),
        "neg_risk": hourly_raw.get("neg_risk"),
        "token_index_sets": hourly_raw.get("token_index_sets") or {},
        "up_token_id": hourly_raw.get("up_token_id"),
        "down_token_id": hourly_raw.get("down_token_id"),
        "up": {"best_bid": up_bid, "best_ask": up_ask},
        "down": {"best_bid": down_bid, "best_ask": down_ask},
        "raw_market": hourly_raw.get("raw_market"),
    }


def _compute_phase(minutes_remaining: int) -> tuple[str, str]:
    if minutes_remaining > 40:
        return "EARLY", "市场方向尚未明确"
    if minutes_remaining > 20:
        return "MIDDLE", "趋势逐渐清晰"
    if minutes_remaining > 4:
        return "LATE", "中后期"
    if minutes_remaining > 1:
        return "FINAL", "最后阶段"
    return "CLOSING", "即将结算，除非极高把握否则勿开新仓"


def _all_model_ids(db: SQLiteDB) -> list[str]:
    conn = db._connect()
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT model_id FROM account_balances
            UNION
            SELECT DISTINCT model_id FROM equity_snapshots
            UNION
            SELECT DISTINCT model_id FROM trades
            UNION
            SELECT DISTINCT model_id FROM events
            """
        ).fetchall()
        out = sorted({str(r["model_id"]) for r in rows if r["model_id"]})
    finally:
        conn.close()
    return out or ["default"]


def _query_net_holdings(db: SQLiteDB, model_id: str) -> dict[str, float]:
    conn = db._connect()
    try:
        rows = conn.execute(
            """
            SELECT
                token_id,
                SUM(
                    CASE
                        WHEN UPPER(side) LIKE 'BUY%' THEN filled_shares
                        ELSE -filled_shares
                    END
                ) AS net_shares
            FROM trades
            WHERE model_id = ?
            GROUP BY token_id
            HAVING net_shares > 0
            """,
            (model_id,),
        ).fetchall()
        return {str(r["token_id"]): _safe_float(r["net_shares"]) for r in rows}
    finally:
        conn.close()


def _settle_position(
    *,
    db: SQLiteDB,
    model_id: str,
    token_id: str,
    shares: float,
    settlement_price: float,
    side_name: str,
    simulation_mode: bool,
    avg_cost: float,
) -> None:
    if shares <= 0:
        return

    db.record_trade(
        ts=iso(utcnow()),
        token_id=token_id,
        side="SELL",
        filled_shares=shares,
        avg_price=settlement_price,
        order={"settlement": True, "winning_side": side_name},
        model_id=model_id,
        is_simulation=simulation_mode,
    )

    revenue = shares * settlement_price
    cost = shares * avg_cost
    pnl = revenue - cost
    db.update_account_balance(
        model_id=model_id,
        cash_change=revenue,
        position_value=0.0,
        unrealized_pnl=0.0,
        realized_pnl_change=pnl,
    )
    pnl_color = chalk.green if pnl >= 0 else chalk.red
    logger.info(
        "  [%s] Settled %s: %.2f shares @ $%.2f -> %s",
        model_id,
        side_name,
        shares,
        settlement_price,
        pnl_color(f"${pnl:+.2f}"),
    )


def _settle_market_if_needed(
    *,
    db: SQLiteDB,
    cached_hourly: dict[str, Any],
    final_btc_price: float | None,
    price_to_beat: float | None,
    simulation_mode: bool,
    model_pm_clients: dict[str, PolymarketClient] | None = None,
    auto_redeem_use_relayer: bool = True,
) -> set[str]:
    up_id = cached_hourly.get("up_token_id")
    down_id = cached_hourly.get("down_token_id")
    up_bid = _safe_float((cached_hourly.get("up") or {}).get("best_bid"), default=-1.0)
    down_bid = _safe_float((cached_hourly.get("down") or {}).get("best_bid"), default=-1.0)
    condition_id = (
        (cached_hourly.get("market") or {}).get("condition_id")
        or cached_hourly.get("condition_id")
        or ""
    )
    token_index_sets_raw = (
        (cached_hourly.get("market") or {}).get("token_index_sets")
        or cached_hourly.get("token_index_sets")
        or {}
    )
    token_index_sets: dict[str, int] = {}
    if isinstance(token_index_sets_raw, dict):
        for k, v in token_index_sets_raw.items():
            try:
                token_index_sets[str(k)] = int(v)
            except Exception:
                continue

    winning_side: str | None = None
    winning_token_id: str | None = None
    losing_token_id: str | None = None

    if up_bid >= 0 and down_bid >= 0:
        winning_side = "UP" if up_bid >= down_bid else "DOWN"
        winning_token_id = up_id if winning_side == "UP" else down_id
        losing_token_id = down_id if winning_side == "UP" else up_id
    elif final_btc_price is not None and price_to_beat is not None:
        winning_side = "UP" if final_btc_price > price_to_beat else "DOWN"
        winning_token_id = up_id if winning_side == "UP" else down_id
        losing_token_id = down_id if winning_side == "UP" else up_id

    settlement_prices: dict[str, float] = {}
    if winning_token_id and losing_token_id:
        settlement_prices[winning_token_id] = 1.0
        settlement_prices[losing_token_id] = 0.0
    else:
        for tid in [up_id, down_id]:
            if isinstance(tid, str) and tid:
                settlement_prices[tid] = 0.0

    logger.info(
        "Settlement prices: %s",
        ", ".join(f"{k[:8]}...=${v:.2f}" for k, v in settlement_prices.items()),
    )

    settled_models: set[str] = set()

    for model_id in _all_model_ids(db):
        all_holdings = _query_net_holdings(db, model_id)
        if not all_holdings:
            continue
        logger.info("  [%s] Holdings to settle: %s", model_id, len(all_holdings))

        can_settle_current_market = True
        holds_current_market = any(tid in {up_id, down_id} for tid in all_holdings)
        if (
            not simulation_mode
            and holds_current_market
            and winning_token_id
            and condition_id
            and model_pm_clients
            and model_id in model_pm_clients
        ):
            # Live mode: redeem on-chain first; only then write local settlement trades.
            try:
                model_pm_clients[model_id].redeem_condition(
                    condition_id=str(condition_id),
                    index_sets=sorted(
                        {
                            int(token_index_sets.get(str(up_id), 1)),
                            int(token_index_sets.get(str(down_id), 2)),
                        }
                    ),
                    use_relayer=auto_redeem_use_relayer,
                    metadata=f"Hourly settlement redeem {str(condition_id)[:10]}",
                )
                logger.info("  [%s] ✅ Settlement redeem submitted", model_id)
            except Exception as redeem_error:
                can_settle_current_market = False
                logger.warning(
                    "  [%s] Settlement redeem skipped/failed, keep positions for retry: %s",
                    model_id,
                    redeem_error,
                )

        for token_id, shares in all_holdings.items():
            positions = db.get_positions([token_id], model_id=model_id)
            avg_cost = _safe_float((positions.get(token_id) or {}).get("avg_price"), default=0.0)
            if token_id in (up_id, down_id):
                if not simulation_mode and not can_settle_current_market:
                    continue
                settlement_price = _safe_float(settlement_prices.get(token_id), default=0.0)
                side_name = "UP" if token_id == up_id else "DOWN"
            else:
                if not simulation_mode:
                    # Live mode: do not force-write stale holdings to 0.
                    continue
                settlement_price = 0.0
                side_name = "OLD"
            _settle_position(
                db=db,
                model_id=model_id,
                token_id=token_id,
                shares=shares,
                settlement_price=settlement_price,
                side_name=side_name,
                simulation_mode=simulation_mode,
                avg_cost=avg_cost,
            )
            if token_id in (up_id, down_id):
                settled_models.add(model_id)

    return settled_models


def _startup_cleanup_stale_positions(
    db: SQLiteDB, current_token_ids: list[str], simulation_mode: bool
) -> None:
    if not simulation_mode:
        logger.info("Startup cleanup: live mode, skip force-zero stale positions")
        return
    logger.info("Startup cleanup: checking for stale positions...")
    for model_id in _all_model_ids(db):
        all_holdings = _query_net_holdings(db, model_id)
        stale = {tid: sz for tid, sz in all_holdings.items() if tid not in current_token_ids}
        if not stale:
            continue
        logger.info("  [%s] Found %d stale positions", model_id, len(stale))
        for token_id, shares in stale.items():
            pos = db.get_positions([token_id], model_id=model_id).get(token_id, {})
            avg_cost = _safe_float(pos.get("avg_price"), default=0.0)
            _settle_position(
                db=db,
                model_id=model_id,
                token_id=token_id,
                shares=shares,
                settlement_price=0.0,
                side_name="OLD",
                simulation_mode=simulation_mode,
                avg_cost=avg_cost,
            )


def _settlement_price_from_position_row(row: dict[str, Any]) -> float:
    size = _safe_float(row.get("size"), default=0.0)
    cur = _safe_float(row.get("currentValue"), default=0.0)
    if size > 0:
        px = cur / size
        if px < 0:
            return 0.0
        if px > 1:
            return 1.0
        return px
    if cur > 0:
        return 1.0
    return 0.0


def _auto_redeem_model_resolved_positions(
    *,
    model_cfg: dict[str, Any],
    db: SQLiteDB,
    s: Settings,
) -> None:
    if s.simulation_mode or not s.auto_redeem_enabled:
        return

    model_id = model_cfg["model_id"]
    pm_client: PolymarketClient = model_cfg["pm_client"]

    try:
        rows = pm_client.get_positions(redeemable_only=True, size_threshold=0.0)
    except Exception as e:
        logger.warning("[%s] Auto-redeem scan failed: %s", model_id, e)
        return

    if not rows:
        return

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        condition_id = str(row.get("conditionId") or "").strip().lower()
        if not condition_id:
            continue
        grouped.setdefault(condition_id, []).append(row)

    if not grouped:
        return

    for condition_id, entries in grouped.items():
        if any(bool(e.get("negativeRisk")) for e in entries):
            logger.warning("[%s] Skip neg-risk auto-redeem for %s", model_id, condition_id[:10])
            continue

        redeem_value = sum(max(0.0, _safe_float(e.get("currentValue"), default=0.0)) for e in entries)
        if redeem_value < float(s.auto_redeem_min_value_usd):
            continue

        index_sets: set[int] = set()
        for e in entries:
            try:
                idx = int(e.get("outcomeIndex"))
            except Exception:
                continue
            if idx >= 0:
                index_sets.add(1 << idx)
        if not index_sets:
            index_sets = {1, 2}

        try:
            redeem_result = pm_client.redeem_condition(
                condition_id=condition_id,
                index_sets=sorted(index_sets),
                use_relayer=s.auto_redeem_use_relayer,
                metadata=f"Auto redeem {condition_id[:10]}",
            )
        except Exception as redeem_error:
            logger.warning(
                "[%s] Auto-redeem failed for %s: %s",
                model_id,
                condition_id[:10],
                redeem_error,
            )
            continue

        holdings = _query_net_holdings(db, model_id)
        settled_token_ids: list[str] = []
        for row in entries:
            token_id = str(row.get("asset") or "").strip()
            if not token_id:
                continue
            shares = _safe_float(holdings.get(token_id), default=0.0)
            if shares <= 0:
                continue
            pos = db.get_positions([token_id], model_id=model_id).get(token_id, {})
            avg_cost = _safe_float(pos.get("avg_price"), default=0.0)
            settlement_price = _settlement_price_from_position_row(row)
            outcome = str(row.get("outcome") or "OUTCOME").upper()
            _settle_position(
                db=db,
                model_id=model_id,
                token_id=token_id,
                shares=shares,
                settlement_price=settlement_price,
                side_name=outcome,
                simulation_mode=False,
                avg_cost=avg_cost,
            )
            settled_token_ids.append(token_id)

        if settled_token_ids:
            try:
                db.clear_session_trades(settled_token_ids, model_id=model_id)
            except Exception as clear_error:
                logger.warning("[%s] Failed to clear redeemed trades: %s", model_id, clear_error)

        tx_hash = None
        if isinstance(redeem_result, dict):
            tx_hash = redeem_result.get("tx_hash")

        db.log(
            ts=iso(utcnow()),
            type_="auto_redeem",
            payload_json=json.dumps(
                {
                    "condition_id": condition_id,
                    "tx_hash": tx_hash,
                    "redeem_value": redeem_value,
                    "token_ids": settled_token_ids,
                    "result": redeem_result,
                },
                ensure_ascii=False,
            ),
            model_id=model_id,
        )
        logger.info(
            "[%s] ✅ Auto-redeemed %s | value≈$%.4f | tx=%s",
            model_id,
            condition_id[:10],
            redeem_value,
            tx_hash or "n/a",
        )

        # Sync cash with real on-chain balance after redemption.
        try:
            real_balance = pm_client.get_account_balance()
            current_cash = db.get_account_balance(model_id).get("cash_balance", 0.0)
            db.update_account_balance(
                model_id=model_id,
                cash_change=real_balance - current_cash,
                position_value=0.0,
                unrealized_pnl=0.0,
                realized_pnl_change=0.0,
                skip_snapshot=True,
            )
        except Exception as bal_err:
            logger.warning("[%s] Failed to sync balance after auto-redeem: %s", model_id, bal_err)


def cleanup_handler(signum, frame):
    """信号处理器：优雅关闭所有连接。"""
    global _cleanup_done
    if _cleanup_done:
        logger.info("Force exit on second Ctrl+C")
        sys.exit(0)

    _cleanup_done = True
    _stop_event.set()
    logger.info("Received shutdown signal, cleaning up...")
    logger.info("Press Ctrl+C again to force exit immediately")

    if _ohlcv_source is not None:
        try:
            _ohlcv_source.close()
            logger.info("OHLCV connections closed")
        except Exception as e:
            logger.error(f"Error closing OHLCV connections: {e}")

    # Keep shutdown deterministic even if background threads hang.
    import os

    os._exit(0)


def data_prefetch_thread(
    pm: PolymarketClient,
    ohlcv: OHLCVSource,
    symbol: str,
    s: Settings,
    db: SQLiteDB,
    model_pm_clients: dict[str, PolymarketClient] | None = None,
):
    """
    数据预获取线程：优化策略
    - 市场信息：首次运行时获取，场次结束时更新
    - 价格/余额：每秒更新
    - 盈亏显示：每秒打印
    """
    global _error_count, _last_error_time

    logger.info("Data prefetch thread started (smart update strategy)")
    current_token_ids: list[str] = []
    market_end_time: dt.datetime | None = None
    first_run = True
    startup_cleanup_done = False

    while not _stop_event.is_set():
        try:
            max_retries = 3
            retry_delay_s = 0.5
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
                    time.sleep(retry_delay_s)

            first_run = False

            if current_btc_price is not None and current_btc_price > 0:
                with cache_lock:
                    cached_data["current_btc_price"] = current_btc_price
            else:
                logger.warning("Unable to get current BTC price, will retry next cycle")

            need_market_update = False
            now_utc = utcnow()

            with cache_lock:
                cached_hourly = cached_data.get("hourly_market")
                cached_price_to_beat = cached_data.get("price_to_beat")

            if cached_hourly is None:
                need_market_update = True
            elif market_end_time and now_utc >= market_end_time:
                need_market_update = True
                logger.info("Current market ended, fetching next market")
                try:
                    final_btc_price = current_btc_price
                    if final_btc_price is None:
                        final_btc_price = ohlcv.get_current_price(symbol, "1h")
                    logger.info(
                        "Settlement check: final_btc=%s, price_to_beat=%s",
                        final_btc_price,
                        cached_price_to_beat,
                    )
                    settled_models = _settle_market_if_needed(
                        db=db,
                        cached_hourly=cached_hourly,
                        final_btc_price=final_btc_price,
                        price_to_beat=_safe_float(cached_price_to_beat, default=0.0)
                        if cached_price_to_beat is not None
                        else None,
                        simulation_mode=s.simulation_mode,
                        model_pm_clients=model_pm_clients,
                        auto_redeem_use_relayer=s.auto_redeem_use_relayer,
                    )

                    settled_token_ids = [
                        t
                        for t in [cached_hourly.get("up_token_id"), cached_hourly.get("down_token_id")]
                        if t
                    ]
                    if settled_token_ids and settled_models:
                        for model_id in settled_models:
                            try:
                                db.clear_session_trades(settled_token_ids, model_id=model_id)
                            except Exception as clear_error:
                                logger.warning(
                                    "Failed to clear trades for [%s]: %s", model_id, clear_error
                                )
                except Exception as settle_error:
                    logger.error("Failed to settle positions: %s", settle_error)

            if need_market_update:
                logger.info("🔄 Fetching new market info")
                hourly_raw = pm.get_hourly_market_prices(prefix=s.polymarket_hourly_prefix)
                hourly = _normalize_hourly(hourly_raw)

                end_time_utc = hourly.get("end_time_utc")
                end_dt = _parse_utc(end_time_utc)
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
                current_token_ids = [
                    t for t in [up_id, down_id] if isinstance(t, str) and t
                ]

                if not startup_cleanup_done:
                    startup_cleanup_done = True
                    if current_token_ids:
                        try:
                            _startup_cleanup_stale_positions(
                                db, current_token_ids, s.simulation_mode
                            )
                        except Exception as cleanup_error:
                            logger.error("Startup cleanup failed: %s", cleanup_error)

                price_to_beat = None
                event_start = (hourly.get("market") or {}).get("event_start_time")
                if isinstance(event_start, str) and event_start:
                    start_dt = _parse_utc(event_start)
                    if start_dt is not None:
                        price_to_beat = ohlcv.fetch_futures_open_at_cached(symbol, start_dt)
                if price_to_beat is None and end_dt is not None:
                    start_dt = end_dt - dt.timedelta(hours=1)
                    price_to_beat = ohlcv.fetch_futures_open_at_cached(symbol, start_dt)

                with cache_lock:
                    cached_data["hourly_market"] = hourly
                    cached_data["token_ids"] = current_token_ids
                    cached_data["price_to_beat"] = price_to_beat

                try:
                    from pm_agent.server import update_market_prices

                    with cache_lock:
                        ptb = cached_data.get("price_to_beat")
                        cbtc = cached_data.get("current_btc_price")

                    update_market_prices(
                        up_bid=(hourly.get("up") or {}).get("best_bid"),
                        up_ask=(hourly.get("up") or {}).get("best_ask"),
                        down_bid=(hourly.get("down") or {}).get("best_bid"),
                        down_ask=(hourly.get("down") or {}).get("best_ask"),
                        slug=hourly.get("slug"),
                        title=(hourly.get("market") or {}).get("question"),
                        end_time=hourly.get("end_time_utc"),
                        price_to_beat=ptb,
                        current_btc_price=cbtc,
                        up_token_id=up_id,
                        down_token_id=down_id,
                    )
                except Exception as e:
                    logger.warning("Failed to update server price cache: %s", e)

            with cache_lock:
                hourly = cached_data.get("hourly_market")

            if hourly and current_token_ids:
                # Refresh orderbook every cycle for best bid/ask.
                orderbooks = pm._fetch_orderbooks(current_token_ids)

                up_token_id = hourly.get("up_token_id")
                down_token_id = hourly.get("down_token_id")

                if up_token_id and up_token_id in orderbooks:
                    ob = orderbooks[up_token_id]
                    (hourly["up"] or {})["best_bid"] = _best(ob, "bids")
                    (hourly["up"] or {})["best_ask"] = _best(ob, "asks")
                if down_token_id and down_token_id in orderbooks:
                    ob = orderbooks[down_token_id]
                    (hourly["down"] or {})["best_bid"] = _best(ob, "bids")
                    (hourly["down"] or {})["best_ask"] = _best(ob, "asks")

                with cache_lock:
                    cached_data["hourly_market"] = hourly

                try:
                    from pm_agent.server import update_market_prices

                    with cache_lock:
                        ptb = cached_data.get("price_to_beat")
                        cbtc = cached_data.get("current_btc_price")

                    update_market_prices(
                        up_bid=(hourly.get("up") or {}).get("best_bid"),
                        up_ask=(hourly.get("up") or {}).get("best_ask"),
                        down_bid=(hourly.get("down") or {}).get("best_bid"),
                        down_ask=(hourly.get("down") or {}).get("best_ask"),
                        slug=hourly.get("slug"),
                        title=(hourly.get("market") or {}).get("question"),
                        end_time=hourly.get("end_time_utc"),
                        price_to_beat=ptb,
                        current_btc_price=cbtc,
                        up_token_id=up_token_id,
                        down_token_id=down_token_id,
                    )
                except Exception as e:
                    logger.warning("Failed to update server price cache: %s", e)

                if s.simulation_mode:
                    balances: dict[str, float] = {}
                else:
                    balances = pm.get_token_balances(current_token_ids)

                with cache_lock:
                    cached_data["balances"] = balances
                    cached_data["last_update"] = time.time()

            _error_count = 0
            _last_error_time = time.time()
            time.sleep(1)
        except Exception as e:
            msg = str(e).lower()
            is_network = any(x in msg for x in ("ssl", "eof", "connection", "timeout"))
            _error_count += 1
            _last_error_time = time.time()

            if is_network:
                logger.warning("Network issue #%d (will retry): %s", _error_count, e)
                wait_time = min(10, 5 + _error_count)
                if _error_count >= 3:
                    logger.warning("Multiple network failures, waiting %ss...", wait_time)
                time.sleep(wait_time)
            else:
                logger.error("Data prefetch error: %s", e, exc_info=True)
                time.sleep(min(10, 5 + _error_count))


def _validate_ai_keys(s: Settings, ai_providers: list[str]) -> None:
    missing_keys: list[str] = []
    for provider_name in ai_providers:
        provider = provider_name.lower()
        if provider == "openai" and not s.openai_api_key:
            missing_keys.append(provider_name)
        elif provider == "deepseek" and not s.deepseek_api_key:
            missing_keys.append(provider_name)
        elif provider == "gemini" and not s.gemini_api_key:
            missing_keys.append(provider_name)
        elif provider == "claude" and not s.claude_api_key:
            missing_keys.append(provider_name)
        elif provider == "qwen" and not s.qwen_api_key:
            missing_keys.append(provider_name)

    if missing_keys:
        raise SystemExit(
            "API key missing for: "
            + ", ".join(missing_keys)
            + ". Set it in environment or .env"
        )


def _build_ai_client_and_display(s: Settings, provider: str) -> tuple[AIClient, str]:
    if provider == "deepseek":
        return (
            AIClient(
                api_key=s.deepseek_api_key,
                model=s.deepseek_model,
                base_url=s.deepseek_base_url,
                timeout_s=120,
                provider=provider,
            ),
            f"DeepSeek ({s.deepseek_model})",
        )
    if provider == "gemini":
        return (
            AIClient(
                api_key=s.gemini_api_key,
                model=s.gemini_model,
                base_url=s.gemini_base_url,
                timeout_s=120,
                provider=provider,
            ),
            f"Gemini ({s.gemini_model})",
        )
    if provider == "claude":
        return (
            AIClient(
                api_key=s.claude_api_key,
                model=s.claude_model,
                base_url=s.claude_base_url,
                timeout_s=120,
                provider=provider,
            ),
            f"Claude ({s.claude_model})",
        )
    if provider == "qwen":
        return (
            AIClient(
                api_key=s.qwen_api_key,
                model=s.qwen_model,
                base_url=s.qwen_base_url,
                timeout_s=120,
                provider=provider,
            ),
            f"Qwen ({s.qwen_model})",
        )
    return (
        AIClient(
            api_key=s.openai_api_key,
            model=s.openai_model,
            timeout_s=120,
            provider="openai",
        ),
        f"OpenAI ({s.openai_model})",
    )


def main():
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)

    env_candidates = [
        Path.cwd() / ".env",
        Path(sys.executable).resolve().parent / ".env",
        Path(__file__).resolve().parents[1] / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]
    for env_path in env_candidates:
        if env_path.is_file():
            # Runtime env should always win over .env values.
            load_dotenv(dotenv_path=env_path, override=False)

    s = load_settings()
    ai_providers = [p.lower() for p in s.get_ai_providers()]
    _validate_ai_keys(s, ai_providers)

    db = SQLiteDB(db_path=s.get_db_path())

    pm = PolymarketClient(
        host=s.polymarket_host,
        chain_id=s.polymarket_chain_id,
        api_key=s.polymarket_api_key,
        api_secret=s.polymarket_api_secret,
        api_passphrase=s.polymarket_api_passphrase,
        private_key=s.polymarket_private_key,
        wallet_type=s.polymarket_wallet_type,
        signature_type=s.polymarket_signature_type,
        funder=s.polymarket_funder,
        relayer_url=s.polymarket_relayer_url,
    )
    ohlcv = OHLCVSource()

    global _ohlcv_source
    _ohlcv_source = ohlcv

    start_background_server()

    if not getattr(sys, "frozen", False):

        def open_browser():
            time.sleep(2)
            webbrowser.open("http://localhost:8000")

        threading.Thread(target=open_browser, daemon=True).start()

    private_keys = s.get_private_keys()
    if private_keys and len(private_keys) < len(ai_providers):
        logger.warning(
            "Private keys (%s) < models (%s). Reusing first private key for all models - "
            "profits will NOT be isolated!",
            len(private_keys),
            len(ai_providers),
        )

    models_config: list[dict[str, Any]] = []
    for idx, provider in enumerate(ai_providers):
        model_private_key = (
            private_keys[idx]
            if idx < len(private_keys)
            else (private_keys[0] if private_keys else s.polymarket_private_key)
        )

        model_pm = PolymarketClient(
            host=s.polymarket_host,
            chain_id=s.polymarket_chain_id,
            api_key=s.polymarket_api_key,
            api_secret=s.polymarket_api_secret,
            api_passphrase=s.polymarket_api_passphrase,
            private_key=model_private_key,
            wallet_type=s.polymarket_wallet_type,
            signature_type=s.polymarket_signature_type,
            funder=s.polymarket_funder,
            relayer_url=s.polymarket_relayer_url,
        )

        ai, display_name = _build_ai_client_and_display(s, provider)
        model_id = f"{provider}-{ai.model}"
        orchestrator = StrategyOrchestrator(mode=s.mode, symbol=s.symbol, ohlcv=ohlcv, ai=ai)
        engine = ExecutionEngine(
            pm=model_pm,
            db=db,
            limits=RiskLimits(
                max_notional_usd=s.max_notional_usd,
                max_position_usd=s.max_position_usd,
                max_daily_trades=s.max_daily_trades,
                slippage_bps=s.slippage_bps,
            ),
            simulation_mode=s.simulation_mode,
            model_id=model_id,
        )

        db.init_account_balance(model_id, s.initial_balance_usd)

        if not s.simulation_mode:
            try:
                real_balance = model_pm.get_account_balance()
                current = db.get_account_balance(model_id).get("cash_balance", 0.0)
                db.update_account_balance(
                    model_id=model_id,
                    cash_change=real_balance - current,
                    position_value=0.0,
                    unrealized_pnl=0.0,
                    realized_pnl_change=0.0,
                    skip_snapshot=True,
                )
                logger.info("  Recorded initial on-chain balance: $%.2f", real_balance)
            except Exception as e:
                logger.warning("  Failed to record initial balance: %s", e)

        models_config.append(
            {
                "provider": provider,
                "model_id": model_id,
                "display_name": display_name,
                "orchestrator": orchestrator,
                "engine": engine,
                "pm_client": model_pm,
            }
        )
        logger.info("  %s initialized (model_id=%s)", display_name, model_id)

    if not models_config:
        raise SystemExit("No model initialized.")

    set_runtime_models([m["model_id"] for m in models_config])

    logger.info(
        "🏁 Multi-Model Arena: %d models competing: %s",
        len(models_config),
        [m["provider"] for m in models_config],
    )
    mode_str = "[SIMULATION MODE]" if s.simulation_mode else "[LIVE MODE]"
    logger.info("🎯 %s - %d model(s) initialized", mode_str, len(models_config))

    # Keep logger tag stable for shared console.
    if len(models_config) == 1:
        one = models_config[0]
        set_ai_model_info(one["provider"], one["model_id"])
    else:
        set_ai_model_info("multi", f"{len(models_config)} models")

    model_pm_clients = {m["model_id"]: m["pm_client"] for m in models_config}
    prefetch_thread = threading.Thread(
        target=data_prefetch_thread,
        args=(pm, ohlcv, s.symbol, s, db, model_pm_clients),
        daemon=True,
    )
    prefetch_thread.start()
    logger.info("🚀 Data prefetch thread started (updating every 1s)")

    max_wait = 5
    for i in range(max_wait):
        with cache_lock:
            if cached_data.get("hourly_market"):
                break
        logger.info("Waiting for initial data cache... (%s/%ss)", i + 1, max_wait)
        time.sleep(1)

    logger.info("Agent running in logger mode")
    logger.info("提示：使用 'python -m pm_agent.live_stats' 启动实时统计仪表盘")

    last_ai_end_time = 0.0
    last_auto_redeem_ts = 0.0

    def process_model_decision(model_cfg: dict[str, Any]) -> dict[str, Any]:
        """处理单个模型的决策和执行。"""
        model_id = model_cfg["model_id"]
        display_name = model_cfg["display_name"]
        orchestrator_instance = model_cfg["orchestrator"]
        engine_instance = model_cfg["engine"]

        logger.info(chalk.bold(chalk.magenta(f"\n🤖 [{display_name}] Making decision...")))
        try:
            with cache_lock:
                hourly = dict(cached_data.get("hourly_market") or {})
                balances = dict(cached_data.get("balances") or {})
                price_to_beat = cached_data.get("price_to_beat")
                current_btc = cached_data.get("current_btc_price")

            up_id = hourly.get("up_token_id")
            down_id = hourly.get("down_token_id")
            token_ids = [t for t in [up_id, down_id] if t]
            current_prices = None
            if up_id and down_id:
                current_prices = {
                    up_id: (hourly.get("up") or {}).get("best_bid"),
                    down_id: (hourly.get("down") or {}).get("best_bid"),
                }

            model_db_positions = db.get_positions(
                token_ids,
                actual_balances=(balances if (token_ids and not s.simulation_mode) else None),
                model_id=model_id,
            )
            model_session_stats = (
                db.get_session_stats(
                    token_ids=token_ids,
                    current_positions=model_db_positions,
                    current_prices=current_prices or {},
                    model_id=model_id,
                )
                if token_ids
                else None
            )
            model_decision_history = db.get_decision_history(
                token_ids=token_ids,
                up_token_id=up_id or "",
                down_token_id=down_id or "",
                actual_balances=balances,
                model_id=model_id,
            )

            warning: list[str] = []
            if token_ids and not model_session_stats:
                warning.append("尚无交易记录，但有持仓。")

            model_positions_list = []
            for label, token_id in (("UP", up_id), ("DOWN", down_id)):
                if not token_id:
                    continue
                pos = model_db_positions.get(token_id, {})
                shares = _safe_float(pos.get("shares"), default=0.0)
                avg_cost = _safe_float(pos.get("avg_price"), default=0.0)
                book = hourly.get("up") if label == "UP" else hourly.get("down")
                mark = _safe_float((book or {}).get("best_bid"), default=0.0)
                pnl = (mark - avg_cost) * shares if mark > 0 else 0.0
                model_positions_list.append(
                    {
                        "token_id": token_id,
                        "side": label.lower(),
                        "shares": shares,
                        "avg_cost": avg_cost,
                        "mark_price": mark,
                        "unrealized_pnl": pnl,
                    }
                )

            model_cash = db.get_account_balance(model_id).get("cash_balance", 0.0)
            account_info = {
                "cash_usdc": model_cash,
                "cash_balance": model_cash,
                "position_intelligence": {
                    "trade_history": db.get_trade_history(token_ids, limit=50) if token_ids else [],
                    "pattern_analysis": {},
                    "warnings": warning,
                    "ai_notes": "",
                    "position_intelligence": model_positions_list,
                },
            }

            end_time_utc = hourly.get("end_time_utc")
            end_dt = _parse_utc(end_time_utc)
            time_to_end_sec = 0
            if end_dt is not None:
                time_to_end_sec = int((end_dt - utcnow()).total_seconds())
                if time_to_end_sec < 0:
                    time_to_end_sec = 0

            price_diff = None
            price_diff_pct = None
            btc_is_winning = "TIE"
            if current_btc and price_to_beat:
                price_diff = float(current_btc) - float(price_to_beat)
                if price_to_beat:
                    price_diff_pct = (price_diff / float(price_to_beat)) * 100
                if price_diff > 0:
                    btc_is_winning = "UP"
                elif price_diff < 0:
                    btc_is_winning = "DOWN"

            minutes_remaining = max(0, int(time_to_end_sec / 60))
            phase, phase_note = _compute_phase(minutes_remaining)
            btc_momentum = "neutral"
            if isinstance(price_diff_pct, (int, float)):
                if price_diff_pct > 0.1:
                    btc_momentum = "up"
                elif price_diff_pct < -0.1:
                    btc_momentum = "down"

            btc_analysis = {
                "current_price": current_btc,
                "price_to_beat": price_to_beat,
                "price_diff": price_diff,
                "price_diff_pct": price_diff_pct,
                "btc_is_winning": btc_is_winning,
                "btc_momentum": btc_momentum,
            }
            session_phase = {
                "phase": phase,
                "minutes_remaining": minutes_remaining,
                "phase_note": phase_note,
            }
            market_overview = {
                "symbol": s.symbol,
                "hourly_market_slug": hourly.get("slug"),
                "question": (hourly.get("market") or {}).get("question"),
                "end_time_utc": end_time_utc,
                "time_to_end_sec": time_to_end_sec,
                "price_to_beat": price_to_beat,
                "up": hourly.get("up"),
                "down": hourly.get("down"),
                "btc_analysis": btc_analysis,
                "session_phase": session_phase,
                "_market_price_warning": (
                    "重要：UP/DOWN价格是结果不是原因。价格=f(price_diff, time_remaining)，"
                    "决策应以技术指标为依据。"
                ),
            }

            candidate_markets = []
            for item in model_positions_list:
                shares = _safe_float(item.get("shares"), default=0.0)
                avg_cost = _safe_float(item.get("avg_cost"), default=0.0)
                current_price = _safe_float(item.get("mark_price"), default=0.0)
                pnl_usdc = (current_price - avg_cost) * shares
                pnl_pct = ((current_price - avg_cost) / avg_cost * 100) if avg_cost > 0 else 0.0
                candidate_markets.append(
                    {
                        "token_id": item["token_id"],
                        "side": item["side"],
                        "shares": shares,
                        "avg_cost": avg_cost,
                        "current_price": current_price,
                        "pnl_pct": pnl_pct,
                        "pnl_usdc": pnl_usdc,
                        "vs_current_market": True,
                    }
                )

            if model_session_stats and model_session_stats.get("trade_count", 0) > 0:
                realized_pnl = _safe_float(model_session_stats.get("realized_pnl"), default=0.0)
                unrealized_pnl = _safe_float(model_session_stats.get("unrealized_pnl"), default=0.0)
                total_pnl = realized_pnl + unrealized_pnl
                pnl_color = chalk.green if total_pnl >= 0 else chalk.red
                pnl_text = pnl_color(f"${total_pnl:+.2f}")
                upnl_text = (
                    f" (Unrealized: ${unrealized_pnl:+.2f})" if abs(unrealized_pnl) >= 1e-9 else ""
                )
                session_history_text = (
                    f"📈 Session Stats: Trades: {int(model_session_stats.get('trade_count', 0))}"
                    f" | P&L: {pnl_text}{upnl_text}"
                )
            else:
                session_history_text = "📈 Session Stats: No trades yet"

            model_context = {
                "system_state": {
                    "cycle_id": str(uuid.uuid4()),
                    "model_id": model_id,
                    "display_name": display_name,
                    "time_since_last_ai": max(0.0, time.time() - last_ai_end_time),
                },
                "market_overview": market_overview,
                "account": account_info,
                "account_info": account_info,
                "positions": model_db_positions,
                "current_positions": model_db_positions,
                "candidate_markets": candidate_markets,
                "session_history": session_history_text,
                "decision_history": model_decision_history,
                "risk_limits": {
                    "max_notional_usd": s.max_notional_usd,
                    "max_position_usd": s.max_position_usd,
                },
            }
            cycle_id = model_context["system_state"]["cycle_id"]

            db.log(
                ts=iso(utcnow()),
                type_="run_start",
                payload_json=json.dumps(
                    {"cycle_id": cycle_id},
                    ensure_ascii=False,
                ),
                model_id=model_id,
            )

            if _stop_event.is_set():
                logger.info("  [%s] Skipping AI call due to shutdown signal", display_name)
                return {"model_id": model_id, "display_name": display_name, "status": "skipped"}

            ai_start_time = time.time()
            try:
                parsed = orchestrator_instance.run_once(cycle_id=cycle_id, context=model_context)
            except KeyboardInterrupt:
                logger.info("  [%s] AI call interrupted", display_name)
                _stop_event.set()
                return {"model_id": model_id, "display_name": display_name, "status": "interrupted"}

            ai_elapsed = time.time() - ai_start_time
            logger.info(chalk.cyan(f"  [{display_name}] AI reasoning took {ai_elapsed:.2f}s"))

            db.log(
                ts=iso(utcnow()),
                type_="ai_raw",
                payload_json=json.dumps(
                    {"cycle_id": cycle_id, "raw": parsed.raw},
                    ensure_ascii=False,
                ),
                model_id=model_id,
            )
            db.log(
                ts=iso(utcnow()),
                type_="ai_decision",
                payload_json=json.dumps(
                    {
                        "cycle_id": cycle_id,
                        "decision": parsed.decision,
                        "reasoning": parsed.reasoning,
                    },
                    ensure_ascii=False,
                ),
                model_id=model_id,
            )

            actions = parsed.decision.get("actions", [])
            if actions:
                action_summaries = []
                for a in actions:
                    act_type = a.get("type", "unknown")
                    side = a.get("side", "")
                    size = a.get("size", 0)
                    price = a.get("price", 0)
                    if act_type in ("open", "close"):
                        action_summaries.append(f"{str(act_type).upper()} {side} {size} shares@{price:.2f}")
                    else:
                        action_summaries.append(str(act_type).upper())
                ai_action_text = " | ".join(action_summaries)
            else:
                ai_action_text = "HOLD/WAIT"

            print(f"  {chalk.bold(chalk.magenta('Decision:'))} {chalk.bright_white(ai_action_text)}")
            reasoning_text = parsed.reasoning or "(empty)"
            if len(reasoning_text) > 200:
                print("  Reasoning:")
                for line in reasoning_text.split("\n"):
                    print(f"    {chalk.bright_white(line)}")
            else:
                print(f"  Reasoning: {chalk.bright_white(reasoning_text)}")

            results = engine_instance.execute(parsed.decision)
            db.log(
                ts=iso(utcnow()),
                type_="run_end",
                payload_json=json.dumps({"cycle_id": cycle_id, "results": results}, ensure_ascii=False),
                model_id=model_id,
            )
            for r in results:
                status = str(r.get("status", "unknown"))
                if status in {"filled", "success", "submitted", "simulated"}:
                    status_emoji = "✓"
                    status_color = chalk.green
                elif status in {"skipped"}:
                    status_emoji = "→"
                    status_color = chalk.dim
                else:
                    status_emoji = "✗"
                    status_color = chalk.red
                print(f"  {status_color(f'[{status_emoji}] {json.dumps(r, ensure_ascii=False)}')}")

            # Update per-model account snapshot after execution.
            all_token_ids = [t for t in [up_id, down_id] if t]
            model_positions = db.get_positions(
                all_token_ids,
                actual_balances=(balances if (all_token_ids and not s.simulation_mode) else None),
                model_id=model_id,
            )
            total_position_value = 0.0
            total_unrealized_pnl = 0.0
            for token_id in all_token_ids:
                pos = model_positions.get(token_id, {})
                shares = _safe_float(pos.get("shares"), default=0.0)
                avg_cost = _safe_float(pos.get("avg_price"), default=0.0)
                if token_id == up_id:
                    current_price = _safe_float((hourly.get("up") or {}).get("best_bid"), default=avg_cost)
                else:
                    current_price = _safe_float(
                        (hourly.get("down") or {}).get("best_bid"), default=avg_cost
                    )
                position_value = shares * current_price
                unrealized_pnl = shares * (current_price - avg_cost)
                total_position_value += position_value
                total_unrealized_pnl += unrealized_pnl

            db.update_account_balance(
                model_id=model_id,
                cash_change=0.0,
                position_value=total_position_value,
                unrealized_pnl=total_unrealized_pnl,
                realized_pnl_change=0.0,
            )

            return {
                "model_id": model_id,
                "display_name": display_name,
                "status": "ok",
                "results": results,
            }
        except Exception as model_error:
            logger.error(
                "Model processing failed [%s]: %s\n%s",
                display_name,
                model_error,
                traceback.format_exc(),
            )
            return {
                "model_id": model_id,
                "display_name": display_name,
                "status": "error",
                "error": str(model_error),
            }

    try:
        while not _stop_event.is_set():
            cycle_start_time = time.time()

            with cache_lock:
                hourly = dict(cached_data.get("hourly_market") or {})
                balances = dict(cached_data.get("balances") or {})
                price_to_beat = cached_data.get("price_to_beat")
                current_btc_price = cached_data.get("current_btc_price")
                data_age = (
                    time.time() - float(cached_data.get("last_update"))
                    if cached_data.get("last_update")
                    else None
                )

            if not hourly:
                logger.info("Waiting for cached data...")
                time.sleep(1)
                continue

            if data_age is not None and data_age > 60:
                logger.warning("Cached data is %.1fs old (prefetch thread may be stuck)", data_age)

            now_ts = time.time()
            if (
                not s.simulation_mode
                and s.auto_redeem_enabled
                and now_ts - last_auto_redeem_ts >= float(s.auto_redeem_interval_sec)
            ):
                for model_cfg in models_config:
                    try:
                        _auto_redeem_model_resolved_positions(model_cfg=model_cfg, db=db, s=s)
                    except Exception as auto_redeem_error:
                        logger.warning(
                            "[%s] auto-redeem loop error: %s",
                            model_cfg["model_id"],
                            auto_redeem_error,
                        )
                last_auto_redeem_ts = now_ts

            up_id = hourly.get("up_token_id")
            down_id = hourly.get("down_token_id")
            token_ids = [t for t in [up_id, down_id] if t]
            current_prices = None
            if up_id and down_id:
                current_prices = {
                    up_id: (hourly.get("up") or {}).get("best_bid"),
                    down_id: (hourly.get("down") or {}).get("best_bid"),
                }

            db_positions = db.get_positions(
                token_ids,
                actual_balances=(balances if (token_ids and not s.simulation_mode) else None),
                model_id="default",
            )

            up_shares = _safe_float(
                (balances.get(up_id) if (not s.simulation_mode and up_id) else None)
                or (db_positions.get(up_id) or {}).get("shares"),
                default=0.0,
            )
            down_shares = _safe_float(
                (balances.get(down_id) if (not s.simulation_mode and down_id) else None)
                or (db_positions.get(down_id) or {}).get("shares"),
                default=0.0,
            )
            has_position = up_shares >= 1 or down_shares >= 1

            base_interval = max(10, int(s.run_interval_sec))
            interval = base_interval if has_position else base_interval * 2
            if has_position:
                logger.debug("有持仓，使用正常间隔：%s秒", interval)
            else:
                logger.debug("无持仓，使用2倍间隔：%s秒", interval)

            up_best_bid = _safe_float((hourly.get("up") or {}).get("best_bid"), default=0.0)
            up_best_ask = _safe_float((hourly.get("up") or {}).get("best_ask"), default=0.0)
            down_best_bid = _safe_float((hourly.get("down") or {}).get("best_bid"), default=0.0)
            down_best_ask = _safe_float((hourly.get("down") or {}).get("best_ask"), default=0.0)

            up_avg_cost = _safe_float((db_positions.get(up_id) or {}).get("avg_price"), default=0.0)
            down_avg_cost = _safe_float((db_positions.get(down_id) or {}).get("avg_price"), default=0.0)

            slug = hourly.get("slug") or "Unknown"
            end_time_utc_str = hourly.get("end_time_utc")
            end_dt = _parse_utc(end_time_utc_str)
            now_utc = utcnow()
            if end_dt is not None:
                time_to_end = max(0, int((end_dt - now_utc).total_seconds()))
                minutes_left = int(time_to_end / 60)
            else:
                time_to_end = 0
                minutes_left = 0

            if not isinstance(current_btc_price, (int, float)) or current_btc_price <= 0:
                logger.error("Invalid BTC price, skipping this cycle")
                time.sleep(interval)
                continue

            up_pos_text = f"UP: {up_shares:.0f} @${up_avg_cost:.4f}" if up_shares >= 1 else "UP: -"
            down_pos_text = (
                f"DOWN: {down_shares:.0f} @${down_avg_cost:.4f}" if down_shares >= 1 else "DOWN: -"
            )

            session_stats = (
                db.get_session_stats(
                    token_ids=token_ids,
                    current_positions=db_positions,
                    current_prices=current_prices or {},
                )
                if token_ids
                else None
            )
            trade_count = int((session_stats or {}).get("trade_count", 0))
            total_pnl = _safe_float((session_stats or {}).get("total_pnl"), default=0.0)
            pnl_sign = "+" if total_pnl >= 0 else ""
            stats_text = (
                f"📈 {trade_count}笔 P&L: ${pnl_sign}{total_pnl:.2f}"
                if trade_count > 0
                else "📈 No trades"
            )

            btc_text = ""
            if price_to_beat:
                price_diff = float(current_btc_price) - float(price_to_beat)
                diff_sign = "+" if price_diff >= 0 else ""
                btc_text = f"💹 ${current_btc_price:.2f} vs 🎯 ${float(price_to_beat):.2f} ({diff_sign}{price_diff:.2f})"

            summary_line = (
                f"📊 {slug} | ⏰ {minutes_left}min | UP:{up_best_bid}/{up_best_ask} "
                f"DOWN:{down_best_bid}/{down_best_ask} | {up_pos_text} | {down_pos_text} | "
                f"{stats_text} {btc_text}"
            )

            print(chalk.separator())
            print(
                format_market_info(
                    slug=slug,
                    question=(hourly.get("market") or {}).get("question") or "Unknown",
                    time_to_end=f"{minutes_left} min",
                )
            )
            print(format_prices(up_best_bid, up_best_ask, down_best_bid, down_best_ask))
            cash_usdc = db.get_account_balance(model_id).get("cash_balance", 0.0)
            print(format_positions(up_shares, up_avg_cost, down_shares, down_avg_cost, cash_usdc))
            print(format_session_stats(session_stats))
            print(chalk.dim(summary_line))

            end_time_utc = hourly.get("end_time_utc")
            end_dt_for_ctx = _parse_utc(end_time_utc)
            time_to_end_sec = (
                max(0, int((end_dt_for_ctx - utcnow()).total_seconds())) if end_dt_for_ctx else 0
            )
            current_btc = _safe_float(current_btc_price, default=0.0)
            ptb = _safe_float(price_to_beat, default=0.0) if price_to_beat else None
            price_diff_pct = None
            btc_is_winning = "TIE"
            if ptb:
                price_diff = current_btc - ptb
                price_diff_pct = (price_diff / ptb) * 100 if ptb else None
                if price_diff > 0:
                    btc_is_winning = "UP"
                elif price_diff < 0:
                    btc_is_winning = "DOWN"

            minutes_remaining = max(0, int(time_to_end_sec / 60))
            phase, phase_note = _compute_phase(minutes_remaining)
            btc_momentum = "neutral"
            if isinstance(price_diff_pct, (int, float)):
                if price_diff_pct > 0.1:
                    btc_momentum = "up"
                elif price_diff_pct < -0.1:
                    btc_momentum = "down"

            btc_analysis = {
                "current_price": current_btc,
                "price_to_beat": ptb,
                "price_diff": (current_btc - ptb) if ptb else None,
                "price_diff_pct": price_diff_pct,
                "btc_is_winning": btc_is_winning,
                "btc_momentum": btc_momentum,
            }
            session_phase = {
                "phase": phase,
                "minutes_remaining": minutes_remaining,
                "phase_note": phase_note,
            }
            market_overview = {
                "symbol": s.symbol,
                "hourly_market_slug": slug,
                "question": (hourly.get("market") or {}).get("question"),
                "end_time_utc": end_time_utc,
                "time_to_end_sec": time_to_end_sec,
                "price_to_beat": ptb,
                "up": hourly.get("up"),
                "down": hourly.get("down"),
                "btc_analysis": btc_analysis,
                "session_phase": session_phase,
                "_market_price_warning": (
                    "重要：UP/DOWN价格是结果不是原因。价格=f(price_diff, time_remaining)，"
                    "不代表市场情绪；决策依据应是技术指标。"
                ),
            }

            candidate_markets: list[dict[str, Any]] = []
            for label, token_id in [("UP", up_id), ("DOWN", down_id)]:
                if not token_id:
                    continue
                pos = db_positions.get(token_id, {})
                shares = _safe_float(pos.get("shares"), default=0.0)
                avg_cost = _safe_float(pos.get("avg_price"), default=0.0)
                mark_price = (
                    _safe_float((hourly.get("up") or {}).get("best_bid"), default=0.0)
                    if label == "UP"
                    else _safe_float((hourly.get("down") or {}).get("best_bid"), default=0.0)
                )
                side = label.lower()
                pnl_pct = ((mark_price - avg_cost) / avg_cost * 100) if avg_cost > 0 else 0.0
                pnl_usdc = (mark_price - avg_cost) * shares
                candidate_markets.append(
                    {
                        "token_id": token_id,
                        "side": side,
                        "shares": shares,
                        "avg_cost": avg_cost,
                        "current_price": mark_price,
                        "pnl_pct": pnl_pct,
                        "pnl_usdc": pnl_usdc,
                        "vs_current_market": True,
                    }
                )

            account_info = db.get_account_balance(model_id)
            account_info["cash_usdc"] = account_info.get("cash_balance", 0.0)

            cycle_id = str(uuid.uuid4())
            context = {
                "system_state": {"cycle_id": cycle_id},
                "market_overview": market_overview,
                "account_info": account_info,
                "current_positions": db_positions,
                "candidate_markets": candidate_markets,
                "session_history": format_session_stats(session_stats),
            }

            for model_cfg in models_config:
                try:
                    result = process_model_decision(model_cfg)
                    if result.get("status") == "error":
                        logger.error(
                            "[%s] model run failed: %s",
                            model_cfg["display_name"],
                            result.get("error"),
                        )
                except Exception as exc:
                    logger.error(
                        "[%s] model run failed: %s", model_cfg["display_name"], exc, exc_info=True
                    )

            # Cycle-end snapshot updates for each model.
            for model_cfg in models_config:
                try:
                    model_id = model_cfg["model_id"]
                    model_positions = db.get_positions(
                        token_ids,
                        actual_balances=(balances if (token_ids and not s.simulation_mode) else None),
                        model_id=model_id,
                    )
                    total_position_value = 0.0
                    total_unrealized_pnl = 0.0

                    for token_id in token_ids:
                        pos = model_positions.get(token_id, {})
                        shares = _safe_float(pos.get("shares"), default=0.0)
                        avg_cost = _safe_float(pos.get("avg_price"), default=0.0)
                        if token_id == up_id:
                            current_price = _safe_float(
                                (hourly.get("up") or {}).get("best_bid"), default=avg_cost
                            )
                        else:
                            current_price = _safe_float(
                                (hourly.get("down") or {}).get("best_bid"), default=avg_cost
                            )
                        position_value = shares * current_price
                        total_position_value += position_value
                        total_unrealized_pnl += shares * (current_price - avg_cost)

                    db.update_account_balance(
                        model_id=model_id,
                        cash_change=0.0,
                        position_value=total_position_value,
                        unrealized_pnl=total_unrealized_pnl,
                        realized_pnl_change=0.0,
                    )
                except Exception as snap_error:
                    logger.error("Snapshot update failed [%s]: %s", model_cfg["model_id"], snap_error)

            total_cycle_time = time.time() - cycle_start_time
            last_ai_end_time = time.time()
            logger.info("Cycle done in %.2fs, sleeping %ss", total_cycle_time, interval)

            for _ in range(max(1, interval)):
                if _stop_event.is_set():
                    break
                time.sleep(1)
    except KeyboardInterrupt:
        _stop_event.set()
    except Exception as e:
        logger.error("Fatal main loop error: %s\n%s", e, traceback.format_exc())
        raise
    finally:
        try:
            if _ohlcv_source is not None:
                _ohlcv_source.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

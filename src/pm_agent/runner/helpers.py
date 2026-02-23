"""pm_agent.runner.helpers — Shared utility functions used by both
``prefetch`` and ``decision_loop`` modules.

These were originally module-level helpers in ``main.py``.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

from pm_agent.db.sqlite import SQLiteDB
from pm_agent.utils.time import iso, utcnow
from pm_agent.utils import chalk
from pm_agent.utils.logging import get_logger

logger = get_logger("pm_agent.runner")


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def safe_float(v: Any, default: float = 0.0) -> float:
    """Convert *v* to float, returning *default* on error or NaN."""
    try:
        out = float(v)
        if out != out:  # NaN guard
            return default
        return out
    except Exception:
        return default


def parse_utc(value: Any) -> dt.datetime | None:
    """Parse an ISO-8601 string to a tz-aware UTC datetime, or return None."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    except ValueError:
        return None


def best_price(ob: Any, side: str) -> float | None:
    """Return the best bid or ask from an order-book object/dict."""
    if isinstance(ob, dict):
        if side in ob and isinstance(ob.get(side), (int, float, str)):
            x = safe_float(ob.get(side), default=-1.0)
            return x if x >= 0 else None
        lv = ob.get(side)
    else:
        lv = getattr(ob, side, None)

    if isinstance(lv, list) and lv:
        prices: list[float] = []
        for level in lv:
            try:
                px = float(level.get("price") if isinstance(level, dict) else getattr(level, "price"))
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


# ---------------------------------------------------------------------------
# Market helpers
# ---------------------------------------------------------------------------

def normalize_hourly(hourly_raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize the raw hourly market dict from PolymarketClient."""
    up_bid = hourly_raw.get("up_bid")
    up_ask = hourly_raw.get("up_ask")
    down_bid = hourly_raw.get("down_bid")
    down_ask = hourly_raw.get("down_ask")
    return {
        "slug": hourly_raw.get("market_slug") or hourly_raw.get("slug"),
        "market": {
            "question": (
                hourly_raw.get("market_title")
                or hourly_raw.get("question")
                or (hourly_raw.get("market") or {}).get("question")
            ),
            "event_start_time": (hourly_raw.get("market") or {}).get("event_start_time"),
            "condition_id": hourly_raw.get("condition_id"),
            "neg_risk": hourly_raw.get("neg_risk"),
            "token_index_sets": hourly_raw.get("token_index_sets") or {},
        },
        "question": (
            hourly_raw.get("market_title")
            or hourly_raw.get("question")
            or (hourly_raw.get("market") or {}).get("question")
        ),
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


def compute_phase(minutes_remaining: int) -> tuple[str, str]:
    """Map minutes remaining to a phase label + note."""
    if minutes_remaining > 40:
        return "EARLY", "市场方向尚未明确"
    if minutes_remaining > 20:
        return "MIDDLE", "趋势逐渐清晰"
    if minutes_remaining > 4:
        return "LATE", "中后期"
    if minutes_remaining > 1:
        return "FINAL", "最后阶段"
    return "CLOSING", "即将结算，除非极高把握否则勿开新仓"


# ---------------------------------------------------------------------------
# Database helpers (settlement / position queries)
# ---------------------------------------------------------------------------

def all_model_ids(db: SQLiteDB) -> list[str]:
    """Return every model_id that has any activity in the database."""
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


def query_net_holdings(db: SQLiteDB, model_id: str) -> dict[str, float]:
    """Return a mapping of token_id → net shares held by *model_id*."""
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
        return {str(r["token_id"]): safe_float(r["net_shares"]) for r in rows}
    finally:
        conn.close()


def settle_position(
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
    """Write a settlement SELL trade and update account balance."""
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


def settlement_price_from_position_row(row: dict[str, Any]) -> float:
    """Derive a settlement price from a Polymarket position row."""
    size = safe_float(row.get("size"), default=0.0)
    cur = safe_float(row.get("currentValue"), default=0.0)
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


# ---------------------------------------------------------------------------
# Settlement orchestration
# ---------------------------------------------------------------------------

def settle_market_if_needed(
    *,
    db: SQLiteDB,
    cached_hourly: dict[str, Any],
    final_btc_price: float | None,
    price_to_beat: float | None,
    simulation_mode: bool,
    model_pm_clients: dict | None = None,
    auto_redeem_use_relayer: bool = True,
) -> set[str]:
    """Settle open positions when a market round has concluded.

    Returns the set of model IDs that had positions settled.
    """
    from pm_agent.polymarket.client import PolymarketClient  # local import avoids circular dep

    up_id = cached_hourly.get("up_token_id")
    down_id = cached_hourly.get("down_token_id")
    up_bid = safe_float((cached_hourly.get("up") or {}).get("best_bid"), default=-1.0)
    down_bid = safe_float((cached_hourly.get("down") or {}).get("best_bid"), default=-1.0)
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

    for model_id in all_model_ids(db):
        all_holdings = query_net_holdings(db, model_id)
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
                    "  [%s] Settlement redeem skipped/failed: %s",
                    model_id,
                    redeem_error,
                )

        for token_id, shares in all_holdings.items():
            positions = db.get_positions([token_id], model_id=model_id)
            avg_cost = safe_float((positions.get(token_id) or {}).get("avg_price"), default=0.0)
            if token_id in (up_id, down_id):
                if not simulation_mode and not can_settle_current_market:
                    continue
                settlement_price = safe_float(settlement_prices.get(token_id), default=0.0)
                side_name = "UP" if token_id == up_id else "DOWN"
            else:
                if not simulation_mode:
                    continue
                settlement_price = 0.0
                side_name = "OLD"
            settle_position(
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


def startup_cleanup_stale_positions(
    db: SQLiteDB,
    current_token_ids: list[str],
    simulation_mode: bool,
) -> None:
    """Zero out stale positions from previous sessions on startup (sim-only)."""
    if not simulation_mode:
        logger.info("Startup cleanup: live mode, skip force-zero stale positions")
        return
    logger.info("Startup cleanup: checking for stale positions...")
    for model_id in all_model_ids(db):
        all_holdings = query_net_holdings(db, model_id)
        stale = {tid: sz for tid, sz in all_holdings.items() if tid not in current_token_ids}
        if not stale:
            continue
        logger.info("  [%s] Found %d stale positions", model_id, len(stale))
        for token_id, shares in stale.items():
            pos = db.get_positions([token_id], model_id=model_id).get(token_id, {})
            avg_cost = safe_float(pos.get("avg_price"), default=0.0)
            settle_position(
                db=db,
                model_id=model_id,
                token_id=token_id,
                shares=shares,
                settlement_price=0.0,
                side_name="OLD",
                simulation_mode=simulation_mode,
                avg_cost=avg_cost,
            )



# ---------------------------------------------------------------------------
# Auto-redeem loop helper
# ---------------------------------------------------------------------------

def auto_redeem_model_resolved_positions(
    *,
    model_cfg: dict,
    db: SQLiteDB,
    s: Any,   # Settings
) -> None:
    """Scan for resolved (redeemable) positions and redeem them on-chain.

    Called once per ``auto_redeem_interval_sec`` inside the main loop.
    Only runs in live mode when ``s.auto_redeem_enabled`` is True.
    """
    if s.simulation_mode or not s.auto_redeem_enabled:
        return

    from pm_agent.polymarket.client import PolymarketClient  # avoid circular import
    import json as _json

    model_id: str = model_cfg["model_id"]
    pm_client: PolymarketClient = model_cfg["pm_client"]

    try:
        rows = pm_client.get_positions(redeemable_only=True, size_threshold=0.0)
    except Exception as e:
        logger.warning("[%s] Auto-redeem scan failed: %s", model_id, e)
        return

    if not rows:
        return

    grouped: dict[str, list] = {}
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

        redeem_value = sum(max(0.0, safe_float(e.get("currentValue"), default=0.0)) for e in entries)
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

        holdings = query_net_holdings(db, model_id)
        settled_token_ids: list[str] = []
        for row in entries:
            token_id = str(row.get("asset") or "").strip()
            if not token_id:
                continue
            shares = safe_float(holdings.get(token_id), default=0.0)
            if shares <= 0:
                continue
            pos = db.get_positions([token_id], model_id=model_id).get(token_id, {})
            avg_cost = safe_float(pos.get("avg_price"), default=0.0)
            sp = settlement_price_from_position_row(row)
            outcome = str(row.get("outcome") or "OUTCOME").upper()
            settle_position(
                db=db,
                model_id=model_id,
                token_id=token_id,
                shares=shares,
                settlement_price=sp,
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
            tx_hash = redeem_result.get("tx_hash") or redeem_result.get("transactionHash")

        db.log(
            ts=iso(utcnow()),
            type_="auto_redeem",
            payload_json=_json.dumps(
                {
                    "condition_id": condition_id,
                    "tx_hash": tx_hash,
                    "redeem_value": redeem_value,
                    "token_ids": settled_token_ids,
                    "result": str(redeem_result),
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

        # Sync cash balance after redemption.
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

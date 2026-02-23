"""pm_agent.runner.decision_loop — Per-model AI decision processing.

Extracted from the ``process_model_decision`` nested function that lived
inside ``main.main()``.  Now a standalone function with explicit parameters
so that it can be tested and imported independently.
"""
from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from typing import Any

from pm_agent.config import Settings
from pm_agent.db.sqlite import SQLiteDB
from pm_agent.utils import chalk
from pm_agent.utils.logging import get_logger
from pm_agent.utils.time import iso, utcnow

from pm_agent.runner.helpers import (
    compute_phase,
    parse_utc,
    safe_float,
)

logger = get_logger("pm_agent.runner.decision_loop")


def process_model_decision(
    *,
    model_cfg: dict[str, Any],
    cache: Any,          # MarketCache (or any object with dict-style .get())
    cache_lock: threading.Lock,
    db: SQLiteDB,
    s: Settings,
    stop_event: threading.Event,
    last_ai_end_time: float,
) -> dict[str, Any]:
    """Run one decision cycle for a single AI model.

    Parameters
    ----------
    model_cfg:
        Dict produced during initialisation; contains keys
        ``model_id``, ``display_name``, ``orchestrator``, ``engine``.
    cache:
        The shared ``MarketCache`` (or any mapping-like object).
    cache_lock:
        Lock protecting *cache* reads (kept for backward compatibility;
        ``MarketCache.__getitem__`` already acquires internally).
    db:
        Shared ``SQLiteDB`` instance.
    s:
        Application ``Settings``.
    stop_event:
        Global shutdown flag.
    last_ai_end_time:
        Timestamp of when the previous AI call finished (used for
        ``time_since_last_ai`` context).

    Returns
    -------
    dict with keys ``model_id``, ``display_name``, ``status``, and optionally
    ``results`` or ``error``.
    """
    model_id: str = model_cfg["model_id"]
    display_name: str = model_cfg["display_name"]
    orchestrator_instance = model_cfg["orchestrator"]
    engine_instance = model_cfg["engine"]
    # Per-provider simulation flag (may differ from global Settings.simulation_mode)
    model_sim_mode: bool = model_cfg.get("simulation_mode", s.simulation_mode)

    logger.info(chalk.bold(chalk.magenta(f"\n🤖 [{display_name}] Making decision...")))
    try:
        with cache_lock:
            hourly = dict(cache.get("hourly_market") or {})
            balances = dict(cache.get("balances") or {})
            price_to_beat = cache.get("price_to_beat")
            current_btc = cache.get("current_btc_price")

        up_id = hourly.get("up_token_id")
        down_id = hourly.get("down_token_id")
        token_ids = [t for t in [up_id, down_id] if t]

        current_prices: dict[str, Any] | None = None
        if up_id and down_id:
            current_prices = {
                up_id: (hourly.get("up") or {}).get("best_bid"),
                down_id: (hourly.get("down") or {}).get("best_bid"),
            }

        model_db_positions = db.get_positions(
            token_ids,
            actual_balances=(balances if (token_ids and not model_sim_mode) else None),
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

        model_positions_list: list[dict[str, Any]] = []
        for label, token_id in (("UP", up_id), ("DOWN", down_id)):
            if not token_id:
                continue
            pos = model_db_positions.get(token_id, {})
            shares = safe_float(pos.get("shares"), default=0.0)
            avg_cost = safe_float(pos.get("avg_price"), default=0.0)
            book = hourly.get("up") if label == "UP" else hourly.get("down")
            mark = safe_float((book or {}).get("best_bid"), default=0.0)
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
        account_info: dict[str, Any] = {
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
        end_dt = parse_utc(end_time_utc)
        time_to_end_sec = 0
        if end_dt is not None:
            time_to_end_sec = max(0, int((end_dt - utcnow()).total_seconds()))

        price_diff: float | None = None
        price_diff_pct: float | None = None
        btc_is_winning = "TIE"
        if current_btc and price_to_beat:
            price_diff = float(current_btc) - float(price_to_beat)
            price_diff_pct = (price_diff / float(price_to_beat)) * 100 if price_to_beat else None
            if price_diff > 0:
                btc_is_winning = "UP"
            elif price_diff < 0:
                btc_is_winning = "DOWN"

        minutes_remaining = max(0, int(time_to_end_sec / 60))
        phase, phase_note = compute_phase(minutes_remaining)
        btc_momentum = "neutral"
        if isinstance(price_diff_pct, (int, float)):
            if price_diff_pct > 0.1:
                btc_momentum = "up"
            elif price_diff_pct < -0.1:
                btc_momentum = "down"

        btc_analysis: dict[str, Any] = {
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
        market_overview: dict[str, Any] = {
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

        candidate_markets: list[dict[str, Any]] = []
        for item in model_positions_list:
            shares = safe_float(item.get("shares"), default=0.0)
            avg_cost = safe_float(item.get("avg_cost"), default=0.0)
            current_price = safe_float(item.get("mark_price"), default=0.0)
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
            realized_pnl = safe_float(model_session_stats.get("realized_pnl"), default=0.0)
            unrealized_pnl = safe_float(model_session_stats.get("unrealized_pnl"), default=0.0)
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

        model_context: dict[str, Any] = {
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
        cycle_id: str = model_context["system_state"]["cycle_id"]

        db.log(
            ts=iso(utcnow()),
            type_="run_start",
            payload_json=json.dumps({"cycle_id": cycle_id}, ensure_ascii=False),
            model_id=model_id,
        )

        if stop_event.is_set():
            logger.info("  [%s] Skipping AI call due to shutdown signal", display_name)
            return {"model_id": model_id, "display_name": display_name, "status": "skipped"}

        ai_start_time = time.time()
        try:
            parsed = orchestrator_instance.run_once(cycle_id=cycle_id, context=model_context)
        except KeyboardInterrupt:
            logger.info("  [%s] AI call interrupted", display_name)
            stop_event.set()
            return {"model_id": model_id, "display_name": display_name, "status": "interrupted"}

        ai_elapsed = time.time() - ai_start_time
        logger.info(chalk.cyan(f"  [{display_name}] AI reasoning took {ai_elapsed:.2f}s"))

        db.log(
            ts=iso(utcnow()),
            type_="ai_raw",
            payload_json=json.dumps(
                {"cycle_id": cycle_id, "prompt": getattr(parsed, "prompt", ""), "raw": parsed.raw},
                ensure_ascii=False
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

        # Print action summary to console.
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
                status_emoji, status_color = "✓", chalk.green
            elif status in {"skipped"}:
                status_emoji, status_color = "→", chalk.dim
            else:
                status_emoji, status_color = "✗", chalk.red
            print(f"  {status_color(f'[{status_emoji}] {json.dumps(r, ensure_ascii=False)}')}")

        # Update per-model account snapshot after execution.
        all_token_ids = [t for t in [up_id, down_id] if t]
        model_positions = db.get_positions(
            all_token_ids,
            actual_balances=(balances if (all_token_ids and not model_sim_mode) else None),
            model_id=model_id,
        )
        total_position_value = 0.0
        total_unrealized_pnl = 0.0
        for token_id in all_token_ids:
            pos = model_positions.get(token_id, {})
            shares = safe_float(pos.get("shares"), default=0.0)
            avg_cost = safe_float(pos.get("avg_price"), default=0.0)
            if token_id == up_id:
                current_price = safe_float((hourly.get("up") or {}).get("best_bid"), default=avg_cost)
            else:
                current_price = safe_float((hourly.get("down") or {}).get("best_bid"), default=avg_cost)
            total_position_value += shares * current_price
            total_unrealized_pnl += shares * (current_price - avg_cost)

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

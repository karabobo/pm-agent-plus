from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import sqlite3
from pathlib import Path
import json
import datetime as dt
import math
import statistics


@dataclass
class SQLiteDB:
    db_path: str = "pm_agent.db"
    timeout: float = 30.0

    def _connect(self) -> sqlite3.Connection:
        """Create SQLite connection with WAL mode and busy timeout."""
        conn = sqlite3.Connection(self.db_path, timeout=self.timeout)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_tables(self) -> None:
        """Initialize database schema."""
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL DEFAULT 'default',
                    ts TEXT NOT NULL,
                    type TEXT NOT NULL,
                    payload_json TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_model_ts ON events(model_id, ts)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(type)")

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL DEFAULT 'default',
                    ts TEXT NOT NULL,
                    token_id TEXT NOT NULL,
                    side TEXT NOT NULL,
                    filled_shares REAL NOT NULL,
                    avg_price REAL NOT NULL,
                    is_simulation BOOLEAN NOT NULL DEFAULT 0,
                    order_json TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_model_ts ON trades(model_id, ts)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_token_id ON trades(token_id)")

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS equity_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    total_equity REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    position_value REAL NOT NULL,
                    cash_balance REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_equity_model_ts ON equity_snapshots(model_id, ts)"
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS account_balances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL UNIQUE,
                    cash_balance REAL NOT NULL,
                    position_value REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.commit()
        finally:
            conn.close()

    def __post_init__(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def log(
        self,
        ts: str,
        type_: str,
        payload_json: str,
        model_id: str = "default",
    ) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO events(model_id, ts, type, payload_json) VALUES(?, ?, ?, ?)",
                (model_id, ts, type_, payload_json),
            )
            conn.commit()
        finally:
            conn.close()

    def record_trade(
        self,
        ts: str,
        token_id: str,
        side: str,
        filled_shares: float,
        avg_price: float,
        order: dict[str, Any],
        model_id: str = "default",
        is_simulation: bool = False,
    ) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO trades(model_id, ts, token_id, side, filled_shares, avg_price, is_simulation, order_json)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    ts,
                    token_id,
                    side,
                    float(filled_shares),
                    float(avg_price),
                    1 if is_simulation else 0,
                    json.dumps(order, ensure_ascii=False),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def clear_session_trades(self, token_ids: list[str], model_id: str = "default") -> int:
        if not token_ids:
            return 0
        placeholders = ",".join(["?"] * len(token_ids))
        conn = self._connect()
        try:
            cur = conn.execute(
                f"DELETE FROM trades WHERE model_id = ? AND token_id IN ({placeholders})",
                [model_id, *token_ids],
            )
            conn.commit()
            return int(cur.rowcount or 0)
        finally:
            conn.close()

    def get_positions(
        self,
        token_ids: list[str],
        actual_balances: dict[str, float] | None = None,
        actual_avg_prices: dict[str, float] | None = None,
        model_id: str = "default",
    ) -> dict[str, dict[str, float]]:
        positions = self._calculate_positions_from_trades(token_ids, model_id)
        if actual_balances is not None:
            for tid in token_ids:
                p = positions.setdefault(tid, {"shares": 0.0, "avg_price": 0.0})
                chain_shares = float(actual_balances.get(tid, p.get("shares", 0.0)) or 0.0)
                p["shares"] = chain_shares
                if chain_shares <= 1e-9:
                    p["avg_price"] = 0.0
                    continue
                if actual_avg_prices is not None and tid in actual_avg_prices:
                    try:
                        chain_avg = float(actual_avg_prices.get(tid, p.get("avg_price", 0.0)) or 0.0)
                        if chain_avg >= 0:
                            p["avg_price"] = chain_avg
                    except Exception:
                        pass
        return positions

    def _calculate_positions_from_trades(
        self, token_ids: list[str], model_id: str = "default"
    ) -> dict[str, dict[str, float]]:
        positions = {tid: {"shares": 0.0, "avg_price": 0.0} for tid in token_ids}
        if not token_ids:
            return positions

        placeholders = ",".join(["?"] * len(token_ids))
        conn = self._connect()
        try:
            rows = conn.execute(
                f"""
                SELECT token_id, side, filled_shares, avg_price
                FROM trades
                WHERE model_id = ? AND token_id IN ({placeholders})
                ORDER BY id ASC
                """,
                [model_id, *token_ids],
            ).fetchall()
        finally:
            conn.close()

        for r in rows:
            tid = str(r["token_id"])
            side = str(r["side"]).upper()
            shares = float(r["filled_shares"] or 0)
            px = float(r["avg_price"] or 0)
            p = positions.setdefault(tid, {"shares": 0.0, "avg_price": 0.0})

            if side.startswith("BUY"):
                old_shares = p["shares"]
                new_shares = old_shares + shares
                if new_shares > 0:
                    p["avg_price"] = (p["avg_price"] * old_shares + px * shares) / new_shares
                p["shares"] = new_shares
            elif side.startswith("SELL"):
                p["shares"] = max(0.0, p["shares"] - shares)
                if p["shares"] == 0:
                    p["avg_price"] = 0.0

        return positions

    def get_profit_stats(
        self, token_ids: list[str] | None = None, model_id: str = "default"
    ) -> dict[str, Any]:
        conn = self._connect()
        try:
            if token_ids:
                placeholders = ",".join(["?"] * len(token_ids))
                rows = conn.execute(
                    f"""
                    SELECT side, filled_shares, avg_price
                    FROM trades
                    WHERE model_id = ? AND token_id IN ({placeholders})
                    ORDER BY id ASC
                    """,
                    [model_id, *token_ids],
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT side, filled_shares, avg_price FROM trades WHERE model_id = ? ORDER BY id ASC",
                    (model_id,),
                ).fetchall()
        finally:
            conn.close()

        buys = 0.0
        sells = 0.0
        trade_count = 0
        for r in rows:
            shares = float(r["filled_shares"] or 0)
            px = float(r["avg_price"] or 0)
            notional = shares * px
            if str(r["side"]).upper().startswith("BUY"):
                buys += notional
            else:
                sells += notional
            trade_count += 1

        realized_pnl = sells - buys
        return {
            "trade_count": trade_count,
            "buy_notional": buys,
            "sell_notional": sells,
            "realized_pnl": realized_pnl,
        }

    def get_sharpe_stats(self, model_id: str = "default", limit: int = 2000) -> dict[str, Any]:
        """Compute Sharpe stats from account equity snapshots.

        Returns annualized Sharpe (risk-free rate assumed 0) and supporting
        metadata. If data is insufficient, sharpe values are returned as None.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT total_equity, created_at
                FROM equity_snapshots
                WHERE model_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (model_id, int(limit)),
            ).fetchall()
        finally:
            conn.close()

        if len(rows) < 3:
            return {
                "sharpe_ratio": None,
                "sharpe_raw": None,
                "samples": 0,
                "median_interval_sec": None,
            }

        ordered = list(reversed(rows))
        returns: list[float] = []
        intervals: list[float] = []
        prev_equity: float | None = None
        prev_ts: dt.datetime | None = None

        for r in ordered:
            equity = float(r["total_equity"] or 0.0)
            ts_raw = str(r["created_at"] or "").strip()
            ts: dt.datetime | None = None
            if ts_raw:
                try:
                    ts = dt.datetime.fromisoformat(ts_raw)
                except Exception:
                    ts = None

            if prev_equity is not None and prev_equity > 0 and equity > 0:
                returns.append((equity - prev_equity) / prev_equity)
            prev_equity = equity

            if prev_ts is not None and ts is not None:
                delta_sec = (ts - prev_ts).total_seconds()
                if delta_sec > 0:
                    intervals.append(delta_sec)
            prev_ts = ts if ts is not None else prev_ts

        if len(returns) < 2:
            return {
                "sharpe_ratio": None,
                "sharpe_raw": None,
                "samples": len(returns),
                "median_interval_sec": statistics.median(intervals) if intervals else None,
            }

        mean_ret = statistics.fmean(returns)
        stdev_ret = statistics.stdev(returns)
        if stdev_ret <= 1e-12:
            return {
                "sharpe_ratio": None,
                "sharpe_raw": None,
                "samples": len(returns),
                "median_interval_sec": statistics.median(intervals) if intervals else None,
            }

        sharpe_raw = mean_ret / stdev_ret
        median_interval_sec = statistics.median(intervals) if intervals else None
        annualization = 1.0
        if median_interval_sec and median_interval_sec > 0:
            periods_per_year = (365.0 * 24.0 * 3600.0) / median_interval_sec
            annualization = math.sqrt(max(periods_per_year, 1.0))

        return {
            "sharpe_ratio": sharpe_raw * annualization,
            "sharpe_raw": sharpe_raw,
            "samples": len(returns),
            "median_interval_sec": median_interval_sec,
        }

    def get_trade_history(
        self,
        token_ids: list[str] | None = None,
        limit: int = 100,
        model_id: str = "default",
    ) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            if token_ids:
                placeholders = ",".join(["?"] * len(token_ids))
                rows = conn.execute(
                    f"""
                    SELECT * FROM trades
                    WHERE model_id = ? AND token_id IN ({placeholders})
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    [model_id, *token_ids, int(limit)],
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM trades WHERE model_id = ? ORDER BY id DESC LIMIT ?",
                    (model_id, int(limit)),
                ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                try:
                    d["order"] = json.loads(d.get("order_json") or "{}")
                except Exception:
                    d["order"] = {}
                out.append(d)
            return out
        finally:
            conn.close()

    def get_session_stats(
        self,
        token_ids: list[str],
        current_positions: dict[str, dict],
        current_prices: dict[str, float],
        model_id: str = "default",
    ) -> dict[str, Any]:
        stats = self.get_profit_stats(token_ids, model_id)

        position_value = 0.0
        unrealized_pnl = 0.0
        for tid in token_ids:
            p = current_positions.get(tid, {})
            shares = float(p.get("shares", 0) or 0)
            avg = float(p.get("avg_price", 0) or 0)
            px = float(current_prices.get(tid, avg) or avg)
            position_value += shares * px
            unrealized_pnl += shares * (px - avg)

        stats["position_value"] = position_value
        stats["unrealized_pnl"] = unrealized_pnl
        stats["total_pnl"] = float(stats.get("realized_pnl", 0)) + unrealized_pnl
        return stats

    def get_decision_history(
        self,
        token_ids: list[str],
        up_token_id: str,
        down_token_id: str,
        actual_balances: dict[str, float],
        model_id: str = "default",
    ) -> dict[str, Any]:
        conn = self._connect()
        try:
            ev_rows = conn.execute(
                """
                SELECT ts, type, payload_json
                FROM events
                WHERE model_id = ?
                ORDER BY id DESC
                LIMIT 50
                """,
                (model_id,),
            ).fetchall()
            tr_rows = conn.execute(
                """
                SELECT ts, token_id, side, filled_shares, avg_price
                FROM trades
                WHERE model_id = ?
                ORDER BY id DESC
                LIMIT 50
                """,
                (model_id,),
            ).fetchall()
        finally:
            conn.close()

        events = []
        for r in ev_rows:
            payload = r["payload_json"]
            try:
                payload = json.loads(payload) if payload else None
            except Exception:
                pass
            events.append({"ts": r["ts"], "type": r["type"], "payload": payload})

        trades = [dict(r) for r in tr_rows]
        last_decision = next((e for e in events if e["type"] in {"decision", "ai_decision"}), None)

        return {
            "model_id": model_id,
            "token_ids": token_ids,
            "up_token_id": up_token_id,
            "down_token_id": down_token_id,
            "actual_balances": actual_balances,
            "events": events,
            "recent_trades": trades,
            "last_round_ai_thinking": last_decision["payload"] if last_decision else None,
        }

    def get_all_trades_for_session(self, token_ids: list[str]) -> list[dict]:
        if not token_ids:
            return []
        placeholders = ",".join(["?"] * len(token_ids))
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT * FROM trades WHERE token_id IN ({placeholders}) ORDER BY id ASC",
                token_ids,
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def delete_trade_by_id(self, trade_id: int) -> None:
        conn = self._connect()
        try:
            conn.execute("DELETE FROM trades WHERE id = ?", (int(trade_id),))
            conn.commit()
        finally:
            conn.close()

    def save_equity_snapshot(
        self,
        ts: str,
        model_id: str,
        total_equity: float,
        realized_pnl: float,
        unrealized_pnl: float,
        position_value: float,
        cash_balance: float,
    ) -> None:
        snapshot_ts = str(ts or "").strip()
        if not snapshot_ts:
            snapshot_ts = dt.datetime.now(dt.timezone.utc).isoformat()
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO equity_snapshots(model_id, ts, total_equity, realized_pnl, unrealized_pnl, position_value, cash_balance)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    snapshot_ts,
                    float(total_equity),
                    float(realized_pnl),
                    float(unrealized_pnl),
                    float(position_value),
                    float(cash_balance),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_equity_history(self, model_id: str = "default", limit: int = 1000) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT model_id, ts, total_equity, realized_pnl, unrealized_pnl, position_value, cash_balance
                FROM equity_snapshots
                WHERE model_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (model_id, int(limit)),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def delete_auto_clear_trades(self, token_ids: list[str]) -> int:
        if not token_ids:
            return 0
        placeholders = ",".join(["?"] * len(token_ids))
        conn = self._connect()
        try:
            cur = conn.execute(
                f"""
                DELETE FROM trades
                WHERE token_id IN ({placeholders})
                  AND order_json LIKE '%"auto_clear": true%'
                """,
                token_ids,
            )
            conn.commit()
            return int(cur.rowcount or 0)
        finally:
            conn.close()

    def init_account_balance(self, model_id: str, initial_balance: float = 1000) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO account_balances(model_id, cash_balance, position_value, unrealized_pnl, realized_pnl)
                VALUES(?, ?, 0, 0, 0)
                ON CONFLICT(model_id) DO NOTHING
                """,
                (model_id, float(initial_balance)),
            )
            conn.commit()
        finally:
            conn.close()

    def get_account_balance(self, model_id: str) -> dict[str, float]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT cash_balance, position_value, unrealized_pnl, realized_pnl FROM account_balances WHERE model_id = ?",
                (model_id,),
            ).fetchone()
        finally:
            conn.close()

        if not row:
            self.init_account_balance(model_id, 1000)
            return {
                "cash_balance": 1000.0,
                "position_value": 0.0,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
            }

        return {
            "cash_balance": float(row["cash_balance"]),
            "position_value": float(row["position_value"]),
            "unrealized_pnl": float(row["unrealized_pnl"]),
            "realized_pnl": float(row["realized_pnl"]),
        }

    def update_account_balance(
        self,
        model_id: str,
        cash_change: float = 0.0,
        position_value: float | None = None,
        unrealized_pnl: float | None = None,
        realized_pnl_change: float = 0.0,
        skip_snapshot: bool = False,
    ) -> None:
        bal = self.get_account_balance(model_id)
        new_cash = bal["cash_balance"] + float(cash_change)
        new_pos = float(position_value) if position_value is not None else bal["position_value"]
        new_unreal = (
            float(unrealized_pnl) if unrealized_pnl is not None else bal["unrealized_pnl"]
        )
        new_real = bal["realized_pnl"] + float(realized_pnl_change)

        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO account_balances(model_id, cash_balance, position_value, unrealized_pnl, realized_pnl)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(model_id)
                DO UPDATE SET
                    cash_balance=excluded.cash_balance,
                    position_value=excluded.position_value,
                    unrealized_pnl=excluded.unrealized_pnl,
                    realized_pnl=excluded.realized_pnl,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (model_id, new_cash, new_pos, new_unreal, new_real),
            )
            conn.commit()
        finally:
            conn.close()

        if not skip_snapshot:
            total_equity = new_cash + new_pos
            snapshot_ts = dt.datetime.now(dt.timezone.utc).isoformat()
            self.save_equity_snapshot(
                ts=snapshot_ts,
                model_id=model_id,
                total_equity=total_equity,
                realized_pnl=new_real,
                unrealized_pnl=new_unreal,
                position_value=new_pos,
                cash_balance=new_cash,
            )

    def count_todays_trades(self, model_id: str = "default") -> int:
        """Count BUY trades executed today (UTC date) for the given model.

        Used by ExecutionEngine to enforce max_daily_trades limits.
        Only counts BUY-side trades (opening new positions) so that
        close/sell actions are never blocked by the daily cap.
        """
        import datetime as _dt

        today = _dt.datetime.now(_dt.timezone.utc).date().isoformat()  # e.g. "2026-02-23"
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM trades
                WHERE model_id = ?
                  AND UPPER(side) LIKE 'BUY%'
                  AND ts >= ?
                """,
                (model_id, today),
            ).fetchone()
            return int(row["cnt"]) if row else 0
        finally:
            conn.close()

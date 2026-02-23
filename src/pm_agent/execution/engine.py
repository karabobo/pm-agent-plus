from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from pm_agent.execution.risk import RiskLimits, enforce_action, RiskError
from pm_agent.polymarket.client import PolymarketClient
from pm_agent.db.sqlite import SQLiteDB
from pm_agent.utils.time import utcnow, iso
from pm_agent.utils.logging import get_logger

PRIORITY = {"close": 0, "open": 1, "hold": 2, "wait": 3}
logger = get_logger("pm_agent.execution")


@dataclass
class ExecutionEngine:
    pm: PolymarketClient
    db: SQLiteDB
    limits: RiskLimits
    simulation_mode: bool = False
    model_id: str = "default"

    def _sort_actions(self, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(actions or [], key=lambda a: PRIORITY.get(str(a.get("type", "wait")), 99))

    def execute(self, decision: dict[str, Any]) -> list[dict[str, Any]]:
        actions = self._sort_actions(decision.get("actions", []))
        results: list[dict[str, Any]] = []

        for a in actions:
            try:
                a2 = enforce_action(a, self.limits)
            except RiskError as e:
                res = {"action": a, "status": "rejected_risk", "error": str(e)}
                results.append(res)
                self.db.log(
                    ts=iso(utcnow()),
                    type_="risk_reject",
                    payload_json=json.dumps(res, ensure_ascii=False),
                    model_id=self.model_id,
                )
                continue

            if a2.get("type") in ("hold", "wait"):
                res = {"action": a2, "status": "skipped"}
                results.append(res)
                self.db.log(
                    ts=iso(utcnow()),
                    type_="action_skip",
                    payload_json=json.dumps(res, ensure_ascii=False),
                    model_id=self.model_id,
                )
                continue

            token_id = a2.get("token_id")
            side = str(a2.get("side", "")).lower()
            price = a2.get("price", 0)
            size = a2.get("size", 0)

            if not token_id:
                res = {"action": a2, "status": "rejected", "error": "missing token_id"}
                results.append(res)
                self.db.log(
                    ts=iso(utcnow()),
                    type_="order_reject",
                    payload_json=json.dumps(res, ensure_ascii=False),
                    model_id=self.model_id,
                )
                continue

            if side not in {"buy_up", "sell_up", "buy_down", "sell_down"}:
                res = {"action": a2, "status": "rejected", "error": "invalid side"}
                results.append(res)
                self.db.log(
                    ts=iso(utcnow()),
                    type_="order_reject",
                    payload_json=json.dumps(res, ensure_ascii=False),
                    model_id=self.model_id,
                )
                continue

            if price <= 0 or size <= 0:
                res = {"action": a2, "status": "rejected", "error": "invalid price/size"}
                results.append(res)
                self.db.log(
                    ts=iso(utcnow()),
                    type_="order_reject",
                    payload_json=json.dumps(res, ensure_ascii=False),
                    model_id=self.model_id,
                )
                continue

            is_buy = side.startswith("buy_")
            if is_buy:
                ai_amount = a2.get("amount")
                if ai_amount is not None and ai_amount > 0:
                    amount = ai_amount
                else:
                    amount = price * size
            else:
                if self.simulation_mode:
                    db_positions = self.db.get_positions([str(token_id)], model_id=self.model_id)
                    current_balance = db_positions.get(str(token_id), {}).get("shares", 0)
                    if size > current_balance:
                        logger.warning(
                            f"[SIMULATION] Sell size {size} exceeds balance {current_balance}, adjusting to {current_balance}"
                        )
                        size = current_balance

                    if current_balance < 0.01:
                        res = {
                            "action": a2,
                            "status": "rejected",
                            "error": f"insufficient balance: {current_balance}",
                        }
                        results.append(res)
                        self.db.log(
                            ts=iso(utcnow()),
                            type_="order_reject",
                            payload_json=json.dumps(res, ensure_ascii=False),
                            model_id=self.model_id,
                        )
                        continue
                else:
                    try:
                        balances = self.pm.get_token_balances([str(token_id)])
                        current_balance = balances.get(str(token_id), 0)
                        if size > current_balance:
                            logger.warning(
                                f"Sell size {size} exceeds balance {current_balance}, adjusting to {current_balance}"
                            )
                            size = current_balance

                        if current_balance < 0.01:
                            res = {
                                "action": a2,
                                "status": "rejected",
                                "error": f"insufficient balance: {current_balance}",
                            }
                            results.append(res)
                            self.db.log(
                                ts=iso(utcnow()),
                                type_="order_reject",
                                payload_json=json.dumps(res, ensure_ascii=False),
                                model_id=self.model_id,
                            )
                            continue
                    except Exception as e:
                        logger.warning(
                            f"Failed to get balance for {token_id}: {e}, using original size"
                        )

                amount = size

            logger.info(
                f"{'[SIMULATION] ' if self.simulation_mode else ''}Submitting market order side={'BUY' if is_buy else 'SELL'} token_id={token_id} amount={amount} price={price} size={size}"
            )

            if self.simulation_mode:
                logger.info("[SIMULATION] Skipping real order submission, recording simulated trade")
                realized_pnl_change = 0
                if not is_buy:
                    positions = self.db.get_positions([token_id], model_id=self.model_id)
                    avg_cost = positions.get(token_id, {}).get("avg_cost", 0)

                self.db.record_trade(
                    ts=iso(utcnow()),
                    token_id=str(token_id),
                    side=("BUY" if is_buy else "SELL"),
                    filled_shares=float(size),
                    avg_price=round(float(price), 2),
                    order={
                        "simulated": True,
                        "side": ("BUY" if is_buy else "SELL"),
                        "token_id": token_id,
                        "amount": amount,
                        "price": price,
                        "size": size,
                    },
                    model_id=self.model_id,
                    is_simulation=True,
                )

                if is_buy:
                    cash_change = -amount
                else:
                    cash_change = size * price
                    realized_pnl_change = (price - avg_cost) * size

                self.db.update_account_balance(
                    model_id=self.model_id,
                    cash_change=cash_change,
                    realized_pnl_change=realized_pnl_change,
                    skip_snapshot=True,
                )

                order = {
                    "simulated": True,
                    "side": ("BUY" if is_buy else "SELL"),
                    "token_id": token_id,
                    "amount": amount,
                    "price": price,
                    "size": size,
                }
                res = {"action": a2, "status": "simulated", "order": order}
                results.append(res)
                self.db.log(
                    ts=iso(utcnow()),
                    type_="order_submit",
                    payload_json=json.dumps(res, ensure_ascii=False),
                    model_id=self.model_id,
                )
                continue

            order = None
            last_error = None
            max_attempts = 1 if is_buy else 2
            balance_retry_attempted = False

            for attempt in range(max_attempts):
                try:
                    order = self.pm.submit_market_order(
                        side=("BUY" if is_buy else "SELL"),
                        token_id=str(token_id),
                        amount=float(amount),
                    )
                    if order and isinstance(order, dict):
                        if order.get("error") or order.get("status") == "error":
                            raise Exception(order.get("error", "Order failed"))
                    break
                except Exception as e:
                    last_error = e

                    if (
                        not is_buy
                        and not balance_retry_attempted
                        and hasattr(e, "is_balance_error")
                        and e.is_balance_error
                    ):
                        balance_retry_attempted = True
                        logger.warning(
                            "⚠️ SELL order balance error detected, querying actual position..."
                        )
                        try:
                            actual_balance = self.pm.get_token_balances([str(token_id)])
                            actual_shares = actual_balance.get(str(token_id), 0)
                            if actual_shares > 0:
                                logger.info(
                                    f"🔄 Retrying SELL with actual balance: {actual_shares:.4f} shares"
                                )
                                amount = actual_shares
                                continue
                            logger.warning(f"❌ No actual balance found for token {token_id}")
                        except Exception as balance_err:
                            logger.error(f"Failed to query actual balance: {balance_err}")

                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"SELL order failed (attempt {attempt + 1}/{max_attempts}): {e}, retrying immediately..."
                        )
                    else:
                        logger.error(f"SELL order failed after {max_attempts} attempts: {e}")

            if order is None and last_error is not None:
                res = {"action": a2, "status": "failed", "error": str(last_error)}
                results.append(res)
                self.db.log(
                    ts=iso(utcnow()),
                    type_="order_fail",
                    payload_json=json.dumps(res, ensure_ascii=False),
                    model_id=self.model_id,
                )
                continue

            res = {"action": a2, "status": "submitted", "order": order}
            results.append(res)
            self.db.log(
                ts=iso(utcnow()),
                type_="order_submit",
                payload_json=json.dumps(res, ensure_ascii=False),
                model_id=self.model_id,
            )

            filled = self._extract_filled_shares(order, is_buy)
            avg_price = self._extract_avg_price(order, is_buy)
            if filled > 0 and avg_price > 0:
                self.db.record_trade(
                    ts=iso(utcnow()),
                    token_id=str(token_id),
                    side=("BUY" if is_buy else "SELL"),
                    filled_shares=float(filled),
                    avg_price=round(float(avg_price), 2),
                    order=order if isinstance(order, dict) else {"raw": str(order)},
                    model_id=self.model_id,
                    is_simulation=False,
                )

        return results

    @staticmethod
    def _extract_filled_shares(order_result: dict[str, Any], is_buy: bool) -> float:
        """提取成交股份数量

        Polymarket订单中:
        - BUY: takingAmount = 买入的股数, makingAmount = 支付的USDC
        - SELL: takingAmount = 收到的USDC, makingAmount = 卖出的股数
        """
        if not isinstance(order_result, dict):
            return 0

        try:
            taking = float(order_result.get("takingAmount", 0) or 0)
            making = float(order_result.get("makingAmount", 0) or 0)
            if is_buy:
                if taking > 0:
                    return taking
            else:
                if making > 0:
                    return making
        except Exception:
            pass

        candidates = [
            order_result.get("sizeFilled"),
            order_result.get("size"),
            order_result.get("filled"),
        ]
        for val in candidates:
            if val is None:
                continue
            try:
                parsed = float(val)
                if parsed > 0:
                    return parsed
            except Exception:
                continue
        return 0.0

    @staticmethod
    def _extract_avg_price(order_result: dict[str, Any], is_buy: bool) -> float:
        """提取平均成交价格"""
        if not isinstance(order_result, dict):
            return 0.0

        try:
            taking = float(order_result.get("takingAmount", 0) or 0)
            making = float(order_result.get("makingAmount", 0) or 0)
            if taking > 0 and making > 0:
                if is_buy:
                    return making / taking
                return taking / making
        except Exception:
            return 0.0

        return 0.0

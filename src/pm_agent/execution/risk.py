from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RiskLimits:
    max_notional_usd: float = 200.0
    max_position_usd: float = 100.0
    max_daily_trades: int = 24
    slippage_bps: int = 30


class RiskError(RuntimeError):
    pass


def enforce_action(action: dict[str, Any], limits: RiskLimits) -> dict[str, Any]:
    price = float(action.get("price", 0) or 0)
    size = float(action.get("size", 0) or 0)
    notional = float(action.get("amount", price * size) or (price * size))
    side = str(action.get("side", "")).lower()
    is_buy = side.startswith("buy_")

    if action.get("type") in ("open", "close") and (price <= 0 or price > 1):
        raise RiskError(f"Price out of bounds [0,1]: {price}")

    if size <= 0:
        raise RiskError(f"Size must be positive: {size}")

    if is_buy and notional > limits.max_notional_usd + 1:
        raise RiskError(
            f"Notional {notional:.2f} exceeds max_notional_usd {limits.max_notional_usd}"
        )

    risk = action.setdefault("risk", {})
    requested_slippage = int(risk.get("max_slippage_bps", limits.slippage_bps))
    risk["max_slippage_bps"] = min(requested_slippage, limits.slippage_bps)
    risk["max_notional_usd"] = limits.max_notional_usd
    return action

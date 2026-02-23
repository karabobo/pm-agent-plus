from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class UserPromptInputs:
    system_state: dict[str, Any]
    market_overview: dict[str, Any]
    account_info: dict[str, Any]
    current_positions: dict[str, Any]
    candidate_markets: list[dict[str, Any]]
    session_history: dict[str, Any] | str | None = None
    decision_history: dict[str, Any] | list[dict[str, Any]] | None = None


def build_user_prompt(inp: UserPromptInputs) -> str:
    payload = {
        "system_state": inp.system_state,
        "btc_market_overview": inp.market_overview,
        "account": inp.account_info,
        "positions": inp.current_positions,
        "candidate_markets": inp.candidate_markets,
    }
    if inp.session_history:
        payload["session_trading_history"] = inp.session_history
    if inp.decision_history:
        payload["decision_memory"] = inp.decision_history
    return "Context JSON:\n" + json.dumps(payload, ensure_ascii=False, indent=2)

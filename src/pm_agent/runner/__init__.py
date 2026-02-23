"""pm_agent.runner — Background thread and per-model decision logic.

Submodules
----------
helpers
    Pure utility functions shared by ``prefetch`` and ``decision_loop``
    (safe_float, parse_utc, settle_position, settle_market_if_needed, …)

prefetch
    ``data_prefetch_thread`` — runs in a daemon thread started by ``main()``.
    Keeps the shared ``MarketCache`` updated every second with BTC price,
    market info, and wallet balances.

decision_loop
    ``process_model_decision`` — called once per model per cycle.
    Assembles context, calls the AI orchestrator, executes trades, and
    updates per-model account snapshots.
"""

from pm_agent.runner.helpers import (
    safe_float,
    parse_utc,
    best_price,
    normalize_hourly,
    compute_phase,
    all_model_ids,
    query_net_holdings,
    settle_position,
    settlement_price_from_position_row,
    settle_market_if_needed,
    startup_cleanup_stale_positions,
    auto_redeem_model_resolved_positions,
)
from pm_agent.runner.prefetch import data_prefetch_thread
from pm_agent.runner.decision_loop import process_model_decision

__all__ = [
    "safe_float",
    "parse_utc",
    "best_price",
    "normalize_hourly",
    "compute_phase",
    "all_model_ids",
    "query_net_holdings",
    "settle_position",
    "settlement_price_from_position_row",
    "settle_market_if_needed",
    "startup_cleanup_stale_positions",
    "auto_redeem_model_resolved_positions",
    "data_prefetch_thread",
    "process_model_decision",
]

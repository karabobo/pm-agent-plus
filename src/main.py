from __future__ import annotations

import dataclasses
import datetime as dt
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
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
from pm_agent.runner import (
    data_prefetch_thread,
    safe_float as _safe_float,
    parse_utc as _parse_utc,
    best_price as _best,
    normalize_hourly as _normalize_hourly,
    compute_phase as _compute_phase,
    all_model_ids as _all_model_ids,
    query_net_holdings as _query_net_holdings,
    settle_position as _settle_position,
    settle_market_if_needed as _settle_market_if_needed,
    startup_cleanup_stale_positions as _startup_cleanup_stale_positions,
    auto_redeem_model_resolved_positions as _auto_redeem_model_resolved_positions,
    process_model_decision as _run_model_decision,
)

logger = get_logger("pm_agent.main")

_ohlcv_source: OHLCVSource | None = None
_stop_event = threading.Event()


@dataclass
class MarketCache:
    """Thread-safe cache for shared market data.

    Replaces the bare ``cached_data`` dict + ``cache_lock`` globals with a
    typed container whose ``update()`` / ``snapshot()`` helpers always acquire
    the internal lock, eliminating the risk of forgetting to hold the lock.
    """

    # RLock avoids self-deadlock when legacy code acquires cache_lock and then
    # accesses cache.get()/__getitem__ which also acquires the same lock.
    _lock: threading.Lock = field(default_factory=threading.RLock, repr=False)
    hourly_market: dict[str, Any] | None = None
    balances: dict[str, float] = field(default_factory=dict)
    balances_by_model: dict[str, dict[str, float]] = field(default_factory=dict)
    onchain_cash_by_model: dict[str, float] = field(default_factory=dict)
    position_details_by_model: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    token_ids: list[str] = field(default_factory=list)
    price_to_beat: float | None = None
    current_btc_price: float | None = None
    last_update: str | None = None
    market_context: dict[str, Any] | None = None  # shared per-cycle computed market analysis

    def update(self, **kwargs: Any) -> None:
        """Atomically update one or more fields."""
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def snapshot(self) -> dict[str, Any]:
        """Return a shallow copy of all public fields under the lock."""
        with self._lock:
            return {
                "hourly_market": self.hourly_market,
                "balances": dict(self.balances),
                "balances_by_model": {
                    str(mid): dict(bal or {}) for mid, bal in self.balances_by_model.items()
                },
                "onchain_cash_by_model": {
                    str(mid): float(cash or 0.0)
                    for mid, cash in self.onchain_cash_by_model.items()
                },
                "position_details_by_model": {
                    str(mid): {
                        str(token_id): {
                            "shares": float((vals or {}).get("shares", 0.0) or 0.0),
                            "avg_price": float((vals or {}).get("avg_price", 0.0) or 0.0),
                        }
                        for token_id, vals in (details or {}).items()
                    }
                    for mid, details in self.position_details_by_model.items()
                },
                "token_ids": list(self.token_ids),
                "price_to_beat": self.price_to_beat,
                "current_btc_price": self.current_btc_price,
                "last_update": self.last_update,
            }

    # ------------------------------------------------------------------
    # Backward-compatible dict-style access so existing ``cached_data[k]``
    # and ``cached_data.get(k)`` call-sites require zero changes.
    # ------------------------------------------------------------------
    _FIELDS = frozenset(
        {
            "hourly_market",
            "balances",
            "balances_by_model",
            "onchain_cash_by_model",
            "position_details_by_model",
            "token_ids",
            "price_to_beat",
            "current_btc_price",
            "last_update",
            "market_context",
        }
    )

    def __getitem__(self, key: str) -> Any:
        if key not in self._FIELDS:
            raise KeyError(key)
        with self._lock:
            return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in self._FIELDS:
            raise KeyError(key)
        with self._lock:
            setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default



_market_cache = MarketCache()
# Expose the lock for legacy code that still acquires it directly.
cache_lock = _market_cache._lock
# Alias for callers that read cached_data[key] directly (backward compat).
cached_data = _market_cache  # type: ignore[assignment]

_error_count = 0
_last_error_time = 0.0
_cleanup_done = False


# ---------------------------------------------------------------------------
# Helper aliases (kept for backward compat; now live in pm_agent.runner)
# ---------------------------------------------------------------------------
# _safe_float, _parse_utc, _best, _normalize_hourly, _compute_phase,
# _all_model_ids, _query_net_holdings, _settle_position,
# _settle_market_if_needed, _startup_cleanup_stale_positions
# are all imported from pm_agent.runner above.


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


# data_prefetch_thread is imported from pm_agent.runner.prefetch

def _validate_ai_keys(s: Settings, ai_providers: list[str]) -> None:
    def _gemini_requires_api_key(base_url: str | None) -> bool:
        lowered = (base_url or "").strip().lower()
        if not lowered:
            return True
        # Native Google Gemini endpoints require API key auth.
        if "generativelanguage.googleapis.com" in lowered:
            return True
        if "googleapis.com" in lowered and "generatecontent" in lowered:
            return True
        # Third-party/local OpenAI-compatible Gemini gateways may handle
        # auth upstream and can run without GEMINI_API_KEY.
        return False

    missing_keys: list[str] = []
    missing_url: list[str] = []
    for provider_name in ai_providers:
        provider = provider_name.lower()
        if provider == "openai" and not s.openai_api_key:
            missing_keys.append(provider_name)
        elif provider == "deepseek" and not s.deepseek_api_key:
            missing_keys.append(provider_name)
        elif (
            provider == "gemini"
            and not s.gemini_api_key
            and _gemini_requires_api_key(s.gemini_base_url)
        ):
            missing_keys.append(provider_name)
        elif provider == "claude" and not s.claude_api_key:
            missing_keys.append(provider_name)
        elif provider == "qwen" and not s.qwen_api_key:
            missing_keys.append(provider_name)
        elif provider == "grok" and not s.grok_api_key:
            missing_keys.append(provider_name)
        elif provider == "glm" and not s.glm_api_key:
            missing_keys.append(provider_name)
        elif provider == "custom":
            if not s.custom_api_key:
                missing_keys.append(provider_name)
            if not s.custom_base_url:
                missing_url.append(provider_name)

    if missing_keys:
        raise SystemExit(
            "API key missing for: "
            + ", ".join(missing_keys)
            + ". Set it in environment or .env"
        )
    if missing_url:
        raise SystemExit(
            "CUSTOM_BASE_URL is required when using 'custom' provider. "
            "Set CUSTOM_BASE_URL in .env"
        )


def _build_ai_client_and_display(
    s: Settings,
    provider: str,
    model_override: str | None = None,
) -> tuple[AIClient, str]:
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
        qwen_model = (model_override or s.qwen_model).strip()
        return (
            AIClient(
                api_key=s.qwen_api_key,
                model=qwen_model,
                base_url=s.qwen_base_url,
                timeout_s=120,
                provider=provider,
            ),
            f"Qwen ({qwen_model})",
        )
    if provider == "grok":
        return (
            AIClient(
                api_key=s.grok_api_key,
                model=s.grok_model,
                base_url=s.grok_base_url,
                timeout_s=120,
                provider=provider,
            ),
            f"Grok ({s.grok_model})",
        )
    if provider == "glm":
        return (
            AIClient(
                api_key=s.glm_api_key,
                model=s.glm_model,
                base_url=s.glm_base_url,
                timeout_s=120,
                provider=provider,
            ),
            f"GLM ({s.glm_model})",
        )
    if provider == "custom":
        return (
            AIClient(
                api_key=s.custom_api_key,
                model=s.custom_model,
                base_url=s.custom_base_url,
                timeout_s=120,
                provider=provider,
            ),
            f"Custom ({s.custom_model})",
        )
    # Default: OpenAI
    return (
        AIClient(
            api_key=s.openai_api_key,
            model=s.openai_model,
            base_url=s.openai_base_url or None,
            timeout_s=120,
            provider="openai",
            reasoning_effort=s.openai_reasoning_effort,
            force_stream=s.openai_stream,
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
    provider_configs: list[tuple[str, str | None]] = []
    for provider in ai_providers:
        if provider == "qwen":
            # Allow running multiple Qwen models with one provider entry:
            # QWEN_MODEL=model_a,model_b,model_c
            qwen_models = [m.strip() for m in (s.qwen_model or "").split(",") if m.strip()]
            if qwen_models:
                for qwen_model in qwen_models:
                    provider_configs.append((provider, qwen_model))
            else:
                provider_configs.append((provider, None))
        else:
            provider_configs.append((provider, None))

    def _provider_model_label(provider_name: str, model_override: str | None) -> str:
        p = provider_name.lower().strip()
        if p == "qwen":
            model_name = (model_override or s.qwen_model).strip()
        elif p == "gemini":
            model_name = (s.gemini_model or "").strip()
        elif p == "openai":
            model_name = (s.openai_model or "").strip()
        elif p == "deepseek":
            model_name = (s.deepseek_model or "").strip()
        elif p == "claude":
            model_name = (s.claude_model or "").strip()
        elif p == "grok":
            model_name = (s.grok_model or "").strip()
        elif p == "glm":
            model_name = (s.glm_model or "").strip()
        elif p == "custom":
            model_name = (s.custom_model or "").strip()
        else:
            model_name = ""
        return f"{provider_name}:{model_name or '-'}"

    model_slots = [_provider_model_label(p, m) for p, m in provider_configs]
    private_keys = s.get_private_keys()
    if len(private_keys) != len(model_slots):
        raise SystemExit(
            "POLYMARKET_PRIVATE_KEYS must provide exactly one private key per running model. "
            f"models={len(model_slots)} ({', '.join(model_slots)}), "
            f"keys={len(private_keys)}. "
            "Set POLYMARKET_PRIVATE_KEYS in model order."
        )
    normalized_private_keys = [k.strip().lower() for k in private_keys]
    if len(set(normalized_private_keys)) != len(normalized_private_keys):
        raise SystemExit(
            "POLYMARKET_PRIVATE_KEYS contains duplicate keys. "
            "Each running model must use a unique private key."
        )

    raw_funders = s.get_funders()
    model_funders: list[str | None]
    if len(raw_funders) > 1:
        if len(raw_funders) != len(model_slots):
            raise SystemExit(
                "POLYMARKET_FUNDERS must provide exactly one funder per running model "
                "when multiple funders are configured. "
                f"models={len(model_slots)} ({', '.join(model_slots)}), "
                f"funders={len(raw_funders)}."
            )
        model_funders = [f.strip() or None for f in raw_funders]
    elif len(raw_funders) == 1:
        model_funders = [raw_funders[0].strip() or None for _ in model_slots]
    else:
        model_funders = [None for _ in model_slots]

    db = SQLiteDB(db_path=s.get_db_path())

    pm = PolymarketClient(
        host=s.polymarket_host,
        chain_id=s.polymarket_chain_id,
        api_key=s.polymarket_api_key,
        api_secret=s.polymarket_api_secret,
        api_passphrase=s.polymarket_api_passphrase,
        builder_api_key=s.polymarket_builder_api_key,
        builder_api_secret=s.polymarket_builder_secret,
        builder_api_passphrase=s.polymarket_builder_passphrase,
        private_key=private_keys[0],
        wallet_type=s.polymarket_wallet_type,
        signature_type=s.polymarket_signature_type,
        funder=model_funders[0],
        relayer_url=s.polymarket_relayer_url,
    )
    ohlcv = OHLCVSource()

    global _ohlcv_source
    _ohlcv_source = ohlcv

    models_config: list[dict[str, Any]] = []
    assigned_wallet_keys: list[str] = []
    model_id_counts: dict[str, int] = {}
    for idx, (provider, model_override) in enumerate(provider_configs):
        model_private_key = private_keys[idx]

        model_pm = PolymarketClient(
            host=s.polymarket_host,
            chain_id=s.polymarket_chain_id,
            api_key=s.polymarket_api_key,
            api_secret=s.polymarket_api_secret,
            api_passphrase=s.polymarket_api_passphrase,
            builder_api_key=s.polymarket_builder_api_key,
            builder_api_secret=s.polymarket_builder_secret,
            builder_api_passphrase=s.polymarket_builder_passphrase,
            private_key=model_private_key,
            wallet_type=s.polymarket_wallet_type,
            signature_type=s.polymarket_signature_type,
            funder=model_funders[idx],
            relayer_url=s.polymarket_relayer_url,
        )
        assigned_wallet_keys.append((model_private_key or "").strip().lower())

        ai, display_name = _build_ai_client_and_display(s, provider, model_override=model_override)
        base_model_id = f"{provider}-{ai.model}"
        duplicate_idx = model_id_counts.get(base_model_id, 0) + 1
        model_id_counts[base_model_id] = duplicate_idx
        model_id = base_model_id if duplicate_idx == 1 else f"{base_model_id}-{duplicate_idx}"
        if duplicate_idx > 1:
            logger.warning(
                "Duplicate provider/model detected (%s). Assigned unique model_id=%s for isolation.",
                base_model_id,
                model_id,
            )
        display_name_instance = display_name if duplicate_idx == 1 else f"{display_name} #{duplicate_idx}"
        orchestrator = StrategyOrchestrator(symbol=s.symbol, ohlcv=ohlcv, ai=ai)
        provider_sim_mode = s.get_provider_simulation_mode(provider)
        engine = ExecutionEngine(
            pm=model_pm,
            db=db,
            limits=RiskLimits(
                max_notional_usd=s.max_notional_usd,
                max_position_usd=s.max_position_usd,
                max_daily_trades=s.max_daily_trades,
                slippage_bps=s.slippage_bps,
            ),
            simulation_mode=provider_sim_mode,
            model_id=model_id,
            order_type=s.order_type,
        )

        db.init_account_balance(model_id, s.initial_balance_usd)

        if not provider_sim_mode:
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
                "display_name": display_name_instance,
                "orchestrator": orchestrator,
                "engine": engine,
                "pm_client": model_pm,
                "simulation_mode": provider_sim_mode,
            }
        )
        sim_label = "[SIM]" if provider_sim_mode else "[LIVE]"
        logger.info("  %s %s initialized (model_id=%s)", sim_label, display_name_instance, model_id)

    if not models_config:
        raise SystemExit("No model initialized.")

    # Enable on-chain balance injection only for live models that have a unique wallet key.
    # Models sharing one wallet will use per-model trade ledger only to avoid cross-contaminated
    # positions/P&L in runtime accounting and prompts.
    live_key_counts: dict[str, int] = {}
    for i, model_cfg in enumerate(models_config):
        if model_cfg.get("simulation_mode", s.simulation_mode):
            continue
        key = assigned_wallet_keys[i]
        if key:
            live_key_counts[key] = live_key_counts.get(key, 0) + 1

    duplicate_wallet_models: list[str] = []
    for i, model_cfg in enumerate(models_config):
        msim = model_cfg.get("simulation_mode", s.simulation_mode)
        key = assigned_wallet_keys[i]
        use_actual_balances = (not msim) and bool(key) and live_key_counts.get(key, 0) == 1
        model_cfg["use_actual_balances"] = use_actual_balances
        if not msim and key and live_key_counts.get(key, 0) > 1:
            duplicate_wallet_models.append(model_cfg["model_id"])

    if duplicate_wallet_models:
        logger.warning(
            "Detected shared live wallet across models (%s). Disabled per-model on-chain balance "
            "injection for these models to avoid position/P&L cross-contamination. "
            "Use POLYMARKET_PRIVATE_KEYS with one unique key per live model for true isolation.",
            duplicate_wallet_models,
        )

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

    decision_executor: ThreadPoolExecutor | None = None
    if s.model_parallel and len(models_config) > 1:
        desired_workers = s.model_max_workers if s.model_max_workers > 0 else len(models_config)
        max_workers = max(1, min(len(models_config), desired_workers))
        if max_workers > 1:
            decision_executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="pm-model",
            )
            logger.info("Model decision execution mode: parallel (%d workers)", max_workers)
        else:
            logger.info("Model decision execution mode: serial (MODEL_MAX_WORKERS=1)")
    else:
        logger.info("Model decision execution mode: serial")

    live_model_ids = [
        m["model_id"] for m in models_config if not m.get("simulation_mode", s.simulation_mode)
    ]
    model_pm_clients = {m["model_id"]: m["pm_client"] for m in models_config if m["model_id"] in live_model_ids}
    prefetch_thread = threading.Thread(
        target=data_prefetch_thread,
        kwargs=dict(
            pm=pm,
            ohlcv=ohlcv,
            symbol=s.symbol,
            s=s,
            db=db,
            model_pm_clients=model_pm_clients,
            live_model_ids=live_model_ids,
            cache=_market_cache,
            cache_lock=cache_lock,
            stop_event=_stop_event,
        ),
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

    last_ai_end_time = 0.0
    last_auto_redeem_ts = 0.0

    def _call_model_decision(model_cfg: dict[str, Any]) -> dict[str, Any]:
        """Thin wrapper: delegates to pm_agent.runner.decision_loop."""
        return _run_model_decision(
            model_cfg=model_cfg,
            cache=_market_cache,
            cache_lock=cache_lock,
            db=db,
            s=s,
            stop_event=_stop_event,
            last_ai_end_time=last_ai_end_time,
        )

    try:
        while not _stop_event.is_set():
            cycle_start_time = time.time()

            with cache_lock:
                hourly = dict(cached_data.get("hourly_market") or {})
                balances = dict(cached_data.get("balances") or {})
                raw_balances_by_model = cached_data.get("balances_by_model") or {}
                balances_by_model = (
                    {str(mid): dict(vals or {}) for mid, vals in raw_balances_by_model.items()}
                    if isinstance(raw_balances_by_model, dict)
                    else {}
                )
                raw_position_details_by_model = cached_data.get("position_details_by_model") or {}
                position_details_by_model = (
                    {
                        str(mid): {
                            str(tid): dict(detail or {})
                            for tid, detail in (details or {}).items()
                        }
                        for mid, details in raw_position_details_by_model.items()
                    }
                    if isinstance(raw_position_details_by_model, dict)
                    else {}
                )
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
            # auto_redeem runs per-model, using each model's own simulation_mode
            if (
                s.auto_redeem_enabled
                and now_ts - last_auto_redeem_ts >= float(s.auto_redeem_interval_sec)
            ):
                for model_cfg in models_config:
                    if model_cfg.get("simulation_mode", s.simulation_mode):
                        continue  # skip auto-redeem for simulation models
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

            # Determine interval based on whether ANY live model has an open position.
            # Each model's per-model positions are fetched inside _call_model_decision.
            has_any_position = False
            for model_cfg in models_config:
                mid = model_cfg["model_id"]
                msim = model_cfg.get("simulation_mode", s.simulation_mode)
                use_actual_balances = bool(
                    model_cfg.get("use_actual_balances", (not msim))
                )
                model_balances = balances_by_model.get(mid) or balances
                model_position_details = position_details_by_model.get(mid) or {}
                model_actual_avg_prices = {
                    str(tid): _safe_float((detail or {}).get("avg_price"), default=0.0)
                    for tid, detail in model_position_details.items()
                }
                mdp = db.get_positions(
                    token_ids,
                    actual_balances=(model_balances if (token_ids and use_actual_balances) else None),
                    actual_avg_prices=(
                        model_actual_avg_prices if (token_ids and use_actual_balances) else None
                    ),
                    model_id=mid,
                )
                up_shares_m = _safe_float(
                    (model_balances.get(up_id) if (use_actual_balances and up_id) else None)
                    or (mdp.get(up_id) or {}).get("shares"),
                    default=0.0,
                )
                down_shares_m = _safe_float(
                    (model_balances.get(down_id) if (use_actual_balances and down_id) else None)
                    or (mdp.get(down_id) or {}).get("shares"),
                    default=0.0,
                )
                if up_shares_m >= 1 or down_shares_m >= 1:
                    has_any_position = True
                    break

            base_interval = max(10, int(s.run_interval_sec))
            interval = base_interval if has_any_position else base_interval * 2
            if has_any_position:
                logger.debug("有持仓模型，使用正常间隔：%s秒", interval)
            else:
                logger.debug("无持仓，使用2倍间隔：%s秒", interval)

            up_best_bid = _safe_float((hourly.get("up") or {}).get("best_bid"), default=0.0)
            up_best_ask = _safe_float((hourly.get("up") or {}).get("best_ask"), default=0.0)
            down_best_bid = _safe_float((hourly.get("down") or {}).get("best_bid"), default=0.0)
            down_best_ask = _safe_float((hourly.get("down") or {}).get("best_ask"), default=0.0)

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

            # ── Console: market-level summary (shared, not per-model) ──
            btc_text = ""
            if price_to_beat:
                price_diff_show = float(current_btc_price) - float(price_to_beat)
                diff_sign = "+" if price_diff_show >= 0 else ""
                btc_text = (
                    f"💹 ${current_btc_price:.2f} vs 🎯 ${float(price_to_beat):.2f}"
                    f" ({diff_sign}{price_diff_show:.2f})"
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
            if btc_text:
                logger.info(btc_text)

            # ── Build shared market context (passed to each model's decision_loop) ──
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
                btc_is_winning = "UP" if price_diff > 0 else ("DOWN" if price_diff < 0 else "TIE")

            minutes_remaining = max(0, int(time_to_end_sec / 60))
            phase, phase_note = _compute_phase(minutes_remaining)
            btc_momentum = "neutral"
            if isinstance(price_diff_pct, (int, float)):
                btc_momentum = "up" if price_diff_pct > 0.1 else ("down" if price_diff_pct < -0.1 else "neutral")

            # market_context is public/shared; per-model account/positions are
            # fetched inside decision_loop using their own model_id.
            _market_cache["market_context"] = {
                "symbol": s.symbol,
                "hourly_market_slug": slug,
                "question": (hourly.get("market") or {}).get("question"),
                "end_time_utc": end_time_utc,
                "time_to_end_sec": time_to_end_sec,
                "price_to_beat": ptb,
                "up": hourly.get("up"),
                "down": hourly.get("down"),
                "btc_analysis": {
                    "current_price": current_btc,
                    "price_to_beat": ptb,
                    "price_diff": (current_btc - ptb) if ptb else None,
                    "price_diff_pct": price_diff_pct,
                    "btc_is_winning": btc_is_winning,
                    "btc_momentum": btc_momentum,
                },
                "session_phase": {
                    "phase": phase,
                    "minutes_remaining": minutes_remaining,
                    "phase_note": phase_note,
                },
                "_market_price_warning": (
                    "重要：UP/DOWN价格是结果不是原因。价格=f(price_diff, time_remaining)，"
                    "不代表市场情绪；决策依据应是技术指标。"
                ),
            }

            cycle_id = str(uuid.uuid4())

            # ── Run each model's decision loop (fully isolated) ──
            if decision_executor is None:
                for model_cfg in models_config:
                    try:
                        result = _call_model_decision(model_cfg)
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
            else:
                futures = {
                    decision_executor.submit(_call_model_decision, model_cfg): model_cfg
                    for model_cfg in models_config
                }
                for future in as_completed(futures):
                    model_cfg = futures[future]
                    try:
                        result = future.result()
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
                    msim = model_cfg.get("simulation_mode", s.simulation_mode)
                    use_actual_balances = bool(
                        model_cfg.get("use_actual_balances", (not msim))
                    )
                    model_balances = balances_by_model.get(model_id) or balances
                    model_position_details = position_details_by_model.get(model_id) or {}
                    model_actual_avg_prices = {
                        str(tid): _safe_float((detail or {}).get("avg_price"), default=0.0)
                        for tid, detail in model_position_details.items()
                    }
                    model_positions = db.get_positions(
                        token_ids,
                        actual_balances=(
                            model_balances if (token_ids and use_actual_balances) else None
                        ),
                        actual_avg_prices=(
                            model_actual_avg_prices if (token_ids and use_actual_balances) else None
                        ),
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
        if decision_executor is not None:
            decision_executor.shutdown(wait=False, cancel_futures=True)
        try:
            if _ohlcv_source is not None:
                _ohlcv_source.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

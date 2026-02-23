from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import sys

from dotenv import dotenv_values


def _candidate_env_paths() -> list[Path]:
    if getattr(sys, "frozen", False):
        return [Path(sys.executable).resolve().parent / ".env"]

    here = Path(__file__).resolve()
    out = [Path.cwd() / ".env"]

    # src/pm_agent/config.py -> try src/.env, repo/.env and parent repo/.env.
    for idx in (1, 2, 3):
        try:
            out.append(here.parents[idx] / ".env")
        except IndexError:
            break
    return out


def _load_dotenv_map() -> dict[str, str]:
    for p in _candidate_env_paths():
        if p.is_file():
            raw = dotenv_values(p)
            return {k: str(v) for k, v in raw.items() if v is not None}
    return {}


def _getenv(name: str, default: str | None, dotenv_map: dict[str, str]) -> str | None:
    v = os.getenv(name)
    if v is None or v == "":
        v = dotenv_map.get(name)
    if v is None or v == "":
        return default
    return v


def _as_bool(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _as_int(v: str | None, default: int) -> int:
    try:
        return int(str(v))
    except Exception:
        return default


def _as_float(v: str | None, default: float) -> float:
    try:
        return float(str(v))
    except Exception:
        return default


@dataclass
class Settings:
    ai_provider: str = "openai"

    openai_api_key: str = ""
    openai_model: str = ""
    openai_base_url: str = ""

    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com/v1"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3-flash-preview"
    gemini_base_url: str = ""

    claude_api_key: str = ""
    claude_model: str = "claude-sonnet-4-20250514"
    claude_base_url: str = "https://api.anthropic.com/v1/messages"

    qwen_api_key: str = ""
    qwen_model: str = "qwen-max"
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    polymarket_host: str = "https://clob.polymarket.com"
    polymarket_chain_id: int = 137
    polymarket_api_key: str | None = None
    polymarket_api_secret: str | None = None
    polymarket_api_passphrase: str | None = None
    polymarket_hourly_prefix: str = "bitcoin-up-or-down"
    polymarket_private_keys: str = ""
    polymarket_private_key: str = ""
    polymarket_wallet_type: str = "auto"
    polymarket_signature_type: str = "2"
    polymarket_funder: str | None = None

    private_key: str | None = None
    signature_type: str | None = None
    funder: str | None = None

    simulation_mode: bool = False
    mode: str = "conservative"
    model_id: str = "default"

    initial_balance_usd: float = 1000.0
    max_notional_usd: float = 200.0
    max_position_usd: float = 100.0
    max_daily_trades: int = 24
    slippage_bps: int = 30
    min_position_shares: float = 0.01
    run_interval_sec: int = 60
    auto_redeem_enabled: bool = True
    auto_redeem_interval_sec: int = 60
    auto_redeem_min_value_usd: float = 0.01
    auto_redeem_use_relayer: bool = True

    order_type: str = "GTC"
    symbol: str = "BTC"
    polymarket_relayer_url: str = "https://relayer-v2.polymarket.com/"

    db_path: str = "pm_agent.db"
    sim_db_path: str = ""
    live_db_path: str = ""

    def get_ai_providers(self) -> list[str]:
        return [p.strip() for p in self.ai_provider.split(",") if p.strip()]

    def __post_init__(self) -> None:
        self.ai_provider = (self.ai_provider or "openai").lower()
        self.private_key = self.polymarket_private_key
        self.signature_type = self.polymarket_signature_type
        self.funder = self.polymarket_funder
        self.mode = (self.mode or "conservative").lower()
        self.polymarket_wallet_type = (self.polymarket_wallet_type or "auto").strip().lower()
        if self.auto_redeem_interval_sec < 5:
            self.auto_redeem_interval_sec = 5
        if self.auto_redeem_min_value_usd < 0:
            self.auto_redeem_min_value_usd = 0.0
        if not self.polymarket_relayer_url:
            self.polymarket_relayer_url = "https://relayer-v2.polymarket.com/"
        if not self.gemini_base_url:
            self.gemini_base_url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent"
            )
        if not self.polymarket_funder and self.polymarket_private_key:
            try:
                from pm_agent.utils.web3_utils import get_proxy_address

                derived_funder = get_proxy_address(
                    self.polymarket_private_key,
                    self.polymarket_chain_id,
                )
                if derived_funder:
                    object.__setattr__(self, "polymarket_funder", derived_funder)
                    object.__setattr__(self, "funder", derived_funder)
            except Exception:
                pass

    def get_private_keys(self) -> list[str]:
        if self.polymarket_private_keys:
            return [p.strip() for p in self.polymarket_private_keys.split(",") if p.strip()]
        if self.polymarket_private_key:
            return [self.polymarket_private_key.strip()]
        return []

    def get_db_path(self) -> str:
        if self.simulation_mode:
            if self.sim_db_path:
                return self.sim_db_path
            base = self.db_path or "pm_agent.db"
            base_path = Path(base)
            if base_path.suffix:
                return str(base_path.with_name(f"{base_path.stem}_sim{base_path.suffix}"))
            return str(base_path.with_name(f"{base_path.name}_sim"))
        return self.live_db_path or self.db_path


def load_settings() -> Settings:
    env = _load_dotenv_map()
    # Backward-compatible shared key: prefer provider-specific keys first.
    shared_api_key = _getenv("FASTAPI_KEY", "", env) or ""

    gemini_model = _getenv("GEMINI_MODEL", "gemini-3-flash-preview", env) or "gemini-3-flash-preview"

    return Settings(
        ai_provider=_getenv("AI_PROVIDER", "openai", env) or "openai",
        openai_api_key=_getenv("OPENAI_API_KEY", shared_api_key, env) or "",
        openai_model=_getenv("OPENAI_MODEL", "", env) or "",
        openai_base_url=_getenv("OPENAI_BASE_URL", "", env) or "",
        deepseek_api_key=_getenv("DEEPSEEK_API_KEY", shared_api_key, env) or "",
        deepseek_model=_getenv("DEEPSEEK_MODEL", "deepseek-chat", env) or "deepseek-chat",
        deepseek_base_url=_getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1", env)
        or "https://api.deepseek.com/v1",
        gemini_api_key=_getenv("GEMINI_API_KEY", shared_api_key, env) or "",
        gemini_model=gemini_model,
        gemini_base_url=(
            _getenv("GEMINI_BASE_URL", "", env)
            or f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent"
        ),
        claude_api_key=_getenv("CLAUDE_API_KEY", shared_api_key, env) or "",
        claude_model=_getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514", env)
        or "claude-sonnet-4-20250514",
        claude_base_url=_getenv("CLAUDE_BASE_URL", "https://api.anthropic.com/v1/messages", env)
        or "https://api.anthropic.com/v1/messages",
        qwen_api_key=_getenv("QWEN_API_KEY", shared_api_key, env) or "",
        qwen_model=_getenv("QWEN_MODEL", "qwen-max", env) or "qwen-max",
        qwen_base_url=_getenv(
            "QWEN_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            env,
        )
        or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        polymarket_host=_getenv("POLYMARKET_HOST", "https://clob.polymarket.com", env)
        or "https://clob.polymarket.com",
        polymarket_chain_id=_as_int(_getenv("POLYMARKET_CHAIN_ID", "137", env), 137),
        polymarket_api_key=_getenv("POLYMARKET_API_KEY", None, env),
        polymarket_api_secret=_getenv("POLYMARKET_API_SECRET", None, env),
        polymarket_api_passphrase=_getenv("POLYMARKET_API_PASSPHRASE", None, env),
        polymarket_hourly_prefix=_getenv("POLYMARKET_HOURLY_PREFIX", "bitcoin-up-or-down", env)
        or "bitcoin-up-or-down",
        polymarket_private_keys=_getenv("POLYMARKET_PRIVATE_KEYS", "", env) or "",
        polymarket_private_key=_getenv("POLYMARKET_PRIVATE_KEY", "", env) or "",
        polymarket_wallet_type=_getenv("POLYMARKET_WALLET_TYPE", "auto", env) or "auto",
        polymarket_signature_type=_getenv("POLYMARKET_SIGNATURE_TYPE", "2", env) or "2",
        polymarket_funder=_getenv("POLYMARKET_FUNDER", None, env),
        simulation_mode=_as_bool(_getenv("SIMULATION_MODE", "false", env), False),
        mode=_getenv("MODE", "conservative", env) or "conservative",
        model_id=_getenv("MODEL_ID", "default", env) or "default",
        initial_balance_usd=_as_float(_getenv("INITIAL_BALANCE_USD", "1000", env), 1000.0),
        max_notional_usd=_as_float(_getenv("MAX_NOTIONAL_USD", "200", env), 200.0),
        max_position_usd=_as_float(_getenv("MAX_POSITION_USD", "100", env), 100.0),
        max_daily_trades=_as_int(_getenv("MAX_DAILY_TRADES", "24", env), 24),
        slippage_bps=_as_int(_getenv("SLIPPAGE_BPS", "30", env), 30),
        min_position_shares=_as_float(_getenv("MIN_POSITION_SHARES", "0.01", env), 0.01),
        run_interval_sec=_as_int(_getenv("RUN_INTERVAL_SEC", "60", env), 60),
        auto_redeem_enabled=_as_bool(_getenv("AUTO_REDEEM_ENABLED", "true", env), True),
        auto_redeem_interval_sec=_as_int(_getenv("AUTO_REDEEM_INTERVAL_SEC", "60", env), 60),
        auto_redeem_min_value_usd=_as_float(
            _getenv("AUTO_REDEEM_MIN_VALUE_USD", "0.01", env), 0.01
        ),
        auto_redeem_use_relayer=_as_bool(_getenv("AUTO_REDEEM_USE_RELAYER", "true", env), True),
        order_type=_getenv("ORDER_TYPE", "GTC", env) or "GTC",
        symbol=_getenv("SYMBOL", "BTC", env) or "BTC",
        polymarket_relayer_url=_getenv(
            "POLYMARKET_RELAYER_URL", "https://relayer-v2.polymarket.com/", env
        )
        or "https://relayer-v2.polymarket.com/",
        db_path=_getenv("DB_PATH", "pm_agent.db", env) or "pm_agent.db",
        sim_db_path=_getenv("SIM_DB_PATH", "", env) or "",
        live_db_path=_getenv("LIVE_DB_PATH", "", env) or "",
    )

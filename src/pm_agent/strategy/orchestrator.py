from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pm_agent.ai.client import AIClient
from pm_agent.data.ohlcv import OHLCVSource
from pm_agent.indicators.core import add_indicators
from pm_agent.parse.decision import ParsedDecision, parse_full_decision_response
from pm_agent.prompts.system_prompt import SystemPromptConfig, build_system_prompt
from pm_agent.prompts.user_prompt import UserPromptInputs, build_user_prompt
from pm_agent.utils.logging import get_logger
from pm_agent.utils.time import iso, utcnow

logger = get_logger("pm_agent.strategy")


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


@dataclass
class StrategyOrchestrator:
    mode: str
    symbol: str
    ohlcv: OHLCVSource
    ai: AIClient

    def compute_indicators_snapshot(self) -> dict[str, Any]:
        snap: dict[str, Any] = {}
        current_price: float | None = None
        current_price_ts: str | None = None

        for tf in ("5m", "15m", "30m", "1h"):
            df = self.ohlcv.fetch_futures_ohlcv_with_ws(self.symbol, tf)
            if df.empty:
                logger.warning("%s OHLCV is empty for %s", tf, self.symbol)
                continue

            df = add_indicators(df).tail(200)
            last_ts = df.index[-1] if len(df.index) else None
            last = df.iloc[-1]

            logger.info(
                "".join(
                    [
                        f"{tf} rows={len(df)}",
                        f" last_ts={last_ts}",
                        f" ema20={_to_float(last.get('ema_20')):.6f}",
                        f" ema50={_to_float(last.get('ema_50')):.6f}",
                        f" ret1={_to_float(last.get('returns_1')):.6f}",
                        f" mom14={_to_float(last.get('momentum_14')):.6f}",
                        f" ewma_mom14={_to_float(last.get('ewma_momentum_14')):.6f}",
                        f" macd={_to_float(last.get('macd')):.6f}",
                        f" macd_signal={_to_float(last.get('macd_signal')):.6f}",
                        f" macd_hist={_to_float(last.get('macd_hist')):.6f}",
                        f" rsi14={_to_float(last.get('rsi_14')):.4f}",
                        f" mfi14={_to_float(last.get('mfi_14')):.4f}",
                        f" tr5={_to_float(last.get('tr_5')):.6f}",
                        f" donchian5={_to_float(last.get('donchian_breakout_5')):.2f}",
                        f" jump20_3={_to_float(last.get('jump_spike_20_3')):.0f}",
                        f" price_pct15={_to_float(last.get('price_pct_15m')):.2f}",
                        f" price_pct30={_to_float(last.get('price_pct_30m')):.2f}",
                        f" rv_parkinson14={_to_float(last.get('rv_parkinson_14')):.6f}",
                        f" rv_gk14={_to_float(last.get('rv_gk_14')):.6f}",
                        f" rv_rs14={_to_float(last.get('rv_rs_14')):.6f}",
                        f" vol_sma20={_to_float(last.get('vol_sma_20')):.6f}",
                        f" vol_z20={_to_float(last.get('vol_z_20')):.4f}",
                    ]
                )
            )

            last_dict = last.to_dict()
            snap[tf] = {k: _to_float(v) for k, v in last_dict.items()}

            if tf == "5m":
                current_price = _to_float(last.get("close"), default=0.0)
                current_price_ts = str(last_ts) if last_ts is not None else None

        if current_price is not None:
            snap["current_price"] = current_price
            if current_price_ts:
                snap["current_price_ts"] = current_price_ts

        fetch_open_interest = getattr(self.ohlcv, "fetch_open_interest", None)
        if callable(fetch_open_interest):
            oi = fetch_open_interest(self.symbol)
            if oi is not None:
                snap["open_interest"] = oi

        fetch_funding_rate = getattr(self.ohlcv, "fetch_funding_rate", None)
        if callable(fetch_funding_rate):
            fr = fetch_funding_rate(self.symbol)
            if fr is not None:
                snap["funding_rate"] = fr["funding_rate"]
                snap["funding_time"] = str(fr["funding_time"])

        fetch_oi_history = getattr(self.ohlcv, "fetch_oi_history", None)
        if callable(fetch_oi_history):
            try:
                oi_hist = fetch_oi_history(self.symbol, "1h", 24)
                if not oi_hist.empty:
                    oi_change = (
                        (oi_hist["open_interest"].iloc[-1] - oi_hist["open_interest"].iloc[0])
                        / oi_hist["open_interest"].iloc[0]
                    ) * 100
                    snap["oi_change_24h_pct"] = float(oi_change)
            except Exception as e:
                logger.warning("Failed to calculate OI trend: %s", e)

        return snap

    def build_prompts(
        self,
        cycle_id: str,
        context: dict[str, Any],
        indicators_snapshot: dict[str, Any],
    ) -> tuple[str, str]:
        limits = context.get("risk_limits", {})
        max_notional = limits.get("max_notional_usd")
        max_position = limits.get("max_position_usd")
        limit_text = ""
        if max_notional is not None or max_position is not None:
            limit_text = (
                f" Risk limits: max_notional_usd={max_notional}, "
                f"max_position_usd={max_position}."
            )

        sys_cfg = SystemPromptConfig(
            role_definition=(
                "You are a professional Polymarket crypto trader focused on BTC hourly "
                "UP/DOWN markets. You are intelligent and adaptive. This is a 60-minute "
                "session - you decide whether to scalp, swing, or hold to settlement!"
                + limit_text
            ),
            mode=self.mode,
            trade_frequency=(
                "Evaluate each cycle and decide your trading style. You can:\n"
                "- SCALP: Quick in-and-out trades (minutes)\n"
                "- SWING: Hold for 15-30 minutes\n"
                "- HOLD TO SETTLEMENT: Ride the trend to the end"
            ),
            entry_standards=(
                "INDICATOR USAGE:\n"
                "- PRICE vs TARGET (CRITICAL): If Price < Target, you are in DOWN territory. "
                "EMA bullish alignment is IRRELEVANT if gap is widening.\n"
                "- EMA20 vs EMA50: Lagging indicator. Use only for general context. DO NOT "
                "catch falling knives just because EMAs are crossed up.\n"
                "- Returns: Recent price change (log return)\n"
                "- Momentum: Rolling sum of returns (trend strength)\n"
                "- EWMA Momentum: Exponentially-weighted momentum (reacts faster)\n"
                "- MACD histogram: Momentum is KING. If 5m MACD is negative, DO NOT BUY UP.\n"
                "- RSI: Overbought (>70) / Oversold (<30) for reversals\n"
                "- MFI (Money Flow Index):\n"
                "  * >80 (1H timeframe) = DANGER ZONE. Upside is capped. DO NOT BUY UP.\n"
                "  * <20 = Potential bounce, but require volume confirmation.\n"
                "- True Range (short window): Range-based volatility over recent candles\n"
                "- Donchian Breakout (short window): +1 breakout above recent high, -1 breakdown "
                "below recent low\n"
                "- Jump/Spike: return exceeds 3 sigma of 20-period rolling std (1 = spike)\n"
                "- Price Percentile (15m/30m): position within recent range (0=low, 100=high)\n"
                "- Realized Vol (Parkinson/GK/RS): OHLC-based volatility (higher = more risk)\n"
                "- Volume Z-score: >1.5 confirms moves, <-1 = low participation\n"
                "- Open Interest (OI): Total contracts held. Rising OI + Rising Price = Strong "
                "uptrend. Rising OI + Falling Price = Strong downtrend\n"
                "- OI Change 24h: % change in open interest over 24h. Big increase = new money "
                "entering, decrease = positions closing\n"
                "- Funding Rate (FR): Positive = longs pay shorts (bullish sentiment), Negative "
                "= shorts pay longs (bearish sentiment). Extreme rates (>0.1% or <-0.1%) = "
                "potential reversal\n\n"
                "PREDICTION MARKET REALITY CHECK:\n"
                "- Probability Price < 0.30 means the market knows something (70% chance of losing).\n"
                "- Probability Price > 0.70 means strong consensus.\n"
                "- DO NOT fight the consensus (Price < 0.30) if Time < 30 mins. It's not "
                "'cheap', it's 'dying'."
            ),
            decision_process=(
                "=== FLEXIBLE DECISION PROCESS ===\n\n"
                "STEP 1: DEADLY TRIAD CHECK (AVOID STUPID LOSSES)\n"
                "- IF 1H MFI > 80 (Overbought) AND You want to Buy UP -> STOP. FORBIDDEN.\n"
                "- IF Market Price < 0.30 AND Time < 30m AND You want to Buy UP -> STOP. FORBIDDEN.\n"
                "- IF 5m MACD < 0 (Negative) AND Price < Target AND You want to Buy UP -> STOP. FORBIDDEN.\n\n"
                "STEP 2: ASSESS TIME (session_phase)\n"
                "- How much time is left?\n"
                "- Time changes EVERYTHING. More time = more flexibility.\n\n"
                "STEP 3: ASSESS CONFIDENCE\n"
                "- Check btc_analysis: Is BTC supporting your direction?\n"
                "- Check indicators: Confirming or diverging?\n"
                "- Be HONEST: How confident are you really?\n\n"
                "STEP 4: CHECK POSITION STATUS (if you have one)\n"
                "- Look at position_intelligence: pnl_pct, status\n"
                "- Combine with TIME and CONFIDENCE:\n"
                "  * Loss + Time + Confidence = Can hold\n"
                "  * Loss + No time + Against you = Cut\n"
                "  * Profit + Early in session = Consider taking profits to avoid reversals\n"
                "  * Profit + Late + Confident = Can hold to settlement\n\n"
                "STEP 5: DECIDE ACTION\n"
                "- HOLD: Position working OR loss but confident with time\n"
                "- CLOSE: Locking profit OR cutting loss (time/confidence says so)\n"
                "- OPEN: Good entry price + confidence + time makes sense\n"
                "- WAIT: No clear edge, sit on hands\n\n"
                "KEY PRINCIPLE:\n"
                "There are NO fixed rules. You combine TIME + CONFIDENCE + P&L\n"
                "to make intelligent, flexible decisions. Trust your analysis!"
            ),
            hard_constraints=[
                "Price in [0,1].",
                "Only output the specified format.",
                "Use UP/DOWN terminology only.",
                "Set token_id to the corresponding up_token_id/down_token_id from candidate_markets.",
                "BINARY MARKET: Winner=$1.00, Loser=$0.00.",
                "LOW PRICE + NEAR SETTLEMENT (< 10 min) = DO NOT BUY the losing side.",
                "FLEXIBLE P&L: No fixed % for stop-loss/take-profit. Combine TIME + CONFIDENCE + P&L.",
                "BE HONEST: About your confidence level. Don't hold losing trades just on hope.",
                f"Never exceed max_notional_usd={max_notional} or max_position_usd={max_position}.",
            ],
        )
        system_prompt = build_system_prompt(sys_cfg)

        market_overview = context.get("market_overview")
        if not isinstance(market_overview, dict):
            market_overview = {}
        account_info = context.get("account")
        if not isinstance(account_info, dict):
            account_info = context.get("account_info")
        if not isinstance(account_info, dict):
            account_info = {}
        current_positions = context.get("positions")
        if not isinstance(current_positions, dict):
            current_positions = context.get("current_positions")
        if not isinstance(current_positions, dict):
            current_positions = {}
        candidate_markets = context.get("candidate_markets")
        if not isinstance(candidate_markets, list):
            candidate_markets = []

        user_inp = UserPromptInputs(
            system_state={"utc_time": iso(utcnow()), "cycle_id": cycle_id},
            market_overview=market_overview | {"indicators_snapshot": indicators_snapshot},
            account_info=account_info,
            current_positions=current_positions,
            candidate_markets=candidate_markets,
            session_history=context.get("session_history"),
            decision_history=context.get("decision_history"),
        )
        user_prompt = build_user_prompt(user_inp)
        return system_prompt, user_prompt

    def run_once(self, cycle_id: str, context: dict[str, Any]) -> ParsedDecision:
        indicators_snapshot = self.compute_indicators_snapshot()
        system_prompt, user_prompt = self.build_prompts(cycle_id, context, indicators_snapshot)
        raw = self.ai.call_with_messages(system_prompt, user_prompt)
        parsed = parse_full_decision_response(raw)
        parsed.prompt = user_prompt
        return parsed

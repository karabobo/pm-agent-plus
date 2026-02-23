from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def returns_log(close: pd.Series) -> pd.Series:
    close = close.replace(0, np.nan)
    return np.log(close / close.shift(1)).fillna(0)


def momentum(returns: pd.Series, period: int = 14) -> pd.Series:
    return returns.rolling(period).sum().fillna(0)


def ewma_momentum(returns: pd.Series, span: int = 14) -> pd.Series:
    return returns.ewm(span=span, adjust=False).mean().fillna(0)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    line = fast_ema - slow_ema
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.fillna(0)


def donchian_breakout(
    close: pd.Series, high: pd.Series, low: pd.Series, window: int = 5
) -> pd.Series:
    upper = high.rolling(window).max()
    lower = low.rolling(window).min()
    breakout = np.where(close >= upper, 1, np.where(close <= lower, -1, 0))
    return pd.Series(breakout, index=close.index).fillna(0)


def jump_spike_flag(returns: pd.Series, window: int = 20, z_threshold: float = 3.0) -> pd.Series:
    rolling_std = returns.rolling(window).std().replace(0, np.nan)
    z = returns / rolling_std
    flag = (z.abs() >= z_threshold).astype(float)
    return flag.fillna(0)


def price_percentile_position(close: pd.Series, window_minutes: int) -> pd.Series:
    if isinstance(close.index, pd.DatetimeIndex):
        window = f"{int(window_minutes)}min"
        rolling_min = close.rolling(window).min()
        rolling_max = close.rolling(window).max()
    else:
        rolling_min = close.rolling(window_minutes).min()
        rolling_max = close.rolling(window_minutes).max()
    denom = (rolling_max - rolling_min).replace(0, np.nan)
    pct = ((close - rolling_min) / denom) * 100
    return pct.clip(lower=0, upper=100).fillna(50)


def _safe_log_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    a = a.replace(0, np.nan)
    b = b.replace(0, np.nan)
    return np.log(a / b)


def realized_vol_parkinson(high: pd.Series, low: pd.Series, period: int = 14) -> pd.Series:
    hl = _safe_log_ratio(high, low)
    var = hl.pow(2).rolling(period).mean() / (4 * np.log(2))
    return np.sqrt(var.clip(lower=0))


def realized_vol_gk(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    open_: pd.Series,
    period: int = 14,
) -> pd.Series:
    hl = _safe_log_ratio(high, low)
    co = _safe_log_ratio(close, open_)
    var = 0.5 * hl.pow(2) - (2 * np.log(2) - 1) * co.pow(2)
    var = var.rolling(period).mean()
    return np.sqrt(var.clip(lower=0))


def realized_vol_rs(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    open_: pd.Series,
    period: int = 14,
) -> pd.Series:
    ho = _safe_log_ratio(high, open_)
    hc = _safe_log_ratio(high, close)
    lo = _safe_log_ratio(low, open_)
    lc = _safe_log_ratio(low, close)
    var = ho * hc + lo * lc
    var = var.rolling(period).mean()
    return np.sqrt(var.clip(lower=0))


def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    direction = typical_price.diff()
    positive_flow = money_flow.where(direction > 0, 0.0)
    negative_flow = money_flow.where(direction < 0, 0.0)
    positive_mf = positive_flow.rolling(period).sum()
    negative_mf = negative_flow.rolling(period).sum()
    mfr = positive_mf / negative_mf.replace(0, np.nan)
    return (100 - 100 / (1 + mfr)).fillna(50)


def kdj(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 9,
    k_smooth: int = 3,
    d_smooth: int = 3,
):
    lowest_low = low.rolling(period).min()
    highest_high = high.rolling(period).max()
    rsv = (((close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)) * 100).fillna(50)
    k = rsv.ewm(alpha=1 / k_smooth, adjust=False).mean()
    d = k.ewm(alpha=1 / d_smooth, adjust=False).mean()
    j = 3 * k - 2 * d
    return k.fillna(50), d.fillna(50), j.fillna(50)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema_20"] = ema(out["close"], 20)
    out["ema_50"] = ema(out["close"], 50)

    out["returns_1"] = returns_log(out["close"])
    out["momentum_14"] = momentum(out["returns_1"], 14)
    out["ewma_momentum_14"] = ewma_momentum(out["returns_1"], 14)

    macd_line, macd_sig, macd_hist = macd(out["close"])
    out["macd"] = macd_line
    out["macd_signal"] = macd_sig
    out["macd_hist"] = macd_hist

    out["rsi_14"] = rsi(out["close"], 14)
    out["tr_5"] = true_range(out["high"], out["low"], out["close"]).rolling(5).mean().fillna(0)
    out["donchian_breakout_5"] = donchian_breakout(out["close"], out["high"], out["low"], 5)
    out["jump_spike_20_3"] = jump_spike_flag(out["returns_1"], 20, 3)

    out["price_pct_15m"] = price_percentile_position(out["close"], 15)
    out["price_pct_30m"] = price_percentile_position(out["close"], 30)

    out["rv_parkinson_14"] = realized_vol_parkinson(out["high"], out["low"], 14)
    out["rv_gk_14"] = realized_vol_gk(out["high"], out["low"], out["close"], out["open"], 14)
    out["rv_rs_14"] = realized_vol_rs(out["high"], out["low"], out["close"], out["open"], 14)

    out["vol_sma_20"] = out["volume"].rolling(20).mean()
    out["vol_z_20"] = (
        (out["volume"] - out["vol_sma_20"])
        / out["volume"].rolling(20).std().replace(0, np.nan)
    ).fillna(0)
    out["mfi_14"] = mfi(out["high"], out["low"], out["close"], out["volume"], 14)

    return out

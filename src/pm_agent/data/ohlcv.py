from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional
import datetime as dt
import json
import threading
import time

import httpx
import numpy as np
import pandas as pd
from websocket import create_connection, WebSocketTimeoutException

from pm_agent.utils.logging import get_logger

logger = get_logger("pm_agent.ohlcv")
Interval = Literal["1m", "5m", "15m", "30m", "1h", "4h"]


@dataclass
class OHLCVSource:
    proxy: Optional[str] = None
    timeout_s: float = 20.0
    rest_host: str = "https://api.binance.com"
    ws_host: str = "wss://stream.binance.com:9443"
    kline_limit: int = 500
    cache_ttl_s: float = 30.0

    _open_cache: dict[str, float] = field(default_factory=dict)
    _oi_cache: dict[str, tuple[float, float]] = field(default_factory=dict)
    _fr_cache: dict[str, tuple[dict[str, Any], float]] = field(default_factory=dict)
    _ws_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    _ws_threads: dict[str, threading.Thread] = field(default_factory=dict)
    _ws_connections: dict[str, Any] = field(default_factory=dict)
    _ws_lock: threading.Lock = field(default_factory=threading.Lock)
    _stop_flag: threading.Event = field(default_factory=threading.Event)
    _http_client: Optional[httpx.Client] = field(default=None, init=False, repr=False)
    _client_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        logger.info("Initialized OHLCVSource with persistent HTTP connection")

    def _get_client(self) -> httpx.Client:
        """Get or create persistent HTTP client (thread-safe)."""
        with self._client_lock:
            if self._http_client is None or self._http_client.is_closed:
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Accept": "application/json",
                    "Accept-Language": "en-US,en;q=0.9",
                }
                kwargs: dict[str, Any] = {
                    "timeout": self.timeout_s,
                    "headers": headers,
                    "limits": httpx.Limits(max_keepalive_connections=5, max_connections=10, keepalive_expiry=15),
                    "transport": httpx.HTTPTransport(retries=2),
                    "verify": False,
                }
                try:
                    if self.proxy:
                        # new httpx versions use proxy=; old versions use proxies=
                        test_client = httpx.Client(proxy=self.proxy)
                        test_client.close()
                        kwargs["proxy"] = self.proxy
                except TypeError:
                    if self.proxy:
                        kwargs["proxies"] = self.proxy
                self._http_client = httpx.Client(**kwargs)
            return self._http_client

    def close(self) -> None:
        with self._ws_lock:
            self._stop_flag.set()
            for conn in self._ws_connections.values():
                try:
                    conn.close()
                except Exception:
                    pass
            self._ws_connections.clear()

        with self._client_lock:
            if self._http_client is not None:
                try:
                    self._http_client.close()
                except Exception:
                    pass
                self._http_client = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        s = (symbol or "BTC").upper()
        return s if s.endswith("USDT") else f"{s}USDT"

    def _ws_key(self, symbol: str, interval: Interval) -> str:
        return f"{self._normalize_symbol(symbol)}:{interval}"

    def wait_for_websocket_ready(self, symbol: str, interval: Interval = "1h", timeout: float = 10.0) -> bool:
        end = time.time() + timeout
        key = self._ws_key(symbol, interval)
        while time.time() < end:
            data = self._ws_cache.get(key)
            if data and data.get("close") is not None:
                return True
            self._ensure_ws(symbol, interval)
            time.sleep(0.2)
        return False

    def _ensure_ws(self, symbol: str, interval: Interval) -> None:
        key = self._ws_key(symbol, interval)
        with self._ws_lock:
            thread = self._ws_threads.get(key)
            if thread is not None and thread.is_alive():
                return
            thread = threading.Thread(target=self._ws_loop, args=(symbol, interval), daemon=True)
            self._ws_threads[key] = thread
            thread.start()

    def _ws_loop(self, symbol: str, interval: Interval) -> None:
        sym = self._normalize_symbol(symbol).lower()
        key = self._ws_key(symbol, interval)
        stream = f"{sym}@kline_{interval}"
        ws_url = f"{self.ws_host}/ws/{stream}"

        while not self._stop_flag.is_set():
            ws = None
            try:
                ws = create_connection(ws_url, timeout=self.timeout_s)
                with self._ws_lock:
                    self._ws_connections[key] = ws

                while not self._stop_flag.is_set():
                    msg = ws.recv()
                    if not msg:
                        continue
                    payload = json.loads(msg)
                    k = payload.get("k") or {}
                    self._ws_cache[key] = {
                        "timestamp": int(k.get("t", 0)),
                        "open": float(k.get("o", 0) or 0),
                        "high": float(k.get("h", 0) or 0),
                        "low": float(k.get("l", 0) or 0),
                        "close": float(k.get("c", 0) or 0),
                        "volume": float(k.get("v", 0) or 0),
                        "is_closed": bool(k.get("x", False)),
                        "raw": payload,
                    }
            except WebSocketTimeoutException:
                continue
            except Exception as ex:
                logger.warning("WebSocket loop error (%s %s): %s", symbol, interval, ex)
                time.sleep(1.0)
            finally:
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass
                with self._ws_lock:
                    self._ws_connections.pop(key, None)

    def fetch_futures_ohlcv(
        self, symbol: str, interval: Interval = "1h", limit: Optional[int] = None
    ) -> pd.DataFrame:
        symbol_n = self._normalize_symbol(symbol)
        lim = int(limit or self.kline_limit)
        c = self._get_client()

        urls = [
            f"{self.rest_host}/fapi/v1/klines",
            f"{self.rest_host}/api/v3/klines",
        ]
        data = None
        for url in urls:
            try:
                r = c.get(url, params={"symbol": symbol_n, "interval": interval, "limit": lim})
                r.raise_for_status()
                data = r.json()
                if isinstance(data, list):
                    break
            except Exception:
                continue

        if not isinstance(data, list):
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        rows = []
        for x in data:
            try:
                ts = dt.datetime.fromtimestamp(int(x[0]) / 1000, tz=dt.timezone.utc)
                rows.append(
                    {
                        "timestamp": ts,
                        "open": float(x[1]),
                        "high": float(x[2]),
                        "low": float(x[3]),
                        "close": float(x[4]),
                        "volume": float(x[5]),
                    }
                )
            except Exception:
                continue

        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        return df

    def fetch_open_interest(self, symbol: str) -> Optional[float]:
        """
        获取当前期货持仓量（Open Interest）。
        使用短缓存降低请求频率，出现限流时回退缓存。
        """
        sym = self._normalize_symbol(symbol)
        now = time.time()
        if sym in self._oi_cache:
            value, ts = self._oi_cache[sym]
            if now - ts < self.cache_ttl_s:
                return value

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        }

        try:
            with httpx.Client(timeout=self.timeout_s) as c:
                r = c.get(
                    "https://fapi.binance.com/fapi/v1/openInterest",
                    params={"symbol": sym},
                    headers=headers,
                )
                r.raise_for_status()
                data = r.json()
                value = float(data.get("openInterest", 0) or 0)
                self._oi_cache[sym] = (value, now)
                return value
        except httpx.HTTPStatusError as e:
            if e.response is not None and e.response.status_code == 418:
                logger.warning("Binance OI API rate limit (418) for %s", sym)
                if sym in self._oi_cache:
                    cached_value, cached_ts = self._oi_cache[sym]
                    age = now - cached_ts
                    logger.info("  Using cached OI: %.0f (age: %.0fs)", cached_value, age)
                    return cached_value
                logger.warning("  No cached OI available, will retry later")
                return None
            logger.warning("Failed to fetch open interest: %s", e)
            if sym in self._oi_cache:
                return self._oi_cache[sym][0]
            return None
        except Exception as e:
            logger.warning("Failed to fetch open interest: %s", e)
            if sym in self._oi_cache:
                return self._oi_cache[sym][0]
            return None

    def fetch_funding_rate(self, symbol: str) -> Optional[dict[str, Any]]:
        """
        获取最近一期资金费率。
        返回: {"funding_rate": float, "funding_time": datetime}
        """
        sym = self._normalize_symbol(symbol)
        now = time.time()
        if sym in self._fr_cache:
            cached_value, cached_ts = self._fr_cache[sym]
            if now - cached_ts < self.cache_ttl_s:
                return cached_value

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        }

        try:
            with httpx.Client(timeout=self.timeout_s) as c:
                r = c.get(
                    "https://fapi.binance.com/fapi/v1/fundingRate",
                    params={"symbol": sym, "limit": 1},
                    headers=headers,
                )
                r.raise_for_status()
                rows = r.json()
                if not isinstance(rows, list) or not rows:
                    return None
                row = rows[0]
                out = {
                    "funding_rate": float(row.get("fundingRate", 0) or 0),
                    "funding_time": dt.datetime.fromtimestamp(
                        int(row.get("fundingTime", 0)) / 1000,
                        tz=dt.timezone.utc,
                    ),
                }
                self._fr_cache[sym] = (out, now)
                return out
        except httpx.HTTPStatusError as e:
            if e.response is not None and e.response.status_code == 418:
                logger.warning("Binance FR API rate limit (418) for %s", sym)
                if sym in self._fr_cache:
                    cached_value, cached_ts = self._fr_cache[sym]
                    age = now - cached_ts
                    logger.info(
                        "  Using cached FR: %.6f (age: %.0fs)",
                        float(cached_value.get("funding_rate", 0) or 0),
                        age,
                    )
                    return cached_value
                logger.warning("  No cached FR value available, will retry later")
                return None
            logger.warning("Failed to fetch funding rate: %s", e)
            if sym in self._fr_cache:
                return self._fr_cache[sym][0]
            return None
        except Exception as e:
            logger.warning("Failed to fetch funding rate: %s", e)
            if sym in self._fr_cache:
                return self._fr_cache[sym][0]
            return None

    def fetch_oi_history(self, symbol: str, interval: str = "1h", limit: int = 100) -> pd.DataFrame:
        """获取期货持仓量历史。"""
        sym = self._normalize_symbol(symbol)
        c = self._get_client()
        try:
            r = c.get(
                "https://fapi.binance.com/futures/data/openInterestHist",
                params={"symbol": sym, "period": interval, "limit": limit},
            )
            r.raise_for_status()
            rows = r.json()
            data = []
            for item in rows:
                ts = pd.to_datetime(item["timestamp"], unit="ms", utc=True)
                data.append(
                    {
                        "ts": ts,
                        "open_interest": float(item["sumOpenInterest"]),
                        "open_interest_value": float(item["sumOpenInterestValue"]),
                    }
                )
            df = pd.DataFrame(data).set_index("ts")
            df.index.name = "ts"
            return df
        except Exception as e:
            logger.warning("Failed to fetch OI history: %s", e)
            return pd.DataFrame()

    def fetch_futures_open_at(self, symbol: str, start_dt: dt.datetime) -> Optional[float]:
        start = start_dt.replace(second=0, microsecond=0)
        end = start + dt.timedelta(minutes=2)
        df = self.fetch_spot_ohlcv_placeholder(symbol, start, end)
        if df.empty:
            return None
        try:
            return float(df.iloc[0]["open"])
        except Exception:
            return None

    def get_current_price(
        self,
        symbol: str,
        interval: Interval = "1h",
        wait_for_ready: bool = False,
        wait_timeout: float = 10.0,
        allow_rest_fallback: bool = True,
    ) -> Optional[float]:
        if wait_for_ready:
            self.wait_for_websocket_ready(symbol, interval, wait_timeout)

        self._ensure_ws(symbol, interval)
        latest = self._fetch_latest_kline_ws(symbol, interval)
        if latest and latest.get("close") is not None:
            return float(latest["close"])

        if allow_rest_fallback:
            df = self.fetch_futures_ohlcv(symbol, interval, limit=1)
            if not df.empty:
                return float(df.iloc[-1]["close"])
        return None

    def fetch_futures_open_at_cached(self, symbol: str, start_dt: dt.datetime) -> Optional[float]:
        key = f"{self._normalize_symbol(symbol)}:{start_dt.replace(second=0, microsecond=0).isoformat()}"
        if key in self._open_cache:
            return self._open_cache[key]
        value = self.fetch_futures_open_at(symbol, start_dt)
        if value is not None:
            self._open_cache[key] = value
        return value

    def get_ws_status(self, symbol: str, interval: Interval = "1h") -> dict[str, Any]:
        key = self._ws_key(symbol, interval)
        t = self._ws_threads.get(key)
        data = self._ws_cache.get(key)
        return {
            "key": key,
            "thread_alive": bool(t and t.is_alive()),
            "has_data": data is not None,
            "last_timestamp": data.get("timestamp") if data else None,
        }

    def _fetch_latest_kline_ws(self, symbol: str, interval: Interval) -> Optional[dict[str, Any]]:
        return self._ws_cache.get(self._ws_key(symbol, interval))

    @staticmethod
    def resample_ohlcv(df: pd.DataFrame, interval: Interval) -> pd.DataFrame:
        if df.empty:
            return df
        rule_map = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1H",
            "4h": "4H",
        }
        rule = rule_map[interval]
        return (
            df.resample(rule)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna(subset=["open", "high", "low", "close"])
        )

    def fetch_futures_ohlcv_with_ws(
        self, symbol: str, interval: Interval = "1h", limit: Optional[int] = None
    ) -> pd.DataFrame:
        df = self.fetch_futures_ohlcv(symbol, interval, limit=limit)
        self._ensure_ws(symbol, interval)
        latest = self._fetch_latest_kline_ws(symbol, interval)
        if latest and latest.get("timestamp"):
            ts = dt.datetime.fromtimestamp(int(latest["timestamp"]) / 1000, tz=dt.timezone.utc)
            row = {
                "open": float(latest.get("open", 0) or 0),
                "high": float(latest.get("high", 0) or 0),
                "low": float(latest.get("low", 0) or 0),
                "close": float(latest.get("close", 0) or 0),
                "volume": float(latest.get("volume", 0) or 0),
            }
            if df.empty or ts > df.index[-1]:
                df = pd.concat([df, pd.DataFrame({k: [v] for k, v in row.items()}, index=[ts])])
            else:
                for k, v in row.items():
                    df.loc[df.index[-1], k] = v
        return df.tail(limit or self.kline_limit)

    def fetch_spot_ohlcv_placeholder(
        self, symbol: str, start: dt.datetime, end: dt.datetime
    ) -> pd.DataFrame:
        symbol_n = self._normalize_symbol(symbol)
        c = self._get_client()
        try:
            r = c.get(
                f"{self.rest_host}/api/v3/klines",
                params={
                    "symbol": symbol_n,
                    "interval": "1m",
                    "startTime": int(start.timestamp() * 1000),
                    "endTime": int(end.timestamp() * 1000),
                    "limit": 1000,
                },
            )
            r.raise_for_status()
            data = r.json()
        except Exception:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        rows = []
        for x in data:
            try:
                ts = dt.datetime.fromtimestamp(int(x[0]) / 1000, tz=dt.timezone.utc)
                rows.append(
                    {
                        "timestamp": ts,
                        "open": float(x[1]),
                        "high": float(x[2]),
                        "low": float(x[3]),
                        "close": float(x[4]),
                        "volume": float(x[5]),
                    }
                )
            except Exception:
                continue
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return pd.DataFrame(rows).set_index("timestamp").sort_index()

    @staticmethod
    def resample(df_1m: pd.DataFrame, interval: Interval) -> pd.DataFrame:
        return OHLCVSource.resample_ohlcv(df_1m, interval)

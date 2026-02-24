from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional
import datetime as dt
from zoneinfo import ZoneInfo

import httpx

from pm_agent.utils.logging import get_logger
from pm_agent.polymarket.clob import (
    submit_market_order as _submit_market_order,
    get_account_balance as _get_account_balance,
    get_token_balances as _get_token_balances,
    get_clob_client as _get_clob_client,
    get_positions as _get_positions,
    redeem_condition as _redeem_condition,
    get_user_address as _get_user_address,
)


class PolymarketError(RuntimeError):
    pass


@dataclass
class PolymarketClient:
    host: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    api_passphrase: Optional[str] = None
    builder_api_key: Optional[str] = None
    builder_api_secret: Optional[str] = None
    builder_api_passphrase: Optional[str] = None
    chain_id: int = 137
    proxy: Optional[str] = None
    private_key: Optional[str] = None
    wallet_type: Optional[str] = None
    signature_type: Optional[str] = None
    funder: Optional[str] = None
    relayer_url: Optional[str] = None
    timeout_s: float = 20.0

    _logger = get_logger("pm_agent.polymarket")
    _cached_funder: Optional[str] = None

    def _web_client(self) -> httpx.Client:
        return httpx.Client(timeout=self.timeout_s, follow_redirects=True)

    def _make_settings(self, *, with_relayer: bool = False) -> SimpleNamespace:
        """Build a unified settings namespace for clob helpers."""
        actual_funder = self._cached_funder or self.funder
        ns = SimpleNamespace(
            private_key=self.private_key,
            wallet_type=self.wallet_type,
            signature_type=self.signature_type,
            funder=actual_funder,
            polymarket_host=self.host,
            polymarket_chain_id=self.chain_id,
            polymarket_api_key=self.api_key,
            polymarket_api_secret=self.api_secret,
            polymarket_api_passphrase=self.api_passphrase,
            polymarket_builder_api_key=self.builder_api_key,
            polymarket_builder_secret=self.builder_api_secret,
            polymarket_builder_passphrase=self.builder_api_passphrase,
            poly_builder_api_key=self.builder_api_key,
            poly_builder_secret=self.builder_api_secret,
            poly_builder_passphrase=self.builder_api_passphrase,
            polymarket_private_key=self.private_key,
            polymarket_wallet_type=self.wallet_type,
            polymarket_signature_type=self.signature_type,
            polymarket_funder=actual_funder,
        )
        if with_relayer:
            ns.polymarket_relayer_url = self.relayer_url
        return ns

    @staticmethod
    def _month_slug(dt_et: dt.datetime) -> str:
        return dt_et.strftime("%b").lower()

    @staticmethod
    def _hour_slug(dt_et: dt.datetime) -> str:
        return dt_et.strftime("%H")

    @staticmethod
    def _hourly_slug_for_time(prefix: str, dt_et: dt.datetime) -> str:
        # Example verified from real pages:
        # bitcoin-up-or-down-february-21-8am-et
        month = dt_et.strftime("%B").lower()
        day = dt_et.day
        hour_12 = dt_et.strftime("%I").lstrip("0") or "12"
        ampm = dt_et.strftime("%p").lower()
        return f"{prefix}-{month}-{day}-{hour_12}{ampm}-et"

    @staticmethod
    def _legacy_hourly_slug_for_time(prefix: str, dt_et: dt.datetime) -> str:
        # Backward-compat fallback for old naming patterns.
        mon = dt_et.strftime("%b").lower()
        day = dt_et.day
        year = dt_et.year
        hour = dt_et.strftime("%H")
        return f"{prefix}-{mon}-{day}-{year}-{hour}"

    def current_hour_slug(self, prefix: str = "bitcoin-up-or-down") -> str:
        et = dt.datetime.now(ZoneInfo("America/New_York"))
        event_start_et = et.replace(minute=0, second=0, microsecond=0)
        slug = self._hourly_slug_for_time(prefix, event_start_et)
        self._logger.info(
            "Generated slug: %s (ET time: %s)",
            slug,
            event_start_et.strftime("%Y-%m-%d %H:%M %Z"),
        )
        return slug

    def _hourly_slug_candidates(
        self, prefix: str = "bitcoin-up-or-down", look_around_hours: int = 2
    ) -> list[str]:
        et = dt.datetime.now(ZoneInfo("America/New_York")).replace(
            minute=0, second=0, microsecond=0
        )
        offsets = [0]
        for i in range(1, max(0, look_around_hours) + 1):
            offsets.extend([-i, i])

        out: list[str] = []
        for off in offsets:
            t = et + dt.timedelta(hours=off)
            out.append(self._hourly_slug_for_time(prefix, t))
        for off in offsets:
            t = et + dt.timedelta(hours=off)
            out.append(self._legacy_hourly_slug_for_time(prefix, t))

        dedup: list[str] = []
        seen: set[str] = set()
        for s in out:
            if s not in seen:
                seen.add(s)
                dedup.append(s)
        return dedup

    def _clob_client(self):
        # Auto-derive proxy funder for magic_link / proxy wallet types.
        wallet_type = str(self.wallet_type or "auto").strip().lower()
        should_derive_proxy_funder = wallet_type in {
            "auto",
            "magic",
            "magic-link",
            "magic link",
            "magic_link",
            "magiclink",
            "proxy",
            "polymarket_proxy",
        }
        if (
            not self._cached_funder
            and not self.funder
            and self.private_key
            and should_derive_proxy_funder
        ):
            try:
                from pm_agent.utils.web3_utils import get_proxy_address

                derived_proxy = get_proxy_address(self.private_key, self.chain_id)
                if derived_proxy:
                    self._cached_funder = derived_proxy
                    self._logger.info("✅ 自动查询到代理钱包地址: %s", derived_proxy)
            except Exception as e:
                self._logger.warning("⚠️ 无法自动查询代理钱包地址: %s", e)

        return _get_clob_client(self._make_settings())

    def _fetch_market_info_once(self, slug: str) -> dict[str, Any] | None:
        with self._web_client() as c:
            # Gamma API returns either list or object depending on endpoint shape.
            urls = [
                f"https://gamma-api.polymarket.com/events/slug/{slug}",
                f"https://gamma-api.polymarket.com/events?slug={slug}",
                f"https://gamma-api.polymarket.com/markets?slug={slug}",
            ]
            for url in urls:
                try:
                    r = c.get(url)
                    r.raise_for_status()
                    data = r.json()
                    if isinstance(data, list) and data:
                        return data[0]
                    if isinstance(data, dict) and data:
                        return data
                except Exception:
                    continue
        return None

    def fetch_market_info(self, slug: str) -> dict[str, Any]:
        info = self._fetch_market_info_once(slug)
        if info:
            return info

        raise PolymarketError(f"Unable to fetch market info for slug: {slug}")

    def _resolve_hourly_market(
        self, prefix: str = "bitcoin-up-or-down"
    ) -> tuple[str, dict[str, Any]]:
        candidates = self._hourly_slug_candidates(prefix, look_around_hours=2)
        for idx, slug in enumerate(candidates):
            info = self._fetch_market_info_once(slug)
            if info:
                if idx > 0:
                    self._logger.info("Hourly market fallback matched slug: %s", slug)
                elif idx == 0:
                    et_now = dt.datetime.now(ZoneInfo("America/New_York")).replace(
                        minute=0, second=0, microsecond=0
                    )
                    self._logger.info(
                        "Generated slug: %s (ET time: %s)",
                        slug,
                        et_now.strftime("%Y-%m-%d %H:%M %Z"),
                    )
                return slug, info
        raise PolymarketError(
            f"Unable to fetch market info for hourly candidates: {', '.join(candidates[:5])}..."
        )

    def _fetch_orderbooks(self, token_ids: list[str]) -> dict[str, Any]:
        client = self._clob_client()
        out: dict[str, Any] = {}
        for tid in token_ids:
            try:
                out[tid] = client.get_order_book(tid)
            except Exception as ex:
                self._logger.warning("get_order_book failed for %s: %s", tid, ex)
                out[tid] = None
        return out

    def get_hourly_market_prices(self, prefix: str = "bitcoin-up-or-down") -> dict[str, Any]:
        slug, info = self._resolve_hourly_market(prefix)

        def _extract_token_ids(payload: dict[str, Any]) -> list[str]:
            import json

            ids: list[str] = []

            # Markets endpoint shape.
            if isinstance(payload.get("clobTokenIds"), list):
                ids.extend(str(t) for t in payload["clobTokenIds"] if t)
            elif isinstance(payload.get("clobTokenIds"), str):
                try:
                    parsed = json.loads(payload["clobTokenIds"])
                    if isinstance(parsed, list):
                        ids.extend(str(t) for t in parsed if t)
                except Exception:
                    pass

            # Events endpoint shape.
            if not ids and isinstance(payload.get("markets"), list):
                for m in payload["markets"]:
                    if not isinstance(m, dict):
                        continue
                    tids = m.get("clobTokenIds")
                    if isinstance(tids, list):
                        ids.extend(str(t) for t in tids if t)
                    elif isinstance(tids, str):
                        try:
                            parsed = json.loads(tids)
                            if isinstance(parsed, list):
                                ids.extend(str(t) for t in parsed if t)
                        except Exception:
                            pass
                    if len(ids) >= 2:
                        break

            # Older/custom shapes.
            if not ids and isinstance(payload.get("tokens"), list):
                for t in payload["tokens"]:
                    if isinstance(t, dict) and t.get("token_id"):
                        ids.append(str(t["token_id"]))

            if not ids and isinstance(payload.get("outcomes"), list):
                # Some payloads store dict outcomes with token_id.
                for t in payload["outcomes"]:
                    if isinstance(t, dict) and t.get("token_id"):
                        ids.append(str(t["token_id"]))

            dedup: list[str] = []
            seen: set[str] = set()
            for tid in ids:
                if tid not in seen:
                    seen.add(tid)
                    dedup.append(tid)
            return dedup[:2]

        token_ids = _extract_token_ids(info)

        orderbooks = self._fetch_orderbooks(token_ids) if token_ids else {}

        def _best(ob: Any, side: str) -> Optional[float]:
            if isinstance(ob, dict):
                lv = ob.get(side)
            else:
                # py-clob-client returns OrderBookSummary objects with attrs: bids/asks
                lv = getattr(ob, side, None)
            if isinstance(lv, list) and lv:
                prices: list[float] = []
                for level in lv:
                    try:
                        if isinstance(level, dict):
                            px = float(level.get("price"))
                        else:
                            px = float(getattr(level, "price"))
                        if px >= 0:
                            prices.append(px)
                    except Exception:
                        continue
                if not prices:
                    return None
                side_l = str(side).lower()
                if side_l == "bids":
                    return max(prices)
                if side_l == "asks":
                    return min(prices)
                return prices[0]
            return None

        up_token_id = token_ids[0] if len(token_ids) > 0 else None
        down_token_id = token_ids[1] if len(token_ids) > 1 else None

        up_ob = orderbooks.get(up_token_id)
        down_ob = orderbooks.get(down_token_id)

        market_title = info.get("question") or info.get("title")
        market_end_time = info.get("endDate") or info.get("endDateIso") or info.get("end_time")
        market_condition_id = info.get("conditionId")
        market_neg_risk = bool(info.get("negRisk"))
        token_index_sets: dict[str, int] = {}
        if (not market_title or not market_end_time) and isinstance(info.get("markets"), list):
            for m in info["markets"]:
                if isinstance(m, dict):
                    market_title = market_title or m.get("question") or m.get("title")
                    market_end_time = market_end_time or m.get("endDate") or m.get("endDateIso")
                    market_condition_id = market_condition_id or m.get("conditionId")
                    market_neg_risk = market_neg_risk or bool(m.get("negRisk"))
                    outcomes = m.get("outcomes")
                    if isinstance(outcomes, str):
                        import json

                        try:
                            outcomes = json.loads(outcomes)
                        except Exception:
                            outcomes = None
                    token_ids_raw = m.get("clobTokenIds")
                    if isinstance(token_ids_raw, str):
                        import json

                        try:
                            token_ids_raw = json.loads(token_ids_raw)
                        except Exception:
                            token_ids_raw = None
                    if isinstance(token_ids_raw, list):
                        for idx, tid in enumerate(token_ids_raw):
                            if not tid:
                                continue
                            token_index_sets[str(tid)] = 1 << idx
                    if market_title and market_end_time:
                        break
        if not token_index_sets:
            for idx, tid in enumerate(token_ids):
                if tid:
                    token_index_sets[str(tid)] = 1 << idx

        return {
            "market_slug": slug,
            "market_title": market_title,
            "market_end_time": market_end_time,
            "condition_id": market_condition_id,
            "neg_risk": market_neg_risk,
            "token_index_sets": token_index_sets,
            "up_token_id": up_token_id,
            "down_token_id": down_token_id,
            "up_bid": _best(up_ob, "bids"),
            "up_ask": _best(up_ob, "asks"),
            "down_bid": _best(down_ob, "bids"),
            "down_ask": _best(down_ob, "asks"),
            "raw_market": info,
        }

    def submit_market_order(
        self,
        side: str,
        token_id: str,
        amount: float,
        order_type: Optional[Any] = None,
    ) -> dict:
        return _submit_market_order(
            self._make_settings(),
            side=side,
            token_id=token_id,
            amount=amount,
            order_type=order_type,
        )

    def get_account_balance(self) -> float:
        return _get_account_balance(self._make_settings())

    def get_token_balances(self, token_ids: list[str]) -> dict[str, float]:
        return _get_token_balances(self._make_settings(), token_ids)

    def get_user_address(self) -> str:
        return _get_user_address(self._make_settings())

    def get_positions(
        self,
        *,
        user_address: str | None = None,
        redeemable_only: bool = False,
        size_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        return _get_positions(
            self._make_settings(),
            user_address=user_address,
            redeemable_only=redeemable_only,
            size_threshold=size_threshold,
        )

    def get_redeemable_positions(
        self,
        *,
        user_address: str | None = None,
        min_value_usd: float = 0.0,
    ) -> list[dict[str, Any]]:
        rows = self.get_positions(
            user_address=user_address,
            redeemable_only=True,
            size_threshold=0.0,
        )
        if min_value_usd <= 0:
            return rows
        out: list[dict[str, Any]] = []
        for row in rows:
            try:
                cur = float(row.get("currentValue", 0.0))
            except Exception:
                cur = 0.0
            if cur >= float(min_value_usd):
                out.append(row)
        return out

    def redeem_condition(
        self,
        *,
        condition_id: str,
        index_sets: list[int] | None = None,
        use_relayer: bool = True,
        metadata: str | None = None,
    ) -> dict[str, Any]:
        return _redeem_condition(
            self._make_settings(with_relayer=True),
            condition_id=condition_id,
            index_sets=index_sets,
            use_relayer=use_relayer,
            metadata=metadata,
        )

    def get_account_info_placeholder(self) -> dict[str, Any]:
        return {"balance": self.get_account_balance()}

    def get_positions_placeholder(self) -> dict[str, Any]:
        return {}

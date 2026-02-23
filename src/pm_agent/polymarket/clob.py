from __future__ import annotations

from typing import Any, TYPE_CHECKING
import os

import httpx
from pm_agent.utils.logging import get_logger

if TYPE_CHECKING:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import (
        ApiCreds,
        MarketOrderArgs,
        OrderType,
        BalanceAllowanceParams,
        AssetType,
    )
    from py_clob_client.order_builder.constants import BUY, SELL

logger = get_logger("pm_agent.clob")
_client_cache: dict[str, Any] = {}

_PROXY_WALLET_TYPES = {
    "magic",
    "magic-link",
    "magic link",
    "magic_link",
    "magiclink",
    "proxy",
    "polymarket_proxy",
    "safe",
    "gnosis",
    "gnosis_safe",
}


def _require_clob():
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds, MarketOrderArgs, OrderType
    from py_clob_client.order_builder.constants import BUY, SELL

    return ClobClient, ApiCreds, MarketOrderArgs, OrderType, BUY, SELL


def _require_balance_types():
    from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

    return BalanceAllowanceParams, AssetType


def _require_exception_types():
    from py_clob_client.exceptions import PolyApiException

    return PolyApiException


def _require_signature_constants():
    # py-order-utils constants used by py-clob-client builder
    from py_order_utils.model import EOA, POLY_PROXY, POLY_GNOSIS_SAFE

    return EOA, POLY_PROXY, POLY_GNOSIS_SAFE


def _get_setting(settings: Any, name: str, alt: str | None = None):
    if hasattr(settings, name):
        return getattr(settings, name)
    if alt and hasattr(settings, alt):
        return getattr(settings, alt)
    return None


def _resolve_signature_type(settings: Any) -> int | None:
    wallet_type_raw = _get_setting(settings, "wallet_type", "polymarket_wallet_type")
    wallet_type = str(wallet_type_raw or "auto").strip().lower()

    sig_raw = _get_setting(settings, "signature_type", "polymarket_signature_type")
    if sig_raw is not None and str(sig_raw).strip() != "":
        try:
            sig_from_env = int(sig_raw)
        except Exception:
            sig_from_env = None
    else:
        sig_from_env = None

    eoa, poly_proxy, poly_safe = _require_signature_constants()
    by_wallet_type = {
        "eoa": eoa,
        "magic": poly_proxy,
        "magic-link": poly_proxy,
        "magic link": poly_proxy,
        "magic_link": poly_proxy,
        "magiclink": poly_proxy,
        "proxy": poly_proxy,
        "polymarket_proxy": poly_proxy,
        "safe": poly_safe,
        "gnosis_safe": poly_safe,
        "gnosis": poly_safe,
    }

    if wallet_type in by_wallet_type:
        return by_wallet_type[wallet_type]
    if sig_from_env is not None:
        return sig_from_env
    return None


def _signature_type_for_balance(settings: Any) -> int:
    sig = _resolve_signature_type(settings)
    if sig is not None:
        return int(sig)
    # Default remains compatible with previous runtime behavior.
    raw = _get_setting(settings, "signature_type", "polymarket_signature_type")
    try:
        return int(raw)
    except Exception:
        return 2


def get_clob_client(settings: Any):
    """
    获取或创建 CLOB 客户端实例。
    支持：
    - `POLYMARKET_WALLET_TYPE=eoa`
    - `POLYMARKET_WALLET_TYPE=magic_link` (映射为 proxy 签名类型)
    - `POLYMARKET_WALLET_TYPE=proxy`
    - `POLYMARKET_WALLET_TYPE=safe`
    """
    private_key = _get_setting(settings, "private_key", "polymarket_private_key")
    if not private_key:
        raise RuntimeError("Missing POLYMARKET_PRIVATE_KEY, cannot trade")

    funder = _get_setting(settings, "funder", "polymarket_funder")
    wallet_type = str(_get_setting(settings, "wallet_type", "polymarket_wallet_type") or "auto")
    signature_type = _resolve_signature_type(settings)

    host = _get_setting(settings, "polymarket_host", "host") or "https://clob.polymarket.com"
    chain_id = int(_get_setting(settings, "polymarket_chain_id", "chain_id") or 137)

    cache_key = (
        f"{host}|{chain_id}|{str(signature_type)}|{wallet_type.lower()}|"
        f"{(funder or 'auto')}|{str(private_key)[:10]}"
    )
    if cache_key in _client_cache:
        return _client_cache[cache_key]

    ClobClient, ApiCreds, _, _, _, _ = _require_clob()

    logger.info("🔧 创建 CLOB 客户端 (funder: %s)", funder or "None")
    client = ClobClient(
        host,
        key=str(private_key).strip(),
        chain_id=int(chain_id),
        signature_type=signature_type,
        funder=(str(funder).strip() if funder else None),
    )

    # EOA wallet does not require explicit funder: order builder falls back to signer address.
    if not funder and signature_type not in (None, 0):
        logger.warning("⚠️  FUNDER 地址未配置，ClobClient 可能无法正常工作")
        logger.warning("   请在 .env 中设置 POLYMARKET_FUNDER 或确保自动查询成功")

    api_key = _get_setting(settings, "api_key", "polymarket_api_key")
    api_secret = _get_setting(settings, "api_secret", "polymarket_api_secret")
    api_passphrase = _get_setting(settings, "api_passphrase", "polymarket_api_passphrase")

    if api_key and api_secret and api_passphrase:
        logger.info("✅ 使用环境变量中的 API 凭证")
        api_credentials = ApiCreds(
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=api_passphrase,
        )
        client.set_api_creds(api_credentials)
    else:
        logger.info("🔑 未找到 API 凭证，正在从私钥自动派生...")
        api_credentials = client.create_or_derive_api_creds()
        client.set_api_creds(api_credentials)
        logger.info("✅ API 凭证自动生成成功")
        logger.info(
            "   API Key: %s...%s",
            api_credentials.api_key[:10],
            api_credentials.api_key[-10:],
        )
        logger.info("   💡 提示：这些凭证会自动保存，无需手动配置到 .env 文件")

    _client_cache[cache_key] = client
    return client


def get_account_balance(settings: Any) -> float:
    """获取账户 USDC 余额（单位：USDC）。"""
    try:
        BalanceAllowanceParams, AssetType = _require_balance_types()
        client = get_clob_client(settings)

        params = BalanceAllowanceParams(
            asset_type=AssetType.COLLATERAL,
            signature_type=_signature_type_for_balance(settings),
        )
        result = client.get_balance_allowance(params)

        if isinstance(result, dict):
            balance_raw = result.get("balance", "0")
            balance_wei = float(balance_raw)
            return balance_wei / 1_000_000

        logger.warning("Unexpected balance response: %s", result)
        return 0.0
    except Exception as e:
        logger.error("Balance fetch failed: %s", e)
        return 0.0


def get_token_balances(settings: Any, token_ids: list[str]) -> dict[str, float]:
    """
    获取指定代币的余额（股份数量）。
    返回值单位为 shares（把链上 micro 单位除以 1e6）。
    """
    import logging
    import time

    dbg = logging.getLogger("pm_agent")
    balances: dict[str, float] = {}
    if not token_ids:
        return balances

    try:
        BalanceAllowanceParams, AssetType = _require_balance_types()
        client = get_clob_client(settings)

        for token_id in token_ids:
            max_retries = 3
            retry_delay = 0.5
            for attempt in range(max_retries):
                try:
                    params = BalanceAllowanceParams(
                        asset_type=AssetType.CONDITIONAL,
                        token_id=token_id,
                        signature_type=_signature_type_for_balance(settings),
                    )
                    result = client.get_balance_allowance(params)
                    dbg.debug(
                        "[get_token_balances] token=%s..., result=%s",
                        token_id[:8],
                        result,
                    )
                    if isinstance(result, dict):
                        balance_raw = result.get("balance", "0")
                        balance_shares = float(balance_raw) / 1_000_000
                        balances[token_id] = balance_shares
                        dbg.debug(
                            "[get_token_balances] token=%s..., balance_raw=%s, balance_shares=%.6f",
                            token_id[:8],
                            balance_raw,
                            balance_shares,
                        )
                    else:
                        balances[token_id] = 0.0
                        dbg.warning(
                            "[get_token_balances] result不是dict: token=%s..., result_type=%s, result=%s",
                            token_id[:8],
                            type(result),
                            result,
                        )
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        dbg.debug(
                            "Retry %s/%s for token %s... after error: %s",
                            attempt + 1,
                            max_retries,
                            token_id[:16],
                            e,
                        )
                    else:
                        raise

    except Exception as e:
        import traceback

        dbg.error("Token balance fetch failed: %s", e)
        dbg.error("  Token IDs: %s", token_ids)
        dbg.error(
            "  Settings debug: private_key=%s, funder=%s",
            bool(
                _get_setting(settings, "private_key", None)
                or _get_setting(settings, "polymarket_private_key", None)
            ),
            _get_setting(settings, "funder", None)
            or _get_setting(settings, "polymarket_funder", None),
        )
        dbg.error("  Exception type: %s", type(e).__name__)
        dbg.error("  Exception details: %s", str(e))
        try:
            PolyApiException = _require_exception_types()
            if PolyApiException and isinstance(e, PolyApiException):
                dbg.error("  Status code: %s", getattr(e, "status_code", "N/A"))
                dbg.error("  Error message: %s", getattr(e, "error_msg", "N/A"))
                if hasattr(e, "__cause__") and e.__cause__:
                    dbg.error("  Root cause: %s: %s", type(e.__cause__).__name__, e.__cause__)
        except Exception:
            pass
        dbg.debug("  Full traceback:\n%s", traceback.format_exc())

    return balances


def _private_key_with_prefix(settings: Any) -> str:
    private_key = _get_setting(settings, "private_key", "polymarket_private_key")
    if not private_key:
        raise RuntimeError("Missing private key")
    private_key = str(private_key).strip()
    return private_key if private_key.startswith("0x") else f"0x{private_key}"


def _signer_address(settings: Any) -> str:
    from eth_account import Account

    return Account.from_key(_private_key_with_prefix(settings)).address


def get_user_address(settings: Any) -> str:
    funder = _get_setting(settings, "funder", "polymarket_funder")
    if funder:
        return str(funder).strip()
    return _signer_address(settings)


def get_positions(
    settings: Any,
    *,
    user_address: str | None = None,
    redeemable_only: bool = False,
    size_threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Fetch positions from Polymarket data-api.
    """
    user = (user_address or get_user_address(settings) or "").strip().lower()
    if not user:
        return []

    params: dict[str, str] = {"user": user}
    if redeemable_only:
        params["redeemable"] = "true"
    if size_threshold > 0:
        params["sizeThreshold"] = str(size_threshold)
    else:
        params["sizeThreshold"] = "0"

    with httpx.Client(timeout=20.0, follow_redirects=True) as c:
        resp = c.get("https://data-api.polymarket.com/positions", params=params)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    return []


def _is_proxy_wallet(settings: Any) -> bool:
    wallet_type = str(_get_setting(settings, "wallet_type", "polymarket_wallet_type") or "auto").strip().lower()
    funder = str(_get_setting(settings, "funder", "polymarket_funder") or "").strip().lower()
    signer = _signer_address(settings).lower()
    return wallet_type in _PROXY_WALLET_TYPES or (funder and funder != signer)


def _condition_bytes(condition_id: str) -> bytes:
    cond = (condition_id or "").strip()
    if not cond:
        raise ValueError("condition_id is required")
    if cond.startswith("0x"):
        cond = cond[2:]
    raw = bytes.fromhex(cond)
    if len(raw) != 32:
        raise ValueError(f"condition_id must be 32 bytes, got {len(raw)}")
    return raw


def _redeem_direct_onchain(
    settings: Any,
    *,
    condition_id: str,
    index_sets: list[int],
) -> dict[str, Any]:
    try:
        from web3 import Web3
    except Exception as e:
        raise RuntimeError("web3 is required for EOA auto-redeem") from e

    from py_clob_client.config import get_contract_config

    chain_id = int(_get_setting(settings, "polymarket_chain_id", "chain_id") or 137)
    cfg = get_contract_config(chain_id, False)
    private_key = _private_key_with_prefix(settings)
    signer = _signer_address(settings)

    rpc_candidates = [
        _get_setting(settings, "polygon_rpc_url"),
        _get_setting(settings, "polymarket_rpc_url"),
        os.getenv("POLYGON_RPC_URL"),
        os.getenv("POLYMARKET_RPC_URL"),
        "https://polygon.llamarpc.com",
        "https://rpc-mainnet.matic.network",
        "https://matic-mainnet.chainstacklabs.com",
    ]

    w3 = None
    for rpc in rpc_candidates:
        if not rpc:
            continue
        try:
            _w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 8}))
            if _w3.is_connected():
                w3 = _w3
                break
        except Exception:
            continue

    if w3 is None:
        raise RuntimeError("Unable to connect to Polygon RPC for EOA redeem")

    redeem_abi = [
        {
            "inputs": [
                {"internalType": "address", "name": "collateralToken", "type": "address"},
                {"internalType": "bytes32", "name": "parentCollectionId", "type": "bytes32"},
                {"internalType": "bytes32", "name": "conditionId", "type": "bytes32"},
                {"internalType": "uint256[]", "name": "indexSets", "type": "uint256[]"},
            ],
            "name": "redeemPositions",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function",
        }
    ]
    ctf = w3.eth.contract(address=w3.to_checksum_address(cfg.conditional_tokens), abi=redeem_abi)

    cond = _condition_bytes(condition_id)
    call = ctf.functions.redeemPositions(
        w3.to_checksum_address(cfg.collateral),
        bytes(32),
        cond,
        [int(x) for x in index_sets if int(x) > 0],
    )

    tx: dict[str, Any] = {
        "from": w3.to_checksum_address(signer),
        "nonce": w3.eth.get_transaction_count(w3.to_checksum_address(signer), "pending"),
        "chainId": chain_id,
        "gasPrice": int(w3.eth.gas_price),
    }
    try:
        est = int(call.estimate_gas({"from": w3.to_checksum_address(signer)}))
        tx["gas"] = max(220_000, int(est * 1.2))
    except Exception:
        tx["gas"] = 350_000

    built = call.build_transaction(tx)
    signed = w3.eth.account.sign_transaction(built, private_key=private_key)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
    status = int(receipt.get("status", 0))
    if status != 1:
        raise RuntimeError(f"EOA redeem reverted: tx={w3.to_hex(tx_hash)}")

    return {
        "tx_hash": w3.to_hex(tx_hash),
        "status": status,
        "mode": "eoa",
        "gas_used": int(receipt.get("gasUsed", 0)),
    }


def _redeem_via_relayer(
    settings: Any,
    *,
    condition_id: str,
    index_sets: list[int],
    metadata: str | None = None,
) -> dict[str, Any]:
    from eth_abi import encode
    from py_builder_signing_sdk.config import BuilderConfig
    from py_builder_signing_sdk.sdk_types import BuilderApiKeyCreds
    from py_builder_relayer_client.client import RelayClient
    from py_builder_relayer_client.models import OperationType, SafeTransaction
    from py_clob_client.config import get_contract_config

    chain_id = int(_get_setting(settings, "polymarket_chain_id", "chain_id") or 137)
    cfg = get_contract_config(chain_id, False)
    relayer_url = (
        str(_get_setting(settings, "polymarket_relayer_url") or "").strip()
        or os.getenv("POLYMARKET_RELAYER_URL")
        or "https://relayer-v2.polymarket.com/"
    )

    client = get_clob_client(settings)
    creds = getattr(client, "creds", None)
    if not creds:
        raise RuntimeError("CLOB API creds missing, cannot authenticate relayer")

    builder_cfg = BuilderConfig(
        local_builder_creds=BuilderApiKeyCreds(
            key=creds.api_key,
            secret=creds.api_secret,
            passphrase=creds.api_passphrase,
        )
    )
    relay = RelayClient(
        relayer_url=relayer_url,
        chain_id=chain_id,
        private_key=_private_key_with_prefix(settings),
        builder_config=builder_cfg,
    )

    expected_safe = relay.get_expected_safe()
    configured_funder = str(_get_setting(settings, "funder", "polymarket_funder") or "").strip()
    if configured_funder and expected_safe.lower() != configured_funder.lower():
        logger.warning(
            "Configured funder (%s) differs from expected safe (%s); relayer uses expected safe.",
            configured_funder,
            expected_safe,
        )
    if not relay.get_deployed(expected_safe):
        raise RuntimeError(f"Safe not deployed for signer: {expected_safe}")

    cond = _condition_bytes(condition_id)
    selector = bytes.fromhex("01b7037c")
    calldata = selector + encode(
        ["address", "bytes32", "bytes32", "uint256[]"],
        [cfg.collateral, bytes(32), cond, [int(x) for x in index_sets if int(x) > 0]],
    )

    tx = SafeTransaction(
        to=cfg.conditional_tokens,
        operation=OperationType.Call,
        data="0x" + calldata.hex(),
        value="0",
    )
    response = relay.execute([tx], metadata=metadata or f"Auto redeem {condition_id[:10]}")
    mined = response.wait()
    if mined is None:
        raise RuntimeError(f"Relayer redeem failed or timed out for {condition_id}")
    return {
        "tx_hash": mined.get("transactionHash") or response.hash,
        "status": mined.get("state"),
        "mode": "relayer",
        "safe": expected_safe,
    }


def redeem_condition(
    settings: Any,
    *,
    condition_id: str,
    index_sets: list[int] | None = None,
    use_relayer: bool = True,
    metadata: str | None = None,
) -> dict[str, Any]:
    """
    Redeem a resolved condition.
    - Proxy/safe wallets: use Polymarket relayer.
    - EOA wallets: send direct on-chain redeem tx.
    """
    partition = sorted({int(x) for x in (index_sets or [1, 2]) if int(x) > 0})
    if not partition:
        raise ValueError("index_sets cannot be empty")

    wants_proxy_path = _is_proxy_wallet(settings)
    if wants_proxy_path:
        if not use_relayer:
            raise RuntimeError("Proxy wallet redeem requires relayer")
        return _redeem_via_relayer(
            settings,
            condition_id=condition_id,
            index_sets=partition,
            metadata=metadata,
        )

    if use_relayer:
        try:
            return _redeem_via_relayer(
                settings,
                condition_id=condition_id,
                index_sets=partition,
                metadata=metadata,
            )
        except Exception:
            logger.info("Relayer redeem unavailable for EOA, falling back to on-chain redeem")
    return _redeem_direct_onchain(settings, condition_id=condition_id, index_sets=partition)


def submit_market_order(
    settings: Any,
    *,
    side: str,
    token_id: str,
    amount: float,
    order_type,
):
    """
    Submit market order.
    BUY: amount is USDC notional.
    SELL: amount is shares.
    """
    if amount <= 0:
        raise ValueError("amount must be > 0")
    if not token_id:
        raise ValueError("token_id is required")

    side_upper = side.upper()
    _, _, MarketOrderArgs, OrderType, BUY, SELL = _require_clob()
    if side_upper not in {BUY, SELL}:
        raise ValueError(f"side must be BUY or SELL, got: {side_upper}")

    client = get_clob_client(settings)

    if side_upper == BUY:
        order_args = MarketOrderArgs(token_id=token_id, amount=float(amount))
    else:
        order_args = MarketOrderArgs(token_id=token_id, size=float(amount))

    ot = order_type or OrderType.GTC
    try:
        signed = client.create_market_order(order_args, side=side_upper)
        return client.post_order(signed, ot)
    except Exception:
        # Compatibility fallback for older py-clob-client variants.
        signed = client.create_market_order(order_args)
        return client.post_order(signed, ot)

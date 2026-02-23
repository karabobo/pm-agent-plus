from __future__ import annotations

from eth_account import Account
from web3 import Web3

from pm_agent.utils.logging import get_logger

logger = get_logger("pm_agent.utils.web3")


def get_proxy_address(private_key: str, chain_id: int = 137):
    """
    Derive a Polymarket proxy address through the known proxy-factory contract.
    Returns None on failure.
    """
    try:
        pk = private_key if private_key.startswith("0x") else "0x" + private_key
        account = Account.from_key(pk)
        eoa_address = account.address

        rpcs = [
            "https://polygon-rpc.com",
            "https://rpc-mainnet.matic.network",
            "https://matic-mainnet.chainstacklabs.com",
            "https://polygon.llamarpc.com",
        ]

        w3 = None
        for rpc in rpcs:
            try:
                provider = Web3.HTTPProvider(rpc, request_kwargs={"timeout": 5})
                temp_w3 = Web3(provider)
                if temp_w3.is_connected():
                    w3 = temp_w3
                    break
            except Exception:
                continue

        if not w3:
            logger.warning("Could not connect to any Polygon RPC to query proxy address")
            return None

        factory_address = w3.to_checksum_address("0xaacFeEa03eb1561C4e67d661e40682Bd20E3541b")
        abi = [
            {
                "inputs": [{"internalType": "address", "name": "_owner", "type": "address"}],
                "name": "computeProxyAddress",
                "outputs": [{"internalType": "address", "name": "", "type": "address"}],
                "stateMutability": "view",
                "type": "function",
            }
        ]
        contract = w3.eth.contract(address=factory_address, abi=abi)
        proxy_address = contract.functions.computeProxyAddress(eoa_address).call()
        if proxy_address:
            return proxy_address
        return None
    except Exception as e:
        logger.error("Failed to derive proxy address: %s", e)
        return None

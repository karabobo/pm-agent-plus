from __future__ import annotations

import logging
from pm_agent.utils import chalk

_current_ai_model = "N/A"


def set_ai_model_info(provider: str, model: str) -> None:
    global _current_ai_model
    provider = (provider or "unknown").upper()
    model = model or "unknown"
    _current_ai_model = f"{provider}:{model}"


def get_ai_model_info() -> str:
    return _current_ai_model


class ChalkFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG: chalk.cyan,
        logging.INFO: chalk.bright_white,
        logging.WARNING: chalk.yellow,
        logging.ERROR: chalk.red,
        logging.CRITICAL: chalk.red,
    }

    def format(self, record: logging.LogRecord) -> str:
        record.ai_model = get_ai_model_info()
        msg = super().format(record)
        color = self.LEVEL_COLORS.get(record.levelno)
        return color(msg) if color else msg


def get_logger(name: str = "pm_agent") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "%(asctime)s | %(levelname)s | %(ai_model)s | %(message)s"
    handler.setFormatter(ChalkFormatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def format_market_info(slug: str, question: str, time_to_end: str | None) -> str:
    lines = [
        chalk.bold(chalk.cyan(f"Market: {slug}")),
        chalk.dim(f"Question: {question}"),
    ]
    if time_to_end:
        lines.append(chalk.dim(f"Time To End: {time_to_end}"))
    return "\n".join(lines)


def format_prices(up_bid: float, up_ask: float, down_bid: float, down_ask: float) -> str:
    return (
        f"{chalk.green('UP')}: bid={chalk.bold(str(up_bid))} ask={up_ask}  |  "
        f"{chalk.red('DOWN')}: bid={chalk.bold(str(down_bid))} ask={down_ask}"
    )


def format_positions(
    up_shares: float,
    up_cost: float,
    down_shares: float,
    down_cost: float,
    cash: float,
) -> str:
    lines = [chalk.bold(chalk.cyan("Positions & Balance:"))]
    lines.append(
        chalk.green(f"UP: {up_shares:.2f} shares @ ${up_cost:.4f}")
        if up_shares >= 1
        else chalk.dim("UP: -")
    )
    lines.append(
        chalk.red(f"DOWN: {down_shares:.2f} shares @ ${down_cost:.4f}")
        if down_shares >= 1
        else chalk.dim("DOWN: -")
    )
    lines.append(chalk.cyan(f"Cash: ${cash:.2f} USDC"))
    return "\n".join(lines)


def format_session_stats(stats: dict | None) -> str:
    if not stats or stats.get("trade_count", 0) == 0:
        return chalk.dim("Session Stats: No trades yet")

    pnl = float(stats.get("realized_pnl", 0) or 0)
    trade_count = int(stats.get("trade_count", 0) or 0)
    pnl_color = chalk.green if pnl >= 0 else chalk.red
    pnl_sign = "+" if pnl >= 0 else ""
    pnl_text = pnl_color(f"{pnl_sign}${pnl:.2f}")
    return chalk.bold(chalk.cyan("Session Stats:")) + f"\nTrades: {trade_count} | P&L: {pnl_text}"

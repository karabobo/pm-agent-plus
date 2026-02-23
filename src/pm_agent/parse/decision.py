from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL | re.IGNORECASE)
_DECISION_RE = re.compile(r"<decision>(.*?)</decision>", re.DOTALL | re.IGNORECASE)


def _clean(s: str) -> str:
    return s.replace("\x00", "").strip()


@dataclass
class ParsedDecision:
    reasoning: str
    decision: dict[str, Any]
    raw: str
    prompt: str = ""


class DecisionParseError(RuntimeError):
    pass


def parse_full_decision_response(raw: str) -> ParsedDecision:
    raw2 = _clean(raw)
    reasoning_match = _REASONING_RE.search(raw2)
    decision_match = _DECISION_RE.search(raw2)

    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    decision_text = decision_match.group(1).strip() if decision_match else ""

    if not decision_text:
        m = re.search(r"\{\s*\"actions\"\s*:\s*\[.*\]\s*\}", raw2, re.DOTALL)
        if m:
            decision_text = m.group(0)

    decision_text = _clean(decision_text)
    decision_text = re.sub(r",\s*([}\]])", r"\1", decision_text)

    try:
        decision = json.loads(decision_text) if decision_text else {"actions": []}
    except json.JSONDecodeError as ex:
        raise DecisionParseError(f"Invalid decision JSON: {ex}") from ex

    if not isinstance(decision, dict):
        raise DecisionParseError("Decision payload must be an object")

    actions = decision.get("actions")
    if actions is None:
        decision["actions"] = []
    elif not isinstance(actions, list):
        raise DecisionParseError("decision.actions must be a list")

    return ParsedDecision(reasoning=reasoning, decision=decision, raw=raw2)

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any

_REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL | re.IGNORECASE)
_DECISION_RE = re.compile(r"<decision>(.*?)</decision>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_UNQUOTED_KEY_RE = re.compile(r"(^|[{,]\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s*:)", re.MULTILINE)


def _clean(s: str) -> str:
    return s.replace("\x00", "").strip()


def _strip_code_fence(text: str) -> str:
    m = _CODE_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _extract_first_braced_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escape = False
    quote = ""
    for i, ch in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_string = False
            continue

        if ch in ('"', "'"):
            in_string = True
            quote = ch
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return ""


def _normalize_json_like(text: str) -> str:
    t = _strip_code_fence(text)
    t = t.replace("“", '"').replace("”", '"').replace("’", "'")
    t = re.sub(r",\s*([}\]])", r"\1", t)
    return t.strip()


def _quote_unquoted_keys(text: str) -> str:
    return _UNQUOTED_KEY_RE.sub(r'\1"\2"\3', text)


def _load_json_like(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    candidates: list[str] = []
    seen: set[str] = set()

    def _add(candidate: str) -> None:
        c = (candidate or "").strip()
        if c and c not in seen:
            seen.add(c)
            candidates.append(c)

    normalized = _normalize_json_like(text)
    _add(normalized)
    _add(_extract_first_braced_object(normalized))

    for candidate in list(candidates):
        _add(_quote_unquoted_keys(candidate))

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj

    for candidate in candidates:
        py_like = re.sub(r"\btrue\b", "True", candidate, flags=re.IGNORECASE)
        py_like = re.sub(r"\bfalse\b", "False", py_like, flags=re.IGNORECASE)
        py_like = re.sub(r"\bnull\b", "None", py_like, flags=re.IGNORECASE)
        try:
            obj = ast.literal_eval(py_like)
        except (SyntaxError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj

    return None


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
        if "actions" in raw2.lower():
            decision_text = _extract_first_braced_object(raw2)

    decision_text = _clean(decision_text)
    decision = _load_json_like(decision_text) if decision_text else {"actions": []}
    if decision is None:
        snippet = decision_text[:200].replace("\n", "\\n")
        raise DecisionParseError(f"Invalid decision JSON: unable to parse payload: {snippet}")

    if not isinstance(decision, dict):
        raise DecisionParseError("Decision payload must be an object")

    actions = decision.get("actions")
    if actions is None:
        decision["actions"] = []
    elif not isinstance(actions, list):
        raise DecisionParseError("decision.actions must be a list")

    # If model returns prose instead of structured action JSON, keep the
    # pipeline alive with an explicit WAIT action so UI/stats still update.
    if not decision.get("actions") and raw2:
        fallback_reason = (reasoning or raw2).strip()
        decision["actions"] = [
            {
                "type": "wait",
                "market": "",
                "token_id": "",
                "side": "",
                "price": 0,
                "size": 0,
                "amount": 0,
                "time_in_force": "GTC",
                "risk": {"max_slippage_bps": 30, "max_notional_usd": 0},
                "rationale": fallback_reason[:1000],
            }
        ]
        if not reasoning:
            reasoning = fallback_reason[:1200]

    return ParsedDecision(reasoning=reasoning, decision=decision, raw=raw2)

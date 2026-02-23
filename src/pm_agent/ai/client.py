from __future__ import annotations

from dataclasses import dataclass
import json
import requests
import httpx
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from pm_agent.utils.logging import get_logger


@dataclass
class AIClient:
    api_key: str
    model: str
    base_url: str | None = None
    timeout_s: float = 120.0
    provider: str = "openai"

    def _client(self) -> OpenAI:
        if self.base_url:
            return OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout_s)
        return OpenAI(api_key=self.api_key, timeout=self.timeout_s)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError, Exception)),
        reraise=True,
    )
    def call_with_messages(self, system_prompt: str, user_prompt: str) -> str:
        def _ensure_chat_url(url: str) -> str:
            trimmed = url.rstrip("/")
            if trimmed.endswith("chat/completions"):
                return trimmed
            return f"{trimmed}/chat/completions"

        logger = get_logger("pm_agent.ai")
        provider = (self.provider or "openai").lower()

        if provider in {"openai", "deepseek", "qwen"} and self.base_url:
            url = _ensure_chat_url(self.base_url)
            body = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
            }
            response = requests.request(
                "POST",
                url,
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json; charset=utf-8",
                },
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            payload = response.json()
            try:
                return str(payload["choices"][0]["message"]["content"])
            except (KeyError, ValueError, TypeError):
                logger.warning("Unexpected openai/deepseek/qwen payload: %s", payload)
                return str(payload)

        if provider == "gemini" and self.base_url:
            body = {
                "contents": [
                    {
                        "parts": [
                            {"text": system_prompt},
                            {"text": user_prompt},
                        ]
                    }
                ]
            }
            response = requests.request(
                "POST",
                self.base_url,
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Content-Type": "application/json; charset=utf-8",
                    "x-goog-api-key": self.api_key,
                    "Authorization": f"Bearer {self.api_key}",
                },
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            payload = response.json()
            try:
                return str(payload["candidates"][0]["content"]["parts"][0]["text"])
            except (KeyError, ValueError, TypeError):
                logger.warning("Unexpected gemini payload: %s", payload)
                return str(payload)

        if provider == "claude" and self.base_url:
            body = {
                "model": self.model,
                "max_tokens": 2048,
                "messages": [
                    {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"},
                ],
            }
            response = requests.request(
                "POST",
                self.base_url,
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json; charset=utf-8",
                },
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            payload = response.json()
            try:
                content = payload.get("content")
                if isinstance(content, list) and content:
                    first = content[0]
                    if isinstance(first, dict):
                        return str(first.get("text", ""))
                return str(payload["choices"][0]["message"]["content"])
            except (KeyError, ValueError, TypeError):
                logger.warning("Unexpected claude payload: %s", payload)
                return str(payload)

        client = self._client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return str(resp.choices[0].message.content)

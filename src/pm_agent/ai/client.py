from __future__ import annotations

from dataclasses import dataclass
import json
import re
import requests
import httpx
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from pm_agent.utils.logging import get_logger


def _is_retryable_ai_error(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True
    if isinstance(exc, requests.HTTPError):
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        return status_code is None or status_code == 429 or status_code >= 500
    if isinstance(exc, RuntimeError):
        message = str(exc).lower()
        return "no assistant output extracted from /responses sse payload" in message
    return False


@dataclass
class AIClient:
    api_key: str
    model: str
    base_url: str | None = None
    timeout_s: float = 120.0
    provider: str = "openai"
    reasoning_effort: str | None = None
    force_stream: bool = False

    def _client(self) -> OpenAI:
        if self.base_url:
            return OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout_s)
        return OpenAI(api_key=self.api_key, timeout=self.timeout_s)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception(_is_retryable_ai_error),
        reraise=True,
    )
    def call_with_messages(self, system_prompt: str, user_prompt: str) -> str:
        api_key = str(self.api_key or "").strip()

        def _json_headers(
            *,
            bearer: bool = False,
            goog: bool = False,
            x_api_key: bool = False,
        ) -> dict[str, str]:
            headers = {"Content-Type": "application/json; charset=utf-8"}
            if bearer and api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            if goog and api_key:
                headers["x-goog-api-key"] = api_key
            if x_api_key and api_key:
                headers["x-api-key"] = api_key
            return headers

        def _ensure_chat_url(url: str) -> str:
            trimmed = url.rstrip("/")
            if trimmed.endswith("chat/completions"):
                return trimmed
            return f"{trimmed}/chat/completions"

        def _ensure_responses_url(url: str) -> str:
            trimmed = url.rstrip("/")
            if trimmed.endswith("responses"):
                return trimmed
            return f"{trimmed}/responses"

        def _model_prefers_responses(model_name: str) -> bool:
            model = (model_name or "").strip().lower()
            return model.startswith("gpt-5") or "codex" in model

        def _is_gemini_openai_compat_url(url: str) -> bool:
            lowered = (url or "").strip().lower()
            if not lowered:
                return False
            # Native Gemini REST usually points to generativelanguage/googleapis
            # and uses :generateContent payloads.
            if "generativelanguage.googleapis.com" in lowered:
                return False
            if "googleapis.com" in lowered and "generatecontent" in lowered:
                return False
            # Local/third-party Gemini gateways are often OpenAI-compatible (/v1).
            return "/v1" in lowered or lowered.endswith("chat/completions")

        def _extract_responses_text(payload: dict) -> str:
            output_text = payload.get("output_text")
            if isinstance(output_text, str) and output_text.strip():
                return output_text

            chunks: list[str] = []
            output = payload.get("output")
            if isinstance(output, list):
                for item in output:
                    if not isinstance(item, dict):
                        continue
                    content = item.get("content")
                    if not isinstance(content, list):
                        continue
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        text = part.get("text")
                        if isinstance(text, str):
                            chunks.append(text)
            if chunks:
                return "".join(chunks)
            return ""

        def _extract_text_from_responses_payload(raw_text: str) -> str:
            text = (raw_text or "").strip()
            if not text:
                return ""

            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                return ""

            if not isinstance(payload, dict):
                return ""

            parsed = _extract_responses_text(payload)
            if parsed:
                return parsed

            response_obj = payload.get("response")
            if isinstance(response_obj, dict):
                parsed = _extract_responses_text(response_obj)
                if parsed:
                    return parsed

            try:
                message = payload["choices"][0]["message"]["content"]
            except (KeyError, TypeError, IndexError):
                return ""
            return str(message) if isinstance(message, str) and message.strip() else ""

        def _extract_text_from_sse(raw_text: str) -> str:
            deltas: list[str] = []
            done_chunks: list[str] = []
            completed_text = ""
            completed_response_obj: dict | None = None

            for line in raw_text.splitlines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue

                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue

                event_type = str(event.get("type", ""))
                if event_type == "response.output_text.delta":
                    delta = event.get("delta")
                    if isinstance(delta, str):
                        deltas.append(delta)
                    continue

                if event_type == "response.output_text.done":
                    done_text = event.get("text")
                    if isinstance(done_text, str):
                        done_chunks.append(done_text)
                    continue

                if event_type == "response.content_part.done":
                    part = event.get("part")
                    if isinstance(part, dict):
                        part_text = part.get("text")
                        if isinstance(part_text, str) and part_text:
                            done_chunks.append(part_text)
                    continue

                if event_type == "response.output_item.done":
                    item = event.get("item")
                    if isinstance(item, dict):
                        maybe = _extract_responses_text({"output": [item]})
                        if maybe:
                            done_chunks.append(maybe)
                    continue

                if event_type == "response.completed":
                    response_obj = event.get("response")
                    if isinstance(response_obj, dict):
                        completed_response_obj = response_obj
                        maybe_text = _extract_responses_text(response_obj)
                        if isinstance(maybe_text, str) and maybe_text.strip():
                            completed_text = maybe_text

            if deltas:
                return "".join(deltas)
            if done_chunks:
                return "".join(done_chunks)
            if completed_response_obj is not None:
                maybe_text = _extract_responses_text(completed_response_obj)
                if maybe_text:
                    return maybe_text
            # Fallback: extract output_text.done text with regex when JSON line parsing fails.
            m = re.search(
                r'"type"\s*:\s*"response\.output_text\.done".*?"text"\s*:\s*"((?:\\.|[^"\\])*)"',
                raw_text,
                re.DOTALL,
            )
            if m:
                try:
                    return json.loads(f'"{m.group(1)}"')
                except json.JSONDecodeError:
                    pass
            return completed_text

        def _summarize_sse(raw_text: str) -> dict[str, object]:
            event_types: list[str] = []
            errors: list[object] = []
            non_json_data_lines = 0
            last_data_line = ""

            for line in raw_text.splitlines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                last_data_line = data[:200]
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    non_json_data_lines += 1
                    continue

                typ = obj.get("type")
                if isinstance(typ, str):
                    event_types.append(typ)
                if typ == "response.completed":
                    response_obj = obj.get("response")
                    if isinstance(response_obj, dict):
                        err = response_obj.get("error")
                        if err:
                            errors.append(err)
                if typ == "response.failed":
                    response_obj = obj.get("response")
                    if isinstance(response_obj, dict):
                        err = response_obj.get("error")
                        if err:
                            errors.append(err)
                if typ == "error":
                    err = obj.get("error")
                    errors.append(err if err else obj)

            return {
                "event_types": sorted(set(event_types)),
                "non_json_data_lines": non_json_data_lines,
                "errors": errors[:3],
                "last_data_line": last_data_line[:400],
            }

        def _call_responses_api(system_prompt: str, user_prompt: str) -> str:
            url = _ensure_responses_url(self.base_url or "")
            body_base = {
                "model": self.model,
                "instructions": system_prompt,
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": user_prompt}],
                    }
                ],
            }
            if provider == "openai":
                effort = str(self.reasoning_effort or "").strip().lower()
                if effort and effort != "none":
                    body_base["reasoning"] = {"effort": effort}
            headers = {
                **_json_headers(bearer=True)
            }

            last_text = ""
            last_http_error: requests.HTTPError | None = None
            stream_modes = (True,) if self.force_stream else (False, True)
            for stream in stream_modes:
                body = body_base | {"stream": stream}
                try:
                    response = requests.request(
                        "POST",
                        url,
                        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                        headers=headers,
                        timeout=self.timeout_s,
                    )
                    response.raise_for_status()
                except requests.HTTPError as ex:
                    last_http_error = ex
                    # Some OpenAI-compatible gateways only accept streaming /responses.
                    if not stream:
                        continue
                    raise

                text = response.text.strip()
                if not text:
                    continue
                last_text = text

                parsed_json = _extract_text_from_responses_payload(text)
                if parsed_json:
                    return parsed_json

                if stream:
                    parsed_sse = _extract_text_from_sse(text)
                    if parsed_sse:
                        return parsed_sse
                    # Do not pass raw SSE body downstream; retry instead.
                    if "data:" in text or "event:" in text:
                        sse_diag = _summarize_sse(text)
                        logger.warning(
                            "Empty assistant text from /responses SSE for %s; "
                            "events=%s non_json_data=%s response_error=%s last_data=%s",
                            provider,
                            sse_diag.get("event_types"),
                            sse_diag.get("non_json_data_lines"),
                            sse_diag.get("errors"),
                            sse_diag.get("last_data_line"),
                        )
                        raise RuntimeError(
                            "No assistant output extracted from /responses SSE payload"
                        )

                # Some gateways may return plain model text instead of structured JSON.
                if not stream and ("<decision>" in text or "\"actions\"" in text):
                    return text

            if last_http_error is not None and not last_text:
                raise last_http_error
            logger.warning("Unexpected /responses payload for %s: %s", provider, last_text[:800])
            return last_text

        def _call_openai_compat_chat(
            *,
            url: str,
            model_name: str,
            system_prompt_text: str,
            user_prompt_text: str,
            include_temperature: bool = True,
            merge_system_into_user: bool = False,
        ) -> dict:
            if merge_system_into_user:
                # Some Gemini-compatible gateways ignore `system` role semantics.
                # Merge system + user into one user message to preserve constraints.
                merged_prompt = (
                    "System Instructions:\n"
                    f"{system_prompt_text}\n\n"
                    "User Context:\n"
                    f"{user_prompt_text}"
                )
                messages = [{"role": "user", "content": merged_prompt}]
            else:
                messages = [
                    {"role": "system", "content": system_prompt_text},
                    {"role": "user", "content": user_prompt_text},
                ]

            body = {
                "model": model_name,
                "messages": messages,
            }
            if include_temperature:
                body["temperature"] = 0.2

            response = requests.request(
                "POST",
                url,
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers=_json_headers(bearer=True),
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            return response.json()

        def _extract_chat_choice_content(payload: dict) -> str:
            try:
                message = payload["choices"][0]["message"]
            except (KeyError, TypeError, IndexError):
                return ""
            if isinstance(message, dict):
                content = message.get("content")
            else:
                return ""

            if isinstance(content, str):
                return content
            if isinstance(content, list):
                out: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if isinstance(text, str):
                            out.append(text)
                return "".join(out)
            return ""

        def _fallback_wait_response(reason: str) -> str:
            rationale = (reason or "Gemini returned empty response").strip()
            action = {
                "type": "wait",
                "market": "",
                "token_id": "",
                "side": "",
                "price": 0,
                "size": 0,
                "amount": 0,
                "time_in_force": "GTC",
                "risk": {"max_slippage_bps": 30, "max_notional_usd": 0},
                "rationale": rationale[:1000],
            }
            decision = {"actions": [action]}
            return (
                f"<reasoning>{rationale[:1200]}</reasoning>\n"
                f"<decision>{json.dumps(decision, ensure_ascii=False)}</decision>"
            )

        logger = get_logger("pm_agent.ai")
        provider = (self.provider or "openai").lower()
        gemini_openai_compat = (
            provider == "gemini"
            and bool(self.base_url)
            and _is_gemini_openai_compat_url(str(self.base_url))
        )

        # ----------------------------------------------------------------
        # OpenAI-compatible Chat Completions (includes grok, glm, custom)
        # ----------------------------------------------------------------
        _openai_compat = {"openai", "deepseek", "qwen", "grok", "glm", "custom"}
        if (provider in _openai_compat or gemini_openai_compat) and self.base_url:
            if provider == "openai" and _model_prefers_responses(self.model):
                return _call_responses_api(system_prompt, user_prompt)

            url = _ensure_chat_url(self.base_url)

            # Some OpenAI-compatible Gemini gateways intermittently return 5xx
            # or timeout on specific model IDs. Retry with a deterministic
            # lower-latency model first, then alias pool as last resort.
            if gemini_openai_compat:
                primary_model = (self.model or "").strip() or "gemini-3-flash"
                fallback_models = ["gemini-2.5-flash", "gemini-auto"]
                model_candidates = [primary_model]
                for fallback_model in fallback_models:
                    if primary_model.lower() != fallback_model.lower():
                        model_candidates.append(fallback_model)

                last_error: Exception | None = None
                for idx, model_name in enumerate(model_candidates):
                    try:
                        payload = _call_openai_compat_chat(
                            url=url,
                            model_name=model_name,
                            system_prompt_text=system_prompt,
                            user_prompt_text=user_prompt,
                            include_temperature=False,
                            merge_system_into_user=True,
                        )
                        content = _extract_chat_choice_content(payload).strip()
                        if not content:
                            should_fallback = idx < (len(model_candidates) - 1)
                            if should_fallback:
                                logger.warning(
                                    "Gemini returned empty content on model=%s, retrying with %s",
                                    model_name,
                                    model_candidates[idx + 1],
                                )
                                continue
                            logger.warning(
                                "Gemini returned empty content (model=%s, payload=%s)",
                                model_name,
                                payload,
                            )
                            return _fallback_wait_response(
                                f"Gemini empty completion on model={model_name}; fallback WAIT."
                            )
                        if idx > 0:
                            logger.warning(
                                "Gemini fallback succeeded via model=%s (primary=%s)",
                                model_name,
                                primary_model,
                            )
                        return content
                    except requests.HTTPError as ex:
                        last_error = ex
                        status = ex.response.status_code if ex.response is not None else None
                        should_fallback = (
                            idx < (len(model_candidates) - 1)
                            and (status is None or status >= 500 or status == 429)
                        )
                        if should_fallback:
                            logger.warning(
                                "Gemini chat failed (model=%s, status=%s), retrying with %s",
                                model_name,
                                status,
                                model_candidates[idx + 1],
                            )
                            continue
                        return _fallback_wait_response(
                            f"Gemini HTTP error status={status}; fallback WAIT."
                        )
                    except (requests.Timeout, requests.ConnectionError) as ex:
                        last_error = ex
                        should_fallback = idx < (len(model_candidates) - 1)
                        if should_fallback:
                            logger.warning(
                                "Gemini chat transport error on %s, retrying with %s: %s",
                                model_name,
                                model_candidates[idx + 1],
                                ex,
                            )
                            continue
                        return _fallback_wait_response(
                            f"Gemini transport error: {ex}; fallback WAIT."
                        )

                if last_error is not None:
                    return _fallback_wait_response(
                        f"Gemini fallback failed: {last_error}; fallback WAIT."
                    )

            body = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
            }
            if provider == "openai":
                effort = str(self.reasoning_effort or "").strip().lower()
                if effort and effort != "none":
                    body["reasoning_effort"] = effort
            if provider == "glm":
                # Align with NVIDIA OpenAI-compatible GLM-5 template settings.
                body.update(
                    {
                        "temperature": 1,
                        "top_p": 1,
                        "max_tokens": 16384,
                        "chat_template_kwargs": {
                            "enable_thinking": True,
                            "clear_thinking": False,
                        },
                    }
                )

            response = requests.request(
                "POST",
                url,
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers=_json_headers(bearer=True),
                timeout=self.timeout_s,
            )
            try:
                response.raise_for_status()
            except requests.HTTPError:
                if provider == "openai":
                    err_text = (response.text or "").lower()
                    if (
                        "chat/completions endpoint not supported" in err_text
                        or "codex channel" in err_text
                    ):
                        logger.info(
                            "Model %s does not support chat/completions on %s; "
                            "retrying via /responses",
                            self.model,
                            self.base_url,
                        )
                        return _call_responses_api(system_prompt, user_prompt)
                raise
            payload = response.json()
            try:
                return str(payload["choices"][0]["message"]["content"])
            except (KeyError, ValueError, TypeError):
                logger.warning("Unexpected %s payload: %s", provider, payload)
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
                headers=_json_headers(bearer=True, goog=True),
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
                headers=_json_headers(bearer=True, x_api_key=True)
                | {"anthropic-version": "2023-06-01"},
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
        if provider == "openai" and _model_prefers_responses(self.model):
            kwargs: dict[str, object] = {
                "model": self.model,
                "instructions": system_prompt,
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": user_prompt}],
                    }
                ],
            }
            effort = str(self.reasoning_effort or "").strip().lower()
            if effort and effort != "none":
                kwargs["reasoning"] = {"effort": effort}
            resp = client.responses.create(**kwargs)
            return str(getattr(resp, "output_text", "") or "")

        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return str(resp.choices[0].message.content)

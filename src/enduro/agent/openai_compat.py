"""Adapter for OpenAI-compatible Chat Completions APIs (DeepSeek, Z.ai GLM, ...).

- History is append-only: assistant messages go back exactly as returned, including
  `reasoning_content` — DeepSeek requires the chain of thought of every earlier turn
  to be sent back on requests that carry tools.
- Provider-specific switches (thinking mode, reasoning effort) come from the model
  profile as extra request fields, so the adapter itself stays provider-neutral.
- Cost is estimated from the profile's per-million-token prices; providers report the
  cached part of the prompt under different names, both are read.
"""

from __future__ import annotations

import json
from typing import Any

import openai

from enduro.agent.llm import (
    INVALID_ARGUMENTS,
    LLMError,
    LLMTurn,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
    UsageLimitError,
)

# Providers answer "out of credit" with 402 (DeepSeek) or a 429 that does not clear by
# retrying soon; both mean: pause, the next calls would fail the same way.
_OUT_OF_CREDIT = ("insufficient balance", "insufficient_quota")


def tool_call_input(arguments: str | None) -> dict[str, Any]:
    try:
        value = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return {INVALID_ARGUMENTS: arguments}
    return value if isinstance(value, dict) else {INVALID_ARGUMENTS: arguments}


def usage_from(raw: Any, prices: dict[str, float]) -> Usage:
    if raw is None:
        return Usage()
    prompt = raw.prompt_tokens or 0
    details = getattr(raw, "prompt_tokens_details", None)
    cached = getattr(raw, "prompt_cache_hit_tokens", None)  # DeepSeek
    if cached is None:
        cached = getattr(details, "cached_tokens", None) or 0
    fresh = max(0, prompt - cached)
    output = raw.completion_tokens or 0
    cost = (
        fresh * prices.get("input", 0.0)
        + cached * prices.get("cache_read", 0.0)
        + output * prices.get("output", 0.0)
    ) / 1e6
    return Usage(
        input_tokens=fresh,
        output_tokens=output,
        cache_read_tokens=cached,
        cost_usd=cost,
    )


class OpenAICompatSession:
    def __init__(
        self,
        client: openai.AsyncOpenAI,
        model: str,
        max_tokens: int,
        extra_body: dict[str, Any],
        prices: dict[str, float],
        system: str,
        tools: list[ToolSpec],
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._extra_body = extra_body
        self._prices = prices
        self._tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]
        self._messages: list[dict[str, Any]] = [{"role": "system", "content": system}]

    async def send(self, text: str) -> LLMTurn:
        self._messages.append({"role": "user", "content": text})
        return await self._turn()

    async def send_tool_results(self, results: list[ToolResult], text: str = "") -> LLMTurn:
        for r in results:
            content = f"ERROR: {r.content}" if r.is_error else r.content
            self._messages.append({"role": "tool", "tool_call_id": r.call_id, "content": content})
        if text:
            self._messages.append({"role": "user", "content": text})
        return await self._turn()

    async def _turn(self) -> LLMTurn:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=self._messages,
                tools=self._tools,
                max_tokens=self._max_tokens,
                extra_body=self._extra_body or None,
            )
        except openai.APIStatusError as e:
            message = f"{self._model} API error {e.status_code}: {e.message}"
            if e.status_code == 402 or any(m in str(e.message).lower() for m in _OUT_OF_CREDIT):
                raise UsageLimitError(message, None) from e
            raise LLMError(message) from e
        except openai.APIConnectionError as e:  # includes timeouts
            raise LLMError(f"{self._model} API connection error: {e}") from e
        if not response.choices:
            raise LLMError(f"{self._model} returned no choices")

        choice = response.choices[0]
        message = choice.message
        # Append exactly what came back, reasoning included: append-only history.
        assistant: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning is None and message.model_extra:
            reasoning = message.model_extra.get("reasoning_content")
        if reasoning:
            assistant["reasoning_content"] = reasoning
        calls = message.tool_calls or []
        if calls:
            assistant["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.function.name, "arguments": c.function.arguments},
                }
                for c in calls
            ]
        self._messages.append(assistant)
        if choice.finish_reason == "content_filter":
            raise LLMError(f"{self._model} refused (content filter)")
        stop = {"tool_calls": "tool_use", "stop": "end_turn", "length": "max_tokens"}
        return LLMTurn(
            text=message.content or "",
            tool_calls=[
                ToolCall(c.id, c.function.name, tool_call_input(c.function.arguments))
                for c in calls
            ],
            stop_reason=stop.get(choice.finish_reason or "", choice.finish_reason or ""),
            usage=usage_from(response.usage, self._prices),
            model=response.model or self._model,
        )


class OpenAICompatClient:
    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str,
        max_tokens: int = 16_000,
        extra_body: dict[str, Any] | None = None,
        prices: dict[str, float] | None = None,
        timeout_s: float = 300.0,
    ) -> None:
        self.name = model
        self._max_tokens = max_tokens
        self._extra_body = extra_body or {}
        self._prices = prices or {}
        # Retries cover transient 5xx / 429 / connection errors of a single model call.
        self._client = openai.AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=2
        )

    def session(self, system: str, tools: list[ToolSpec]) -> OpenAICompatSession:
        return OpenAICompatSession(
            self._client,
            self.name,
            self._max_tokens,
            self._extra_body,
            self._prices,
            system,
            tools,
        )

    async def close(self) -> None:
        await self._client.close()

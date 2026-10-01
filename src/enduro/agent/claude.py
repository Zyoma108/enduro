"""Claude adapter (Anthropic Messages API) for the provider-agnostic agent interface.

- History is append-only: assistant turns go back exactly as returned (including
  thinking blocks), which keeps preserved thinking valid and the prompt cache warm.
- The system prompt and tool definitions are stable across ticks and cached; within an
  episode, automatic caching re-uses the growing prefix.
- Thinking is adaptive (always on for Claude Opus 5.5); depth is set via `effort`.
- Server-side refusal fallback is on (`fallbacks: "default"`): if a safety classifier
  declines, the API re-runs the request on the recommended fallback model.
"""

from __future__ import annotations

from typing import Any

import anthropic

from enduro.agent.llm import LLMError, LLMTurn, ToolCall, ToolResult, ToolSpec, Usage

FALLBACK_BETA = "server-side-fallback-2026-07-01"

# USD per million tokens: (input, output, cache read). Cache writes (5 min TTL) cost
# 1.25x input. Keep in sync with Anthropic pricing.
PRICES: dict[str, tuple[float, float, float]] = {
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20),
    "claude-haiku-4-5": (1.0, 5.0, 0.10),
}


def estimate_cost(model: str, usage: Any) -> float:
    price_in, price_out, price_cache = PRICES.get(model, (0.0, 0.0, 0.0))
    return (
        (usage.input_tokens or 0) * price_in
        + (usage.output_tokens or 0) * price_out
        + (usage.cache_read_input_tokens or 0) * price_cache
        + (usage.cache_creation_input_tokens or 0) * price_in * 1.25
    ) / 1e6


class ClaudeSession:
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        model: str,
        effort: str,
        max_tokens: int,
        system: str,
        tools: list[ToolSpec],
    ) -> None:
        self._client = client
        self._model = model
        self._effort = effort
        self._max_tokens = max_tokens
        self._system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        self._tools = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]
        self._messages: list[dict[str, Any]] = []

    async def send(self, text: str) -> LLMTurn:
        self._messages.append({"role": "user", "content": text})
        return await self._turn()

    async def send_tool_results(self, results: list[ToolResult], text: str = "") -> LLMTurn:
        content: list[dict[str, Any]] = [
            {
                "type": "tool_result",
                "tool_use_id": r.call_id,
                "content": r.content,
                **({"is_error": True} if r.is_error else {}),
            }
            for r in results
        ]
        if text:
            content.append({"type": "text", "text": text})
        self._messages.append({"role": "user", "content": content})
        return await self._turn()

    async def _turn(self) -> LLMTurn:
        try:
            response = await self._client.beta.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=self._system,
                tools=self._tools,
                messages=self._messages,
                output_config={"effort": self._effort},
                cache_control={"type": "ephemeral"},
                fallbacks="default",
                betas=[FALLBACK_BETA],
            )
        except anthropic.APIStatusError as e:
            raise LLMError(f"Claude API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError(f"Claude API connection error: {e}") from e

        # Append exactly what came back (thinking blocks included): append-only history.
        self._messages.append({"role": "assistant", "content": response.content})
        usage = Usage(
            input_tokens=response.usage.input_tokens or 0,
            output_tokens=response.usage.output_tokens or 0,
            cache_read_tokens=response.usage.cache_read_input_tokens or 0,
            cache_write_tokens=response.usage.cache_creation_input_tokens or 0,
            cost_usd=estimate_cost(response.model, response.usage),
        )
        if response.stop_reason == "refusal":
            raise LLMError(f"model refused: {response.stop_details}")
        return LLMTurn(
            text="".join(b.text for b in response.content if b.type == "text"),
            tool_calls=[
                ToolCall(b.id, b.name, dict(b.input))
                for b in response.content
                if b.type == "tool_use"
            ],
            stop_reason=response.stop_reason or "",
            usage=usage,
            model=response.model,
        )


class ClaudeClient:
    def __init__(
        self,
        model: str = "claude-opus-5-5",
        effort: str = "medium",
        max_tokens: int = 16_000,
        api_key: str | None = None,
    ) -> None:
        self.name = model
        self._effort = effort
        self._max_tokens = max_tokens
        # Without an explicit key the SDK resolves ANTHROPIC_API_KEY / auth profiles.
        self._client = (
            anthropic.AsyncAnthropic(api_key=api_key) if api_key else anthropic.AsyncAnthropic()
        )

    def session(self, system: str, tools: list[ToolSpec]) -> ClaudeSession:
        return ClaudeSession(self._client, self.name, self._effort, self._max_tokens, system, tools)

    async def close(self) -> None:
        await self._client.close()

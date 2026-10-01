"""Provider-agnostic LLM interface for a tool-using agent.

An adapter turns (system prompt, tools, user text, tool results) into one model turn.
The adapter owns the provider-specific conversation format: it appends the assistant
turns exactly as the provider returned them (append-only), so provider features such
as reasoning blocks and prompt caching keep working. Other providers plug in by
implementing `LLMSession` / `LLMClient`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]  # JSON Schema


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True, slots=True)
class LLMTurn:
    text: str  # visible text the model wrote this turn
    tool_calls: list[ToolCall]
    stop_reason: str  # "end_turn" | "tool_use" | "max_tokens" | "refusal" | ...
    usage: Usage = field(default_factory=Usage)
    model: str = ""  # the model that actually answered (may differ after a fallback)


class LLMError(RuntimeError):
    pass


class LLMSession(Protocol):
    """One agent episode: a fixed system prompt and tool set, an append-only history."""

    async def send(self, text: str) -> LLMTurn: ...

    async def send_tool_results(self, results: list[ToolResult], text: str = "") -> LLMTurn:
        """Return results for every tool call of the previous turn (all in one message)."""
        ...


class LLMClient(Protocol):
    name: str  # e.g. "claude-opus-5-5"

    def session(self, system: str, tools: list[ToolSpec]) -> LLMSession: ...

"""Provider-agnostic LLM interface for a tool-using agent.

An adapter turns (system prompt, tools, user text, tool results) into one model turn.
The adapter owns the provider-specific conversation format: it appends the assistant
turns exactly as the provider returned them (append-only), so provider features such
as reasoning blocks and prompt caching keep working. Other providers plug in by
implementing `LLMSession` / `LLMClient`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
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


class UsageLimitError(LLMError):
    """The model account is out of usage until `resets_at_ms` (None: reset time unknown).
    Every request fails until then, so the agent should pause instead of retrying."""

    def __init__(self, message: str, resets_at_ms: int | None) -> None:
        super().__init__(message)
        self.resets_at_ms = resets_at_ms


class LLMSession(Protocol):
    """One agent episode: a fixed system prompt and tool set, an append-only history."""

    async def send(self, text: str) -> LLMTurn: ...

    async def send_tool_results(self, results: list[ToolResult], text: str = "") -> LLMTurn:
        """Return results for every tool call of the previous turn (all in one message)."""
        ...


class LLMClient(Protocol):
    name: str  # e.g. "claude-opus-5-5"

    def session(self, system: str, tools: list[ToolSpec]) -> LLMSession: ...


ToolCaller = Callable[[str, dict[str, Any]], Awaitable[ToolResult]]  # name, args -> result
UsageSink = Callable[[LLMTurn], None]


class EpisodeBackend(Protocol):
    """Runs one agent episode end to end: the model may call tools any number of times.

    `call_tool` executes a tool in the agent process; `record` journals each model turn
    (or one aggregate turn when the backend cannot see individual turns); `is_done`
    tells whether the agent already called finish_tick.
    """

    name: str

    async def run_episode(
        self,
        system: str,
        situation: str,
        tools: list[ToolSpec],
        call_tool: ToolCaller,
        record: UsageSink,
        is_done: Callable[[], bool],
    ) -> str:
        """Returns the model's final text."""
        ...

    async def close(self) -> None: ...


class ApiLoopBackend:
    """Drives an LLMClient turn by turn: we execute tools between model calls."""

    def __init__(self, llm: LLMClient, max_calls: int) -> None:
        self.llm = llm
        self.name = llm.name
        self.max_calls = max_calls

    async def run_episode(
        self,
        system: str,
        situation: str,
        tools: list[ToolSpec],
        call_tool: ToolCaller,
        record: UsageSink,
        is_done: Callable[[], bool],
    ) -> str:
        session = self.llm.session(system, tools)
        turn = await session.send(situation)
        for call_no in range(self.max_calls):
            record(turn)
            if not turn.tool_calls:
                return turn.text
            results = []
            for c in turn.tool_calls:
                result = await call_tool(c.name, c.input)
                results.append(ToolResult(c.id, result.content, result.is_error))
            if is_done() or call_no == self.max_calls - 1:
                if not is_done():
                    raise LLMError(f"episode exceeded {self.max_calls} model calls")
                return turn.text
            turn = await session.send_tool_results(results)
        return turn.text

    async def close(self) -> None:
        close = getattr(self.llm, "close", None)
        if close is not None:
            await close()

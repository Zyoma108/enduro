"""Episode backend that runs the model through the Claude Code CLI (`claude -p`).

Uses the account Claude Code is logged into (e.g. a Claude subscription) instead of
API billing. Each tick runs one headless `claude -p` process:
  - our system prompt replaces Claude Code's default one;
  - all built-in tools are disabled (`--tools ""`), only our MCP tools are allowed,
    and anything else is denied without asking (`--permission-mode dontAsk`);
  - it runs in a dedicated working directory outside the project, so the project's
    CLAUDE.md (instructions for *developing* Enduro) never reaches the trader;
  - ANTHROPIC_API_KEY is removed from its environment so it never silently switches
    to API billing.
Tools execute in the agent process through the local MCP server (`ToolServer`).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from enduro.agent.llm import LLMError, LLMTurn, ToolCaller, ToolResult, ToolSpec, Usage, UsageSink
from enduro.agent.mcp_server import ToolServer

log = logging.getLogger(__name__)

DEFAULT_WORKDIR = Path.home() / ".cache" / "enduro" / "claude-agent"
# Variables that would change how the nested CLI authenticates or behaves.
_STRIPPED_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDECODE")


def build_command(
    claude_bin: str,
    model: str,
    effort: str,
    system: str,
    mcp_config: dict[str, Any],
    allowed_tools: list[str],
) -> list[str]:
    return [
        claude_bin,
        "-p",
        "--output-format",
        "json",
        "--model",
        model,
        "--effort",
        effort,
        "--system-prompt",
        system,
        "--tools",
        "",
        "--mcp-config",
        json.dumps(mcp_config),
        "--strict-mcp-config",
        "--allowedTools",
        ",".join(allowed_tools),
        "--permission-mode",
        "dontAsk",
        "--no-session-persistence",
    ]


def parse_result(stdout: bytes, returncode: int, stderr: bytes) -> dict[str, Any]:
    try:
        data = json.loads(stdout.decode("utf-8", "replace") or "{}")
    except json.JSONDecodeError:
        data = {}
    if returncode != 0 or data.get("is_error") or not data:
        detail = data.get("result") or stderr.decode("utf-8", "replace").strip()[-500:]
        raise LLMError(f"claude -p failed (exit {returncode}, {data.get('subtype')}): {detail}")
    return data


def turn_from_result(data: dict[str, Any], model: str) -> LLMTurn:
    usage = data.get("usage") or {}
    return LLMTurn(
        text=data.get("result") or "",
        tool_calls=[],  # tool calls are journaled by the agent as they execute
        stop_reason=data.get("subtype") or "",
        usage=Usage(
            input_tokens=usage.get("input_tokens") or 0,
            output_tokens=usage.get("output_tokens") or 0,
            cache_read_tokens=usage.get("cache_read_input_tokens") or 0,
            cache_write_tokens=usage.get("cache_creation_input_tokens") or 0,
            # API-equivalent cost as reported by Claude Code; on a subscription it is
            # not billed, but it shows how much of the plan's usage a tick consumes.
            cost_usd=float(data.get("total_cost_usd") or 0.0),
        ),
        model=model,
    )


class ClaudeCodeBackend:
    def __init__(
        self,
        model: str,
        effort: str,
        timeout_s: float = 300.0,
        claude_bin: str = "claude",
        workdir: Path = DEFAULT_WORKDIR,
    ) -> None:
        self.name = f"claude-code:{model}"
        self.model = model
        self.effort = effort
        self.timeout_s = timeout_s
        self.claude_bin = claude_bin
        self.workdir = workdir
        self._server: ToolServer | None = None
        self._call: ToolCaller | None = None

    async def _ensure_server(self, tools: list[ToolSpec]) -> ToolServer:
        if self._server is None:

            async def forward(name: str, args: dict[str, Any]) -> ToolResult:
                assert self._call is not None, "tool call outside an episode"
                return await self._call(name, args)

            self._server = ToolServer(tools, forward)
            url = await self._server.start()
            log.info("MCP tool server for Claude Code at %s", url.rsplit("/", 1)[0] + "/…")
        return self._server

    async def run_episode(
        self,
        system: str,
        situation: str,
        tools: list[ToolSpec],
        call_tool: ToolCaller,
        record: UsageSink,
        is_done: Callable[[], bool],
    ) -> str:
        server = await self._ensure_server(tools)
        self._call = call_tool
        self.workdir.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items() if k not in _STRIPPED_ENV}
        command = build_command(
            self.claude_bin,
            self.model,
            self.effort,
            system,
            server.mcp_config(),
            server.tool_names(),
        )
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.workdir,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(situation.encode("utf-8")), self.timeout_s
            )
        except (TimeoutError, asyncio.CancelledError) as e:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await process.wait()
            if isinstance(e, TimeoutError):
                raise LLMError(f"claude -p did not finish within {self.timeout_s:.0f}s") from e
            raise
        finally:
            self._call = None
        data = parse_result(stdout, process.returncode or 0, stderr)
        turn = turn_from_result(data, self.model)
        record(turn)
        return turn.text

    async def close(self) -> None:
        if self._server is not None:
            await self._server.stop()

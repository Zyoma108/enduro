"""Serves the agent's tools over MCP (streamable HTTP) from inside the agent process.

Used when the model runs in Claude Code (`claude -p`): Claude Code connects to this
server and calls the tools, while all state (focus streams, radar, risk, exchange
connections) stays in our process. Bound to 127.0.0.1 only, and the endpoint path
carries a random secret so other local processes cannot call trading tools.
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets
from collections.abc import Awaitable, Callable, Generator, Sequence
from typing import Any

import uvicorn
from mcp import types
from mcp.server.lowlevel import Server

from enduro.agent.llm import ToolResult, ToolSpec

ToolCaller = Callable[[str, dict[str, Any]], Awaitable[ToolResult]]

SERVER_NAME = "enduro"


class _QuietUvicorn(uvicorn.Server):
    """uvicorn without its own SIGINT/SIGTERM handling: the agent owns shutdown."""

    @contextlib.contextmanager
    def capture_signals(self) -> Generator[None]:
        yield


class ToolServer:
    def __init__(self, specs: Sequence[ToolSpec], call: ToolCaller) -> None:
        self._specs = list(specs)
        self._call = call
        self._secret = secrets.token_urlsafe(16)
        self._uvicorn: _QuietUvicorn | None = None
        self._task: asyncio.Task | None = None
        self.url = ""

        async def list_tools(ctx: Any, params: Any) -> types.ListToolsResult:
            return types.ListToolsResult(
                tools=[
                    types.Tool(name=s.name, description=s.description, input_schema=s.input_schema)
                    for s in self._specs
                ]
            )

        async def call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
            result = await self._call(params.name, dict(params.arguments or {}))
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=result.content)],
                is_error=result.is_error,
            )

        self._server = Server(SERVER_NAME, on_list_tools=list_tools, on_call_tool=call_tool)

    async def start(self) -> str:
        path = f"/mcp/{self._secret}"
        app = self._server.streamable_http_app(
            streamable_http_path=path, stateless_http=True, json_response=True
        )
        config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="on")
        self._uvicorn = _QuietUvicorn(config)
        self._task = asyncio.create_task(self._uvicorn.serve())
        while not self._uvicorn.started:
            if self._task.done():
                self._task.result()  # surface the startup error
            await asyncio.sleep(0.05)
        port = self._uvicorn.servers[0].sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}{path}"
        return self.url

    async def stop(self) -> None:
        if self._uvicorn is not None:
            self._uvicorn.should_exit = True
        if self._task is not None:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.wait_for(self._task, 5)

    def mcp_config(self) -> dict[str, Any]:
        return {"mcpServers": {SERVER_NAME: {"type": "http", "url": self.url}}}

    def tool_names(self) -> list[str]:
        """Names as Claude Code exposes them to the model."""
        return [f"mcp__{SERVER_NAME}__{s.name}" for s in self._specs]

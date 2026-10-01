import httpx2
from mcp import Client

from enduro.agent.llm import ToolResult, ToolSpec
from enduro.agent.mcp_server import ToolServer

SPECS = [
    ToolSpec(
        "echo", "Echo text back", {"type": "object", "properties": {"text": {"type": "string"}}}
    ),
    ToolSpec("fail", "Always fails", {"type": "object", "properties": {}}),
]


async def test_tool_server_round_trip_over_http():
    calls = []

    async def call(name, args):
        calls.append((name, args))
        if name == "fail":
            return ToolResult("", "error: nope", is_error=True)
        return ToolResult("", f"echo: {args['text']}")

    server = ToolServer(SPECS, call)
    url = await server.start()
    try:
        assert url.startswith("http://127.0.0.1:") and "/mcp/" in url
        async with Client(url) as client:
            tools = await client.list_tools()
            assert [t.name for t in tools.tools] == ["echo", "fail"]
            assert tools.tools[0].input_schema["properties"]["text"]["type"] == "string"

            ok = await client.call_tool("echo", {"text": "привет"})
            assert not ok.is_error and ok.content[0].text == "echo: привет"
            bad = await client.call_tool("fail", {})
            assert bad.is_error and "nope" in bad.content[0].text
        assert calls == [("echo", {"text": "привет"}), ("fail", {})]

        # Without the secret path the endpoint does not exist.
        base = url.rsplit("/mcp/", 1)[0]
        async with httpx2.AsyncClient() as http:
            response = await http.post(f"{base}/mcp", json={})
            assert response.status_code == 404
        assert server.mcp_config()["mcpServers"]["enduro"]["url"] == url
        assert server.tool_names() == ["mcp__enduro__echo", "mcp__enduro__fail"]
    finally:
        await server.stop()

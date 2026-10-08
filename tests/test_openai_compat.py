"""OpenAI-compatible adapter (DeepSeek, GLM) against recorded-shape responses."""

import httpx2 as httpx
import openai
import pytest
from openai.types.chat import ChatCompletion

from enduro.agent.llm import (
    INVALID_ARGUMENTS,
    ApiLoopBackend,
    LLMError,
    ToolResult,
    ToolSpec,
    UsageLimitError,
)
from enduro.agent.openai_compat import OpenAICompatClient, tool_call_input

TOOLS = [ToolSpec("get_radar", "radar", {"type": "object", "properties": {}})]
PRICES = {"input": 1.0, "output": 4.0, "cache_read": 0.1}


def completion(content="", calls=(), reasoning=None, finish="stop", usage=None):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    if calls:
        message["tool_calls"] = [
            {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}
            for cid, name, args in calls
        ]
    return ChatCompletion.model_validate(
        {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "deepseek-v4-pro",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": usage
            or {
                "prompt_tokens": 1000,
                "completion_tokens": 200,
                "total_tokens": 1200,
                "prompt_cache_hit_tokens": 800,
                "prompt_cache_miss_tokens": 200,
            },
        }
    )


def client_with(responses, sent):
    client = OpenAICompatClient(
        "deepseek-v4-pro",
        base_url="https://api.example.com",
        api_key="k",
        extra_body={"thinking": {"type": "enabled"}},
        prices=PRICES,
    )

    async def create(**kwargs):
        sent.append(kwargs)
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    client._client.chat.completions.create = create
    return client


async def test_tool_loop_sends_reasoning_back_and_prices_the_turn():
    sent = []
    client = client_with(
        [
            completion(reasoning="think", calls=[("c1", "get_radar", "{}")], finish="tool_calls"),
            completion(content="done"),
        ],
        sent,
    )
    session = client.session("SYSTEM", TOOLS)
    turn = await session.send("tick")
    assert turn.stop_reason == "tool_use" and turn.tool_calls[0].name == "get_radar"
    # 200 fresh input + 800 cached + 200 output
    assert turn.usage.cost_usd == pytest.approx((200 * 1.0 + 800 * 0.1 + 200 * 4.0) / 1e6)
    assert turn.usage.cache_read_tokens == 800
    final = await session.send_tool_results([ToolResult("c1", "radar rows")])
    assert final.text == "done" and final.stop_reason == "end_turn"
    second = sent[1]["messages"]
    assert second[0] == {"role": "system", "content": "SYSTEM"}
    assistant = second[2]
    assert assistant["reasoning_content"] == "think"  # DeepSeek needs it back with tools
    assert assistant["tool_calls"][0]["id"] == "c1"
    assert second[3] == {"role": "tool", "tool_call_id": "c1", "content": "radar rows"}
    assert sent[0]["extra_body"] == {"thinking": {"type": "enabled"}}
    assert sent[0]["tools"][0]["function"]["name"] == "get_radar"


def test_non_json_arguments_are_flagged():
    assert tool_call_input('{"top": 5}') == {"top": 5}
    assert tool_call_input("top=5") == {INVALID_ARGUMENTS: "top=5"}
    assert tool_call_input("[1]") == {INVALID_ARGUMENTS: "[1]"}


async def test_invalid_arguments_go_back_as_a_tool_error():
    sent = []
    client = client_with(
        [
            completion(calls=[("c1", "get_radar", "not json")], finish="tool_calls"),
            completion(content="ok"),
        ],
        sent,
    )
    called = []

    async def call_tool(name, args):
        called.append(name)
        return ToolResult("", "")

    backend = ApiLoopBackend(client, max_calls=4)
    await backend.run_episode("S", "tick", TOOLS, call_tool, lambda turn: None, lambda: False)
    assert called == []
    # The session keeps one history list, so look the tool message up rather than by index.
    tool_message = next(m for m in sent[1]["messages"] if m["role"] == "tool")
    assert "must be a JSON object" in tool_message["content"]


def status_error(code, message):
    request = httpx.Request("POST", "https://api.example.com/chat/completions")
    response = httpx.Response(code, request=request, json={"error": {"message": message}})
    cls = openai.RateLimitError if code == 429 else openai.APIStatusError
    return cls(message, response=response, body=None)


async def test_out_of_credit_pauses_other_errors_fail_the_tick():
    sent = []
    client = client_with([status_error(402, "Insufficient Balance")], sent)
    with pytest.raises(UsageLimitError):
        await client.session("S", TOOLS).send("tick")
    client = client_with([status_error(500, "boom")], sent)
    with pytest.raises(LLMError) as info:
        await client.session("S", TOOLS).send("tick")
    assert not isinstance(info.value, UsageLimitError)


def test_profiles_parse_and_unknown_profile_is_rejected():
    from enduro.config import ModelProfile, Settings

    settings = Settings(
        agent={"model_profile": "ds"},
        models={
            "ds": {
                "backend": "openai",
                "model": "deepseek-v4-pro",
                "base_url": "https://api.deepseek.com",
                "credentials": "deepseek",
            }
        },
    )
    assert settings.models["ds"].backend == "openai"
    with pytest.raises(ValueError):
        ModelProfile(backend="openai", model="x")  # no endpoint
    with pytest.raises(ValueError):
        Settings(agent={"model_profile": "nope"}, models={})

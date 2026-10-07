import itertools
import json
from datetime import UTC, datetime

import pytest

from enduro.agent.claude_code import build_command, parse_result, turn_from_result
from enduro.agent.llm import LLMError


def test_command_locks_claude_code_down_to_our_tools():
    config = {"mcpServers": {"enduro": {"type": "http", "url": "http://127.0.0.1:1/mcp/s"}}}
    cmd = build_command("claude", "claude-opus-5-5", "medium", "SPIRIT", config, ["mcp__enduro__a"])
    arg = dict(itertools.pairwise(cmd))
    assert cmd[:2] == ["claude", "-p"]
    assert arg["--system-prompt"] == "SPIRIT"  # replaces Claude Code's default prompt
    assert arg["--tools"] == ""  # no built-in tools (Bash, Edit, ...)
    assert json.loads(arg["--mcp-config"]) == config
    assert "--strict-mcp-config" in cmd
    assert arg["--allowedTools"] == "mcp__enduro__a"
    assert arg["--permission-mode"] == "dontAsk"
    assert arg["--output-format"] == "json"
    assert "--no-session-persistence" in cmd


def test_parse_result_and_usage():
    out = json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": "done",
            "total_cost_usd": 0.12,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 900,
                "cache_read_input_tokens": 4000,
                "cache_creation_input_tokens": 300,
            },
        }
    ).encode()
    turn = turn_from_result(parse_result(out, 0, b""), "claude-opus-5-5")
    assert turn.text == "done" and turn.stop_reason == "success"
    assert (turn.usage.output_tokens, turn.usage.cache_read_tokens) == (900, 4000)
    assert turn.usage.cost_usd == 0.12


@pytest.mark.parametrize(
    ("stdout", "code", "stderr", "fragment"),
    [
        (b"", 1, b"Not logged in", "Not logged in"),
        (
            json.dumps({"is_error": True, "subtype": "error", "result": "rate limited"}).encode(),
            0,
            b"",
            "rate limited",
        ),
        (b"garbage", 0, b"boom", "boom"),
    ],
)
def test_parse_result_errors(stdout, code, stderr, fragment):
    with pytest.raises(LLMError, match=fragment):
        parse_result(stdout, code, stderr)


def test_usage_limit_is_its_own_error_with_the_reset_time():
    from enduro.agent.llm import UsageLimitError

    out = json.dumps(
        {
            "is_error": True,
            "subtype": "success",
            "result": "You've hit your session limit · resets 10:50pm (Asia/Yekaterinburg)",
        }
    ).encode()
    now = datetime(2026, 10, 7, 17, 22, tzinfo=UTC)  # 22:22 in Yekaterinburg (UTC+5)
    with pytest.raises(UsageLimitError) as info:
        parse_result(out, 1, b"", now=now)
    reset = datetime(2026, 10, 7, 17, 50, tzinfo=UTC)
    assert info.value.resets_at_ms == int(reset.timestamp() * 1000)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("resets 1am (UTC)", datetime(2026, 10, 8, 1, 0)),  # already past today: tomorrow
        ("resets Oct 9, 10am (UTC)", datetime(2026, 10, 9, 10, 0)),
        ("resets 12pm (UTC)", datetime(2026, 10, 7, 12, 0)),
        ("no reset time here", None),
    ],
)
def test_limit_reset_ms(text, expected):
    from enduro.agent.claude_code import limit_reset_ms

    now = datetime(2026, 10, 7, 10, 0, tzinfo=UTC)
    got = limit_reset_ms(text, now)
    want = None if expected is None else int(expected.replace(tzinfo=UTC).timestamp() * 1000)
    assert got == want

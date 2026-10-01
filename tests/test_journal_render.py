import json

from enduro.journal.journal import Journal
from enduro.journal.render import follow, format_line, format_record

TS = 1_790_850_941_688  # 2026-10-01 10:35:41 UTC


def test_open_decision_shows_size_risk_and_thesis():
    record = {
        "ts": TS,
        "kind": "risk",
        "intent": {
            "symbol": "NEAR/USDT:USDT",
            "side": "short",
            "stop_loss": 5.162,
            "take_profit": 5.05,
            "risk_pct": 0.5,
        },
        "decision": {
            "approved": True,
            "reasons": [],
            "qty": 121.8,
            "entry_price": 5.127,
            "risk_usd": 4.95,
            "risk_pct": 0.4997,
        },
        "thesis": "lower high\nvolume fading",
    }
    assert format_line(record) == (
        "10:35:41 RISK ok: short NEAR stop 5.162 take 5.05 · qty 121.8 @~5.127"
        " · risk 4.95 USDT (0.50%)\n    lower high\n    volume fading"
    )
    record["decision"] = {"approved": False, "reasons": ["too many trades"]}
    assert format_record(record).startswith(
        "RISK REJECTED: short NEAR stop 5.162 take 5.05 · too many trades"
    )


def test_protection_close_alert_and_note_lines():
    protect = {
        "kind": "risk",
        "protection": {"stop_loss": 5.153, "take_profit": None},
        "ok": True,
        "position": {"symbol": "NEAR/USDT:USDT", "side": "short", "stop_loss": 5.162},
    }
    assert format_record(protect) == "PROTECT NEAR short: stop 5.162 → 5.153"
    close = {
        "kind": "order",
        "action": "close",
        "request": {"symbol": "QNT/USDT:USDT", "position_side": "short"},
        "result": {"filled": 1.49, "avg_price": 278.38, "fee": 0.2281, "status": "closed"},
        "reason": "bounce with buyers",
        "entry_price": 277.53,
        "gross_pnl_usdt": -1.2665,
    }
    assert format_record(close) == (
        "ORDER close short QNT 1.49 @ 278.38 · fee 0.2281 · gross PnL -1.27 (entry 277.53)"
        "\n    bounce with buyers"
    )
    alert = {
        "kind": "alert",
        "action": "fired",
        "close": 0.3788,
        "alert": {"id": 3, "symbol": "STX/USDT:USDT", "direction": "below", "level": 0.379},
    }
    assert format_record(alert) == "ALERT #3 fired: STX closed 0.3788, below 0.379"
    note = {
        "kind": "note",
        "n": 7,
        "focus": None,
        "text": "flat",
        "next_check_s": 300,
        "session_cost_usd": 0.42,
    }
    assert format_record(note) == "NOTE tick 7 · search · next in 300s · $0.420\n    flat"


def test_machinery_only_when_verbose():
    tool = {"ts": TS, "kind": "tool", "name": "get_focus", "input": {}, "result": {"a": 1}}
    assert format_record(tool) is None
    assert format_record(tool, verbose=True) == '    tool get_focus {} → {"a":1}'


def test_journal_echoes_each_record(tmp_path):
    seen = []
    journal = Journal(tmp_path, echo=seen.append)
    journal.write("focus", symbol="SOL/USDT:USDT", reason="hot")
    assert seen[0]["kind"] == "focus" and seen[0]["symbol"] == "SOL/USDT:USDT"


def test_follow_reads_whole_lines_only(tmp_path):
    path = tmp_path / "2026-10-01.jsonl"
    path.write_text(json.dumps({"ts": TS, "kind": "tick", "n": 1}) + "\n" + '{"ts": 1, "ki')
    records = follow(tmp_path, "2026-10-01", poll_s=0)
    assert next(records)["n"] == 1
    with path.open("a") as f:  # the writer finishes the line
        f.write('nd": "tick", "n": 2}\n')
    assert next(records)["n"] == 2

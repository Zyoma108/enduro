import pytest

from enduro.execution.models import InstrumentRules, Position
from enduro.risk.manager import (
    HOUR_MS,
    OpenIntent,
    RiskLimits,
    RiskManager,
    RiskStateStore,
)

FEE = 0.00055
DAY = 86_400_000
T0 = 1_790_726_400_000 + 12 * HOUR_MS  # noon UTC
RULES = InstrumentRules("X", qty_step=0.01, min_qty=0.01, min_notional=5, price_tick=0.01)
BTC = InstrumentRules("BTC", qty_step=0.001, min_qty=0.001, min_notional=5, price_tick=0.1)


def manager(tmp_path, **limits) -> RiskManager:
    return RiskManager(RiskLimits(**limits), FEE, RiskStateStore(tmp_path / "risk.json"))


def evaluate(rm, intent, equity=1000.0, positions=(), rules=RULES, bid=99.99, ask=100.0, now=T0):
    return rm.evaluate_open(
        intent,
        equity=equity,
        positions=list(positions),
        rules=rules,
        bid=bid,
        ask=ask,
        now_ms=now,
    )


def position(symbol="Y", side="long", size=1.0):
    return Position(symbol, side, size, 100.0, 100.0, 0.0, 5.0, None)


def test_size_from_stop_including_fees(tmp_path):
    d = evaluate(manager(tmp_path), OpenIntent("X", "long", stop_loss=99.0))
    loss_per_unit = 1.0 + FEE * (100 + 99)
    assert d.approved and d.reasons == ()
    assert d.entry_price == 100.0  # a long buys at the ask
    assert d.qty == pytest.approx(9.01)  # 10 USDT / 1.109 = 9.013 -> rounded down to step
    assert d.risk_usd == pytest.approx(9.01 * loss_per_unit)
    assert d.risk_pct <= 1.0
    assert d.leverage == pytest.approx(9.01 * 100 / 1000)


def test_short_uses_bid_and_stop_above(tmp_path):
    d = evaluate(manager(tmp_path), OpenIntent("X", "short", stop_loss=101.0))
    assert d.approved and d.entry_price == 99.99


def test_leverage_caps_size_with_a_tight_stop(tmp_path):
    d = evaluate(manager(tmp_path), OpenIntent("X", "long", stop_loss=99.99))
    assert d.qty == pytest.approx(50.0)  # 1000 * 5 / 100
    assert d.leverage == pytest.approx(5.0)
    assert d.risk_pct < 1.0


def test_agent_can_risk_less_but_not_more(tmp_path):
    rm = manager(tmp_path)
    half = evaluate(rm, OpenIntent("X", "long", stop_loss=99.0, risk_pct=0.5))
    full = evaluate(rm, OpenIntent("X", "long", stop_loss=99.0))
    greedy = evaluate(rm, OpenIntent("X", "long", stop_loss=99.0, risk_pct=3.0))
    assert half.risk_pct <= 0.5 < full.risk_pct <= 1.0
    assert greedy.qty == full.qty
    assert not evaluate(rm, OpenIntent("X", "long", stop_loss=99.0, risk_pct=0)).approved


def test_minimum_order_larger_than_budget_is_rejected_not_rounded_up(tmp_path):
    # 100 USDT account, 1% = 1 USDT; BTC min 0.001 with a 4.8% stop risks ~4 USDT
    d = evaluate(
        manager(tmp_path),
        OpenIntent("BTC", "long", stop_loss=80_000),
        equity=100.0,
        rules=BTC,
        bid=83_999.9,
        ask=84_000.0,
    )
    assert not d.approved
    assert "minimum order 0.001" in d.reasons[0]


@pytest.mark.parametrize(
    ("intent", "fragment"),
    [
        (OpenIntent("X", "long", stop_loss=100.5), "must be below"),
        (OpenIntent("X", "short", stop_loss=99.0), "must be above"),
        (OpenIntent("X", "long", stop_loss=99.0, take_profit=99.5), "losing side"),
        (OpenIntent("X", "short", stop_loss=101.0, take_profit=100.5), "losing side"),
    ],
)
def test_stop_and_take_profit_must_be_on_the_right_side(tmp_path, intent, fragment):
    d = evaluate(manager(tmp_path), intent)
    assert not d.approved and fragment in d.reasons[0]


def test_max_open_positions(tmp_path):
    d = evaluate(manager(tmp_path), OpenIntent("X", "long", stop_loss=99.0), positions=[position()])
    assert not d.approved and "Y long" in d.reasons[0]
    flat = position(size=0.0)
    assert evaluate(manager(tmp_path), OpenIntent("X", "long", 99.0), positions=[flat]).approved


def test_trades_per_hour(tmp_path):
    rm = manager(tmp_path, max_trades_per_hour=2)
    rm.record_open(T0 - HOUR_MS + 1_000)
    rm.record_open(T0 - 1_000)
    assert not evaluate(rm, OpenIntent("X", "long", 99.0)).approved
    assert evaluate(rm, OpenIntent("X", "long", 99.0), now=T0 + 2_000).approved  # one aged out


def test_daily_loss_limit_resets_next_utc_day(tmp_path):
    rm = manager(tmp_path, daily_loss_limit_pct=5.0)
    rm.observe_equity(1000.0, T0 - HOUR_MS)  # start of day equity
    assert evaluate(rm, OpenIntent("X", "long", 99.0), equity=960.0).approved
    d = evaluate(rm, OpenIntent("X", "long", 99.0), equity=950.0)
    assert not d.approved and "daily loss 5.00%" in d.reasons[0]
    next_day = evaluate(rm, OpenIntent("X", "long", 99.0), equity=950.0, now=T0 + DAY)
    assert next_day.approved


def test_kill_switch_persists_until_manual_reset(tmp_path):
    rm = manager(tmp_path, max_drawdown_pct=10.0, daily_loss_limit_pct=50.0)
    rm.observe_equity(1000.0, T0)
    rm.observe_equity(1100.0, T0 + DAY)  # new peak
    d = evaluate(rm, OpenIntent("X", "long", 99.0), equity=990.0, now=T0 + 2 * DAY)
    assert not d.approved and "kill switch" in d.reasons[0]

    # survives a restart, and recovering equity does not lift it
    reloaded = manager(tmp_path, max_drawdown_pct=10.0, daily_loss_limit_pct=50.0)
    assert reloaded.state.halted
    assert not evaluate(reloaded, OpenIntent("X", "long", 99.0), equity=1100.0).approved

    reloaded.reset_halt()
    assert evaluate(reloaded, OpenIntent("X", "long", 99.0), equity=990.0).approved
    assert reloaded.state.peak_equity == 990.0


def test_state_file_is_written_atomically(tmp_path):
    rm = manager(tmp_path)
    rm.observe_equity(1000.0, T0)
    assert (tmp_path / "risk.json").exists()
    assert not list(tmp_path.glob("*.tmp"))
    assert RiskStateStore(tmp_path / "risk.json").load().day_start_equity == 1000.0

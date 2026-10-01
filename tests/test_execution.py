import pytest

from enduro.execution.bybit import (
    BybitGateway,
    LiveTradingNotAllowed,
    balance_from_ccxt,
    order_from_ccxt,
    order_params,
    position_from_ccxt,
)
from enduro.execution.models import InstrumentRules, OrderRequest


@pytest.mark.parametrize(
    ("position_side", "action", "side", "idx", "reduce_only"),
    [
        ("long", "open", "buy", 1, False),
        ("long", "close", "sell", 1, True),
        ("short", "open", "sell", 2, False),
        ("short", "close", "buy", 2, True),
    ],
)
def test_hedge_mode_order_mapping(position_side, action, side, idx, reduce_only):
    request = OrderRequest("BTC/USDT:USDT", position_side, action, 0.001, client_order_id="c1")
    params = order_params(request)
    assert request.side == side
    assert params["positionIdx"] == idx
    assert params.get("reduceOnly", False) is reduce_only
    assert params["clientOrderId"] == "c1"


def test_stop_loss_and_take_profit_are_attached_to_opens():
    request = OrderRequest("X", "long", "open", 1, stop_loss=95.0, take_profit=110.0)
    params = order_params(request)
    assert params["stopLoss"] == {"triggerPrice": 95.0}
    assert params["takeProfit"] == {"triggerPrice": 110.0}
    assert params["slTriggerBy"] == params["tpTriggerBy"] == "LastPrice"
    with pytest.raises(ValueError):
        OrderRequest("X", "long", "close", 1, stop_loss=95.0)


def test_order_request_validation():
    with pytest.raises(ValueError):
        OrderRequest("X", "long", "open", 0)
    with pytest.raises(ValueError):
        OrderRequest("X", "long", "open", 1, type="limit")
    assert "clientOrderId" not in order_params(OrderRequest("X", "long", "open", 1))


def test_instrument_rules_rounding():
    rules = InstrumentRules("X", qty_step=0.001, min_qty=0.001, min_notional=5, price_tick=0.1)
    assert rules.round_qty(0.0129) == 0.012  # always down: never order more than asked
    assert rules.round_qty(0.3) == 0.3  # no float artifacts like 0.30000000000000004
    # BTC at 84k: min notional 5 USDT needs 0.0000595 BTC -> rounds up to one step
    assert rules.min_order_qty(84_000) == 0.001
    # a cheap coin: min notional dominates
    cheap = InstrumentRules("Y", qty_step=1, min_qty=1, min_notional=5, price_tick=0.0001)
    assert cheap.min_order_qty(0.2094) == 24  # 5 / 0.2094 = 23.9 -> 24
    exact = InstrumentRules("Z", qty_step=0.1, min_qty=0.1, min_notional=5, price_tick=0.01)
    assert exact.min_order_qty(50) == 0.1  # exactly 5 USDT, no extra step


def test_live_requires_explicit_permission():
    with pytest.raises(LiveTradingNotAllowed):
        BybitGateway("k", "s", environment="live")
    BybitGateway("k", "s", environment="live", allow_live=True)  # constructs without network


def test_demo_uses_demo_endpoints():
    gateway = BybitGateway("k", "s", environment="demo")
    assert "api-demo" in str(gateway._client.urls["api"])


def test_order_from_ccxt():
    order = order_from_ccxt(
        {
            "id": 123,
            "clientOrderId": "c1",
            "symbol": "BTC/USDT:USDT",
            "side": "buy",
            "status": "closed",
            "amount": 0.001,
            "filled": 0.001,
            "average": "84000.5",
            "fee": {"cost": 0.0462, "currency": "USDT"},
            "timestamp": 1,
        }
    )
    assert (order.id, order.status, order.filled, order.avg_price, order.fee) == (
        "123",
        "closed",
        0.001,
        84000.5,
        0.0462,
    )
    bare = order_from_ccxt({"id": "1"})
    assert (bare.status, bare.avg_price, bare.fee) == ("unknown", None, None)


def test_position_from_ccxt():
    pos = position_from_ccxt(
        {
            "symbol": "BTC/USDT:USDT",
            "side": "short",
            "contracts": 0.002,
            "contractSize": 1,
            "entryPrice": 84000,
            "markPrice": 83900,
            "unrealizedPnl": 0.2,
            "leverage": 10,
            "liquidationPrice": "",
        }
    )
    assert (pos.side, pos.size, pos.entry_price, pos.liquidation_price) == (
        "short",
        0.002,
        84000.0,
        None,
    )
    assert position_from_ccxt({"symbol": "X", "side": None}) is None


def test_balance_is_in_usdt_not_usd():
    raw = {
        "info": {
            "result": {
                "list": [
                    {
                        "totalEquity": "999.17322863",  # USD
                        "coin": [{"coin": "USDT", "equity": "1010.5", "usdValue": "1009.7"}],
                    }
                ]
            }
        },
        "USDT": {"free": 990.5, "used": 20.0, "total": 999.93},
    }
    balance = balance_from_ccxt(raw)
    assert balance.equity == 1010.5  # includes unrealized PnL, in USDT
    assert balance.available == 990.5
    assert balance_from_ccxt({"USDT": {"free": 5.0, "total": 7.0}}).equity == 7.0


async def test_unsubscribe_order_books_uses_the_subscribed_depth():
    from enduro.data.ccxt_source import CcxtSource

    source = CcxtSource("bybit", book_limit=1000)
    seen = {}

    async def fake_unwatch(symbols, params=None):
        seen["symbols"], seen["params"] = symbols, params

    source._client.un_watch_order_book_for_symbols = fake_unwatch
    try:
        await source.unsubscribe_order_books(["MOVR/USDT:USDT"])
    finally:
        await source.close()
    assert seen == {"symbols": ["MOVR/USDT:USDT"], "params": {"limit": 1000}}

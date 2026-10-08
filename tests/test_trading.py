"""TradingService against a scripted exchange: chasing limit entries and closes, the
limit take profit, and what happens when an order does not fill."""

import pytest

from enduro.execution.base import OrderNotOpen
from enduro.execution.models import Balance, InstrumentRules, OrderResult, Position
from enduro.journal.journal import Journal
from enduro.risk.manager import OpenIntent, RiskLimits, RiskManager, RiskStateStore
from enduro.trading.service import ChaseSettings, TradingService

SYMBOL = "SOL/USDT:USDT"
RULES = InstrumentRules(SYMBOL, qty_step=0.1, min_qty=0.1, min_notional=5.0, price_tick=0.01)


class Exchange:
    """Fills a resting order on its `fill_on_fetch`-th check; markets fill at the touch."""

    def __init__(self, quotes, fill_on_fetch=None):
        self.quotes = list(quotes)
        self.fill_on_fetch = fill_on_fetch
        self.orders: dict[str, dict] = {}
        self.position: dict | None = None  # {"side", "size", "entry", "stop"}
        self.placed: list = []
        self.amends: list = []
        self.cancels: list = []
        self.environment = "demo"
        self.stopped_at_once = False
        self.reject_post_only = 0  # this many post-only orders are cancelled on arrival

    def touch(self):
        return self.quotes[0] if len(self.quotes) == 1 else self.quotes.pop(0)

    async def quote(self, symbol):
        return self.touch()

    async def balance(self):
        return Balance(1000.0, 1000.0)

    async def positions(self, symbols=None):
        p = self.position
        if not p or p["size"] <= 0:
            return []
        return [
            Position(
                SYMBOL, p["side"], p["size"], p["entry"], p["entry"], 0.0, 5.0, None, p["stop"]
            )
        ]

    async def instrument_rules(self, symbol):
        return RULES

    async def set_leverage(self, symbol, leverage):
        pass

    async def set_protection(self, symbol, side, stop_loss=None, take_profit=None):
        self.position["stop"] = stop_loss

    def _fill(self, order, price):
        order["status"], order["filled"], order["avg"] = "closed", order["qty"], price
        r = order["request"]
        p = self.position
        if r.action == "open" and self.stopped_at_once:
            return  # the exchange stop closed it straight away
        if r.action == "open":
            size = (p["size"] if p else 0.0) + r.qty
            self.position = {
                "side": r.position_side,
                "size": size,
                "entry": price,
                "stop": r.stop_loss,
            }
        else:
            p["size"] = round(p["size"] - r.qty, 10)

    async def place_order(self, request):
        oid = f"o{len(self.orders) + 1}"
        order = {
            "request": request,
            "status": "open",
            "filled": 0.0,
            "avg": None,
            "price": request.price,
            "qty": request.qty,
            "fetches": 0,
        }
        self.orders[oid] = order
        self.placed.append(request)
        if request.post_only and self.reject_post_only > 0:
            self.reject_post_only -= 1
            order["status"] = "canceled"
        if request.type == "market":
            bid, ask = self.touch()
            self._fill(order, ask if request.side == "buy" else bid)
        return self._result(oid)

    def _result(self, oid):
        o = self.orders[oid]
        r = o["request"]
        return OrderResult(
            oid,
            r.client_order_id,
            SYMBOL,
            r.side,
            o["status"],
            o["qty"],
            o["filled"],
            o["avg"],
            0.0002 * o["filled"] * (o["avg"] or 0),
            None,
            o["price"],
        )

    async def fetch_order(self, oid, symbol):
        o = self.orders[oid]
        o["fetches"] += 1
        if o["status"] == "open" and self.fill_on_fetch and o["fetches"] >= self.fill_on_fetch:
            if not o["request"].client_order_id.endswith("-tp"):
                self._fill(o, o["price"])
        return self._result(oid)

    async def wait_for_fill(self, oid, symbol, timeout_s=10.0):
        return self._result(oid)

    async def amend_order(self, oid, symbol, side, price):
        if self.orders[oid]["status"] != "open":
            raise OrderNotOpen(oid)
        self.orders[oid]["price"] = price
        self.amends.append(price)

    async def cancel_order(self, oid, symbol):
        if self.orders[oid]["status"] != "open":
            raise OrderNotOpen(oid)
        self.orders[oid]["status"] = "canceled"
        self.cancels.append(oid)

    async def open_orders(self, symbol=None):
        return [self._result(i) for i, o in self.orders.items() if o["status"] == "open"]


def service(tmp_path, exchange, **chase):
    risk = RiskManager(RiskLimits(), 0.00055, RiskStateStore(tmp_path / "risk.json"))
    settings = ChaseSettings(**{"open_s": 1.0, "close_s": 0.05, "poll_s": 0.001, **chase})
    journal = Journal(tmp_path / "journal")
    return TradingService(exchange, risk, journal, exchange.quote, chase=settings), journal


async def test_entry_chases_the_bid_and_rests_the_take_profit(tmp_path):
    # decision quote, first chase quote, then the bid moves up
    ex = Exchange([(100.0, 100.02), (100.0, 100.02), (100.05, 100.07)], fill_on_fetch=3)
    svc, journal = service(tmp_path, ex)
    result = await svc.open(OpenIntent(SYMBOL, "long", stop_loss=99.0, take_profit=102.0), "x")
    assert result["opened"] is True and result["entry"].startswith("maker")
    entry = ex.placed[0]
    assert (entry.type, entry.post_only, entry.price, entry.stop_loss) == (
        "limit",
        True,
        100.0,
        99.0,
    )
    assert ex.amends == [100.05]  # followed the bid up
    assert result["avg_price"] == 100.05
    tp = ex.placed[-1]
    assert (tp.action, tp.type, tp.price, tp.side) == ("close", "limit", 102.0, "sell")
    assert tp.client_order_id.endswith("-tp") and result["take_profit"] == 102.0
    opened = journal.recent("order", 5)[-1]
    assert opened["action"] == "open" and opened["take_profit"] == 102.0


async def test_entry_is_sized_at_the_chase_bound(tmp_path):
    ex = Exchange([(100.0, 100.02)], fill_on_fetch=1)
    svc, journal = service(tmp_path, ex)
    await svc.open(OpenIntent(SYMBOL, "long", stop_loss=99.0), "x")
    decision = journal.recent("risk", 1)[0]["decision"]
    # bound = ask + 10% of the stop distance = 100.02 + 0.102 -> 100.12 (rounded down)
    assert decision["entry_price"] == pytest.approx(100.12)
    plan = journal.recent("order", 1)[0]["plan"]
    assert plan["chase_bound"] == pytest.approx(100.12)


async def test_runaway_entry_is_cancelled_and_nothing_opens(tmp_path):
    ex = Exchange([(100.0, 100.02), (100.0, 100.02), (100.5, 100.52)])  # never fills
    svc, journal = service(tmp_path, ex)
    result = await svc.open(OpenIntent(SYMBOL, "long", stop_loss=99.0), "x")
    assert result["opened"] is False and "past the chase bound" in result["not_filled"]
    assert ex.position is None and ex.cancels == ["o1"]
    assert journal.recent("order", 1)[0]["action"] == "open_cancelled"
    assert svc.risk.state.opens == []


async def test_unfilled_entry_times_out(tmp_path):
    ex = Exchange([(100.0, 100.02)])
    svc, _ = service(tmp_path, ex, open_s=0.05)
    result = await svc.open(OpenIntent(SYMBOL, "short", stop_loss=101.0), "x")
    assert result["opened"] is False and result["not_filled"].startswith("not filled within")
    assert ex.placed[0].price == 100.02  # a short rests on the ask


async def test_close_cancels_take_profit_then_finishes_at_market(tmp_path):
    ex = Exchange([(100.0, 100.02)], fill_on_fetch=1)
    svc, journal = service(tmp_path, ex)
    await svc.open(OpenIntent(SYMBOL, "long", stop_loss=99.0, take_profit=102.0), "x")
    ex.fill_on_fetch = None  # the closing limit never fills
    result = await svc.close(SYMBOL, "long", reason="thesis broken")
    assert result["closed"] is True and result["maker_qty"] == 0
    close = next(r for r in journal.recent("order", 5) if r["action"] == "close")
    assert len(close["result"]["order_ids"]) >= 2  # the chase order(s) and the market rest
    assert result["market_qty"] == pytest.approx(ex.placed[1].qty)
    assert not [o for o in await ex.open_orders() if o.client_order_id.endswith("-tp")]
    assert ex.placed[-1].type == "market" and ex.placed[-2].post_only


async def test_urgent_close_goes_straight_to_market(tmp_path):
    ex = Exchange([(100.0, 100.02)], fill_on_fetch=1)
    svc, _ = service(tmp_path, ex)
    await svc.open(OpenIntent(SYMBOL, "long", stop_loss=99.0), "x")
    placed = len(ex.placed)
    result = await svc.close(SYMBOL, "long", reason="now", urgent=True)
    assert result["closed"] is True and result["market_qty"] > 0
    assert [r.type for r in ex.placed[placed:]] == ["market"]


async def test_protect_moves_and_removes_the_take_profit(tmp_path):
    ex = Exchange([(100.0, 100.02)], fill_on_fetch=1)
    svc, _ = service(tmp_path, ex)
    await svc.open(OpenIntent(SYMBOL, "long", stop_loss=99.0, take_profit=102.0), "x")
    moved = await svc.protect(SYMBOL, "long", stop_loss=99.5, take_profit=101.5)
    assert moved["updated"] is True and ex.amends[-1] == 101.5
    assert moved["position"]["take_profit"] == 101.5 and moved["position"]["stop_loss"] == 99.5
    removed = await svc.protect(SYMBOL, "long", stop_loss=None, take_profit=0)
    assert removed["position"]["take_profit"] is None


async def test_entry_stopped_out_at_once_is_reported_as_opened(tmp_path):
    ex = Exchange([(100.0, 100.02)], fill_on_fetch=1)
    ex.stopped_at_once = True
    svc, journal = service(tmp_path, ex)
    result = await svc.open(OpenIntent(SYMBOL, "long", stop_loss=99.9), "x")
    assert result["opened"] is True and result["already_closed"] is True
    assert journal.recent("order", 1)[0]["action"] == "open"
    assert len(svc.risk.state.opens) == 1


async def test_rejected_post_only_is_placed_again_a_tick_behind(tmp_path):
    ex = Exchange([(100.0, 100.02)], fill_on_fetch=1)
    ex.reject_post_only = 1
    svc, _ = service(tmp_path, ex)
    result = await svc.open(OpenIntent(SYMBOL, "long", stop_loss=99.0), "x")
    assert result["opened"] is True
    assert [r.price for r in ex.placed[:2]] == [100.0, 99.99]


async def test_repeating_the_same_take_profit_does_not_amend(tmp_path):
    ex = Exchange([(100.0, 100.02)], fill_on_fetch=1)
    svc, _ = service(tmp_path, ex)
    await svc.open(OpenIntent(SYMBOL, "long", stop_loss=99.0, take_profit=102.0), "x")
    result = await svc.protect(SYMBOL, "long", stop_loss=99.5, take_profit=102.0)
    assert result["updated"] is True and "take_profit_error" not in result
    assert ex.amends[-1:] != [102.0]  # no amend to the price it already has

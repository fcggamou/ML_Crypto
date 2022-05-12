import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'bot'))
import factory
import config
from data import Data, BacktestData
from clock import BacktestClock
from exchange import RealExchange, BacktestExchange


factory.create_logger(factory.create_clock())


def test_backtest_buy():
    config.balance_asset = 1
    config.balance_currency = 1000
    clock = BacktestClock(1573666861, 60)
    data = Data(clock)
    ex = BacktestExchange(clock, data)

    ex.buy(0.01)
    assert ex.available_asset() == 1.01
    assert ex.available_currency() == 1000 - (0.01 * ex.ticker() * (1 + config.fee))


def test_backtest_buy_by_currency():
    config.balance_asset = 1
    config.balance_currency = 1000
    clock = BacktestClock(1573666861, 60)
    data = Data(clock)
    ex = BacktestExchange(clock, data)

    ex.buy(amount_currency=100)
    assert abs(ex.available_currency() - 900) < 0.1
    assert round(ex.available_asset(), 2) == round(1 + (100 / ex.ticker()) * (1 - config.fee), 2)


def test_backtest_sell():
    config.balance_asset = 1
    config.balance_currency = 1000
    clock = BacktestClock(1573666861, 60)
    data = Data(clock)
    ex = BacktestExchange(clock, data)

    ex.sell(0.01)
    assert ex.available_asset() == 1 - (0.01 * (1 + config.fee))
    assert ex.available_currency() == 1000 + (0.01 * ex.ticker())


def test_backtest_sell_by_currency():
    config.balance_asset = 1
    config.balance_currency = 1000
    clock = BacktestClock(1573666861, 60)
    data = Data(clock)
    ex = BacktestExchange(clock, data)

    ex.sell(amount_currency=100)
    assert abs(ex.available_currency() == 1100) < 0.1
    assert abs(ex.available_asset() - (1 - (100 / ex.ticker()) * (1 + config.fee))) < 0.1


def test_real_balance():
    ex = RealExchange()
    assert ex.available_asset() >= 0
    assert ex.available_currency() >= 0


def test_real_ticker():
    ex = RealExchange()
    assert ex.ticker() is not None


def test_backtest_ticker():
    config.granularity = '30min'
    config.granularity_s = 30 * 60

    clock = BacktestClock(1610974800, 60)  # 18/12/2020
    data = BacktestData(clock)

    ex = BacktestExchange(clock, data)

    assert abs(ex.ticker() - 36981.4) < 0.01

    for _ in range(0, 30):
        clock.tick()
    assert abs(ex.ticker() - 36808.99) < 0.01

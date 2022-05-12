import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from bot.clock import BacktestClock, RealClock


def test_backtest_clock():
    start_time = 1605281379
    tick_interval_s = 60
    cl = BacktestClock(start_time, tick_interval_s)
    cl.tick()

    assert cl.get_time() == start_time + tick_interval_s

    cl.tick()
    assert cl.get_time() == start_time + tick_interval_s * 2


def test_real_clock():
    tick_interval_s = 1
    cl = RealClock(tick_interval_s)
    start_time = cl.get_time()

    cl.tick()

    assert cl.get_time() == start_time + tick_interval_s

    cl.tick()
    assert cl.get_time() == start_time + tick_interval_s * 2

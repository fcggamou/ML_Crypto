import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'bot'))
import config
from clock import RealClock
from data import Data


def test_candles():
    config.granularity = '5min'
    data = Data(RealClock(60))
    candle = data.candles().loc[1608320700]
    assert candle['Open'] - 22698.99 < 0.01
    assert candle['Close'] - 22721.24 < 0.01
    assert candle['High'] - 22730.98 < 0.01
    assert candle['Low'] - 22695.83 < 0.01

# def test_update_data():
#     import dataloader
#     config.granularity = '30min'
#     config.update_data = True
#     data = Data(RealClock(60))
#     candles = data.candles.copy()
#     candles['ts'] = candles.index.values
#     candles['ts_diff'] = candles['ts'].diff()
#     all_data = dataloader.get_all_data('BTCUSDT', '30min')
#     all_data['ts'] = all_data.index.values
#     all_data['ts_diff'] = all_data['ts'].diff()
#     assert all(candles['ts_diff'].values[-5000:] == 1800)

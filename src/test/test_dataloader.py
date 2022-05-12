import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import dataloader
import file_utils


def test_dataloader_update_data():
    data_1h = file_utils.load_from_file(os.path.join('data', 'BTCUSDT_1h_test.pkl'))
    last_candle_ts = data_1h.index.values[-1]
    len_candles = len(data_1h)
    data_1h = dataloader.update_data(data_1h, 'BTCUSDT', '1h')
    for i in range(0, len(data_1h) - len_candles):
        assert data_1h.loc[last_candle_ts + 60 * 60 * i] is not None

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from clock import BaseClock
import config
from file_utils import load_from_file, dump_to_file
from dataloader import update_data, get_all_data
import time_utils


class Data():

    def __init__(self, clock: BaseClock):
        self.granularity = config.granularity
        self.granularity_models = config.granularity_models
        self._clock = clock
        self.symbol = config.symbol

        self._candles = self.load_data_file(self.symbol, self.granularity)
        self._candles_model = self.load_data_file(self.symbol, self.granularity_models)

    def load_data_file(self, symbol: str, granularity: str):
        file_path = '{}_{}.pkl'.format(symbol, granularity)

        if not os.path.exists(os.path.join(config.data_path, file_path)):
            data = get_all_data(self.symbol, self.granularity)
            dump_to_file(data, os.path.join(config.data_path, file_path))
        else:
            data = load_from_file(os.path.join(config.data_path, file_path))
            if config.update_data:
                data = self.update_data(data, granularity)
                dump_to_file(data, os.path.join(config.data_path, file_path))
        return data

    def update_data(self, data, granularity: str):
        if self._clock.get_time() - data.index.values[-1] >= time_utils.granularityStrToSeconds(granularity) * 2:
            data = update_data(data, self.symbol, granularity)
        return data

    def tick(self):
        self._candles = self.update_data(self._candles, self.granularity)
        self._candles_model = self.update_data(self._candles_model, self.granularity_models)

    def last_candle(self):
        return self._get_last_candle(self._candles)

    def candles(self):
        return self._candles

    def last_candle_models(self):
        return self._get_last_candle(self.candles_models())

    def candles_models(self):
        return self._candles_model

    def _get_last_candle(self, data):
        return data.iloc[-1:]


class BacktestData(Data):

    def __init__(self, clock: BaseClock):
        super().__init__(clock)
        self._last_candle_models = None

    def candles(self):
        return self._candles.loc[:self._clock.get_time()]

    def candles_models(self):
        return self._candles_model.loc[:self._clock.get_time()]

    def last_candle(self):
        return self._get_last_candle(self._candles, config.granularity_s)

    def last_candle_models(self):
        new_candle = self._get_last_candle(self.candles_models(), config.granularity_models_s)
        if len(new_candle) > 0:
            self._last_candle_models = new_candle
        return self._last_candle_models

    def _get_last_candle(self, data, granularity_s):
        loc = int(self._clock.get_time())
        return data.loc[loc:loc + granularity_s]

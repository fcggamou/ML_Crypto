import config
from data import Data
import logger
from ml.prediction import Prediction
from ml.model import load_all_models
from typing import List
from itertools import chain
import numpy as np


class Predictor():

    def __init__(self, data: Data):
        self.models = load_all_models(config.models_path)
        self.latest_predictions: List[Prediction] = []
        self._last_candle = None
        self._data = data

    def tick(self):
        last_candle = self._data.last_candle_models()
        if self._last_candle is None or len(self._last_candle == 0) or self._last_candle.index.values[0] != last_candle.index.values[0]:
            self._last_candle = last_candle
            self.generate_predictions()

    def generate_predictions(self):
        self.latest_predictions = []
        for model in self.models:
            prediction = model.predict_last(self._data.candles_models())
            self.latest_predictions.append(prediction)

        logger.logger.log_prediction(self.latest_predictions, self._data.last_candle_models())


class BacktestPredictor(Predictor):

    def __init__(self, data: Data):
        super().__init__(data)
        self.generate_all_predictions()
        self._all_predictions = self.generate_all_predictions()

    def generate_all_predictions(self):
        data = self._data._candles_model.loc[config.start_time:config.end_time]

        dics = []
        for model in self.models:
            preds = model.predict_all(data)
            dics.append({x.timestamp: x for x in preds})

        keys = set(chain(*[d.keys() for d in dics]))
        return {k: [d.get(k, np.nan) for d in dics] for k in keys}

    def generate_predictions(self):
        self.latest_predictions = self._all_predictions.get(self._data.last_candle_models().index.values[-1], [])
        if len(self.latest_predictions) > 0:
            logger.logger.log_prediction(self.latest_predictions, self._data.last_candle_models())

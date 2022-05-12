from tradingview_ta import TA_Handler, Interval, Exchange
from abc import ABC, abstractmethod
from exchange import BaseExchange
import random
import config
from data import Data
from predictor import Predictor, Prediction
import logger


class BaseStrategy(ABC):

    def __init__(self, exchange: BaseExchange, data: Data):
        self._exchange = exchange
        self._data = data

    @abstractmethod
    def tick(self):
        pass


class RandomStrategy(BaseStrategy):

    def tick(self):
        if random.random() < 0.5:
            self._exchange.buy(amount_currency=self._exchange.available_currency() * 0.95)
        else:
            self._exchange.sell(amount_asset=self._exchange.available_asset() * (1 - config.fee) * 0.95)


class BuyAndHoldStrategy(BaseStrategy):

    def __init__(self, exchange: BaseExchange, data: Data):
        super().__init__(exchange, data)
        self._bought = False

    def tick(self):
        if not self._bought:
            self._exchange.buy(amount_currency=self._exchange.available_currency())
            self._bought = True


class SimpleStrategy(BaseStrategy):

    def __init__(self, exchange: BaseExchange, data: Data, predictor: Predictor):
        super().__init__(exchange, data)
        self._predictor = predictor
        self._current_trade = None
        self._last_predictions = []
        self._ticker = self._exchange.ticker()
        self._last_order_ticker = None
        self._high_target = None
        self._low_target = None
        self._patience_counter_s = config.strat_patience_s
        self._data = data
        self._last_candle = None

    def tick(self):
        self._ticker = self._exchange.ticker()
        self.check_new_predictions()
        self._patience_counter_s = self._patience_counter_s - config.tick_interval_s
        if not self.is_new_candle() and self._current_trade is None:
            return
        if self._current_trade == 'long' or self._current_trade is None:
            self.sell_if_on_target()
        if self._current_trade == 'short' or self._current_trade is None:
            self.buy_if_on_target()

    def is_new_candle(self):
        last_candle = self._data.last_candle()
        if self._last_candle is None or self._last_candle.index.values[0] != last_candle.index.values[0]:
            self._last_candle = last_candle
            return True
        return False

    def check_new_predictions(self):
        if len(self._last_predictions) == 0 or self._last_predictions != self._predictor.latest_predictions:
            self._last_predictions = self._predictor.latest_predictions
            if len(self._last_predictions) > 0:
                self.get_targets()

    def get_targets(self):
        if self._current_trade == 'long' and self._patience_counter_s > 0:
            self._high_target = self._last_order_ticker * (1 + config.strat_profit_target)
            self._low_target = None
        elif self._current_trade == 'short' and self._patience_counter_s > 0:
            self._low_target = self._last_order_ticker * (1 - config.strat_profit_target)
            self._high_target = None
        else:
            self._high_target = self._data.last_candle_models()['Close'].values[-1] * self.high_prediction().value * (1 - config.strat_tolerance)
            if self.low_prediction() is not None:
                self._low_target = self._data.last_candle_models()['Close'].values[-1] * self.low_prediction().value * (1 + config.strat_tolerance)

        if self._high_target is not None:
            logger.logger.log("High target: {:.2f}".format(self._high_target))
        if self._low_target is not None:
            logger.logger.log("Low target: {:.2f}".format(self._low_target))

    def low_prediction(self) -> Prediction:
        preds = [x for x in self._last_predictions if x.variable == 'Low']
        if len(preds) > 0:
            return preds[0]
        return None        

    def high_prediction(self) -> Prediction:
        return [x for x in self._last_predictions if x.variable == 'High'][0]

    def sell_if_on_target(self):
        if self._high_target is not None and self._ticker >= self._high_target:
            self._exchange.sell(amount_currency=config.trade_amount_currency)
            self._current_trade = 'short' if self._current_trade is None else None
            self.open_close_trade()

    def buy_if_on_target(self):
        if self._low_target is not None and self._ticker <= self._low_target:
            self._exchange.buy(amount_currency=config.trade_amount_currency)
            self._current_trade = 'long' if self._current_trade is None else None
            self.open_close_trade()

    def open_close_trade(self):
        self._last_order_ticker = self._ticker
        self._patience_counter_s = config.strat_patience_s
        self.get_targets()


class SimpleStrategyClassification(SimpleStrategy):

    def __init__(self, exchange: BaseExchange, data: Data, predictor: Predictor, profit_target: float, fpr_treshold: float):
        super().__init__(exchange, data, predictor)
        self.profit_target = profit_target
        self.fpr_treshold = fpr_treshold

    def get_targets(self):

        if self._current_trade == 'long':
            if self._patience_counter_s > 0 and not self.stop_loss_hit():
                self._high_target = self._last_order_ticker * (1 + config.strat_profit_target)
            else:
                self._high_target = self._ticker
            self._low_target = None
        elif self._current_trade == 'short':
            if self._patience_counter_s > 0 and not self.stop_loss_hit():
                self._low_target = self._last_order_ticker * (1 - config.strat_profit_target)
            else:
                self._low_target = self._ticker
            self._high_target = None
        else:
            if self._current_trade is None:
                high_signal = self.high_prediction() is not None and 1 - self.high_prediction().fpr > self.fpr_treshold
                low_signal = self.low_prediction() is not None and 1 - self.low_prediction().fpr > self.fpr_treshold
                if high_signal and not low_signal:
                    # self._high_target =
                    # self._data.last_candle_models()['Close'].values[-1] * (1
                    # + self.  profit_target)
                    self._high_target = None
                    self._low_target = self._data.last_candle_models()['Close'].values[-1]
                elif low_signal and not high_signal:
                    # self._low_target =
                    # self._data.last_candle_models()['Close'].values[-1] * (1
                    # - self.  profit_target)
                    self._low_target = None
                    self._high_target = self._data.last_candle_models()['Close'].values[-1]
                else:
                    self._low_target = None
                    self._high_target = None

        if self._high_target is not None:
            logger.logger.log("High target: {:.2f}".format(self._high_target))
        if self._low_target is not None:
            logger.logger.log("Low target: {:.2f}".format(self._low_target))

    def stop_loss_hit(self):
        if config.stop_loss is not None:
            if self._current_trade == 'long':
                return self._last_order_ticker / self._ticker - 1 > config.stop_loss
            else:
                return self._ticker / self._last_order_ticker - 1 > config.stop_loss
        return False



class OnlyLong(BaseStrategy):

    def __init__(self, exchange: BaseExchange, data: Data, predictor: Predictor, profit_target: float, fpr_treshold: float):
        super().__init__(exchange, data)
        self._predictor = predictor
        self._current_trade = None
        self._last_predictions = []
        self._ticker = self._exchange.ticker()
        self._last_order_ticker = None
        self._high_target = None
        self._data = data
        self._last_candle = None
        self.profit_target = profit_target
        self.fpr_treshold = fpr_treshold
        self._patience = config.strat_patience_s

    def tick(self):
        self._ticker = self._exchange.ticker()
        self.check_new_predictions()          
        if not self.is_new_candle() and self._current_trade is None:
            return
        if self._current_trade == 'long' or self._current_trade is None:
            self._patience = self._patience - config.tick_interval_s
            self.sell_if_on_target()

    def is_new_candle(self):
        last_candle = self._data.last_candle()
        if self._last_candle is None or self._last_candle.index.values[0] != last_candle.index.values[0]:
            self._last_candle = last_candle
            return True
        return False

    def check_new_predictions(self):
        if len(self._last_predictions) == 0 or self._last_predictions != self._predictor.latest_predictions:
            self._last_predictions = self._predictor.latest_predictions
            if len(self._last_predictions) > 0:
                self.get_targets()

    def get_targets(self):
        if self._high_target is None:
            if self.high_prediction() is not None and 1 - self.high_prediction().fpr > self.fpr_treshold:
                self._high_target = self._data.last_candle_models()['Close'].values[-1] * (1 + self. profit_target)
                self._current_trade = 'long'
                self.buy()

        if self._high_target is not None:
            logger.logger.log("High target: {:.2f}".format(self._high_target))        

    def high_prediction(self) -> Prediction:
        return [x for x in self._last_predictions if x.variable == 'High'][0]

    def sell_if_on_target(self):
        if self._high_target is not None and (self._ticker >= self._high_target or self._patience <= 0):
            self._exchange.sell(amount_currency=config.trade_amount_currency)
            self._current_trade = 'short' if self._current_trade is None else None
            self.open_close_trade()
            self._high_target = None
            self._patience = config.strat_patience_s

    def buy(self):
        self._patience = config.strat_patience_s
        self._exchange.buy(amount_currency=config.trade_amount_currency)        
        self.open_close_trade()

    def open_close_trade(self):
        self._last_order_ticker = self._ticker                



class NoiseLong(BaseStrategy):

    def __init__(self, exchange: BaseExchange, data: Data, profit_target: float, stop_loss: float):
        super().__init__(exchange, data)
        self._current_trade = None
        self._ticker = self._exchange.ticker()
        self._last_order_ticker = None
        self._high_target = None
        self._data = data
        self._last_candle = None
        self._profit_target = profit_target        
        self._patience = config.strat_patience_s
        self._stop_loss = stop_loss

    def tick(self):
        self._ticker = self._exchange.ticker()
        
        if not self.is_new_candle() and self._current_trade is None:
            return
        if self._current_trade is None:
            self.init_trade()
        else:
            self._patience = self._patience - config.tick_interval_s
            self.sell_if_on_target()

    def is_new_candle(self):
        last_candle = self._data.last_candle()
        if self._last_candle is None or self._last_candle.index.values[0] != last_candle.index.values[0]:
            self._last_candle = last_candle
            return True
        return False

    def init_trade(self):
        if self._high_target is None:          
            self._high_target = self._data.last_candle()['Close'].values[-1] * (1 + self._profit_target)
            self._current_trade = 'long'
            self.buy()
            self._stop_target = self._data.last_candle()['Close'].values[-1] * (1 + self._stop_loss)

        if self._high_target is not None:
            logger.logger.log("High target: {:.2f}".format(self._high_target))

    def sell_if_on_target(self):
        if self._high_target is not None and (self._ticker >= self._high_target or self._ticker <= self._stop_target):
            if self._ticker >= self._high_target:
                print("HIT")
            elif self._ticker <= self._stop_target:
                print("STOP LOSS")
            self._exchange.sell(amount_currency=config.trade_amount_currency)
            self._current_trade = 'short' if self._current_trade is None else None
            self.open_close_trade()
            self._high_target = None
            self._patience = config.strat_patience_s

    def buy(self):
        self._patience = config.strat_patience_s
        self._exchange.buy(amount_currency=config.trade_amount_currency)        
        self.open_close_trade()

    def open_close_trade(self):
        self._last_order_ticker = self._ticker    


class Technical(BaseStrategy):
    

    def __init__(self, exchange: BaseExchange, data: Data, profit_target: float, stop_loss: float):
        super().__init__(exchange, data)
        self._current_trade = None
        self._ticker = self._exchange.ticker()
        self._last_order_ticker = None
        self._high_target = None
        self._data = data
        self._last_candle = None
        self._profit_target = profit_target                
        self._stop_loss = stop_loss
        self._ta = TA_Handler(symbol="BTCUSDT",exchange="binance",screener="crypto",interval=Interval.INTERVAL_15_MINUTES)

    def tick(self):
        self._ticker = self._exchange.ticker()
        self._anal = self._ta.get_analysis().moving_averages['RECOMMENDATION']
        print(self._anal)
        if not self.is_new_candle() and self._current_trade is None:
            return
        if self._current_trade is None:
            self.init_trade()
        else:            
            self.sell_if_on_target()

    def is_new_candle(self):
        last_candle = self._data.last_candle()
        if self._last_candle is None or self._last_candle.index.values[0] != last_candle.index.values[0]:
            self._last_candle = last_candle
            return True
        return False

    def init_trade(self):                        
        if self._high_target is None:                      
            if (self._anal == 'STRONG_BUY'):
                self._high_target = self._exchange.ticker() * (1 + self._profit_target)
                self._current_trade = 'long'
                self.buy()
                self._stop_target = self._exchange.ticker() * (1 + self._stop_loss)

        if self._high_target is not None:
            logger.logger.log("High target: {:.2f}".format(self._high_target))

    def sell_if_on_target(self):
        if self._high_target is not None and (self._ticker >= self._high_target or self._ticker <= self._stop_target):
            if self._ticker >= self._high_target:
                print("HIT")
            elif self._ticker <= self._stop_target:
                print("STOP LOSS")
            self._exchange.sell(amount_currency=config.trade_amount_currency)
            self._current_trade = 'short' if self._current_trade is None else None
            self.open_close_trade()
            self._high_target = None
            self._patience = config.strat_patience_s

    def buy(self):
        self._patience = config.strat_patience_s
        self._exchange.buy(amount_currency=config.trade_amount_currency)        
        self.open_close_trade()

    def open_close_trade(self):
        self._last_order_ticker = self._ticker    

class DummyStrategy(BaseStrategy):

    def tick(self):
        pass

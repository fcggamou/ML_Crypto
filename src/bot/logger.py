from abc import ABC, abstractmethod
import config
from enums import OrderType
from clock import BaseClock
import telegram.ext
from ml.prediction import Prediction
from datetime import datetime
import numpy as np
import file_utils
import os


class BaseLogger(ABC):

    def __init__(self, clock: BaseClock):
        self._orders = []
        self._predictions = {}
        self._current_order = None
        self._trade_profits = []
        self._clock = clock

    @abstractmethod
    def log(self, str):
        pass

    def log_trade(self, order_type: OrderType, amount: float, price: float):
        if self._current_order is None:
            self._current_order = (order_type, amount, price)
        else:
            if order_type == OrderType.BUY:
                profit = (self._current_order[2] / price - 1) * 100
            else:
                profit = (price / self._current_order[2] - 1) * 100
            profit = profit - config.fee * 200
            self._trade_profits.append(profit)
            self._current_order = None

    def log_order(self, order_type: OrderType, amount: float, price: float):
        time = self._clock.get_time()
        self._orders.append((order_type, amount, price, time))
        self.log("{} {:.2f} {:.2f}".format(order_type, amount, price))
        self.log_trade(order_type, amount, price)


    def log_candle(self, candle):
        current_close, current_high, current_low = candle['Close'].values[0], candle['High'].values[0], candle['Low'].values[0]
        self.log("<b>Current:</b>\nClose: {:.2f}\nHigh: {:.2f}\nLow: {:.2f}\n".format(
            current_close, current_high, current_low))

    def log_prediction(self, predictions: Prediction, candle):
        
        current_close = candle['Close'].values[0]
        self.log_candle(candle)
        for pred in predictions:
            print("FPR: " + str(pred.fpr))
        #pred_low: Prediction = 0#[x for x in predictions if x.variable == 'Low'][0]
        ##pred_close: Prediction = [x for x in predictions if x.variable == 'Close'][0]

        #pred_value_high = pred_high.value
        #pred_value_low = 0#pred_low.value
        ##pred_value_close = pred_close.value

        #pred_value_high_per = (pred_high.value - 1) * 100
        #pred_value_low_per = 0#(pred_low.value - 1) * 100
        ##pred_value_close_per = (pred_close.value - 1) * 100

        #date = datetime.fromtimestamp(pred_high.timestamp).strftime('%H:%M')
        #target_date = datetime.fromtimestamp(pred_high.target_timestamp).strftime('%H:%M')
        #self.log_candle(candle)
        #self.log("<b>Prediction:\n{} - {}</b>\nHigh: {:+.2f}%. {:.2f}\nLow: {:+.2f}%. {:.2f}\n".format(date, target_date, pred_value_high_per,
        #                                                                                               current_close * pred_value_high, pred_value_low_per, current_close * pred_value_low))
        # self.log("<b>Prediction:\n{} - {}</b>\nHigh: {:+.2f}%. {:.2f}\nLow: {:+.2f}%. {:.2f}\nClose: {:+.2f}%. {:.2f}\n".format(date, target_date, pred_value_high_per,
        #                                                                                                                        current_close * pred_value_high, pred_value_low_per, current_close * pred_value_low, pred_value_close_per, current_close * pred_value_close))

    def log_result(self, ticker: float, balance_asset: float, balance_currency: float):

        initial_asset_balance = config.balance_asset
        initial_currency_balance = config.balance_currency

        total_initial_asset_balance = config.balance_asset + config.balance_currency / ticker
        total_initial_currency_balance = config.balance_currency + config.balance_asset * ticker

        profit_asset = (balance_asset / initial_asset_balance - 1) * 100
        profit_currency = (balance_currency / initial_currency_balance - 1) * 100

        total_asset = balance_asset + balance_currency / ticker
        total_currency = balance_currency + balance_asset * ticker

        profit_total_asset = (total_asset / total_initial_asset_balance - 1) * 100
        profit_total_currency = (total_currency / total_initial_currency_balance - 1) * 100

        mean_profit_per_trade = np.mean(self._trade_profits)
        total_profit_per_trade = np.sum(self._trade_profits)
        self.log("{}\n".format(datetime.utcfromtimestamp(self._clock.get_time()).strftime('%Y-%m-%d %H:%M:%S')))

        self.log("Asset balance: {:.2f}. {:+.2f}%".format(balance_asset, profit_asset))
        self.log("Currency balance: {:.2f}. {:+.2f}%".format(balance_currency, profit_currency))

        self.log("Total Asset balance: {:.2f}. {:+.2f}%".format(total_asset, profit_total_asset))
        self.log("Total currency balance: {:.2f}. {:+.2f}%".format(total_currency, profit_total_currency))

        self.log("Mean profit per trade: {:+.2f}%".format(mean_profit_per_trade))
        self.log("Total profit per trade: {:+.2f}%".format(total_profit_per_trade))

    def dump_results(self):
        config_kv = [(x, config.__dict__[x]) for x in dir(config) if not x.startswith('_')]
        config_kv = {k: v for (k, v) in config_kv if type(v) is str or type(v) is int or type(v) is float}
        file_utils.dump_to_file([self._orders, config_kv], os.path.join(config.results_folder, config.results_path))


class ConsoleLogger(BaseLogger):

    def log(self, text):
        print(text)


class SilentConsoleLogger(BaseLogger):

    def log(self, text):
        return

    def log_result(self, ticker: float, balance_asset: float, balance_currency: float):

        initial_asset_balance = config.balance_asset
        initial_currency_balance = config.balance_currency

        total_initial_asset_balance = config.balance_asset + config.balance_currency / ticker
        total_initial_currency_balance = config.balance_currency + config.balance_asset * ticker

        profit_asset = (balance_asset / initial_asset_balance - 1) * 100
        profit_currency = (balance_currency / initial_currency_balance - 1) * 100

        total_asset = balance_asset + balance_currency / ticker
        total_currency = balance_currency + balance_asset * ticker

        profit_total_asset = (total_asset / total_initial_asset_balance - 1) * 100
        profit_total_currency = (total_currency / total_initial_currency_balance - 1) * 100

        mean_profit_per_trade = np.mean(self._trade_profits)
        total_profit_per_trade = np.sum(self._trade_profits)

        print("{}\n".format(datetime.utcfromtimestamp(self._clock.get_time()).strftime('%Y-%m-%d %H:%M:%S')))

        print("Asset balance: {:.2f}. {:+.2f}%".format(balance_asset, profit_asset))
        print("Currency balance: {:.2f}. {:+.2f}%".format(balance_currency, profit_currency))

        print("Total Asset balance: {:.2f}. {:+.2f}%".format(total_asset, profit_total_asset))
        print("Total currency balance: {:.2f}. {:+.2f}%".format(total_currency, profit_total_currency))

        print("Mean profit per trade: {:+.2f}%".format(mean_profit_per_trade))
        print("Total profit per trade: {:+.2f}%".format(total_profit_per_trade))


class TelegramLogger(ConsoleLogger):
    def __init__(self, clock: BaseClock):
        super().__init__(clock)
        self.bot = telegram.Bot(config.telegram_token)

    def log(self, text):
        super().log(text)
        self.bot.send_message(chat_id=config.telegram_chat_id, text=str(text), parse_mode="HTML")


logger: BaseLogger = None

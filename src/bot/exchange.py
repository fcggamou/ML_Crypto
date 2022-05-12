from abc import ABC, abstractmethod
from binance.client import Client
from clock import BaseClock
from data import Data
import config
import logger
from enums import OrderType


class BaseExchange(ABC):

    @abstractmethod
    def ticker(self):
        pass

    @abstractmethod
    def _execute_buy(self, amount_asset):
        pass

    @abstractmethod
    def _execute_sell(self, amount_asset):
        pass

    @abstractmethod
    def available_asset(self):
        pass

    @abstractmethod
    def available_currency(self):
        pass

    def buy(self, amount_asset=None, amount_currency=None):
        amount_asset = self.validate_buy(amount_asset, amount_currency)
        self._execute_buy(amount_asset)
        logger.logger.log_order(OrderType.BUY, amount_asset, self.ticker())

    def limit_buy(self, price, amount_asset=None, amount_currency=None):
        price = round(price, 2)
        amount_asset = self.validate_buy(amount_asset, amount_currency, price)
        self._create_limit_buy(price, amount_asset)        
        logger.logger.log_order(OrderType.BUY, amount_asset, price)

    def limit_sell(self, price, amount_asset=None, amount_currency=None):
        price = round(price, 2)
        amount_asset = self.validate_sell(amount_asset, amount_currency, price)
        self._create_limit_sell(price, amount_asset)
        logger.logger.log_order(OrderType.SELL, amount_asset, price)

    def validate_buy(self, amount_asset=None, amount_currency=None, price=None) -> (float, float):
        if price is None:
            price = self.ticker()
        if amount_asset is None and amount_currency is None or amount_asset is not None and amount_currency is not None:
            raise Exception('Only one of amount_asset and amount_currency can be specified')
        if amount_currency is not None:
            amount_asset = amount_currency / ((1 + self._fee) * price)

        cost = amount_asset * self.ticker() * (1 + self._fee)
        if self.available_currency() < cost:
            raise Exception('Not enough funds')
        return round(amount_asset, 6)

    def sell(self, amount_asset=None, amount_currency=None):
        amount_asset = self.validate_sell(amount_asset, amount_currency)
        self._execute_sell(amount_asset)
        logger.logger.log_order(OrderType.SELL, amount_asset, self.ticker())

    def validate_sell(self, amount_asset=None, amount_currency=None, price=None) -> (float, float):
        if price is None:
            price = self.ticker()
        if amount_asset is None and amount_currency is None or amount_asset is not None and amount_currency is not None:
            raise Exception('Only one of amount_asset and amount_currency can be specified')

        if amount_currency is not None:
            amount_asset = amount_currency / price

        cost = amount_asset * (1 + self._fee)
        if self.available_asset() < cost:
            raise Exception('Not enough funds')
        return round(amount_asset, 6)


class RealExchange(BaseExchange):
    def __init__(self):
        self.client = Client(config.api_key, config.api_secret)
        self._fee = config.fee

    def ticker(self):
        return float(self.client.get_symbol_ticker(symbol=config.symbol)['price'])

    def available_asset(self):
        return float(self.client.get_asset_balance(config.asset)['free'])

    def available_currency(self):
        return float(self.client.get_asset_balance(config.currency)['free'])

    def _execute_buy(self, amount_asset):
        self.client.order_market_buy(symbol=config.symbol, quantity=amount_asset)

    def _create_limit_buy(self, price, amount_asset):
        self.client.order_limit_buy(symbol=config.symbol, quantity=amount_asset, price=price)

    def _create_limit_sell(self, price, amount_asset):
        self.client.order_limit_sell(symbol=config.symbol, quantity=amount_asset, price=price)

    def _execute_sell(self, amount_asset):
        self.client.order_market_sell(symbol=config.symbol, quantity=amount_asset)


class BacktestExchange(BaseExchange):

    def __init__(self, clock: BaseClock, data: Data):
        self.clock = clock
        self._data = data
        self._available_asset = config.balance_asset
        self._available_currency = config.balance_currency
        self._fee = config.fee

    def ticker(self):
        return self._data.last_candle()['Close'].values[-1]

    def available_asset(self):
        return self._available_asset

    def available_currency(self):
        return self._available_currency

    def _execute_buy(self, amount_asset):
        cost = amount_asset * self.ticker() * (1 + self._fee)
        self._available_currency = self.available_currency() - cost
        self._available_asset = self.available_asset() + amount_asset

    def _execute_sell(self, amount_asset):
        self._available_asset = self.available_asset() - (amount_asset * (1 + self._fee))
        self._available_currency = self.available_currency() + (amount_asset * self.ticker())


class PaperExchange(BacktestExchange):
    def __init__(self):
        self.client = Client(config.api_key, config.api_secret)
        self._fee = config.fee
        self._available_asset = config.balance_asset
        self._available_currency = config.balance_currency

    def ticker(self):
        return float(self.client.get_symbol_ticker(symbol=config.symbol)['price'])

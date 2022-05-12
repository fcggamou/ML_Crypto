from exchange import BaseExchange
import config
from data import Data
import factory


class Shell():

    def __init__(self):
        self._clock = factory.create_clock()
        self._data = factory.create_data(self._clock)
        self._exchange = factory.create_exchange(self._clock, self._data)
        factory.create_logger(self._clock)

    def start(self):
        while(True):
            command = input("Command: ")
            if command == 'long':
                self.long()
            elif command == 'short':
                self.short()
            else:
                print("Invalid command")


    def long(self):
        self._exchange.buy(amount_currency=config.trade_amount_currency)  
        self._high_target = self._exchange.ticker() * (1 + config.strat_profit_target)
        self._exchange.limit_sell(price=self._high_target, amount_currency=config.trade_amount_currency)

    def short(self):
        self._exchange.sell(amount_currency=config.trade_amount_currency)  
        self._low_target = self._exchange.ticker() * (1 - config.strat_profit_target)
        self._exchange.limit_buy(price=self._low_target, amount_currency=config.trade_amount_currency)
    
        

if __name__ == '__main__':
    shell = Shell()
    shell.start()
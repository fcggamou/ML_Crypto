import tensorflow as tf
import config
import factory
import logger
import warnings
from pandas.core.common import SettingWithCopyWarning
import signal
import atexit
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
tf.get_logger().setLevel('ERROR')


# # # High
# DONE Start / end dates instead of ts
# DONE Train notebook for single model with date
# Resumen de buenos y malos trades en log, indicar cuantas señales al arrancar la simulación
# Improve performance


# Comparar resultados de modelo con corrida de bot y ver si matchea
# Order class
# Timeout on realexchange.ticker
# Test for bot, with dummy model and dummy strat. Check candles match, buy/sell and balance match, final balance and summary match etc.


# # # Low
# Limit orders
# Retry after disconnect telegram, realexchange, etc.


class Bot():
    def __init__(self):
        self.clock = factory.create_clock()
        self.data = factory.create_data(self.clock)
        self.exchange = factory.create_exchange(self.clock, self.data)
        #self.predictor = factory.create_predictor(self.data)        
        self.predictor = None
        self.strategy = factory.create_strat(self.exchange, self.data, self.predictor)
        factory.create_logger(self.clock)

    def trade(self):
        i = 0
        while (True):
            self.data.tick()
            #self.predictor.tick()
            self.strategy.tick()

            self.clock.tick()
            while len(self.data.last_candle()) == 0:
                self.clock.tick()
                print('Fast forwarding {}'.format(self.clock.get_time()))
                if self.should_stop():
                    break
            if self.should_stop():
                break

            if i % config.log_result_ticks == 0:
                logger.logger.log_result(self.exchange.ticker(), self.exchange.available_asset(),
                                         self.exchange.available_currency())

            i = i + 1

    def should_stop(self):
        return config.end_time is not None and self.clock.get_time() >= config.end_time


def handle_exit(bot):
    logger.logger.log_result(bot.exchange.ticker(), bot.exchange.available_asset(),
                             bot.exchange.available_currency())
    logger.logger.dump_results()


if __name__ == '__main__':
    bot = Bot()
    atexit.register(handle_exit, bot)
    signal.signal(signal.SIGTERM, handle_exit)
    signal.signal(signal.SIGINT, handle_exit)
    bot.trade()

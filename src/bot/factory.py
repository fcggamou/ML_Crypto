import enums
import config
from clock import BaseClock, BacktestClock, RealClock
from exchange import BaseExchange, BacktestExchange, RealExchange, PaperExchange
from data import Data, BacktestData
from logger import BaseLogger, ConsoleLogger, TelegramLogger, SilentConsoleLogger
import logger
from strategy import BaseStrategy, BuyAndHoldStrategy, RandomStrategy, DummyStrategy, SimpleStrategy, SimpleStrategyClassification, OnlyLong, NoiseLong, Technical
from predictor import Predictor, BacktestPredictor


def create_data(clock: BaseClock) -> Data:
    if config.clock == enums.Clocks.BACKTEST:
        return BacktestData(clock)
    else:
        return Data(clock)


def create_predictor(data: Data) -> Predictor:
    if config.clock == enums.Clocks.BACKTEST:
        return BacktestPredictor(data)
    else:
        return Predictor(data)


def create_clock() -> BaseClock:
    if config.clock == enums.Clocks.BACKTEST:
        return BacktestClock(config.start_time, config.tick_interval_s)
    elif config.clock == enums.Clocks.REAL:
        return RealClock(config.tick_interval_s)


def create_exchange(clock: BaseClock = None, data: Data = None) -> BaseExchange:
    if config.exchange == enums.Exchanges.BACKTEST:
        return BacktestExchange(clock, data)
    elif config.exchange == enums.Exchanges.REAL:
        return RealExchange()
    elif config.exchange == enums.Exchanges.PAPER:
        return PaperExchange()


def create_logger(clock: BaseClock) -> BaseLogger:
    if config.logger == enums.Logger.CONSOLE:
        logger.logger = ConsoleLogger(clock)
    elif config.logger == enums.Logger.TELEGRAM:
        logger.logger = TelegramLogger(clock)
    elif config.logger == enums.Logger.SILENTCONSOLE:
        logger.logger = SilentConsoleLogger(clock)


def create_strat(ex: BaseExchange, data: Data, predictor: Predictor) -> BaseStrategy:
    if config.strat == enums.Strats.BUY_AND_HOLD:
        return BuyAndHoldStrategy(ex)
    elif config.strat == enums.Strats.RANDOM:
        return RandomStrategy(ex)
    elif config.strat == enums.Strats.DUMMY:
        return DummyStrategy(ex, data)
    elif config.strat == enums.Strats.SIMPLE:
        return SimpleStrategy(ex, data, predictor)
    elif config.strat == enums.Strats.SIMPLECLASSIFICATION:
        return SimpleStrategyClassification(ex, data, predictor, config.strat_profit_target, config.fpr_treshold)
    elif config.strat == enums.Strats.ONLYLONG:
        return OnlyLong(ex, data, predictor, config.strat_profit_target, config.fpr_treshold)
    elif config.strat == enums.Strats.NOISELONG:
        return NoiseLong(ex, data, config.strat_profit_target, config.stop_loss)
    elif config.strat == enums.Strats.TECHNICAL:
        return Technical(ex, data, config.strat_profit_target, config.stop_loss)
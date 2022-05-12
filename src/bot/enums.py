from aenum import Enum


class Strats(Enum):
    RANDOM = 1
    BUY_AND_HOLD = 2
    DUMMY = 3
    SIMPLE = 4
    SIMPLECLASSIFICATION = 5
    ONLYLONG = 6
    NOISELONG = 7
    TECHNICAL = 8


class Exchanges(Enum):
    BACKTEST = 1
    REAL = 2
    PAPER = 3


class Clocks(Enum):
    BACKTEST = 1
    REAL = 2


class Logger(Enum):
    CONSOLE = 1
    DB = 2
    TELEGRAM = 3
    SILENTCONSOLE = 4


class OrderType(Enum):
    BUY = 1
    SELL = 2

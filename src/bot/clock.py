from abc import ABC, abstractmethod
import time
import math
from datetime import datetime


class BaseClock(ABC):

    @abstractmethod
    def tick(self):
        pass

    @abstractmethod
    def get_time(self):
        pass


class RealClock(BaseClock):
    def __init__(self, tick_interval_s):
        self.tick_interval_s = tick_interval_s

    def tick(self):
        t = datetime.utcnow()
        if self.tick_interval_s >= 60:
            correction = (t.second + t.microsecond / 1000000.0)
        else:
            correction = 0
        time.sleep(self.tick_interval_s - correction)

    def get_time(self):
        return math.floor(time.time())


class BacktestClock(BaseClock):
    def __init__(self, start_time, tick_interval_s):
        self.current_time = start_time
        self.tick_interval_s = tick_interval_s

    def tick(self):
        self.current_time = self.current_time + self.tick_interval_s

    def get_time(self):
        return self.current_time

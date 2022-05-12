import tzlocal
import datetime
import pandas as pd
import time
import sys

this = sys.modules[__name__]
this.current_time = None  # type: ignore


def dateToEpoc(date: datetime.datetime):
    #my_timezone = tzlocal.get_localzone()    
    #return my_timezone.timestamp()
    #mytime_with_timezone = my_timezone.localize(mytime)
    return int(date.timestamp())


def epocToDate(timestamp):
    datetime_without_timezone = datetime.datetime.fromtimestamp(int(timestamp))
    my_timezone = tzlocal.get_localzone()
    return datetime_without_timezone
    #datetime_with_timezone = my_timezone.localize(datetime_without_timezone)
    #return datetime_with_timezone


def granularityStrToSeconds(str):
    return pd.to_timedelta(str).total_seconds()


def get_time():
    if (this.current_time is None):
        return time.time()
    return this.current_time


def set_time(time_input):
    this.current_time = time_input


def sleep(seconds):
    if (this.current_time):
        this.current_time = this.current_time + seconds
    else:
        time.sleep(seconds)


def np_datetime_to_timestamp(np_dt):
    return int(np_dt.astype('datetime64[s]').astype(int))


def interval_to_binance_interval(interval: str) -> str:
    return interval.replace("min", "m")

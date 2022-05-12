from binance.client import Client

import time
import pandas as pd
import numpy as np
import logging
import time_utils

log = logging.getLogger(__name__)


def update_data(data: pd.DataFrame, symbol, interval) -> pd.DataFrame:
    """ Update data with newer klines """
    new_data = get_missing_data(data, symbol, interval)
    log.debug('Missing exchange data: %s', str(new_data))
    if new_data.empty:
        return data
    # Ignore duplicates
    if new_data.index.values[0] in data.index:
        data = data.drop(data.tail(1).index)
    return pd.concat([data, new_data])


def get_missing_data(data: pd.DataFrame, symbol, interval) -> pd.DataFrame:
    latest = int(data.index[-1] * 1000)
    jsonData = np.array(get_historical_klines(symbol, interval, latest))
    if jsonData.size == 0:
        return pd.DataFrame()
    df = pd.DataFrame(jsonData[:-1, 1:11], columns=['Open', 'High', 'Low', 'Close',
                                                    'Vol', 'Close_Time', 'Quote_Asset_Vol', 'Trades', 'Base_Vol', 'Quote_Vol'], dtype='float32')

    granularity_s = int(time_utils.granularityStrToSeconds(interval))
    df.index = map(lambda x: int(x[:-3]) + granularity_s, jsonData[:-1, 0])
    df = df.drop(['Close_Time'], axis=1)
    # Ignore incomplete (current) candle
    if is_unfinished_candle(df, interval):
        log.info('Discarding unfinished candle with open TS: %d',
                 df.index[:-1])
        df = df[:-1]
    return df


def is_unfinished_candle(data: pd.DataFrame, interval: str) -> bool:
    ''' Indicates whether the last candle in the dataframe corresponds to the current unfinished candle '''

    latest_open_ts = pd.Timestamp.utcnow().floor(interval).timestamp()

    log.info('Latest data candle open TS: %d', data.index[-1])
    log.info('Latest candle open TS: %d', latest_open_ts)

    result: bool = data.index[-1] == latest_open_ts

    return result


def get_all_data(symbol, interval, startMs=1587661700000) -> pd.DataFrame:
    jsonData = np.array(get_historical_klines(symbol, interval, startMs))
    df = pd.DataFrame(jsonData[:-1, 1:11], columns=['Open', 'High', 'Low', 'Close',
                                                    'Vol', 'Close_Time', 'Quote_Asset_Vol', 'Trades', 'Base_Vol', 'Quote_Vol'], dtype='float32')
    granularity_s = int(time_utils.granularityStrToSeconds(interval))
    df.index = map(lambda x: int(x[:-3]) + granularity_s, jsonData[:-1, 0])
    df = df.drop(['Close_Time'], axis=1)
    return df


def get_historical_klines(symbol, interval, startMs, endMs=None):
    """Get Historical Klines from Binance

    See dateparse docs for valid start and end string formats http://dateparser.readthedocs.io/en/latest/

    If using offset strings for dates add "UTC" to date string e.g. "now UTC", "11 hours ago UTC"

    :param symbol: Name of symbol pair e.g BNBBTC
    :type symbol: str
    :param interval: Biannce Kline interval
    :type interval: str
    :param start_str: Start date string in UTC format
    :type start_str: str
    :param end_str: optional - end date string in UTC format
    :type end_str: str

    :return: list of OHLCV values:
    [
        [
        1499040000000,      // Open time
        "0.01634790",       // Open
        "0.80000000",       // High
        "0.01575800",       // Low
        "0.01577100",       // Close
        "148976.11427815",  // Volume
        1499644799999,      // Close time
        "2434.19055334",    // Quote asset volume
        308,                // Number of trades
        "1756.87402397",    // Taker buy base asset volume
        "28.46694368",      // Taker buy quote asset volume
        "17928899.62484339" // Ignore.
        ]
    ]
    """
    # create the Binance client, no need for api key
    client = Client("", "")

    # init our list
    output_data = []

    # setup the max limit
    limit = 500
    intervalMs = int(pd.to_timedelta(interval).total_seconds() * 1000)
    interval = time_utils.interval_to_binance_interval(interval)
    idx = 0
    # it can be difficult to know when a symbol was listed on Binance so allow
    # start time to be before list date
    symbol_existed = False
    while True:
        # fetch the klines from start_ts up to max 500 entries or the end_ts if
        # set
        temp_data = client.get_klines(symbol=symbol,
                                      interval=interval,
                                      limit=limit,
                                      startTime=startMs,
                                      endTime=endMs)
        print("Downloading data ts=" + str(startMs))
        # handle the case where our start date is before the symbol pair listed
        # on Binance
        if not symbol_existed and len(temp_data) > 0:
            symbol_existed = True

        if symbol_existed and len(temp_data) > 0:
            # append this loops data to our output data
            output_data += temp_data

            # update our start timestamp using the last value in the array and
            # add the interval timeframe
            startMs = temp_data[len(temp_data) - 1][0] + intervalMs
        else:
            # it wasn't listed yet, increment our start date
            startMs += intervalMs

        idx += 1
        # check if we received less than the required limit and exit the loop
        if len(temp_data) < limit:
            # exit the while loop
            break

        # sleep after every 3rd call to be kind to the API
        if idx % 3 == 0:
            time.sleep(0.1)

    return output_data

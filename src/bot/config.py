import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import datetime
import time_utils
import enums


# Real
# tick_interval_s = 1
# clock = enums.Clocks.BACKTEST
# exchange = enums.Exchanges.REAL
# strat_patience = 15 * 60

# Backtest
tick_interval_s = 60
clock = enums.Clocks.REAL
exchange = enums.Exchanges.PAPER
strat_patience_s = 60 * 60 * 8

# Clock
start_time_dt = datetime.datetime(2021, 6, 1, 0, 0) 
end_time_dt = datetime.datetime(2023, 10, 1, 0, 0)

# Logger
logger = enums.Logger.CONSOLE
results_path = 'results_{}.pkl'.format(datetime.datetime.now().strftime("%H-%M - %d%m"))
results_folder = 'results'
log_result_ticks = 60 * 24

# Strategy
strat = enums.Strats.TECHNICAL

# Simple strategy
strat_profit_target = 0.0025
strat_tolerance = 0.005
stop_loss = -1
trade_amount_currency = 50
fpr_treshold = 0.99

# Data
granularity = '1min'
granularity_models = '30min'
data_path = 'data'
update_data = True

# Exchange
symbol = 'BTCUSDT'
balance_asset = 0.1
balance_currency = 1000
fee = 0.00075
asset = 'BTC'
currency = 'USDT'

# Predictor
models_path = "models"

# Telegram
granularity_s = time_utils.granularityStrToSeconds(granularity)
granularity_models_s = time_utils.granularityStrToSeconds(granularity_models)

start_time = int(time_utils.dateToEpoc(start_time_dt))
end_time = int(time_utils.dateToEpoc(end_time_dt))
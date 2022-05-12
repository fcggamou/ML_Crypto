# import sys
# import os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'bot'))
# import inputgenerator
# import ml.model
# from bot.bot import Bot
# import config
# import enums


# def bot_test():

#     ig1 = inputgenerator.InputGenerator(None, '5m', 4, 1, ['Close'], 'High', output_granularity='30m', scaler=None)
#     ig2 = inputgenerator.InputGenerator(None, '5m', 4, 1, ['Close'], 'Low', output_granularity='30m', scaler=None)
#     model_h = ml.model.DummyModel(ig1)
#     model_l = ml.model.DummyModel(ig2)
#     model_h.save(os.path.join('data', 'models', 'model_h.pkl'))
#     model_l.save(os.path.join('data', 'models', 'model_l.pkl'))
#     config.models_path = os.path.join('data', 'models')
#     config.clock = enums.Clocks.BACKTEST
#     config.exchange = enums.Exchanges.BACKTEST
#     config.start_time = 1609551900  # Jan 2, 2021 1:45:00 AM GMT
#     config.logger = enums.Logger.CONSOLE
#     config.strat = enums.Strats.SIMPLECLASSIFICATION

#     # Simple strategy
#     config.strat_profit_target = 0.003
#     config.strat_tolerance = 0.005
#     config.trade_amount_currency = 15
#     config.fpr_treshold = 0.95

#     # Data
#     config.granularity = '1min'
#     config.granularity_models = '5min'
#     config.data_path = 'data'
#     config.update_data = True
#     bot = Bot()

#     ts = config.start_time
#     ts_model = config.start_time
#     i = 0
#     price_should_match = False
#     while (ts < 60 * 60 * 30):
#         last_candle = bot.data.last_candle()
#         last_candle_model = bot.data.last_candle_models()
#         predictions = bot.predictor.latest_predictions

#         assert last_candle.index.values[0] == ts
#         assert last_candle_model.index.values[0] == ts_model

#         if price_should_match:
#             assert last_candle.Close.values[0] == last_candle_model.Close.values[0]

#             if len(predictions) > 0:
#                 assert predictions[0].timestamp == ts_model
#                 assert predictions[0].value == last_candle_model.Close.values[0]

#         bot.clock.tick()
#         bot.data.tick()
#         bot.predictor.tick()

#         i += 1
#         ts += 60
#         if i % 5 == 0:
#             ts_model += 5 * 60
#             price_should_match = True
#         else:
#             price_should_match = False

#     #    self.data.tick()
#     #     self.predictor.tick()
#     #     self.strategy.tick()
#     #     self.clock.tick()

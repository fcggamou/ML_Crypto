import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import file_utils
from enums import OrderType
import plotly.graph_objects as go
import time_utils
import glob
import numpy as np

class Plotter():

    def __init__(self, path=None):
        if path is None:
            path = max(glob.glob(os.path.join('results', '*')), key=os.path.getctime)

        results = file_utils.load_from_file(path)
        self.orders = results[0]
        self.config = results[1]
        self._short_legend_added = False
        self._long_legend_added = False
        self._profits = []

    def trade_to_profit(self, orders) -> float:
        profit: float = 0
        if orders[2] == 'long':
            profit = orders[1][2] / orders[0][2]
        else:
            profit = orders[0][2] / orders[1][2]
        profit = profit - self.config['fee'] * 2
        return round(100 * profit - 100, 2)

    def trade_to_line(self, orders):

        show_legend = False
        if orders[2] == 'long':
            color = 'darkgreen'
            name = 'Long'
            legendgroup = 'Long'
            if not self._long_legend_added:
                show_legend = True
                self._long_legend_added = True
        else:
            color = 'indianred'
            name = 'Short'
            legendgroup = 'Short'
            if not self._short_legend_added:
                show_legend = True
                self._short_legend_added = True

        profit = self.trade_to_profit(orders)
        self._profits.append(profit)        
        print(str(profit) + "%")
        line = go.Scatter(x=[time_utils.epocToDate(orders[0][3]), time_utils.epocToDate(orders[1][3])],
                          y=[orders[0][2], orders[1][2]], hovertext=profit, hoverinfo='text', hoveron="points+fills",
                          showlegend=show_legend, line=dict(color=color, width=2), name=name,
                          legendgroup=legendgroup)

        return line

    def plot(self):
        start_time = self.orders[0][3]
        end_time = self.orders[-1][3]
        file_path = '{}_{}.pkl'.format(self.config['symbol'], self.config['granularity'])

        data = file_utils.load_from_file(os.path.join(self.config['data_path'], file_path))
        data = data.loc[start_time:end_time]
        data['Date'] = data.index.map(time_utils.epocToDate)

        trades = []
        trade_type = None
        for order in sorted(self.orders, key=lambda x: x[3]):
            if trade_type is None:
                if order[0] == OrderType.BUY:
                    trade_type = 'long'
                else:
                    trade_type = 'short'
                order_trade = order
            else:
                trades.append((order_trade, order, trade_type))
                trade_type = None
                order_trade = None

        lines = [self.trade_to_line(x) for x in trades]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='BTC/USDT price'))
        fig.add_traces(lines)

        print('======')
        print('max: ', end='')        
        print(np.max(self._profits))

        print('min: ', end='')
        print(np.min(self._profits))

        print('mean: ', end='')
        print(np.mean(self._profits))

        print('sum: ', end='')
        print(np.sum(self._profits))
        fig.show()


p = Plotter()
p.plot()

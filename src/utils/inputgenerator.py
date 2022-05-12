import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import time_utils
import ta  # noqa


class InputGenerator(object):

    @property
    def name(self):
        return self._name

    def __init__(self, target_func, granularity, input_window, output_window, features, target_variable,
                 group_by=False, output_granularity=None, custom_ta=[], scaler='minmax', scale_range=(0, 1), diff_percent=None, static_features=[]):
        self.granularity = granularity
        self.iw = input_window
        self.ow = output_window
        self.features = features
        self.target_variable = target_variable
        self.min_window = self.iw
        self.target_func = target_func
        self.scaler = scaler
        self.scale_range = scale_range
        self.scalerX = None
        self.scalerY = DummyScaler()
        self.diff_percent = diff_percent
        self.output_dim = 1
        self.group_by = group_by
        self.custom_ta = custom_ta
        self._name = str(granularity) + "-" + str(input_window) + "-" + str(
            output_window) + " " + str(features)
        self.output_granularity = output_granularity if output_granularity is not None else granularity
        self.inputs_per_output = int(pd.to_timedelta(
            self.output_granularity) / pd.to_timedelta(granularity))
        self.stop_percentage = None
        self.static_features = static_features

    def create_scaler(self):
        if self.scaler == 'minmax':
            return MinMaxScaler(self.scale_range)
        elif self.scaler == 'std':
            return StandardScaler()
        else:
            return DummyScaler()

    def output_granularity_seconds(self):
        return time_utils.granularityStrToSeconds(self.output_granularity)

    def granularity_seconds(self):
        return time_utils.granularityStrToSeconds(self.granularity)

    def prepare_data(self, data, for_training):

        self.group_data(data)
        self.add_custom_data(data)

        self.min_window = self.iw + np.count_nonzero(data.isnull().values)
        data = data.dropna().apply(pd.to_numeric, downcast="float")

        return data

    def group_data(self, data):
        if (self.group_by):
            data = data.groupby(pd.Grouper(freq=self.granularity)).apply(
                group_by_interval)
            data = data.fillna(0)
            data = data.replace(to_replace=0, method='ffill')

    def add_custom_data(self, data):
        for custom_ta in self.custom_ta:
            exec(custom_ta)

    def scale_x(self, data):
        if (self.scalerX is None):
            self.scalerX = self.create_scaler()
            self.scalerX.fit(data)

        scaled_data = pd.DataFrame(self.scalerX.transform(data))
        scaled_data.columns = data.columns
        scaled_data.index = data.index

        return scaled_data.apply(pd.to_numeric, downcast="float")

    def scale_y(self, data):
        return data

    def get_io(self, data, for_training=True):
        data = self.prepare_data(data, for_training)

        if for_training:
            output = self.get_output(data)
        x = np.zeros(shape=(len(data), len(self.features) * self.iw + self.output_dim + len(self.static_features) + 1))
        x[:, 0] = data.index.values

        if for_training:
            if self.output_dim > 1:
                x[:, 1:self.output_dim + 1] = self.scale_y(output).values
            else:
                x[:, 1] = self.scale_y(output).values.reshape(1, -1)[0]

        data = data[list(dict.fromkeys(self.features + self.static_features))]
        data = self.scale_x(data)

        j = 1 + self.output_dim
        for input_var in self.features:
            for i in range(self.iw - 1, -1, -1):
                x[:, j] = data[input_var].shift(i)
                j = j + 1

        k = j
        for input_var in self.static_features:
            x[:, j] = data[input_var]
            k = k + 1

        x = x[~np.isnan(x).any(axis=1)]
        if len(self.static_features) > 0:
            input = [x[:, 1 + self.output_dim:j], x[:, j:]]
        else:
            input = x[:, 1 + self.output_dim:j]
        return x[:, 0], input, x[:, 1:self.output_dim + 1]

    def get_input_for_prediction(self, data):
        ts, x, _ = self.get_io(data.iloc[-self.min_window:], False)
        return ts[-1], x[-1]

    def inverse_transform_x(self, x):
        if self.scalerX is not None:
            return self.scalerX.inverse_transform(x)
        return x

    def get_output(self, data):

        y = data[self.target_variable]
        if self.target_func is not None:
            y = y.rolling(window=self.inputs_per_output)
            if self.target_func == 'max':
                y = y.max()
            elif self.target_func == 'min':
                y = y.min()
            elif self.target_func == 'mean':
                y = y.mean()
            elif self.target_func == 'max_diff':
                close_lag = data['Close'].shift(1)
                y = y.apply(lambda x, close_lag=close_lag: x.max() / close_lag.loc[x.index[0]])
            elif self.target_func == 'min_diff':
                close_lag = data['Close'].shift(1)
                y = y.apply(lambda x, close_lag=close_lag: x.min() / close_lag.loc[x.index[0]])
            elif self.target_func == 'close_diff':
                close_lag = data['Close'].shift(1)
                y = y.apply(lambda x, close_lag=close_lag: x.iloc[-1] / close_lag.loc[x.index[0]])
            elif self.target_func == 'is_diff_higher':
                close_lag = data['Close'].shift(1)
                y = y.apply(lambda x, close_lag=close_lag, diff_percent=self.diff_percent: int((x.max() / close_lag.loc[x.index[0]]) - 1 >= diff_percent))
            elif self.target_func == 'is_diff_higher_sl':
                close_lag = data['Close'].shift(1)
                high = data['High'].rolling(window=self.inputs_per_output).apply(lambda x, close_lag=close_lag, diff_percent=self.diff_percent: int(x.max() / close_lag.loc[x.index[0]] - 1 >= diff_percent))
                low = data['Low'].rolling(window=self.inputs_per_output).apply(lambda x, close_lag=close_lag, diff_percent=self.diff_percent: int(x.min() / close_lag.loc[x.index[0]] - 1 <= -diff_percent))
                y = np.logical_and(high, np.logical_not(low))
            elif self.target_func == 'is_diff_lower':
                close_lag = data['Close'].shift(1)
                y = y.apply(lambda x, close_lag=close_lag, diff_percent=self.diff_percent: int((x.min() / close_lag.loc[x.index[0]]) - 1 <= diff_percent))
            elif self.target_func == 'multi_diff':
                close_lag = data['Close'].shift(1)
                y = pd.DataFrame()
                high = data['High'].rolling(window=self.inputs_per_output).apply(lambda x, close_lag=close_lag, diff_percent=self.diff_percent: int(x.max() / close_lag.loc[x.index[0]] - 1 >= diff_percent))
                low = data['Low'].rolling(window=self.inputs_per_output).apply(lambda x, close_lag=close_lag, diff_percent=self.diff_percent: int(x.min() / close_lag.loc[x.index[0]] - 1 <= -diff_percent))
                y['High'] = np.logical_and(high, np.logical_not(low))
                y['Low'] = np.logical_and(low, np.logical_not(high))
                y['Same'] = np.logical_not(np.logical_or(y['High'], y['Low'])).astype(int)
                self.output_dim = 3
            else:
                y = y.apply(self.target_func)

        y = y.shift(-self.inputs_per_output)
        return y


class InputGeneratorRegression(InputGenerator):
    def __init__(self, target_func, granularity, input_window, output_window, features, target_variable,
                 group_by=False, output_granularity=None, custom_ta=[], scaler='minmax', scale_range=(0, 1), static_features=[]):
        super().__init__(target_func, granularity, input_window, output_window, features, target_variable,
                         group_by, output_granularity, custom_ta, scaler, scale_range, static_features=static_features)
        self.scalerY = None

    def scale_y(self, data):
        y = np.array(data).reshape(-1, 1)
        if (self.scalerY is None):
            self.scalerY = self.create_scaler()
            self.scalerY.fit(y)

        scaled_data = pd.DataFrame(self.scalerY.transform(y), columns=['y'])
        scaled_data.index = data.index

        return scaled_data.apply(pd.to_numeric, downcast="float")


class InputGeneratorEnsemble(InputGenerator):

    def get_io_array(self, data):
        return data, data

    def get_input_for_prediction(data):
        return data


def split(x, y, test_size, validation_size):
    """Splits the dataset into train, validation and test sets.
    [train:validation_size:test_size] """

    x_val = x[len(x) - validation_size - test_size:-test_size]
    y_val = y[len(y) - validation_size - test_size:-test_size]

    x_train = x[:len(x) - test_size - validation_size]
    y_train = y[:len(y) - test_size - validation_size]

    x_test = x[len(x) - test_size:]
    y_test = y[len(y) - test_size:]

    return x_train, x_test, x_val, y_train, y_test, y_val


def group_by_interval(data_row):
    if len(data_row) == 0:
        return None
    d = {'Close': data_row['Close'][-1],
         'Vol': data_row['Vol'].sum(),
         'High': data_row['High'].max(),
         'Low': data_row['Low'].min(),
         'Open': data_row['Open'][0]}

    return pd.Series(d)


class DummyScaler(object):

    def fit(self, data):
        pass

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x


def to_vol_ticks(df, vol_tick):
    def grouper(df, vol_tick):
        df['new_bin'] = 0
        cum_vol = 0
        for i in df.index:
            if cum_vol >= vol_tick:
                df.loc[i, 'new_bin'] = 1
                cum_vol = 0
            cum_vol += df.loc[i, "Vol"]
        df['group'] = df['new_bin'].cumsum()
        return df.drop("new_bin", axis=1)

    df['TS'] = df.index
    df = grouper(df, vol_tick)
    adict = {'TS': 'last', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Vol': 'sum', 'Open': 'first'}
    df = df.groupby('group')[['TS', 'High', 'Low', 'Close', 'Vol', 'Open']].agg(adict)
    df.index = df['TS']
    return df.drop('TS', axis=1)

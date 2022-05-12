import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import inputgenerator
from pandas import DataFrame
import numpy as np
import pandas as pd


def test_basic():
    ig = inputgenerator.InputGeneratorRegression(
        'max', '5m', 4, 1, ['Close', 'High', 'Low'], 'High', output_granularity='1h', scaler=None)
    assert ig.features == ['Close', 'High', 'Low']
    assert ig.iw == 4
    assert ig.ow == 1
    assert ig.granularity_seconds() == 5 * 60
    assert ig.output_granularity_seconds() == 60 * 60
    assert ig.target_variable == 'High'
    assert ig.inputs_per_output == 12


def test_get_io():
    ig = inputgenerator.InputGeneratorRegression(
        None, '5m', 4, 1, ['Close', 'High', 'Low'], 'Close', output_granularity='1h', scaler=None)
    df = DataFrame([[1608642300, 1000, 1100, 900],
                    [1608642600, 1100, 1200, 1000],
                    [1608642900, 1200, 1300, 1100],
                    [1608643200, 1300, 1400, 1200],
                    [1608643500, 1400, 1500, 1300]], columns=['TS', 'Close', 'High', 'Low'])
    df.set_index(df['TS'], inplace=True)
    ts, x, y = ig.get_io(df, False)

    io = pd.DataFrame(np.column_stack((ts, y, x)), columns=[
                      'TS', 'y', 'Close3', 'Close2', 'Close1', 'Close0', 'High3', 'High2', 'High1', 'High0', 'Low3', 'Low2', 'Low1', 'Low0'])
    assert io is not None
    assert len(io) == 2
    assert io.iloc[0]['Close0'] == 1300
    assert io.iloc[0]['Close1'] == 1200
    assert io.iloc[0]['Close2'] == 1100
    assert io.iloc[0]['Close3'] == 1000
    assert io.iloc[0]['High0'] == 1400
    assert io.iloc[0]['High1'] == 1300
    assert io.iloc[0]['High2'] == 1200
    assert io.iloc[0]['High3'] == 1100
    assert io.iloc[0]['Low0'] == 1200
    assert io.iloc[0]['Low1'] == 1100
    assert io.iloc[0]['Low2'] == 1000
    assert io.iloc[0]['Low3'] == 900

    assert io.iloc[1]['Close0'] == 1400
    assert io.iloc[1]['Close1'] == 1300
    assert io.iloc[1]['Close2'] == 1200
    assert io.iloc[1]['Close3'] == 1100
    assert io.iloc[1]['High0'] == 1500
    assert io.iloc[1]['High1'] == 1400
    assert io.iloc[1]['High2'] == 1300
    assert io.iloc[1]['High3'] == 1200
    assert io.iloc[1]['Low0'] == 1300
    assert io.iloc[1]['Low1'] == 1200
    assert io.iloc[1]['Low2'] == 1100
    assert io.iloc[1]['Low3'] == 1000


def test_get_io_scaled():
    ig = inputgenerator.InputGeneratorRegression(
        None, '5m', 4, 1, ['Close', 'High', 'Low'], 'Close', output_granularity='5m', scaler='minmax')
    df = DataFrame([[1608642300, 1000, 1100, 900],
                    [1608642600, 1100, 1200, 1000],
                    [1608642900, 1200, 1300, 1100],
                    [1608643200, 1300, 1400, 1200],
                    [1608643500, 1400, 1500, 1300]], columns=['TS', 'Close', 'High', 'Low'])
    df.set_index(df['TS'], inplace=True)
    ts, x, y = ig.get_io(df, True)
    io = pd.DataFrame(np.column_stack((ts, y, x)), columns=[
                      'TS', 'y', 'Close3', 'Close2', 'Close1', 'Close0', 'High3', 'High2', 'High1', 'High0', 'Low3', 'Low2', 'Low1', 'Low0'])
    assert io is not None
    assert len(io) == 1
    assert abs(io.iloc[0]['y'] - 1) < 0.0001
    assert abs(io.iloc[0]['Close3'] - 0) < 0.0001
    assert abs(io.iloc[0]['Close2'] - 0.25) < 0.0001
    assert abs(io.iloc[0]['Close1'] - 0.5) < 0.0001
    assert abs(io.iloc[0]['Close0'] - 0.75) < 0.0001
    assert abs(io.iloc[0]['High3'] - 0) < 0.0001
    assert abs(io.iloc[0]['High2'] - 0.25) < 0.0001
    assert abs(io.iloc[0]['High1'] - 0.5) < 0.0001
    assert abs(io.iloc[0]['High0'] - 0.75) < 0.0001
    assert abs(io.iloc[0]['Low3'] - 0) < 0.0001
    assert abs(io.iloc[0]['Low2'] - 0.25) < 0.0001
    assert abs(io.iloc[0]['Low1'] - 0.5) < 0.0001
    assert abs(io.iloc[0]['Low0'] - 0.75) < 0.0001


def test_get_input_for_training_1():
    ig = inputgenerator.InputGeneratorRegression(
        None, '5m', 4, 1, ['Close', 'High', 'Low'], 'Close', output_granularity='5m', scaler=None)
    df = DataFrame([[1608642300, 1000, 1100, 900],
                    [1608642600, 1100, 1200, 1000],
                    [1608642900, 1200, 1300, 1100],
                    [1608643200, 1300, 1400, 1200],
                    [1608643800, 1400, 1500, 1300],
                    [1608644100, 1500, 1600, 1400],
                    [1608644300, 1600, 1700, 1500],
                    [1608644600, 1700, 1800, 1600]], columns=['TS', 'Close', 'High', 'Low'])

    df.set_index(df['TS'], inplace=True)
    ts, x, y = ig.get_io(df)
    assert all(x[0] == [1000, 1100, 1200, 1300, 1100, 1200, 1300, 1400, 900, 1000, 1100, 1200])
    assert y[0] == 1400

    assert all(x[1] == [1100, 1200, 1300, 1400, 1200, 1300, 1400, 1500, 1000, 1100, 1200, 1300])
    assert y[1] == 1500

    assert all(x[2] == [1200, 1300, 1400, 1500, 1300, 1400, 1500, 1600, 1100, 1200, 1300, 1400])
    assert y[2] == 1600

    assert all(x[3] == [1300, 1400, 1500, 1600, 1400, 1500, 1600, 1700, 1200, 1300, 1400, 1500])
    assert y[3] == 1700


def test_get_input_for_training_2():
    ig = inputgenerator.InputGeneratorRegression(
        None, '5m', 4, 1, ['Close', 'High', 'Low'], 'Close', output_granularity='10m', scaler=None)
    df = DataFrame([[1608642300, 1000, 1100, 900],
                    [1608642600, 1100, 1200, 1000],
                    [1608642900, 1200, 1300, 1100],
                    [1608643200, 1300, 1400, 1200],
                    [1608643500, 1400, 1500, 1300],
                    [1608643800, 1500, 1600, 1400],
                    [1608644100, 1600, 1700, 1500],
                    [1608644400, 1700, 1800, 1600]], columns=['TS', 'Close', 'High', 'Low'])

    df.set_index(df['TS'], inplace=True)
    ts, x, y = ig.get_io(df)
    assert all(x[0] == [1000, 1100, 1200, 1300, 1100, 1200, 1300, 1400, 900, 1000, 1100, 1200])
    assert y[0] == 1500

    assert all(x[1] == [1100, 1200, 1300, 1400, 1200, 1300, 1400, 1500, 1000, 1100, 1200, 1300])
    assert y[1] == 1600

    assert all(x[2] == [1200, 1300, 1400, 1500, 1300, 1400, 1500, 1600, 1100, 1200, 1300, 1400])
    assert y[2] == 1700


def test_get_input_for_training_3():
    ig = inputgenerator.InputGeneratorRegression(
        'max', '5m', 4, 1, ['Close', 'High', 'Low'], 'High', output_granularity='10m', scaler=None)
    df = DataFrame([[1608642300, 1000, 1100, 900],
                    [1608642600, 1100, 1200, 1000],
                    [1608642900, 1200, 1300, 1100],
                    [1608643200, 1300, 1400, 1200],
                    [1608643500, 1400, 1600, 1300],
                    [1608643800, 1500, 1500, 1400],
                    [1608644100, 1600, 1700, 1500],
                    [1608644400, 1700, 1698, 1600]], columns=['TS', 'Close', 'High', 'Low'])

    df.set_index(df['TS'], inplace=True)
    ts, x, y = ig.get_io(df)
    assert all(x[0] == [1000, 1100, 1200, 1300, 1100, 1200, 1300, 1400, 900, 1000, 1100, 1200])
    assert y[0] == 1600

    assert all(x[1] == [1100, 1200, 1300, 1400, 1200, 1300, 1400, 1600, 1000, 1100, 1200, 1300])
    assert y[1] == 1700

    assert all(x[2] == [1200, 1300, 1400, 1500, 1300, 1400, 1600, 1500, 1100, 1200, 1300, 1400])
    assert y[2] == 1700


def test_get_input_for_training_4():
    ig = inputgenerator.InputGeneratorRegression(
        'min', '5m', 4, 1, ['Close', 'High', 'Low'], 'High', output_granularity='15m', scaler=None)
    df = DataFrame([[1608642300, 1000, 1100, 900],
                    [1608642600, 1100, 1200, 1000],
                    [1608642900, 1200, 1300, 1100],
                    [1608643200, 1300, 1400, 1200],
                    [1608643500, 1400, 1600, 1300],
                    [1608643800, 1500, 1500, 1400],
                    [1608644100, 1600, 1700, 1500],
                    [1608644400, 1700, 1698, 1600]], columns=['TS', 'Close', 'High', 'Low'])

    df.set_index(df['TS'], inplace=True)
    ts, x, y = ig.get_io(df)
    assert all(x[0] == [1000, 1100, 1200, 1300, 1100, 1200, 1300, 1400, 900, 1000, 1100, 1200])
    assert y[0] == 1500

    assert all(x[1] == [1100, 1200, 1300, 1400, 1200, 1300, 1400, 1600, 1000, 1100, 1200, 1300])
    assert y[1] == 1500


def test_get_input_for_prediction():
    ig = inputgenerator.InputGeneratorRegression(
        'max', '5m', 4, 1, ['Close', 'High', 'Low'], 'High', output_granularity='10m', scaler=None)
    df = DataFrame([[1608642300, 1000, 1100, 900],
                    [1608642600, 1100, 1200, 1000],
                    [1608642900, 1200, 1300, 1100],
                    [1608643200, 1300, 1400, 1200],
                    [1608643500, 1400, 1600, 1300],
                    [1608643800, 1500, 1500, 1400],
                    [1608644100, 1600, 1700, 1500],
                    [1608644400, 1700, 1698, 1600]], columns=['TS', 'Close', 'High', 'Low'])

    df.set_index(df['TS'], inplace=True)
    ts, x = ig.get_input_for_prediction(df)

    assert all(x == [1400, 1500, 1600, 1700, 1600, 1500, 1700, 1698, 1300, 1400, 1500, 1600])
    assert ts == 1608644400


def test_get_io_array_with_ts():
    ig = inputgenerator.InputGeneratorRegression(
        'max', '5m', 4, 1, ['Close', 'High', 'Low'], 'High', output_granularity='10m', scaler=None)
    df = DataFrame([[1608642300, 1000, 1100, 900],
                    [1608642600, 1100, 1200, 1000],
                    [1608642900, 1200, 1300, 1100],
                    [1608643200, 1300, 1400, 1200],
                    [1608643500, 1400, 1500, 1300],
                    [1608643800, 1500, 1600, 1400]], columns=['TS', 'Close', 'High', 'Low'])

    df.set_index(df['TS'], inplace=True)
    ts, x, _ = ig.get_io(df, False)
    assert len(x) == 3
    assert all(x[0] == [1000, 1100, 1200, 1300, 1100, 1200, 1300, 1400, 900, 1000, 1100, 1200])
    assert ts[0] == 1608643200

    assert all(x[1] == [1100, 1200, 1300, 1400, 1200, 1300, 1400, 1500, 1000, 1100, 1200, 1300])
    assert ts[1] == 1608643500

    assert all(x[2] == [1200, 1300, 1400, 1500, 1300, 1400, 1500, 1600, 1100, 1200, 1300, 1400])
    assert ts[2] == 1608643800


def test_get_input_with_ts():
    ig = inputgenerator.InputGeneratorRegression(
        'max', '5m', 4, 1, ['Close', 'High', 'Low'], 'High', output_granularity='10m', scaler=None)
    df = DataFrame([[1608642300, 1000, 1100, 900],
                    [1608642600, 1100, 1200, 1000],
                    [1608642900, 1200, 1300, 1100],
                    [1608643200, 1300, 1400, 1200],
                    [1608643500, 1400, 1600, 1300]], columns=['TS', 'Close', 'High', 'Low'])

    df.set_index(df['TS'], inplace=True)
    ts, x, _ = ig.get_io(df, False)
    assert len(x) == 2
    assert len(ts) == 2
    assert all(x[0] == [1000, 1100, 1200, 1300, 1100, 1200, 1300, 1400, 900, 1000, 1100, 1200])
    assert ts[0] == 1608643200

    assert all(x[1] == [1100, 1200, 1300, 1400, 1200, 1300, 1400, 1600, 1000, 1100, 1200, 1300])
    assert ts[1] == 1608643500


def test_custom_data():
    ig = inputgenerator.InputGeneratorRegression(
        None, '5m', 4, 1, ['Close', 'High', 'Low', 'Close_diff'], 'Close', output_granularity='1h', scaler=None, custom_ta=["data['Close_diff'] = data['Close'].diff()"])
    df = DataFrame([[1608642300, 1000, 1100, 900],
                    [1608642600, 1100, 1200, 1000],
                    [1608642900, 1200, 1300, 1100],
                    [1608643200, 1500, 1400, 1200],
                    [1608643500, 1400, 1500, 1300]], columns=['TS', 'Close', 'High', 'Low'])
    df.set_index(df['TS'], inplace=True)
    ts, x, _ = ig.get_io(df, False)
    io = pd.DataFrame(np.column_stack((ts, x)), columns=[
                      'TS', 'Close3', 'Close2', 'Close1', 'Close0', 'High3', 'High2', 'High1', 'High0', 'Low3', 'Low2', 'Low1', 'Low0', 'Close_diff3', 'Close_diff2', 'Close_diff1', 'Close_diff0'])
    assert io is not None
    assert ig.min_window == 5
    assert len(io) == 1
    assert io.iloc[0]['Close0'] == 1400
    assert io.iloc[0]['Close1'] == 1500
    assert io.iloc[0]['Close2'] == 1200
    assert io.iloc[0]['Close3'] == 1100
    assert io.iloc[0]['High0'] == 1500
    assert io.iloc[0]['High1'] == 1400
    assert io.iloc[0]['High2'] == 1300
    assert io.iloc[0]['High3'] == 1200
    assert io.iloc[0]['Low0'] == 1300
    assert io.iloc[0]['Low1'] == 1200
    assert io.iloc[0]['Low2'] == 1100
    assert io.iloc[0]['Low3'] == 1000
    assert io.iloc[0]['Close_diff0'] == -100
    assert io.iloc[0]['Close_diff1'] == 300
    assert io.iloc[0]['Close_diff2'] == 100
    assert io.iloc[0]['Close_diff3'] == 100

    ts, x = ig.get_input_for_prediction(df)
    assert all(x == [1100, 1200, 1500, 1400, 1200, 1300, 1400, 1500, 1000, 1100, 1200, 1300, 100, 100, 300, -100])


def test_custom_data_2():
    ig = inputgenerator.InputGeneratorRegression(
        None, '5m', 4, 1, ['Close', 'High_4'], 'Close', output_granularity='1h', scaler=None, custom_ta=["data['High_4'] = data['High'].rolling(4).max()"])
    df = DataFrame([[1608642300, 1000, 1100, 900],
                    [1608642600, 1100, 1200, 1000],
                    [1608642900, 1200, 1300, 1100],
                    [1608643200, 1300, 1400, 1200],
                    [1608643500, 1400, 1600, 1300],
                    [1608643500, 1500, 1500, 1400],
                    [1608643500, 1600, 1700, 1500],
                    [1608643500, 1700, 1698, 1600]], columns=['TS', 'Close', 'High', 'Low'])
    df.set_index(df['TS'], inplace=True)
    ts, x, _ = ig.get_io(df, False)
    io = pd.DataFrame(np.column_stack((ts, x)), columns=[
                      'TS', 'Close3', 'Close2', 'Close1', 'Close0', 'High_43', 'High_42', 'High_41', 'High_40'])
    assert io is not None
    assert ig.min_window == 7
    assert len(io) == 2
    assert io.iloc[0]['Close0'] == 1600
    assert io.iloc[0]['Close1'] == 1500
    assert io.iloc[0]['Close2'] == 1400
    assert io.iloc[0]['Close3'] == 1300
    assert io.iloc[0]['High_40'] == 1700
    assert io.iloc[0]['High_41'] == 1600
    assert io.iloc[0]['High_42'] == 1600
    assert io.iloc[0]['High_43'] == 1400

    assert io.iloc[1]['Close0'] == 1700
    assert io.iloc[1]['Close1'] == 1600
    assert io.iloc[1]['Close2'] == 1500
    assert io.iloc[1]['Close3'] == 1400
    assert io.iloc[1]['High_40'] == 1700
    assert io.iloc[1]['High_41'] == 1700
    assert io.iloc[1]['High_42'] == 1600
    assert io.iloc[1]['High_43'] == 1600

    ts, x = ig.get_input_for_prediction(df)
    assert all(x == [1400, 1500, 1600, 1700, 1600, 1600, 1700, 1700])


def test_max_diff():
    ig = inputgenerator.InputGeneratorRegression(
        'max_diff', '5m', 4, 1, ['Close', 'High'], 'High', output_granularity='15m', scaler=None)

    df = DataFrame([[1608642300, 1000, 1100, 900],
                    [1608642600, 1200, 1200, 1000],
                    [1608642900, 1200, 1300, 1100],
                    [1608643200, 1300, 1400, 1200],
                    [1608643500, 1400, 1600, 1300],
                    [1608643800, 1500, 1500, 1400],
                    [1608644100, 1600, 1700, 1500],
                    [1608644400, 1700, 1798, 1600]], columns=['TS', 'Close', 'High', 'Low'])
    df.set_index(df['TS'], inplace=True)
    ts, x, y = ig.get_io(df, True)
    io = pd.DataFrame(np.column_stack((ts, y, x)), columns=[
                      'TS', 'y', 'Close3', 'Close2', 'Close1', 'Close0', 'High3', 'High2', 'High1', 'High0'])
    assert len(io) == 2
    assert io.iloc[0]['y'] - 1700 / 1300 < 0.1
    assert io.iloc[1]['y'] - 1798 / 1400 < 0.1


def test_min_diff():
    ig = inputgenerator.InputGeneratorRegression(
        'min_diff', '5m', 4, 1, ['Close', 'High'], 'Low', output_granularity='15m', scaler=None)

    df = DataFrame([[1608642300, 1000, 1100, 900],
                    [1608642600, 1200, 1200, 1000],
                    [1608642900, 1200, 1300, 1100],
                    [1608643200, 1300, 1400, 1200],
                    [1608643500, 1400, 1600, 1300],
                    [1608643800, 1500, 1500, 1400],
                    [1608644100, 1600, 1700, 1500],
                    [1608644400, 1700, 1798, 1600]], columns=['TS', 'Close', 'High', 'Low'])
    df.set_index(df['TS'], inplace=True)
    ts, x, y = ig.get_io(df, True)
    io = pd.DataFrame(np.column_stack((ts, y, x)), columns=[
                      'TS', 'y', 'Close3', 'Close2', 'Close1', 'Close0', 'High3', 'High2', 'High1', 'High0'])
    assert len(io) == 2
    assert io.iloc[0]['y'] - 1300 / 1300 < 0.1
    assert io.iloc[1]['y'] - 1400 / 1400 < 0.1


def test_close_diff():
    ig = inputgenerator.InputGeneratorRegression(
        'close_diff', '5m', 4, 1, ['Close', 'High'], 'Close', output_granularity='15m', scaler=None)

    df = DataFrame([[1608642300, 1000, 1100, 900],
                    [1608642600, 1200, 1200, 1000],
                    [1608642900, 1200, 1300, 1100],
                    [1608643200, 1300, 1400, 1200],
                    [1608643500, 1400, 1600, 1300],
                    [1608643800, 1500, 1500, 1400],
                    [1608644100, 1600, 1700, 1500],
                    [1608644400, 1700, 1798, 1600]], columns=['TS', 'Close', 'High', 'Low'])
    df.set_index(df['TS'], inplace=True)
    ts, x, y = ig.get_io(df, True)
    io = pd.DataFrame(np.column_stack((ts, y, x)), columns=[
                      'TS', 'y', 'Close3', 'Close2', 'Close1', 'Close0', 'High3', 'High2', 'High1', 'High0'])
    assert len(io) == 2
    assert io.iloc[0]['y'] - 1600 / 1300 < 0.1
    assert io.iloc[1]['y'] - 1700 / 1400 < 0.1


def test_multi_diff():
    ig = inputgenerator.InputGenerator(
        'multi_diff', '5m', 4, 1, ['Close', 'High'], 'Low', output_granularity='15m', scaler=None, diff_percent=0.01)

    df = DataFrame([[1608642300, 1000, 1100, 900],
                    [1608642600, 1200, 1200, 1000],
                    [1608642900, 1200, 1300, 1100],
                    [1608643200, 1300, 1301, 1200],
                    [1608643500, 1400, 1314, 1300],
                    [1608643800, 1500, 1401, 1400],
                    [1608644100, 1501, 1402, 1500],
                    [1608644400, 1502, 1403, 1600],
                    [1608644700, 1502, 1403, 1445],
                    [1608645000, 1502, 1600, 1400]], columns=['TS', 'Close', 'High', 'Low'])
    df.set_index(df['TS'], inplace=True)
    ts, x, y = ig.get_io(df, True)
    io = pd.DataFrame(np.column_stack((ts, y[:, 0].reshape(-1, 1), y[:, 1].reshape(-1, 1), y[:, 2].reshape(-1, 1), x)), columns=[
                      'TS', 'High', 'Low', 'Same', 'Close3', 'Close2', 'Close1', 'Close0', 'High3', 'High2', 'High1', 'High0'])
    assert len(io) == 4
    assert io.iloc[0]['High'] == 1
    assert io.iloc[0]['Low'] == 0
    assert io.iloc[0]['Same'] == 0
    assert io.iloc[1]['High'] == 0
    assert io.iloc[1]['Low'] == 0
    assert io.iloc[1]['Same'] == 1
    assert io.iloc[2]['High'] == 0
    assert io.iloc[2]['Low'] == 1
    assert io.iloc[2]['Same'] == 0
    assert io.iloc[3]['High'] == 0
    assert io.iloc[3]['Low'] == 0
    assert io.iloc[3]['Same'] == 1


def test_diff_sl():
    ig = inputgenerator.InputGenerator(
        'is_diff_higher_sl', '5m', 4, 1, ['Close', 'High'], 'High', output_granularity='15m', scaler=None, diff_percent=0.01)

    df = DataFrame([[1608642300, 1000, 1100, 900],
                    [1608642600, 1200, 1200, 1000],
                    [1608642900, 1200, 1300, 1100],
                    [1608643200, 1300, 1301, 1200],
                    [1608643500, 1400, 1314, 1300],
                    [1608643800, 1500, 1401, 1400],
                    [1608644100, 1501, 1402, 1500],
                    [1608644400, 1502, 1403, 1600],
                    [1608644700, 1502, 1403, 1445],
                    [1608645000, 1502, 1600, 1400]], columns=['TS', 'Close', 'High', 'Low'])
    df.set_index(df['TS'], inplace=True)
    ts, x, y = ig.get_io(df, True)
    io = pd.DataFrame(np.column_stack((ts, y.reshape(-1, 1), x)), columns=[
                      'TS', 'y', 'Close3', 'Close2', 'Close1', 'Close0', 'High3', 'High2', 'High1', 'High0'])
    assert len(io) == 4
    assert io.iloc[0]['y'] == 1
    assert io.iloc[1]['y'] == 0
    assert io.iloc[2]['y'] == 0
    assert io.iloc[3]['y'] == 0


def test_classification():

    ig = inputgenerator.InputGenerator(
        'is_diff_higher', '5m', 4, 1, ['Close', 'High'], 'High', output_granularity='15m', scaler=None, diff_percent=0.01)

    df = DataFrame([[1608642300, 1000, 1100, 900],
                    [1608642600, 1200, 1200, 1000],
                    [1608642900, 1200, 1300, 1100],
                    [1608643200, 1300, 1400, 1200],
                    [1608643500, 1400, 1301, 1300],
                    [1608643800, 1396, 1315, 1400],
                    [1608644100, 1600, 1302, 1500],
                    [1608644400, 1700, 1411, 1600],
                    [1608644400, 1700, 1405, 1600]], columns=['TS', 'Close', 'High', 'Low'])
    df.set_index(df['TS'], inplace=True)
    ts, x, y = ig.get_io(df, True)
    io = pd.DataFrame(np.column_stack((ts, y, x)), columns=[
                      'TS', 'y', 'Close3', 'Close2', 'Close1', 'Close0', 'High3', 'High2', 'High1', 'High0'])
    assert len(io) == 3
    assert io.iloc[0]['y'] == 1
    assert io.iloc[1]['y'] == 0
    assert io.iloc[2]['y'] == 1


def test_static_features():

    ig = inputgenerator.InputGenerator(
        'is_diff_higher', '5m', 4, 1, ['Close', 'High'], 'High', output_granularity='15m', scaler=None, diff_percent=0.01, static_features=['Low'])

    df = DataFrame([[1608642300, 1000, 1100, 900],
                    [1608642600, 1200, 1200, 1000],
                    [1608642900, 1200, 1300, 1100],
                    [1608643200, 1300, 1400, 1200],
                    [1608643500, 1400, 1301, 1300],
                    [1608643800, 1396, 1315, 1400],
                    [1608644100, 1600, 1302, 1500],
                    [1608644400, 1700, 1411, 1600],
                    [1608644400, 1700, 1405, 1600]], columns=['TS', 'Close', 'High', 'Low'])
    df.set_index(df['TS'], inplace=True)
    ts, x, y = ig.get_io(df, True)
    io = pd.DataFrame(np.column_stack((ts, y, x[0], x[1])), columns=[
                      'TS', 'y', 'Close3', 'Close2', 'Close1', 'Close0', 'High3', 'High2', 'High1', 'High0', 'Low'])
    assert io.loc[0]['Low'] == 1200
    assert io.loc[1]['Low'] == 1300
    assert io.loc[2]['Low'] == 1400


def test_vol_ticks():

    df = DataFrame([[1608642300, 1000, 1000, 1100, 900, 80],
                    [1608642600, 1200, 1200, 1200, 1000, 20],
                    [1608642900, 1200, 1200, 1300, 1100, 30],
                    [1608643200, 1300, 1300, 1400, 1200, 60],
                    [1608643500, 1400, 1400, 1301, 1300, 15],
                    [1608643800, 1396, 1396, 1315, 1400, 110],
                    [1608644100, 1600, 1600, 1302, 1500, 100],
                    [1608644400, 1700, 1700, 1411, 1600, 50],
                    [1608644700, 1700, 1700, 1405, 1600, 40],
                    [1608645000, 1700, 1700, 1405, 1600, 20]], columns=['TS', 'Open', 'Close', 'High', 'Low', 'Vol'])
    df.index = df['TS']
    df = df.drop('TS', axis=1)
    df = inputgenerator.to_vol_ticks(df, 100)

    assert df is not None
    assert len(df) == 5
    assert df.iloc[0]['Close'] == 1200
    assert df.iloc[0]['Open'] == 1000
    assert df.iloc[0]['High'] == 1200
    assert df.iloc[0]['Low'] == 900
    assert df.iloc[0]['Vol'] == 100

    assert df.iloc[1]['Close'] == 1400
    assert df.iloc[1]['Open'] == 1200
    assert df.iloc[1]['High'] == 1400
    assert df.iloc[1]['Low'] == 1100
    assert df.iloc[1]['Vol'] == 105

    assert df.iloc[2]['Close'] == 1396
    assert df.iloc[2]['Open'] == 1396
    assert df.iloc[2]['High'] == 1315
    assert df.iloc[2]['Low'] == 1400
    assert df.iloc[2]['Vol'] == 110

    assert df.iloc[3]['Close'] == 1600
    assert df.iloc[3]['Open'] == 1600
    assert df.iloc[3]['High'] == 1302
    assert df.iloc[3]['Low'] == 1500
    assert df.iloc[3]['Vol'] == 100

    assert df.iloc[4]['Close'] == 1700
    assert df.iloc[4]['Open'] == 1700
    assert df.iloc[4]['High'] == 1411
    assert df.iloc[4]['Low'] == 1600
    assert df.iloc[4]['Vol'] == 110


def test_vol_ticks_2():

    ig = inputgenerator.InputGenerator(
        'is_diff_higher', '5m', 2, 1, ['Close', 'High'], 'High', output_granularity='10m', scaler=None, diff_percent=0.01)
    df = DataFrame([[1608642300, 1000, 1000, 1100, 900, 80],
                    [1608642600, 1200, 1200, 1200, 1000, 20],
                    [1608642900, 1200, 1200, 1300, 1100, 30],
                    [1608643200, 1300, 1300, 1400, 1200, 60],
                    [1608643500, 1400, 1400, 1301, 1300, 15],
                    [1608643800, 1396, 1396, 1315, 1400, 110],
                    [1608644100, 1600, 1600, 1302, 1500, 100],
                    [1608644400, 1700, 1700, 1411, 1600, 50],
                    [1608644700, 1700, 1700, 1405, 1600, 40],
                    [1608645000, 1700, 1700, 1405, 1600, 20]], columns=['TS', 'Open', 'Close', 'High', 'Low', 'Vol'])
    df.index = df['TS']
    df = df.drop('TS', axis=1)
    df = inputgenerator.to_vol_ticks(df, 100)
    ts, x, y = ig.get_io(df, True)
    assert ts[0] == 1608643500
    assert all(x[0] == [1200, 1400, 1200, 1400])

    assert ts[1] == 1608643800
    assert all(x[1] == [1400, 1396, 1400, 1315])

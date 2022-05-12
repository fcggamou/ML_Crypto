import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import inputgenerator
from pandas import DataFrame
import ml.model
from ml.model import SelectFeatures, LastStep
from tensorflow.keras.layers import Input, Dense, Reshape, Permute, BatchNormalization, Conv1D, LSTM, concatenate
from tensorflow.keras.optimizers import Adam


def test_dummy_consistency_check():

    ig = inputgenerator.InputGeneratorRegression(
        None, '5m', 4, 1, ['Close', 'High', 'Low'], 'Close', output_granularity='5m', scaler=None)
    df = DataFrame([[1608642300, 1000, 1100, 900],
                    [1608642600, 1100, 1200, 1000],
                    [1608642900, 1200, 1300, 1100],
                    [1608643200, 1300, 1400, 1200],
                    [1608643500, 1400, 1500, 1300],
                    [1608643800, 1500, 1600, 1400],
                    [1608644100, 1600, 1700, 1500],
                    [1608644400, 1700, 1800, 1600]], columns=['TS', 'Close', 'High', 'Low'])
    df.set_index(df['TS'], inplace=True)
    ts, x, y = ig.get_io(df, True)

    model = ml.model.DummyModel(ig)
    model.initialize()
    model.train(x, y)
    assert model.consistency_check()


def test_dummy_prediction():
    ig = inputgenerator.InputGeneratorRegression(
        None, '5m', 4, 1, ['Close', 'High', 'Low'], 'Close', output_granularity='5m', scaler=None)
    df = DataFrame([[1608642300, 1000, 1100, 900],
                    [1608642600, 1100, 1200, 1000],
                    [1608642900, 1200, 1300, 1100],
                    [1608643200, 1300, 1400, 1200],
                    [1608643500, 1400, 1500, 1300],
                    [1608643800, 1500, 1600, 1400],
                    [1608644100, 1600, 1700, 1500],
                    [1608644400, 1700, 1800, 999]], columns=['TS', 'Close', 'High', 'Low'])
    df.set_index(df['TS'], inplace=True)

    ts, x, y = ig.get_io(df)
    model = ml.model.DummyModel(ig)
    model.initialize()
    model.train(x, y)

    prediction = model.predict_last(df)
    assert prediction is not None
    assert prediction.timestamp == 1608644400
    assert prediction.target_timestamp == 1608644400 + 60 * 5
    assert prediction.variable == 'Close'
    assert prediction.value == 999


def test_save_load_dummy():
    ig = inputgenerator.InputGeneratorRegression(
        None, '5m', 4, 1, ['Close', 'High', 'Low'], 'Close', output_granularity='5m', scaler=None)
    df = DataFrame([[1608642300, 1000, 1100, 900],
                    [1608642600, 1100, 1200, 1000],
                    [1608642900, 1200, 1300, 1100],
                    [1608643200, 1300, 1400, 1200],
                    [1608643500, 1400, 1500, 1300],
                    [1608643800, 1500, 1600, 1400],
                    [1608644100, 1600, 1700, 1500],
                    [1608644400, 1700, 1800, 999]], columns=['TS', 'Close', 'High', 'Low'])
    df.set_index(df['TS'], inplace=True)

    ts, x, y = ig.get_io(df)
    model = ml.model.DummyModel(ig)
    model.initialize()
    model.train(x, y)

    model.save(os.path.join('data', 'test_model.pkl'))

    model = ml.model.load_model(os.path.join('data', 'test_model.pkl'))
    assert model.consistency_check()


def test_simple_model():
    ig = inputgenerator.InputGeneratorRegression(
        None, '5m', 4, 1, ['Close', 'High', 'Low'], 'Close', output_granularity='5m', scaler=None)
    df = DataFrame([[1608642300, 1000, 1100, 900],
                    [1608642600, 1100, 1200, 1000],
                    [1608642900, 1200, 1300, 1100],
                    [1608643200, 1300, 1400, 1200],
                    [1608643500, 1400, 1500, 1300],
                    [1608643800, 1500, 1600, 1400],
                    [1608644100, 1600, 1700, 1500],
                    [1608644400, 1700, 1800, 999]], columns=['TS', 'Close', 'High', 'Low'])
    df.set_index(df['TS'], inplace=True)

    ts, x, y = ig.get_io(df)
    input = ml.model.Input(shape=(len(ig.features) * ig.iw))
    lstm = ml.model.LSTM(4)
    reshape = ml.model.Reshape((ig.iw, len(ig.features)))
    output = ml.model.Dense(1, activation='linear')
    model = ml.model.SimpleModel(ig, layers=[input, reshape, lstm, output], metrics=['mape'], optimizer='adam',
                                 loss='mse', validation_split=0.1, epochs=5, shuffle=True, batch_size=1)
    model.initialize()
    model.train(x, y)

    prediction = model.predict_last(df)
    assert model.consistency_check
    assert prediction is not None
    assert prediction.timestamp == 1608644400
    assert prediction.target_timestamp == 1608644400 + 60 * 5
    assert prediction.variable == 'Close'


def test_simple_model_predict_all():
    ig = inputgenerator.InputGeneratorRegression(
        None, '5m', 4, 1, ['Close', 'High', 'Low'], 'Close', output_granularity='5m', scaler=None)
    df = DataFrame([[1608642300, 1000, 1100, 900],
                    [1608642600, 1100, 1200, 1000],
                    [1608642900, 1200, 1300, 1100],
                    [1608643200, 1300, 1400, 1200],
                    [1608643500, 1400, 1500, 1300],
                    [1608643800, 1500, 1600, 1400],
                    [1608644100, 1600, 1700, 1500],
                    [1608644400, 1700, 1800, 999]], columns=['TS', 'Close', 'High', 'Low'])
    df.set_index(df['TS'], inplace=True)

    ts, x, y = ig.get_io(df)
    input = ml.model.Input(shape=(len(ig.features) * ig.iw))
    lstm = ml.model.LSTM(4)
    reshape = ml.model.Reshape((ig.iw, len(ig.features)))
    output = ml.model.Dense(1, activation='linear')
    model = ml.model.SimpleModel(ig, layers=[input, reshape, lstm, output], optimizer='adam',
                                 loss='mse', validation_split=0.1, epochs=5, shuffle=True, batch_size=1)
    model.initialize()
    model.train(x, y)

    predictions = model.predict_all(df)
    assert len(predictions) == 5
    assert predictions[0].timestamp == 1608643200
    assert predictions[1].timestamp == 1608643500
    assert predictions[2].timestamp == 1608643800
    assert predictions[3].timestamp == 1608644100
    assert predictions[4].timestamp == 1608644400

    assert predictions[0].target_timestamp == 1608643200 + 60 * 5
    assert predictions[1].target_timestamp == 1608643500 + 60 * 5
    assert predictions[2].target_timestamp == 1608643800 + 60 * 5
    assert predictions[3].target_timestamp == 1608644100 + 60 * 5
    assert predictions[4].target_timestamp == 1608644400 + 60 * 5

    assert all(x.variable == 'Close' for x in predictions)


def test_save_load_simple_model():
    ig = inputgenerator.InputGeneratorRegression(
        None, '5m', 4, 1, ['Close', 'High', 'Low'], 'Close', output_granularity='5m', scaler=None)
    df = DataFrame([[1608642300, 1000, 1100, 900],
                    [1608642600, 1100, 1200, 1000],
                    [1608642900, 1200, 1300, 1100],
                    [1608643200, 1300, 1400, 1200],
                    [1608643500, 1400, 1500, 1300],
                    [1608643800, 1500, 1600, 1400],
                    [1608644100, 1600, 1700, 1500],
                    [1608644400, 1700, 1800, 999]], columns=['TS', 'Close', 'High', 'Low'])
    df.set_index(df['TS'], inplace=True)

    ts, x, y = ig.get_io(df)
    input = ml.model.Input(shape=(len(ig.features) * ig.iw))
    lstm = ml.model.LSTM(4)
    reshape = ml.model.Reshape((ig.iw, len(ig.features)))
    output = ml.model.Dense(1, activation='linear')
    model = ml.model.SimpleModel(ig, layers=[input, reshape, lstm, output], optimizer='adam',
                                 loss='mse', validation_split=0.1, epochs=5, shuffle=True, batch_size=1)
    model.initialize()
    model.train(x, y)

    model.save(os.path.join('data', 'test_model.pkl'))

    model = ml.model.load_model(os.path.join('data', 'test_model.pkl'))
    assert model.consistency_check()


def test_save_load_functional_model():

    df = DataFrame([[1608642300, 1000, 1100, 900],
                    [1608642600, 1100, 1200, 1000],
                    [1608642900, 1200, 1300, 1100],
                    [1608643200, 1300, 1400, 1200],
                    [1608643500, 1400, 1500, 1300],
                    [1608643800, 1500, 1600, 1400],
                    [1608644100, 1600, 1700, 1500],
                    [1608644400, 1700, 1800, 999]], columns=['TS', 'Close', 'High', 'Low'])
    df.set_index(df['TS'], inplace=True)

    ig = inputgenerator.InputGenerator('is_diff_lower', '5m', 3, 1, ['Close', 'High', 'Low'], 'Low', output_granularity='10m', diff_percent=-0.01)
    ts, x, y = ig.get_io(df)

    input = Input(shape=(len(ig.features) * ig.iw))
    reshape = Reshape((len(ig.features), ig.iw))(input)
    permute = Permute((2, 1))(reshape)
    bn = BatchNormalization()(permute)

    sf = SelectFeatures(0, 2)(bn)
    cnn = Conv1D(64, 1, 1, 'causal', activation='relu')(sf)
    lstm = LSTM(64)(cnn)

    lstm2 = LSTM(64)(bn)

    last_step = LastStep()(bn)
    conc = concatenate([last_step, lstm, lstm2])
    output = Dense(1, activation='sigmoid')(conc)

    model = ml.model.KerasModel(ig, input, output, optimizer=Adam(0.001), loss='binary_crossentropy', epochs=2, validation_split=0.5, shuffle=True, batch_size=1, patience=5, enable_consistency_check=True)
    model.initialize()
    print(model.model.summary())
    model.train(x, y)

    model.save(os.path.join('data', 'test_model.pkl'))

    model = ml.model.load_model(os.path.join('data', 'test_model.pkl'))
    assert model.consistency_check()


def test_simple_model_reshape():
    ig = inputgenerator.InputGeneratorRegression(
        None, '5m', 4, 1, ['Close', 'High', 'Low'], 'Close', output_granularity='5m', scaler=None)
    df = DataFrame([[1608642300, 1000, 1100, 900],
                    [1608642600, 1100, 1200, 1000],
                    [1608642900, 1200, 1300, 1100],
                    [1608643200, 1300, 1400, 1200],
                    [1608643500, 1400, 1500, 1300],
                    [1608643800, 1500, 1600, 1400]], columns=['TS', 'Close', 'High', 'Low'])
    df.set_index(df['TS'], inplace=True)

    ts, x, y = ig.get_io(df)
    input = ml.model.Input(shape=(len(ig.features) * ig.iw))
    reshape = ml.model.Reshape((len(ig.features), ig.iw))
    swap_ax = ml.model.Permute((2, 1))
    model = ml.model.SimpleModel(ig, layers=[input, reshape, swap_ax], optimizer='adam',
                                 loss='mse', validation_split=0.1, epochs=1, shuffle=True, batch_size=1)
    model.initialize()
    model.train(x, y)
    ts, x, _ = ig.get_io(df, False)
    predictions = model.model.predict(x)

    assert len(predictions) == 3

    assert predictions.shape == (3, 4, 3)
    assert all(predictions[0][0] == [1000, 1100, 900])
    assert all(predictions[0][1] == [1100, 1200, 1000])
    assert all(predictions[0][2] == [1200, 1300, 1100])
    assert all(predictions[0][3] == [1300, 1400, 1200])

    assert all(predictions[1][0] == [1100, 1200, 1000])
    assert all(predictions[1][1] == [1200, 1300, 1100])
    assert all(predictions[1][2] == [1300, 1400, 1200])
    assert all(predictions[1][3] == [1400, 1500, 1300])

    assert all(predictions[2][0] == [1200, 1300, 1100])
    assert all(predictions[2][1] == [1300, 1400, 1200])
    assert all(predictions[2][2] == [1400, 1500, 1300])
    assert all(predictions[2][3] == [1500, 1600, 1400])


def test_simple_model_reshape_3D():
    ig = inputgenerator.InputGeneratorRegression(
        None, '5m', 4, 1, ['Close', 'High', 'Low'], 'Close', output_granularity='5m', scaler=None)
    df = DataFrame([[1608642300, 1000, 1100, 900],
                    [1608642600, 1100, 1200, 1000],
                    [1608642900, 1200, 1300, 1100],
                    [1608643200, 1300, 1400, 1200],
                    [1608643500, 1400, 1500, 1300],
                    [1608643800, 1500, 1600, 1400]], columns=['TS', 'Close', 'High', 'Low'])
    df.set_index(df['TS'], inplace=True)

    ts, x, y = ig.get_io(df)
    input = ml.model.Input(shape=(len(ig.features) * ig.iw))
    reshape = ml.model.Reshape((len(ig.features), ig.iw))
    swap_ax = ml.model.Permute((2, 1))
    reshape_2 = ml.model.Reshape((ig.iw, len(ig.features), 1))
    model = ml.model.SimpleModel(ig, layers=[input, reshape, swap_ax, reshape_2], optimizer='adam',
                                 loss='mse', validation_split=0.1, epochs=1, shuffle=True, batch_size=1)
    model.initialize()
    model.train(x, y)
    ts, x, _ = ig.get_io(df, False)
    predictions = model.model.predict(x)

    assert len(predictions) == 3
    assert predictions.shape == (3, 4, 3, 1)
    assert all(predictions[0][0] == [[1000], [1100], [900]])
    assert all(predictions[0][1] == [[1100], [1200], [1000]])
    assert all(predictions[0][2] == [[1200], [1300], [1100]])
    assert all(predictions[0][3] == [[1300], [1400], [1200]])

    assert all(predictions[1][0] == [[1100], [1200], [1000]])
    assert all(predictions[1][1] == [[1200], [1300], [1100]])
    assert all(predictions[1][2] == [[1300], [1400], [1200]])
    assert all(predictions[1][3] == [[1400], [1500], [1300]])

    assert all(predictions[2][0] == [[1200], [1300], [1100]])
    assert all(predictions[2][1] == [[1300], [1400], [1200]])
    assert all(predictions[2][2] == [[1400], [1500], [1300]])
    assert all(predictions[2][3] == [[1500], [1600], [1400]])


def test_simple_model_last_step():
    ig = inputgenerator.InputGeneratorRegression(
        None, '5m', 4, 1, ['Close', 'High', 'Low'], 'Close', output_granularity='5m', scaler=None)
    df = DataFrame([[1608642300, 1000, 1100, 900],
                    [1608642600, 1100, 1200, 1000],
                    [1608642900, 1200, 1300, 1100],
                    [1608643200, 1300, 1400, 1200],
                    [1608643500, 1400, 1500, 1300],
                    [1608643800, 1500, 1600, 1400]], columns=['TS', 'Close', 'High', 'Low'])
    df.set_index(df['TS'], inplace=True)

    ts, x, y = ig.get_io(df)
    input = ml.model.Input(shape=(len(ig.features) * ig.iw))
    reshape = ml.model.Reshape((len(ig.features), ig.iw))
    swap_ax = ml.model.Permute((2, 1))
    last_step = LastStep()
    model = ml.model.SimpleModel(ig, layers=[input, reshape, swap_ax, last_step], optimizer='adam',
                                 loss='mse', validation_split=0.1, epochs=1, shuffle=True, batch_size=1)
    model.initialize()
    model.train(x, y)
    pred = model.model.predict(x)
    assert all(pred[0] == [1300, 1400, 1200])
    assert all(pred[1] == [1400, 1500, 1300])


def test_simple_model_filter_features():
    ig = inputgenerator.InputGeneratorRegression(
        None, '5m', 4, 1, ['Close', 'High', 'Low', 'TS'], 'Close', output_granularity='5m', scaler=None)
    df = DataFrame([[1608642300, 1000, 1100, 900],
                    [1608642600, 1100, 1200, 1000],
                    [1608642900, 1200, 1300, 1100],
                    [1608643200, 1300, 1400, 1200],
                    [1608643500, 1400, 1500, 1300],
                    [1608643800, 1500, 1600, 1400]], columns=['TS', 'Close', 'High', 'Low'])
    df.set_index(df['TS'], inplace=True)

    ts, x, y = ig.get_io(df)
    input = ml.model.Input(shape=(len(ig.features) * ig.iw))
    reshape = ml.model.Reshape((len(ig.features), ig.iw))
    swap_ax = ml.model.Permute((2, 1))
    select_features = SelectFeatures(0, 2)
    model = ml.model.SimpleModel(ig, layers=[input, reshape, swap_ax, select_features], optimizer='adam',
                                 loss='mse', validation_split=0.1, epochs=1, shuffle=True, batch_size=1)
    model.initialize()
    model.train(x, y)
    pred = model.model.predict(x)
    assert all(pred[0][0] == [1000, 1100])
    assert all(pred[0][1] == [1100, 1200])
    assert all(pred[0][2] == [1200, 1300])
    assert all(pred[0][3] == [1300, 1400])

    assert all(pred[1][0] == [1100, 1200])
    assert all(pred[1][1] == [1200, 1300])
    assert all(pred[1][2] == [1300, 1400])
    assert all(pred[1][3] == [1400, 1500])


def test_model_static_features():
    ig = inputgenerator.InputGeneratorRegression(
        None, '5m', 4, 1, ['Close', 'High', 'Low'], 'Close', output_granularity='5m', scaler=None, static_features=['Low'])
    df = DataFrame([[1608642300, 1000, 1100, 900],
                    [1608642600, 1100, 1200, 1000],
                    [1608642900, 1200, 1300, 1100],
                    [1608643200, 1300, 1400, 1200],
                    [1608643500, 1400, 1500, 1300],
                    [1608643800, 1500, 1600, 1400]], columns=['TS', 'Close', 'High', 'Low'])
    df.set_index(df['TS'], inplace=True)
    ts, x, y = ig.get_io(df)
    input1 = ml.model.Input(shape=(len(ig.features) * ig.iw))
    input2 = ml.model.Input(shape=(len(ig.static_features)))
    reshape = ml.model.Reshape((len(ig.features), ig.iw))(input1)
    swap_ax = ml.model.Permute((2, 1))(reshape)
    flatten = ml.model.Flatten()(swap_ax)

    conc = ml.model.concatenate([flatten, input2])

    model = ml.model.KerasModel(ig, inputs=[input1, input2], outputs=[conc], optimizer='adam',
                                loss='mse', validation_split=0.1, epochs=1, shuffle=True, batch_size=1)
    model.initialize()
    model.train(x, y)
    pred = model.model.predict(x)
    assert pred[0][0] == 1000
    assert pred[0][3] == 1100
    assert pred[0][6] == 1200
    assert pred[0][9] == 1300
    assert pred[0][1] == 1100
    assert pred[0][4] == 1200
    assert pred[0][7] == 1300
    assert pred[0][10] == 1400
    assert pred[0][2] == 900
    assert pred[0][5] == 1000
    assert pred[0][8] == 1100
    assert pred[0][11] == 1200
    assert pred[0][12] == 1200

    assert pred[1][0] == 1100
    assert pred[1][3] == 1200
    assert pred[1][6] == 1300
    assert pred[1][9] == 1400
    assert pred[1][1] == 1200
    assert pred[1][4] == 1300
    assert pred[1][7] == 1400
    assert pred[1][10] == 1500
    assert pred[1][2] == 1000
    assert pred[1][5] == 1100
    assert pred[1][8] == 1200
    assert pred[1][11] == 1300
    assert pred[1][12] == 1300

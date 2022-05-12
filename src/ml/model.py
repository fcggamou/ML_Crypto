from tensorflow.keras.layers import *  # noqa
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential  # noqa
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from inputgenerator import InputGenerator
from tensorflow.keras.models import load_model as lm
from abc import ABC, abstractmethod
from ml.prediction import Prediction
import file_utils
import numpy as np
import tensorflow
import os
from sklearn.metrics import roc_curve
import pandas as pd
import random


def load_all_models(path):
    models = []

    for file in sorted(os.listdir(path)):
        if (os.path.isfile(os.path.join(path, file)) and file.endswith('pkl')):
            models.append(load_model(os.path.join(path, file)))
    return models


def load_model(filename):

    print("loading model: %s ..." % filename)
    model = file_utils.load_from_file(filename)

    if model.model is None and not isinstance(model, DummyModel):
        model.model = lm(str(filename) + ".model.h5", custom_objects={'SelectFeatures': SelectFeatures, 'LastStep': LastStep})
        print(model.model.summary())

        print("checking model consistency ...")
        if model._enable_consistency_check and not model.consistency_check():
            raise Exception("consistency check failed for model %s" % filename)

    print("loaded!")

    return model


def get_roc(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted, drop_intermediate=True)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tpr': pd.Series(tpr, index=i), 'fpr': pd.Series(fpr, index=i), 'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    total_positives = sum(target)
    total_negatives = len(target) - total_positives
    roc['profits'] = roc['tpr'] * total_positives - roc['fpr'] * total_negatives
    return roc


class SelectFeatures(tensorflow.keras.layers.Layer):
    def __init__(self, i, j, **kwargs):
        super().__init__()
        self.i = i
        self.j = j

    def call(self, x):
        return x[:, :, self.i: self.j]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'i': self.i,
            'j': self.j})
        return config


class LastStep(tensorflow.keras.layers.Layer):
    def call(self, x):
        return x[:, -1]


class BaseModel(ABC):

    def __init__(self, ig: InputGenerator):
        self.ig = ig
        self.history = None
        self.roc = None
        self.optimizer = None
        self.model = None
        self._override_model_loading = False

    def predict_last(self, data):
        ts, x = self.ig.get_input_for_prediction(data)
        y = self._predict(np.expand_dims(x, 0))[0]
        return Prediction(ts, ts + self.ig.output_granularity_seconds(), self.ig.target_variable, y[0], self.ig.diff_percent, self.roc)

    def predict_all(self, data):
        ts, x, _ = self.ig.get_io(data, False)
        predictions = self._predict(x)
        return [Prediction(t, t + self.ig.output_granularity_seconds(), self.ig.target_variable, y[0], self.ig.diff_percent, self.roc) for t, y in zip(ts, predictions)]

    def _predict(self, x):
        pred = self.model.predict(x)
        return self.ig.scalerY.inverse_transform(pred)

    @abstractmethod
    def train(self, x, y):
        pass

    def initialize(self):
        if self.model is None:
            self.model = self._create_model()
        else:
            _model = self.model
            self.model = tensorflow.keras.models.clone_model(_model)
            del _model

        if self.model is not None:
            self.model.compile(loss=self.loss, optimizer=self.get_optimizer(self._optimizer, self.lr), metrics=self.metrics)

    def save(self, filename):

        model = self.model
        history = self.history
        self.model = None
        self.history = None
        self.optimizer = None
        file_utils.dump_to_file(self, filename)

        if model is not None:
            self.model = model
            self.model.save('%s.model.h5' % filename)

        self.history = history

    def _after_train(self, x, y):
        self.model = lm('best_model.h5', custom_objects={'SelectFeatures': SelectFeatures, 'LastStep': LastStep})
        self.callbacks = None
        self._generate_consistency_data(x)
        if self.ig.diff_percent is not None and self.ig.output_dim == 1:
            self.roc = get_roc(y, self._predict(x))[::10]

    def _generate_consistency_data(self, x):
        if self._enable_consistency_check:
            self._check_data = x[-50:]
            self._check_result = self._predict(x[-50:])

    def consistency_check(self):
        predictions = self._predict(self._check_data)
        for pred, actual in zip(predictions, self._check_result):
            if abs(pred[0] - actual[0]) > 0.01:
                return False
        return True

    def _create_model(self):
        pass


class DummyModel(BaseModel):

    def __init__(self, ig: InputGenerator):
        super().__init__(ig)
        self._enable_consistency_check = True
        self._override_model_loading = True

    def _predict(self, x):
        if self.ig.diff_percent is None:
            return [[y[-1]] for y in x]
            # return np.expand_dims([y[-1] for y in x], 0)
        else:
            return np.expand_dims([random.random() for y in x], 0)

    def train(self, x, y):
        self._generate_consistency_data(x)

    def initialize(self):
        pass


class SimpleModel(BaseModel):

    def __init__(self, ig: InputGenerator, layers: list, optimizer, loss, lr=0.001, patience=5, metrics=None, enable_consistency_check=True, rlr_factor=0, **train_args):
        super().__init__(ig)
        self.layers = layers
        self.lr = lr
        self._optimizer = optimizer
        self.optimizer = self.get_optimizer(optimizer, lr)
        self.loss = loss
        self.train_args = train_args
        self.metrics = metrics
        self.history = None
        self._patience = patience
        self._enable_consistency_check = enable_consistency_check
        self._rlr_factor = rlr_factor

    def train(self, x, y):

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=self._patience)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        self.callbacks = [es, mc]
        if self._rlr_factor > 0:
            rlr = ReduceLROnPlateau(factor=0.5, patience=3)
            self.callbacks.append(rlr)

        self.history = self.model.fit(x, y, callbacks=self.callbacks, **self.train_args)
        super()._after_train(x, y)

    def _create_model(self):
        model = Sequential()
        for layer in self.layers:
            model.add(layer)
        return model

    def get_optimizer(self, optimizer, lr):
        if optimizer == 'adam':
            return Adam(lr)
        else:
            return optimizer


class KerasModel(SimpleModel):

    def __init__(self, ig: InputGenerator, inputs, outputs, optimizer, loss, lr=0.001, patience=5, metrics=None, enable_consistency_check=True, rlr_factor=0, **train_args):
        super().__init__(ig, [], optimizer, loss, lr, patience, metrics, enable_consistency_check, rlr_factor, **train_args)
        self.model = None
        self._inputs = inputs
        self._outputs = outputs

    def _create_model(self):
        return Model(inputs=self._inputs, outputs=self._outputs)

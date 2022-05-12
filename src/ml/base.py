from IPython.display import display
from tensorflow.keras.models import load_model

import helpers
import file_utils

from db.sqlalchemy.model.models_history import ModelsHistory


class Predictor(object):

    def __init__(self, ig):
        self.name = ""
        self.trained = False
        self.initialized = False
        self.roc = None
        self.keras = False
        self.balance_weights = False
        self._check_data = None
        self._check_result = None
        self.trained = False
        self.features = len(ig.input_variables)
        self.iw = ig.iw
        self.ig = ig
        self.id = None

        self.model = None
        self.dense_input = None
        self.lstm = None
        self.dense_output = None
        self.cnn = None
        self.bidirectional = None
        self.gru = None
        self.bias = None
        self.timesteps = None
        self.batch_norm = None
        self.batch_size = None
        self.epochs = None
        self.optimizer = None
        self.activation = None
        self.validation_split = None
        self.lr = None
        self.shuffle_validation = None

    def train(self, x, y):
        if not self.initialized:
            raise Exception("Cannot train, initialize the model first.")
        self.class_weights = None

        if self.balance_weights:
            positive_ratio = (len(y) / sum(y))[0]
            self.class_weights = {0: 1, 1: positive_ratio}

    def initialize(self):
        self.initialized = True

    def predict(self, x):
        if not self.trained:
            raise Exception("Cannot predict, train the model first.")

    def save(self, filename, db=None, name=None):
        import ml.ml_utils
        if name:
            self.name = name
        # Do not include the keras model object in the pkl file.
        model = self.model
        self.model = None

        self.path = filename
        if db:
            model_history: ModelsHistory = ml.ml_utils.ml_model_to_db_model(
                self)
            id: int = db.add_model(model_history)
            self.id = id
        file_utils.dump_to_file(self, filename)

        if model is not None:
            self.model = model
            self.model.save_weights('%s.model.h5' % filename)

    def after_train(self, x, y):

        self.model = load_model('best_model.h5')

        self.callbacks = None

        # Compress huge ROC curve, we don't need every single datapoint
        self.roc = helpers.get_roc(y, self.model.predict(x))[::10]
        display(self.model.summary())

        self._check_data = x[-100:]
        self._check_result = self.model.predict(x[-100:])

    def consistency_check(self):
        return (self._check_data is None or all(helpers.nearly_equals(x, y) for x, y in zip(self.model.predict(self._check_data), self._check_result)))

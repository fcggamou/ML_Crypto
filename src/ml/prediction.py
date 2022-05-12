class Prediction():

    def __init__(self, timestamp: int, target_timestamp: int, variable: str, value: float, diff_percentage=None, roc=None):
        self.timestamp = int(timestamp)
        self.target_timestamp = int(target_timestamp)
        self.variable = variable
        self.value = value
        self.diff_percentage = diff_percentage
        if roc is not None:
            self.fpr = roc.iloc[(roc['threshold'] - value).abs().argsort()[0]]['fpr']
        else:
            self.fpr = None

    def __str__(self):
        return "TS: {}. Horizon: {}. Variable: {}. Value: {}".format(self.timestamp, self.target_timestamp, self.variable, round(self.value, 2))


from ml.model import BaseModel, SimpleModel, Conv1D, Conv2D, SeparableConv1D, KerasModel
import numpy as np
from IPython.display import display, HTML


def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", "<br>")))


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def kr_to_str(kr):
    return str(kr)


def layer_to_str(layer):
    layer_class = type(layer)
    class_name = layer_class.__name__
    if hasattr(layer, 'units'):
        name = "{} Units:{}".format(class_name, layer.units)
        if hasattr(layer, 'dropout') and layer.dropout > 0:
            name = name + " Dropout:{}".format(layer.dropout)
        if layer.kernel_regularizer is not None:
            name = name + " Kr:{}".format(kr_to_str(layer.kernel_regularizer))
    elif layer_class == Conv1D or layer_class == Conv2D or layer_class == SeparableConv1D:
        name = "{} f:{} ks:{} p:{} s:{} d:{} g:{}".format(
            class_name, layer.filters, layer.kernel_size, layer.padding, layer.strides, layer.dilation_rate, layer.groups)
    else:
        name = class_name
    return name


def model_to_str(model: BaseModel):
    model_class = type(model)
    class_name = model_class.__name__
    if model_class == SimpleModel or model_class == KerasModel:
        bs = str(model.train_args['batch_size'])
        vs = str(model.train_args['validation_split'])
        return "{}\nBS:{} Opti:{} VS:{}\n{}\n{}".format(class_name, str(bs), optimizer_to_str(model), str(vs), '\n'.join([layer_to_str(x) for x in model.model.layers]), ig_to_str(model.ig))
    else:
        return class_name


def ig_to_str(ig):
    return "{}-{}-{}".format(ig.granularity, str(ig.iw), ig.features)


def optimizer_to_str(model):
    optimizer = model.optimizer
    optimizer_class = type(optimizer)
    class_name = optimizer_class.__name__
    name = optimizer if optimizer_class == str else class_name
    if hasattr(model, "lr"):
        name = name + "\nlr: {}".format(str(model.lr))
    return name

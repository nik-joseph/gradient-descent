import numpy as np
from tqdm import trange
from .functions import Function
from numba import cuda


def predict(x, w, b, function):
    """
    :param x: input row
    :param w: model weights
    :param b: bias
    :param function: an class of type Function having activation function and its derivative
    :return: prediction y hat
    """
    x_ar, w_ar = np.array(x, dtype=float), np.array(w, dtype=float)
    if x_ar.shape != w_ar.shape:
        raise Exception("Input and weight not same size in prediction")
    return function.function(b + np.dot(x_ar, w_ar.transpose()))


def stochastic_coordinate(train_data_X, train_data_y,
                          learning_rate=None, epochs=None, function=Function, func_args=None):
    """
    :param train_data_X: input data attributes
    :param train_data_y: input data y hat
    :param learning_rate: learning rate
    :param epochs: number of iterations to train
    :param function: an class of type Function having activation function and its derivative
    :param func_args: holds an arguments required for weight and bias update
    :return: weight and bias of the model w, b
    """
    if func_args is None:
        func_args = {}

    # Initialize weight and bias to 0
    w, b = np.zeros(len(train_data_X[0])), 0

    # Raise required errors
    if not learning_rate:
        raise Exception("Learning rate not set!")
    if not epochs:
        raise Exception("Number of epochs not set!")
    if not issubclass(function, Function):
        raise Exception("Given function not instance of Function!")

    learning_rate = learning_rate / (w.shape[0])

    # Start epoch
    for _ in trange(epochs):
        w, b = function.cuda_train(
            train_data_X, train_data_y, w, b, learning_rate, **func_args
        )
        cuda.synchronize()

    return w, b

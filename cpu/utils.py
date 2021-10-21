import numpy as np
from tqdm import trange
from .functions import Function


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


def stochastic_gradient(train_data_X, train_data_y, learning_rate=None, epochs=None, function=Function, func_args=None):
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

    # Start epoch
    for _ in trange(epochs):
        for input_X, input_y in zip(train_data_X, train_data_y):
            # Compute prediction
            y_hat = predict(input_X, w, b, function=function)

            # Update weight and bias on learning rate and calculated error
            w = w - learning_rate * function.partial_derivative_w(input_X, input_y, y_hat, w, **func_args)
            b = b - learning_rate * function.partial_derivative_b(input_y, y_hat)
    return w, b


def coordinate_gradient(train_data_X, train_data_y, learning_rate=None, epochs=None, function=Function, func_args=None):
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

    # Start epoch
    for _ in trange(epochs):
        for input_X, input_y in zip(train_data_X, train_data_y):
            # Iterate on number of weights
            for index in range(w.size):
                # Compute prediction
                y_hat = predict(input_X, w, b, function=function)

                # Update weight on learning rate and calculate error for current weight based on index
                w[index] = w[index] - learning_rate * function.partial_derivative_w(
                    input_X[index], input_y, y_hat, w[index], **func_args
                )

            # Compute prediction
            y_hat = predict(input_X, w, b, function=function)

            # Update bias
            b = b - learning_rate * function.partial_derivative_b(input_y, y_hat)
    return w, b
